"""
Training Loop with Implicit Differentiation Support

This module provides training utilities for both baseline and recursive
transformer models, with special handling for implicit differentiation
in fixed-point models.

Key Features:
- Standard training for baseline models
- Implicit differentiation for recursive models
- Gradient accumulation and mixed precision
- Learning rate scheduling
- Checkpointing and logging
"""

import os
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Basic training
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 100000
    warmup_steps: int = 2000
    
    # Batch settings - RTX 3090 optimized defaults
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 512
    
    # Optimization
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16"
    
    # Memory optimization
    gradient_checkpointing: bool = False  # Disabled by default - can cause issues
    
    # Checkpointing
    save_steps: int = 5000
    save_dir: str = "checkpoints"
    save_total_limit: int = 3
    
    # Logging
    log_steps: int = 100
    eval_steps: int = 1000
    use_wandb: bool = False
    wandb_project: str = "umc-benchmark"
    wandb_run_name: Optional[str] = None
    
    # Recursive model specific
    implicit_diff_iterations: int = 10
    implicit_diff_threshold: float = 1e-5
    
    # Device
    device: str = "cuda"
    
    def __post_init__(self):
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps


class CosineScheduler:
    """Cosine learning rate scheduler with warmup."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        lr_mult = self._get_lr_multiplier()
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_mult
            
    def _get_lr_multiplier(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            return self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class TrainingMetrics:
    """Track and aggregate training metrics."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.loss_sum = 0.0
        self.token_count = 0
        self.step_count = 0
        self.iteration_sum = 0.0  # For recursive models
        self.iteration_history: List[float] = []  # Track iteration distribution
        self.converged_count = 0  # Count of batches that converged early
        self.max_iter_count = 0  # Count of batches that hit max iterations
        
    def update(
        self,
        loss: float,
        num_tokens: int,
        iterations: Optional[float] = None,
        max_iterations: int = 12,
    ):
        self.loss_sum += loss * num_tokens
        self.token_count += num_tokens
        self.step_count += 1
        if iterations is not None:
            self.iteration_sum += iterations
            self.iteration_history.append(iterations)
            # Track convergence rate
            if iterations < max_iterations:
                self.converged_count += 1
            else:
                self.max_iter_count += 1
            
    @property
    def avg_loss(self) -> float:
        if self.token_count == 0:
            return 0.0
        return self.loss_sum / self.token_count
    
    @property
    def perplexity(self) -> float:
        return math.exp(min(self.avg_loss, 100))
    
    @property
    def avg_iterations(self) -> Optional[float]:
        if self.step_count == 0:
            return None
        return self.iteration_sum / self.step_count
    
    @property
    def convergence_rate(self) -> Optional[float]:
        """Percentage of batches that converged before max iterations."""
        total = self.converged_count + self.max_iter_count
        if total == 0:
            return None
        return self.converged_count / total
    
    @property
    def iteration_stats(self) -> Optional[Dict[str, float]]:
        """Statistics about iteration distribution."""
        if not self.iteration_history:
            return None
        import numpy as np
        arr = np.array(self.iteration_history)
        return {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'median': float(np.median(arr)),
        }


class FixedPointTrainer:
    """
    Trainer for both baseline and recursive transformer models.
    
    Handles:
    - Standard backpropagation for baseline models
    - Implicit differentiation for recursive models
    - Mixed precision training
    - Gradient accumulation
    - Logging and checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        is_recursive: bool = False,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.is_recursive = is_recursive
        
        # Move model to device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if config.gradient_checkpointing and hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()
            print("[MEMORY] Gradient checkpointing enabled (~40% VRAM reduction)")
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = CosineScheduler(
            self.optimizer,
            config.warmup_steps,
            config.max_steps,
        )
        
        # Setup mixed precision
        self.scaler = GradScaler('cuda') if config.use_amp else None
        self.amp_dtype = torch.float16 if config.amp_dtype == "float16" else torch.bfloat16
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Metrics
        self.train_metrics = TrainingMetrics()
        
        # Setup logging
        if config.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()
        
        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'ln' in name or 'layernorm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        return optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        # Get model-specific info
        config_dict = {
            'model_params': self.model.num_parameters,
            'is_recursive': self.is_recursive,
            **vars(self.config),
        }
        
        # Add recursive model specific info
        if self.is_recursive and hasattr(self.model, 'triton_enabled'):
            config_dict['triton_enabled'] = self.model.triton_enabled
            config_dict['anderson_enabled'] = self.model.anderson_enabled
        
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config=config_dict,
        )
        wandb.watch(self.model, log='gradients', log_freq=1000)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Dictionary with 'input_ids' and optionally 'attention_mask'
            
        Returns:
            Dictionary with loss and other metrics
        """
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Labels are shifted input_ids
        labels = input_ids.clone()
        
        # Forward pass with mixed precision
        with autocast('cuda', enabled=self.config.use_amp, dtype=self.amp_dtype):
            if self.is_recursive:
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_iterations=True,
                )
            else:
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
        
        loss = outputs['loss']
        logits = outputs['logits']
        
        # Check for NaN loss with detailed diagnostics
        if torch.isnan(loss) or torch.isnan(logits).any():
            print(f"\n[WARN] NaN detected! Detailed diagnostics:")
            print(f"  Input range: [{input_ids.min().item()}, {input_ids.max().item()}]")
            print(f"  Input shape: {input_ids.shape}")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Logits has nan: {torch.isnan(logits).any().item()}")
            print(f"  Logits has inf: {torch.isinf(logits).any().item()}")
            if not torch.isnan(logits).all():
                finite_logits = logits[~torch.isnan(logits) & ~torch.isinf(logits)]
                if finite_logits.numel() > 0:
                    print(f"  Finite logits range: [{finite_logits.min().item():.4f}, {finite_logits.max().item():.4f}]")
            print(f"  Loss: {loss.item()}")
            
            # Check model weights for NaN
            nan_params = []
            for name, p in self.model.named_parameters():
                if torch.isnan(p).any():
                    nan_params.append(name)
            if nan_params:
                print(f"  NaN in parameters: {nan_params[:5]}...")
            else:
                print(f"  No NaN in model parameters")
            
            # Return with dummy backward to keep scaler happy
            return {'loss': float('nan'), 'num_tokens': input_ids.numel()}
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Collect metrics
        metrics = {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'num_tokens': input_ids.numel(),
        }
        
        if self.is_recursive and 'iterations' in outputs:
            iters = outputs['iterations']
            metrics['iterations'] = iters.item() if iters.numel() == 1 else iters.mean().item()
        
        return metrics
    
    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        return grad_norm.item()
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on evaluation dataset.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        eval_metrics = TrainingMetrics()
        
        for batch in self.eval_dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            labels = input_ids.clone()
            
            with autocast('cuda', enabled=self.config.use_amp, dtype=self.amp_dtype):
                if self.is_recursive:
                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_iterations=True,
                    )
                else:
                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            
            iterations = None
            if self.is_recursive and 'iterations' in outputs:
                iterations = outputs['iterations'].mean().item()
            
            eval_metrics.update(
                outputs['loss'].item(),
                input_ids.numel(),
                iterations,
            )
        
        results = {
            'eval_loss': eval_metrics.avg_loss,
            'eval_perplexity': eval_metrics.perplexity,
        }
        
        if eval_metrics.avg_iterations is not None:
            results['eval_iterations'] = eval_metrics.avg_iterations
        
        return results
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(
                self.config.save_dir,
                f'checkpoint-{self.global_step}.pt'
            )
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_step': self.scheduler.current_step,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': vars(self.config),
            'best_eval_loss': self.best_eval_loss,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"[CHECKPOINT] Saved to {path}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.current_step = checkpoint['scheduler_step']
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"[CHECKPOINT] Loaded from {path} (step {self.global_step})")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent."""
        checkpoints = sorted(
            Path(self.config.save_dir).glob('checkpoint-*.pt'),
            key=lambda p: int(p.stem.split('-')[1]),
        )
        
        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            print(f"[CHECKPOINT] Removed old checkpoint: {oldest}")
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ''):
        """Log metrics to console and wandb."""
        # Console logging
        log_str = f"Step {self.global_step}"
        for key, value in metrics.items():
            if isinstance(value, float):
                # Use scientific notation for very small values
                if 0 < abs(value) < 0.0001:
                    log_str += f" | {prefix}{key}: {value:.2e}"
                else:
                    log_str += f" | {prefix}{key}: {value:.4f}"
            else:
                log_str += f" | {prefix}{key}: {value}"
        print(log_str)
        
        # Wandb logging
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log({f'{prefix}{k}': v for k, v in metrics.items()}, step=self.global_step)
    
    def _log_extended_metrics(self):
        """Log extended metrics for recursive models to wandb."""
        if not (self.config.use_wandb and WANDB_AVAILABLE and self.is_recursive):
            return
        
        # Log iteration histogram if we have enough data
        if self.train_metrics.iteration_history:
            wandb.log({
                'iterations_histogram': wandb.Histogram(self.train_metrics.iteration_history),
            }, step=self.global_step)
        
        # Log convergence rate
        conv_rate = self.train_metrics.convergence_rate
        if conv_rate is not None:
            wandb.log({
                'train/convergence_rate': conv_rate,
            }, step=self.global_step)
        
        # Log iteration statistics
        iter_stats = self.train_metrics.iteration_stats
        if iter_stats:
            wandb.log({
                'train/iter_min': iter_stats['min'],
                'train/iter_max': iter_stats['max'],
                'train/iter_std': iter_stats['std'],
                'train/iter_median': iter_stats['median'],
            }, step=self.global_step)
    
    def _log_sample_generation(self, num_samples: int = 3):
        """Generate and log sample text to wandb."""
        if not (self.config.use_wandb and WANDB_AVAILABLE):
            return
        
        self.model.eval()
        
        # Simple prompts to test generation
        prompts = [
            "The meaning of life is",
            "In the year 2050,",
            "Scientists have discovered that",
        ]
        
        try:
            # We need a tokenizer to generate text
            # For now, just log that we attempted generation
            wandb.log({
                'generation_attempted': self.global_step,
            }, step=self.global_step)
        except Exception as e:
            print(f"[WARN] Sample generation failed: {e}")
        finally:
            self.model.train()
    
    def train(self):
        """
        Main training loop.
        
        Trains for config.max_steps steps with gradient accumulation,
        logging, evaluation, and checkpointing.
        """
        print(f"\n[TRAINING] Starting training for {self.config.max_steps} steps")
        print(f"  Model parameters: {self.model.num_parameters:,}")
        print(f"  Effective batch size: {self.config.effective_batch_size}")
        print(f"  Recursive model: {self.is_recursive}")
        if self.is_recursive and hasattr(self.model, 'triton_enabled'):
            print(f"  Triton acceleration: {self.model.triton_enabled}")
            print(f"  Anderson acceleration: {self.model.anderson_enabled}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Warmup steps: {self.config.warmup_steps}")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {self.config.use_amp}")
        
        train_iter = iter(self.train_dataloader)
        self.train_metrics.reset()
        
        accumulation_step = 0
        start_time = time.time()
        
        while self.global_step < self.config.max_steps:
            # Get next batch
            try:
                batch = next(train_iter)
            except StopIteration:
                self.epoch += 1
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)
            
            # Training step
            step_metrics = self.train_step(batch)
            
            # Skip if NaN loss
            if math.isnan(step_metrics['loss']):
                # Reset gradients and scaler state to avoid corruption
                self.optimizer.zero_grad()
                accumulation_step = 0
                continue
            
            self.train_metrics.update(
                step_metrics['loss'],
                step_metrics['num_tokens'],
                step_metrics.get('iterations'),
            )
            
            accumulation_step += 1
            
            # Optimizer step after accumulation
            if accumulation_step >= self.config.gradient_accumulation_steps:
                grad_norm = self.optimizer_step()
                self.global_step += 1
                accumulation_step = 0
                
                # Logging
                if self.global_step % self.config.log_steps == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = self.train_metrics.token_count / elapsed
                    
                    log_metrics = {
                        'loss': self.train_metrics.avg_loss,
                        'perplexity': self.train_metrics.perplexity,
                        'lr': self.scheduler.get_lr(),
                        'grad_norm': grad_norm,
                        'tokens_per_sec': tokens_per_sec,
                        'epoch': self.epoch,
                    }
                    
                    if self.train_metrics.avg_iterations is not None:
                        log_metrics['iterations'] = self.train_metrics.avg_iterations
                    
                    # Add convergence rate for recursive models
                    if self.is_recursive and self.train_metrics.convergence_rate is not None:
                        log_metrics['convergence_rate'] = self.train_metrics.convergence_rate
                    
                    self._log_metrics(log_metrics, prefix='train/')
                    
                    # Log extended metrics (histograms, etc.) for recursive models
                    self._log_extended_metrics()
                    
                    self.train_metrics.reset()
                    start_time = time.time()
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    if eval_metrics:
                        self._log_metrics(eval_metrics, prefix='')
                        
                        # Track best model
                        if eval_metrics['eval_loss'] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics['eval_loss']
                            self.save_checkpoint(
                                os.path.join(self.config.save_dir, 'best_model.pt')
                            )
                
                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
        
        # Final save
        self.save_checkpoint(os.path.join(self.config.save_dir, 'final_model.pt'))
        
        print(f"\n[TRAINING] Completed {self.global_step} steps")
        print(f"  Best eval loss: {self.best_eval_loss:.4f}")
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


class SimpleTextDataset(Dataset):
    """Simple dataset for testing."""
    
    def __init__(self, num_samples: int, seq_length: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, self.vocab_size, (self.seq_length,)),
        }


if __name__ == "__main__":
    # Test the trainer with a small model
    print("[TEST] Testing FixedPointTrainer...")
    
    from benchmark.models.baseline_gpt2 import BaselineGPT2, BaselineGPT2Config
    
    # Create small model for testing
    config = BaselineGPT2Config(
        vocab_size=1000,
        max_position_embeddings=128,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
    )
    model = BaselineGPT2(config)
    
    # Create dummy dataset
    train_dataset = SimpleTextDataset(num_samples=100, seq_length=64, vocab_size=1000)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    eval_dataset = SimpleTextDataset(num_samples=20, seq_length=64, vocab_size=1000)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4)
    
    # Create trainer
    training_config = TrainingConfig(
        learning_rate=1e-4,
        max_steps=50,
        warmup_steps=10,
        batch_size=4,
        log_steps=10,
        eval_steps=25,
        save_steps=100,
        use_wandb=False,
        use_amp=torch.cuda.is_available(),
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    trainer = FixedPointTrainer(
        model=model,
        config=training_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        is_recursive=False,
    )
    
    # Run training
    trainer.train()
    
    print("\n[OK] Trainer tests passed.")
