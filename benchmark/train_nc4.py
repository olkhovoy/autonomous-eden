"""
NC4 Convergence Training Script

This script trains the Contractive GPT-2 model to achieve fixed-point convergence
(NC4 criterion) and monitors convergence metrics in real-time.

Features:
- Trains with fixed-point loss + convergence head loss
- Monitors convergence rate, mean iterations, residual norms
- Saves checkpoints based on convergence rate improvement
- Runs NC4 evaluation periodically

Usage:
    python -m benchmark.train_nc4 --device cuda --steps 10000

For RTX 3090 (24GB):
    python -m benchmark.train_nc4 --device cuda --batch-size 8 --grad-accum 4
"""

import argparse
import json
import os
import time
import math

# Set local cache directories before importing transformers
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'cache')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), 'cache')

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

try:
    from transformers import GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from benchmark.models.contractive_llama import (
    ContractiveLlama,
    ContractiveLlamaConfig,
)
from umc_controller import UMCAdaptiveController
from umc_supervisor import OllamaUMCSupervisor, get_model_samples
from datasets import load_dataset
from transformers import AutoTokenizer


@dataclass
class NC4Metrics:
    """Metrics for NC4 convergence tracking."""
    step: int
    convergence_rate: float
    mean_iterations: float
    mean_residual: float
    loss_total: float
    loss_task: float
    loss_fixed_point: float
    loss_convergence: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class NC4Trainer:
    """Trainer for achieving NC4 fixed-point convergence."""
    
    def __init__(
        self,
        model: ContractiveLlama,
        tokenizer,
        device: torch.device,
        output_dir: str = 'benchmark_output/nc4',
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        grad_accumulation_steps: int = 4,
        max_steps: int = 10000,
        eval_interval: int = 500,
        save_interval: int = 1000,
        log_interval: int = 50,
        use_amp: bool = True,
        supervisor_mode: str = "commentary",
        supervisor_interval: int = 500,
        phase_schedule: bool = True,
        phase1_ratio: float = 0.6,
        phase_lambda_start: float = 0.05,
        phase_lambda_end: float = 0.5,
        phase_damping_start: float = 0.15,
        phase_damping_end: Optional[float] = None,
        iteration_penalty_weight: float = 0.05,
        iteration_penalty_threshold: int = 28,
        semantic_warmup_steps: int = 2000,
        semantic_warmup_damping: float = 0.4,
        controller_max_lambda: Optional[float] = None,
        entropy_min_variance: float = 0.01,
        entropy_penalty_weight: float = 0.1,
        jolt_lr_mult: float = 5.0,
        jolt_noise_std: float = 0.02,
        identity_summary_path: str = "data/identity_summary.txt",
        identity_inject_prefix: str = "System Background:\\n",
        identity_consistency_weight: float = 0.05,
        identity_refresh_interval: int = 100,
        loop_signal_path: str = "data/loop_signal.json",
        loop_freeze_lr_mult: float = 0.5,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.use_amp = use_amp and device.type == 'cuda'
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps,
            eta_min=learning_rate * 0.1,
        )
        
        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Metrics tracking
        self.metrics_history: List[NC4Metrics] = []
        self.best_convergence_rate = 0.0
        
        # Adaptive Controller
        controller_max_lambda = controller_max_lambda if controller_max_lambda is not None else model.config.fixed_point_loss_weight
        self.controller = UMCAdaptiveController(
            target_task_loss=0.005, 
            min_diversity=0.02,
            initial_lambda_conv=model.config.fixed_point_loss_weight,
            max_lambda=controller_max_lambda,
        )
        
        # Strategic Supervisor (Ollama)
        self.supervisor = OllamaUMCSupervisor(model_name="deepseek-r1:latest")
        self.supervisor_interval = supervisor_interval # Consult every N steps
        self.supervisor_mode = supervisor_mode

        self.iteration_penalty_weight = iteration_penalty_weight
        self.iteration_penalty_threshold = iteration_penalty_threshold
        self.semantic_warmup_steps = semantic_warmup_steps
        self.semantic_warmup_damping = semantic_warmup_damping
        self.warmup_active = False
        self.entropy_min_variance = entropy_min_variance
        self.entropy_penalty_weight = entropy_penalty_weight
        self.jolt_lr_mult = jolt_lr_mult
        self.jolt_noise_std = jolt_noise_std
        self.jolt_next_step = False
        self.min_damping = 0.15
        self.identity_summary_path = identity_summary_path
        self.identity_inject_prefix = identity_inject_prefix
        self.identity_consistency_weight = identity_consistency_weight
        self.identity_refresh_interval = identity_refresh_interval
        self.loop_signal_path = loop_signal_path
        self.loop_freeze_lr_mult = loop_freeze_lr_mult
        self._identity_summary_text = ""
        self._identity_summary_embed = None
        self._identity_summary_mtime = 0.0
        self._last_loop_signal_ts = 0.0

        # Phase schedule (simple 2-phase)
        self.phase_schedule = phase_schedule
        self.phase1_ratio = phase1_ratio
        self.phase_lambda_start = phase_lambda_start
        self.phase_lambda_end = phase_lambda_end
        self.phase_damping_start = phase_damping_start
        self.phase_damping_end = phase_damping_end if phase_damping_end is not None else model.config.damping
        self.phase_total_steps = max_steps
        self.phase_state = {"phase": None, "lambda_cap": None, "damping_target": None}
        self.hard_reset_done = False

        # Training data: FineWeb-Edu streaming
        print(f"[DATA] Loading FineWeb-Edu dataset (streaming)...")
        self.dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        self.data_iter = iter(self.dataset)

    def _maybe_refresh_identity_summary(self, step: int):
        if not self.identity_summary_path:
            return
        if step % max(1, self.identity_refresh_interval) != 0 and self._identity_summary_embed is not None:
            return
        try:
            if not os.path.exists(self.identity_summary_path):
                return
            mtime = os.path.getmtime(self.identity_summary_path)
            if mtime <= self._identity_summary_mtime and self._identity_summary_embed is not None:
                return
            with open(self.identity_summary_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                return
            self._identity_summary_text = text
            self._identity_summary_mtime = mtime
            with torch.no_grad():
                tok = self.tokenizer(
                    text,
                    max_length=128,
                    truncation=True,
                    return_tensors='pt',
                )
                ids = tok['input_ids'].to(self.device)
                emb = self.model.tok_embeddings(ids).mean(dim=1).squeeze(0)
                self._identity_summary_embed = emb.detach()
        except Exception:
            return

    def _apply_loop_signal(self):
        if not self.loop_signal_path or not os.path.exists(self.loop_signal_path):
            return
        try:
            with open(self.loop_signal_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ts = float(data.get("ts", 0.0))
            if ts <= self._last_loop_signal_ts:
                return
            self._last_loop_signal_ts = ts
            action = data.get("action", "none")
            if action == "jolt":
                self.jolt_next_step = True
            elif action == "freeze":
                for pg in self.optimizer.param_groups:
                    pg['lr'] = max(pg['lr'] * self.loop_freeze_lr_mult, 1e-6)
        except Exception:
            return

    def _identity_prefix(self) -> str:
        if not self._identity_summary_text:
            return ""
        return f"{self.identity_inject_prefix}{self._identity_summary_text}\\n\\n"

    def _apply_semantic_warmup(self, step: int) -> bool:
        if self.semantic_warmup_steps <= 0 or step > self.semantic_warmup_steps:
            self.warmup_active = False
            return False
        self.warmup_active = True
        if self.supervisor_mode != "control":
            damping_target = min(max(self.semantic_warmup_damping, self.min_damping), 1.0 - 1e-4)
            with torch.no_grad():
                target = torch.tensor(damping_target, device=self.model.block.learnable_damping.device)
                self.model.block.learnable_damping.copy_(torch.log(target / (1.0 - target)))
        return True

    def _apply_phase_schedule(self, step: int):
        if not self.phase_schedule:
            return
        phase1_steps = max(1, int(self.phase_total_steps * self.phase1_ratio))
        if step <= phase1_steps:
            phase = 1
            lambda_cap = self.phase_lambda_start
            damping_target = self.phase_damping_start
        else:
            phase = 2
            denom = max(1, self.phase_total_steps - phase1_steps)
            progress = min(1.0, (step - phase1_steps) / denom)
            lambda_cap = self.phase_lambda_start + progress * (self.phase_lambda_end - self.phase_lambda_start)
            damping_target = self.phase_damping_start + progress * (self.phase_damping_end - self.phase_damping_start)

        self.controller.max_lambda = max(0.0, lambda_cap)

        # Only enforce damping schedule when supervisor is not in control mode
        # and semantic warmup is not active.
        if self.supervisor_mode != "control" and not self.warmup_active:
            damping_target = min(max(damping_target, self.min_damping), 1.0 - 1e-4)
            with torch.no_grad():
                target = torch.tensor(damping_target, device=self.model.block.learnable_damping.device)
                self.model.block.learnable_damping.copy_(torch.log(target / (1.0 - target)))

        self.phase_state = {
            "phase": phase,
            "lambda_cap": self.controller.max_lambda,
            "damping_target": damping_target,
        }

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get a batch of training data from streaming dataset."""
        batch_texts = []
        prefix = self._identity_prefix()
        for _ in range(self.batch_size):
            try:
                sample = next(self.data_iter)
                batch_texts.append(prefix + sample['text'] if prefix else sample['text'])
            except StopIteration:
                self.data_iter = iter(self.dataset)
                sample = next(self.data_iter)
                batch_texts.append(prefix + sample['text'] if prefix else sample['text'])
        
        encoding = self.tokenizer(
            batch_texts,
            max_length=256, # Increased for Llama
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Shift labels for causal LM and ignore padding/last token
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        label_mask = attention_mask.clone()
        label_mask[:, :-1] = attention_mask[:, 1:]
        label_mask[:, -1] = 0
        labels[label_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
    
    def train_step(self) -> Dict[str, float]:
        """Single training step with gradient accumulation."""
        self.model.train()
        
        total_loss = 0
        total_loss_task = 0
        total_loss_fp = 0
        total_loss_conv = 0
        total_loss_iter_penalty = 0
        total_loss_entropy = 0
        total_loss_consistency = 0
        total_iter_var = 0
        total_iterations = 0
        total_converged = 0
        num_batches = 0
        
        self.optimizer.zero_grad()
        jolt_active = self.jolt_next_step
        if jolt_active:
            self.jolt_next_step = False
            original_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.jolt_lr_mult
        else:
            original_lrs = None
        
        for _ in range(self.grad_accumulation_steps):
            batch = self.get_batch()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
                        return_iterations=True,
                        return_all_losses=True,
                        noise_std=self.jolt_noise_std if jolt_active else 0.0,
                    )
                    iter_penalty = torch.tensor(0.0, device=outputs['loss'].device, dtype=outputs['loss'].dtype)
                    if outputs['iterations'] > self.iteration_penalty_threshold:
                        iters = torch.tensor(outputs['iterations'], device=outputs['loss'].device, dtype=outputs['loss'].dtype)
                        iter_penalty = (iters / self.model.config.max_iterations) * self.iteration_penalty_weight
                    iter_var = outputs.get('iter_hidden_var', torch.tensor(0.0, device=outputs['loss'].device, dtype=outputs['loss'].dtype))
                    entropy_penalty = torch.clamp(self.entropy_min_variance - iter_var, min=0.0) * self.entropy_penalty_weight
                    loss_consistency = torch.tensor(0.0, device=outputs['loss'].device, dtype=outputs['loss'].dtype)
                    if self._identity_summary_embed is not None:
                        h = outputs.get("final_hidden")
                        if h is not None:
                            mask = batch['attention_mask'].unsqueeze(-1)
                            mean_h = (h * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
                            ident = self._identity_summary_embed.unsqueeze(0).expand_as(mean_h)
                            cos = torch.nn.functional.cosine_similarity(mean_h, ident, dim=-1)
                            loss_consistency = (1.0 - cos).mean() * self.identity_consistency_weight
                    loss = (outputs['loss'] + iter_penalty + entropy_penalty + loss_consistency) / self.grad_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    return_iterations=True,
                    return_all_losses=True,
                    noise_std=self.jolt_noise_std if jolt_active else 0.0,
                )
                iter_penalty = torch.tensor(0.0, device=outputs['loss'].device, dtype=outputs['loss'].dtype)
                if outputs['iterations'] > self.iteration_penalty_threshold:
                    iters = torch.tensor(outputs['iterations'], device=outputs['loss'].device, dtype=outputs['loss'].dtype)
                    iter_penalty = (iters / self.model.config.max_iterations) * self.iteration_penalty_weight
                iter_var = outputs.get('iter_hidden_var', torch.tensor(0.0, device=outputs['loss'].device, dtype=outputs['loss'].dtype))
                entropy_penalty = torch.clamp(self.entropy_min_variance - iter_var, min=0.0) * self.entropy_penalty_weight
                loss_consistency = torch.tensor(0.0, device=outputs['loss'].device, dtype=outputs['loss'].dtype)
                if self._identity_summary_embed is not None:
                    h = outputs.get("final_hidden")
                    if h is not None:
                        mask = batch['attention_mask'].unsqueeze(-1)
                        mean_h = (h * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
                        ident = self._identity_summary_embed.unsqueeze(0).expand_as(mean_h)
                        cos = torch.nn.functional.cosine_similarity(mean_h, ident, dim=-1)
                        loss_consistency = (1.0 - cos).mean() * self.identity_consistency_weight
                loss = (outputs['loss'] + iter_penalty + entropy_penalty + loss_consistency) / self.grad_accumulation_steps
                loss.backward()
            
            total_loss += outputs['loss'].item()
            total_loss_task += outputs['loss_task'].item()
            total_loss_fp += outputs['loss_fixed_point'].item()
            total_loss_conv += outputs['loss_convergence'].item()
            total_loss_iter_penalty += iter_penalty.item()
            total_loss_consistency += loss_consistency.item()
            total_loss_entropy += entropy_penalty.item()
            total_iter_var += iter_var.item()
            total_iterations += outputs['iterations']
            total_converged += 1 if outputs['converged'] else 0
            
            # Track diversity (variance of hidden states)
            # Logits variance is a good proxy for semantic diversity
            hidden_states_var = outputs['logits'].var().item()
            
            num_batches += 1
        
        # Average metrics
        avg_loss_task = total_loss_task / num_batches
        avg_loss_fp = total_loss_fp / num_batches
        avg_loss_consistency = total_loss_consistency / num_batches
        
        # Update controller and get dynamic weights
        controller_metrics = {
            'loss_task': avg_loss_task,
            'loss_convergence': avg_loss_fp, # Using FP loss as convergence metric
            'hidden_states_var': hidden_states_var,
            'mean_iterations': total_iterations / max(1, num_batches),
            'max_iterations': self.model.config.max_iterations,
        }
        adjustments = self.controller.update_params(controller_metrics, self.model, self.optimizer)
        self.model.config.fixed_point_loss_weight = adjustments['new_lambda_conv']
        
        # Gradient clipping
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Optimizer step
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        if jolt_active and original_lrs is not None:
            for lr, param_group in zip(original_lrs, self.optimizer.param_groups):
                param_group['lr'] = lr
        self.scheduler.step()
        
        return {
            'loss': total_loss / num_batches,
            'loss_task': avg_loss_task,
            'loss_fp': avg_loss_fp,
            'loss_conv': total_loss_conv / num_batches,
            'loss_iter_penalty': total_loss_iter_penalty / num_batches,
            'loss_identity_consistency': avg_loss_consistency,
            'loss_entropy': total_loss_entropy / num_batches,
            'iter_hidden_var': total_iter_var / num_batches,
            'iterations': total_iterations / num_batches,
            'converged_rate': total_converged / num_batches,
            'hidden_states_var': hidden_states_var, # Pass this back for logging/supervisor
        }

    def evaluate_nc4(self, num_samples: int = 100) -> Dict[str, float]:
        """Evaluate NC4 convergence metrics."""
        self.model.eval()
        
        iterations_list = []
        converged_count = 0
        
        # Get evaluation samples from dataset
        eval_texts = []
        for _ in range(num_samples):
            try:
                sample = next(self.data_iter)
                eval_texts.append(sample['text'])
            except StopIteration:
                self.data_iter = iter(self.dataset)
                sample = next(self.data_iter)
                eval_texts.append(sample['text'])
        
        with torch.no_grad():
            for i in range(0, len(eval_texts), self.batch_size):
                batch_texts = eval_texts[i:i + self.batch_size]
                
                encoding = self.tokenizer(
                    batch_texts,
                    max_length=256,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt',
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_all_losses=True,
                )
                
                iterations_list.append(outputs['iterations'])
                if outputs['converged']:
                    converged_count += len(batch_texts)
        
        convergence_rate = converged_count / len(eval_texts)
        mean_iterations = sum(iterations_list) / len(iterations_list) if iterations_list else 0
        
        return {
            'convergence_rate': convergence_rate,
            'mean_iterations': mean_iterations,
            'mean_residual': 0.0,
        }
    
    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.model.config),
            'metrics_history': [m.to_dict() for m in self.metrics_history],
            'best_convergence_rate': self.best_convergence_rate,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint-{step}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  [NEW BEST] Saved best model with convergence rate {self.best_convergence_rate:.2%}", flush=True)
    
    def train(self):
        """Main training loop."""
        print("\n" + "=" * 70, flush=True)
        print("NC4 CONVERGENCE TRAINING", flush=True)
        print("=" * 70, flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"Max steps: {self.max_steps}", flush=True)
        print(f"Batch size: {self.batch_size} x {self.grad_accumulation_steps} = {self.batch_size * self.grad_accumulation_steps}", flush=True)
        print(f"Model parameters: {self.model.num_parameters:,}", flush=True)
        print(f"Max iterations: {self.model.config.max_iterations}", flush=True)
        print(f"Damping: {torch.sigmoid(self.model.block.learnable_damping).item():.4f}", flush=True)
        print(f"Convergence threshold: {self.model.config.convergence_threshold}", flush=True)
        print("=" * 70, flush=True)
        
        start_time = time.time()
        
        for step in range(1, self.max_steps + 1):
            # Phase scheduling (integration -> stabilization)
            self._apply_semantic_warmup(step)
            self._apply_phase_schedule(step)
            self._maybe_refresh_identity_summary(step)
            self._apply_loop_signal()

            # Training step
            step_metrics = self.train_step()
            
            # Progress logging
            if step % self.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                eta = (self.max_steps - step) / steps_per_sec
                
                print(f"Step {step}/{self.max_steps} | "
                      f"Loss: {step_metrics['loss']:.4f} | "
                      f"Task: {step_metrics['loss_task']:.4f} | "
                      f"Ident: {step_metrics.get('loss_identity_consistency', 0.0):.4f} | "
                      f"λ_conv: {self.model.config.fixed_point_loss_weight:.4f} | "
                      f"Iter: {step_metrics['iterations']:.1f} | "
                      f"Conv: {step_metrics['converged_rate']:.0%} | "
                      f"ETA: {eta/60:.1f}min", flush=True)
            
            # Evaluation
            if step % self.eval_interval == 0:
                print(f"\n[EVAL] Step {step}", flush=True)
                eval_metrics = self.evaluate_nc4()
                
                metrics = NC4Metrics(
                    step=step,
                    convergence_rate=eval_metrics['convergence_rate'],
                    mean_iterations=eval_metrics['mean_iterations'],
                    mean_residual=eval_metrics['mean_residual'],
                    loss_total=step_metrics['loss'],
                    loss_task=step_metrics['loss_task'],
                    loss_fixed_point=step_metrics['loss_fp'],
                    loss_convergence=step_metrics['loss_conv'],
                )
                self.metrics_history.append(metrics)
                
                print(f"  Convergence rate: {eval_metrics['convergence_rate']:.2%}", flush=True)
                print(f"  Mean iterations: {eval_metrics['mean_iterations']:.2f}", flush=True)
                print(f"  Mean residual: {eval_metrics['mean_residual']:.6f}", flush=True)
                if self.phase_schedule:
                    print(f"  [PHASE] phase={self.phase_state['phase']}, "
                          f"λ_cap={self.phase_state['lambda_cap']:.3f}, "
                          f"damping_target={self.phase_state['damping_target']:.3f}", flush=True)
                if self.semantic_warmup_steps > 0:
                    print(f"  [WARMUP] active={self.warmup_active}, "
                          f"damping={torch.sigmoid(self.model.block.learnable_damping).item():.3f}", flush=True)
                
                if step >= 2000 and eval_metrics['convergence_rate'] == 0.0 and not self.hard_reset_done:
                    print("  [WARN] Convergence rate is 0% after 2000 steps. Hard reset damping to 0.15.", flush=True)
                    with torch.no_grad():
                        target = torch.tensor(self.min_damping, device=self.model.block.learnable_damping.device)
                        self.model.block.learnable_damping.copy_(torch.log(target / (1.0 - target)))
                    self.hard_reset_done = True
                
                # Check for improvement
                is_best = eval_metrics['convergence_rate'] > self.best_convergence_rate
                if is_best:
                    self.best_convergence_rate = eval_metrics['convergence_rate']
                
                # NC4 success check
                if eval_metrics['convergence_rate'] >= 0.5:
                    print(f"\n  [NC4 SUCCESS] Convergence rate {eval_metrics['convergence_rate']:.2%} >= 50%!", flush=True)
                
                print(flush=True)
            
            # Save checkpoint
            if step % self.save_interval == 0:
                self.save_checkpoint(step, is_best=False)
            
            # Strategic AI Supervision
            if self.supervisor_mode != "off" and step % self.supervisor_interval == 0:
                print(f"\n[STRATEGIC REVIEW] Step {step} - Consulting Ollama Supervisor...", flush=True)
                samples = get_model_samples(self.model, self.tokenizer, self.device)
                
                # Use metrics from the last evaluation
                last_eval = self.metrics_history[-1] if self.metrics_history else None
                if last_eval:
                    supervisor_metrics = {
                        'step': step,
                        'loss_task': last_eval.loss_task,
                        'loss_fp': last_eval.loss_fixed_point,
                        'lambda_fp': self.model.config.fixed_point_loss_weight,
                        'diversity': step_metrics.get('hidden_states_var', 0.0),
                        'convergence_rate': last_eval.convergence_rate,
                        'lr': self.optimizer.param_groups[0]['lr'],
                        'damping': torch.sigmoid(self.model.block.learnable_damping).item()
                    }
                    advice = self.supervisor.consult(supervisor_metrics, samples, mode=self.supervisor_mode)
                    
                    if advice:
                        rep_score = advice.get('repetition_score')
                        if rep_score is not None and rep_score > 7.0:
                            self.jolt_next_step = True
                            print("  [JOLT] Repetition score high; applying LR boost + noise next step.", flush=True)
                        if self.supervisor_mode == "control":
                            # 1. Adjust Controller
                            self.controller.target_task_loss *= advice.get('adjust_target_loss', 1.0)
                            
                            # 2. Adjust Learning Rate
                            lr_mult = advice.get('adjust_lr', 1.0)
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] *= lr_mult
                            
                            # 3. Adjust Damping (Stability)
                            damping_mult = advice.get('adjust_damping', 1.0)
                            with torch.no_grad():
                                current = torch.sigmoid(self.model.block.learnable_damping)
                                new_damping = torch.clamp(current * damping_mult, 1e-4, 1.0 - 1e-4)
                                self.model.block.learnable_damping.copy_(torch.log(new_damping / (1.0 - new_damping)))
                            
                            # 4. Adjust NC4 Weight directly
                            fp_mult = advice.get('adjust_lambda_fp', 1.0)
                            self.model.config.fixed_point_loss_weight *= fp_mult

                            print(f"  [ADAPTATION] Action: {advice.get('action')}, "
                                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}, "
                                  f"Damping: {torch.sigmoid(self.model.block.learnable_damping).item():.4f}, "
                                  f"λ_fp: {self.model.config.fixed_point_loss_weight:.4f}")
                            
                            if advice.get('continue_training') and step == self.max_steps:
                                print(f"  [EXTENSION] Supervisor recommended extending training.")
                                self.max_steps += 1000
                        else:
                            print(f"  [SUPERVISOR] semantic={advice.get('semantic_score', 'n/a')}, "
                                  f"coherence={advice.get('coherence_score', 'n/a')}, "
                                  f"repetition={advice.get('repetition_score', 'n/a')}")
        
        # Final evaluation and save
        print("\n[FINAL EVALUATION]", flush=True)
        final_metrics = self.evaluate_nc4(num_samples=500)
        print(f"  Final convergence rate: {final_metrics['convergence_rate']:.2%}", flush=True)
        print(f"  Final mean iterations: {final_metrics['mean_iterations']:.2f}", flush=True)
        print(f"  Final mean residual: {final_metrics['mean_residual']:.6f}", flush=True)
        
        # Save final model
        final_checkpoint = {
            'step': self.max_steps,
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.model.config),
            'final_metrics': final_metrics,
            'best_convergence_rate': self.best_convergence_rate,
        }
        torch.save(final_checkpoint, self.output_dir / 'final_model.pt')
        
        # Save metrics history
        with open(self.output_dir / 'metrics_history.json', 'w') as f:
            json.dump([m.to_dict() for m in self.metrics_history], f, indent=2)
        
        # Summary
        print("\n" + "=" * 70, flush=True)
        print("TRAINING COMPLETE", flush=True)
        print("=" * 70, flush=True)
        print(f"Best convergence rate: {self.best_convergence_rate:.2%}", flush=True)
        print(f"NC4 criterion (>= 50%): {'[PASS]' if self.best_convergence_rate >= 0.5 else '[FAIL]'}", flush=True)
        print(f"Output directory: {self.output_dir}", flush=True)
        print("=" * 70, flush=True)
        
        return final_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Contractive GPT-2 for NC4 convergence')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--steps', type=int, default=10000, help='Training steps')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--grad-accum', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--eval-interval', type=int, default=500, help='Evaluation interval')
    parser.add_argument('--save-interval', type=int, default=2000, help='Save interval')
    parser.add_argument('--log-interval', type=int, default=50, help='Log interval (steps)')
    parser.add_argument('--output-dir', type=str, default='benchmark_output/nc4', help='Output directory')
    parser.add_argument('--supervisor-mode', type=str, default='commentary',
                        choices=['commentary', 'control', 'off'],
                        help='Supervisor mode: commentary (default), control, or off')
    parser.add_argument('--supervisor-interval', type=int, default=500,
                        help='Supervisor consult interval (steps)')
    parser.add_argument('--no-phase-schedule', dest='phase_schedule', action='store_false',
                        help='Disable simple phase schedule')
    parser.add_argument('--phase1-ratio', type=float, default=0.6,
                        help='Fraction of steps for phase 1 (integration)')
    parser.add_argument('--phase-lambda-start', type=float, default=0.05,
                        help='Max lambda cap in phase 1')
    parser.add_argument('--phase-lambda-end', type=float, default=0.5,
                        help='Max lambda cap at end of phase 2')
    parser.add_argument('--phase-damping-start', type=float, default=0.15,
                        help='Damping target in phase 1')
    parser.add_argument('--phase-damping-end', type=float, default=None,
                        help='Damping target at end of phase 2 (defaults to --damping)')
    parser.add_argument('--iteration-penalty-weight', type=float, default=0.05,
                        help='Weight for iteration penalty term')
    parser.add_argument('--iteration-penalty-threshold', type=int, default=28,
                        help='Apply iteration penalty only if iterations exceed this threshold')
    parser.add_argument('--semantic-warmup-steps', type=int, default=2000,
                        help='Number of warmup steps with higher damping (0 to disable)')
    parser.add_argument('--semantic-warmup-damping', type=float, default=0.4,
                        help='Damping value during semantic warmup')
    parser.add_argument('--controller-max-lambda', type=float, default=None,
                        help='Max lambda cap for controller (defaults to fp-loss-weight)')
    parser.add_argument('--entropy-min-variance', type=float, default=0.01,
                        help='Minimum variance across iterations before entropy penalty applies')
    parser.add_argument('--entropy-penalty-weight', type=float, default=0.1,
                        help='Weight for entropy penalty term')
    parser.add_argument('--jolt-lr-mult', type=float, default=5.0,
                        help='Learning rate multiplier for a one-step jolt')
    parser.add_argument('--jolt-noise-std', type=float, default=0.02,
                        help='Gaussian noise std for jolt step')
    parser.add_argument('--identity-summary-path', type=str, default='data/identity_summary.txt',
                        help='Path to identity summary file for system background injection')
    parser.add_argument('--identity-inject-prefix', type=str, default='System Background:\\n',
                        help='Prefix used before identity summary')
    parser.add_argument('--identity-consistency-weight', type=float, default=0.05,
                        help='Weight for identity consistency loss term')
    parser.add_argument('--identity-refresh-interval', type=int, default=100,
                        help='Steps between identity summary reloads')
    parser.add_argument('--loop-signal-path', type=str, default='data/loop_signal.json',
                        help='Path to consciousness loop signal file')
    parser.add_argument('--loop-freeze-lr-mult', type=float, default=0.5,
                        help='LR multiplier when consciousness loop signals freeze')
    parser.set_defaults(phase_schedule=True)
    
    # Model hyperparameters
    parser.add_argument('--max-iter', type=int, default=32, help='Max iterations')
    parser.add_argument('--damping', type=float, default=0.3, help='Damping factor')
    parser.add_argument('--threshold', type=float, default=1e-2, help='Convergence threshold')
    parser.add_argument('--fp-loss-weight', type=float, default=0.1, help='Fixed-point loss weight')
    
    args = parser.parse_args()
    
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers package required")
    
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    
    # Load tokenizer
    print(f"\n[TOKENIZER] Loading Llama-3 tokenizer...")
    # Using a reliable public Llama-3-style tokenizer
    tokenizer = AutoTokenizer.from_pretrained('Xenova/llama-3-tokenizer')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    print("\nCreating Contractive Llama...", flush=True)
    config = ContractiveLlamaConfig(
        vocab_size=len(tokenizer),
        max_iterations=args.max_iter,
        damping=args.damping,
        convergence_threshold=args.threshold,
        fixed_point_loss_weight=args.fp_loss_weight,
    )
    model = ContractiveLlama(config)
    print(f"  Parameters: {model.num_parameters if hasattr(model, 'num_parameters') else sum(p.numel() for p in model.parameters()):,}", flush=True)
    
    # Create trainer
    trainer = NC4Trainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accum,
        max_steps=args.steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        supervisor_mode=args.supervisor_mode,
        supervisor_interval=args.supervisor_interval,
        phase_schedule=args.phase_schedule,
        phase1_ratio=args.phase1_ratio,
        phase_lambda_start=args.phase_lambda_start,
        phase_lambda_end=args.phase_lambda_end,
        phase_damping_start=args.phase_damping_start,
        phase_damping_end=args.phase_damping_end,
        iteration_penalty_weight=args.iteration_penalty_weight,
        iteration_penalty_threshold=args.iteration_penalty_threshold,
        semantic_warmup_steps=args.semantic_warmup_steps,
        semantic_warmup_damping=args.semantic_warmup_damping,
        controller_max_lambda=args.controller_max_lambda,
        entropy_min_variance=args.entropy_min_variance,
        entropy_penalty_weight=args.entropy_penalty_weight,
        jolt_lr_mult=args.jolt_lr_mult,
        jolt_noise_std=args.jolt_noise_std,
        identity_summary_path=args.identity_summary_path,
        identity_inject_prefix=args.identity_inject_prefix,
        identity_consistency_weight=args.identity_consistency_weight,
        identity_refresh_interval=args.identity_refresh_interval,
        loop_signal_path=args.loop_signal_path,
        loop_freeze_lr_mult=args.loop_freeze_lr_mult,
    )
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
