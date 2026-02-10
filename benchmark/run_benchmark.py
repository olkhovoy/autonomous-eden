#!/usr/bin/env python3
"""
UMC Fixed-Point Training Benchmark

Main entry point for running the benchmark comparing baseline GPT-2
against recursive fixed-point GPT-2.

Usage:
    python run_benchmark.py --mode train --model baseline
    python run_benchmark.py --mode train --model recursive
    python run_benchmark.py --mode eval --checkpoint path/to/checkpoint.pt
    python run_benchmark.py --mode full  # Train and evaluate both models

Configuration:
    Edit config.yaml or pass arguments to customize training.
"""

import argparse
import os
import sys
import json
import time
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.models.baseline_gpt2 import BaselineGPT2, BaselineGPT2Config, create_gpt2_small
from benchmark.models.recursive_gpt2 import RecursiveGPT2, RecursiveGPT2Config, create_recursive_gpt2_small
from benchmark.training.trainer import FixedPointTrainer, TrainingConfig
from benchmark.metrics.density import KnowledgeDensityMetric, BenchmarkResults


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark run."""
    
    # Model selection
    model_type: str = "both"  # "baseline", "recursive", or "both"
    
    # Data
    dataset_name: str = "openwebtext"
    dataset_subset: Optional[str] = None
    max_train_samples: int = 1000000
    max_eval_samples: int = 10000
    
    # Training - RTX 3090 24GB optimized defaults
    max_steps: int = 100000
    batch_size: int = 4  # Reduced for 24GB VRAM
    learning_rate: float = 3e-4
    warmup_steps: int = 2000
    gradient_accumulation_steps: int = 8  # Effective batch = 4 * 8 = 32
    max_seq_length: int = 512  # Reduced from 1024 for memory
    
    # Evaluation
    eval_steps: int = 5000
    eval_lambada: bool = True
    eval_hellaswag: bool = False
    
    # Checkpointing
    save_steps: int = 10000
    output_dir: str = "benchmark_output"
    
    # Hardware - RTX 3090 optimized
    device: str = "cuda"
    use_amp: bool = True  # Essential for 24GB
    num_workers: int = 2  # Reduced to save RAM
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "umc-benchmark"
    log_steps: int = 100
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BenchmarkConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        # Filter out unknown fields (like _comment)
        valid_fields = {f.name for f in __import__('dataclasses').fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


class TextDataset(Dataset):
    """Dataset wrapper for text data."""
    
    def __init__(
        self,
        texts: list,
        tokenizer,
        max_length: int = 1024,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing without real data."""
    
    def __init__(
        self,
        num_samples: int,
        seq_length: int,
        vocab_size: int,
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate deterministic random data based on index
        torch.manual_seed(idx)
        return {
            'input_ids': torch.randint(0, self.vocab_size, (self.seq_length,)),
            'attention_mask': torch.ones(self.seq_length, dtype=torch.long),
        }


def get_cache_path(config: BenchmarkConfig) -> Path:
    """Get path for cached dataset."""
    cache_dir = Path(config.output_dir) / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{config.dataset_name}_train{config.max_train_samples}_eval{config.max_eval_samples}.pt"


def load_dataset_splits(config: BenchmarkConfig):
    """Load training and evaluation datasets with caching."""
    
    # Handle synthetic dataset explicitly
    if config.dataset_name == "synthetic":
        print(f"[DATA] Using synthetic data...")
        train_dataset = SyntheticDataset(
            min(config.max_train_samples, 10000),
            config.max_seq_length,
            50257,
        )
        eval_dataset = SyntheticDataset(
            min(config.max_eval_samples, 1000),
            config.max_seq_length,
            50257,
        )
        print(f"[DATA] Train samples: {len(train_dataset)}")
        print(f"[DATA] Eval samples: {len(eval_dataset)}")
        return train_dataset, eval_dataset, None
    
    try:
        from transformers import GPT2Tokenizer
        
        # Load tokenizer (cached by transformers library)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Check for cached data
        cache_path = get_cache_path(config)
        
        if cache_path.exists():
            print(f"[DATA] Loading cached dataset from {cache_path}...")
            cached_data = torch.load(cache_path)
            train_texts = cached_data['train_texts']
            eval_texts = cached_data['eval_texts']
            print(f"[DATA] Loaded {len(train_texts)} train + {len(eval_texts)} eval samples from cache")
        else:
            print(f"[DATA] Downloading {config.dataset_name} (will be cached for future runs)...")
            
            from datasets import load_dataset
            
            # Load dataset
            if config.dataset_name == "openwebtext":
                dataset = load_dataset('openwebtext', split='train', streaming=True)
                
                # Collect samples
                train_texts = []
                eval_texts = []
                
                for i, example in enumerate(dataset):
                    if i < config.max_train_samples:
                        train_texts.append(example['text'])
                    elif i < config.max_train_samples + config.max_eval_samples:
                        eval_texts.append(example['text'])
                    else:
                        break
                    
                    if i % 10000 == 0:
                        print(f"  Downloaded {i} samples...")
                
            else:
                raise ValueError(f"Unknown dataset: {config.dataset_name}")
            
            # Save to cache
            print(f"[DATA] Caching dataset to {cache_path}...")
            torch.save({
                'train_texts': train_texts,
                'eval_texts': eval_texts,
                'dataset_name': config.dataset_name,
                'max_train_samples': config.max_train_samples,
                'max_eval_samples': config.max_eval_samples,
            }, cache_path)
            print(f"[DATA] Cache saved ({cache_path.stat().st_size / 1e6:.1f} MB)")
        
        # Create dataset objects
        train_dataset = TextDataset(train_texts, tokenizer, config.max_seq_length)
        eval_dataset = TextDataset(eval_texts, tokenizer, config.max_seq_length)
        
        print(f"[DATA] Train samples: {len(train_dataset)}")
        print(f"[DATA] Eval samples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset, tokenizer
        
    except ImportError as e:
        print(f"[WARN] Could not load real dataset: {e}")
        print("[WARN] Using synthetic data for testing...")
        
        train_dataset = SyntheticDataset(
            min(config.max_train_samples, 10000),
            config.max_seq_length,
            50257,
        )
        eval_dataset = SyntheticDataset(
            min(config.max_eval_samples, 1000),
            config.max_seq_length,
            50257,
        )
        
        return train_dataset, eval_dataset, None


def create_model(model_type: str, config: BenchmarkConfig):
    """Create model based on type."""
    
    if model_type == "baseline":
        print("[MODEL] Creating Baseline GPT-2...")
        model = create_gpt2_small()
        is_recursive = False
    elif model_type == "recursive":
        print("[MODEL] Creating Recursive GPT-2...")
        model = create_recursive_gpt2_small()
        is_recursive = True
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"  Parameters: {model.num_parameters:,}")
    
    return model, is_recursive


def debug_model_forward(model, is_recursive: bool, device: torch.device, use_amp: bool = True):
    """Run a debug forward/backward pass to verify model works."""
    print("\n" + "="*60)
    print("[DEBUG] Testing model forward/backward pass...")
    print("="*60)
    model.train()
    
    # Create dummy input with full vocab range (like real data)
    batch_size, seq_len = 2, 64
    vocab_size = 50257  # GPT-2 vocab size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Input range: [{input_ids.min().item()}, {input_ids.max().item()}]")
    print(f"  Device: {device}")
    print(f"  Model type: {'Recursive' if is_recursive else 'Baseline'}")
    print(f"  Mixed precision: {use_amp}")
    
    try:
        # Check embeddings first
        with torch.no_grad():
            token_emb = model.token_embedding(input_ids)
            print(f"  Token embedding shape: {token_emb.shape}")
            print(f"  Token embedding has nan: {torch.isnan(token_emb).any().item()}")
            print(f"  Token embedding range: [{token_emb.min().item():.4f}, {token_emb.max().item():.4f}]")
        
        # Test WITHOUT AMP first to establish baseline
        print("\n  [TEST 1] Forward pass WITHOUT mixed precision:")
        if is_recursive:
            outputs = model(input_ids, labels=labels, return_iterations=True)
            iterations = outputs.get('iterations', None)
            if iterations is not None:
                print(f"    Iterations: {iterations.item():.1f}")
        else:
            outputs = model(input_ids, labels=labels)
        
        loss_fp32 = outputs['loss']
        logits_fp32 = outputs['logits']
        print(f"    Logits has nan: {torch.isnan(logits_fp32).any().item()}")
        print(f"    Loss: {loss_fp32.item():.4f}")
        
        if torch.isnan(logits_fp32).any():
            print("  [ERROR] NaN in FP32 forward - model has fundamental issue!")
            return False
        
        # Now test with AMP if enabled
        from torch.amp import autocast
        
        print(f"\n  [TEST 2] Forward pass WITH mixed precision (enabled={use_amp}):")
        model.zero_grad()
        with autocast('cuda', enabled=use_amp):
            if is_recursive:
                outputs = model(input_ids, labels=labels, return_iterations=True)
            else:
                outputs = model(input_ids, labels=labels)
            
            loss = outputs['loss']
            logits = outputs['logits']
        
        print(f"  Logits shape: {logits.shape}")
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        print(f"  Logits has nan: {has_nan}")
        print(f"  Logits has inf: {has_inf}")
        if not has_nan and not has_inf:
            print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        else:
            finite_mask = ~torch.isnan(logits) & ~torch.isinf(logits)
            if finite_mask.any():
                finite_logits = logits[finite_mask]
                print(f"  Finite logits range: [{finite_logits.min().item():.4f}, {finite_logits.max().item():.4f}]")
                print(f"  NaN count: {torch.isnan(logits).sum().item()}")
                print(f"  Inf count: {torch.isinf(logits).sum().item()}")
            else:
                print(f"  All logits are NaN/Inf!")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Loss is nan: {torch.isnan(loss).item()}")
        
        if torch.isnan(loss):
            print("[DEBUG] Loss is NaN - skipping backward pass")
            return False
        
        # Test backward
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0.0
        num_params_with_grad = 0
        num_params_nan_grad = 0
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                num_params_with_grad += 1
                if torch.isnan(p.grad).any():
                    num_params_nan_grad += 1
                    print(f"  [WARN] NaN gradient in: {name}")
        
        total_grad_norm = total_grad_norm ** 0.5
        print(f"  Params with gradients: {num_params_with_grad}")
        print(f"  Params with NaN gradients: {num_params_nan_grad}")
        print(f"  Total gradient norm: {total_grad_norm:.4f}")
        
        model.zero_grad()
        print("="*60)
        print("[DEBUG] Model test PASSED")
        print("="*60 + "\n")
        return True
        
    except Exception as e:
        print(f"[DEBUG] Model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        return False


def train_model(
    model_type: str,
    config: BenchmarkConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> str:
    """
    Train a model and return the checkpoint path.
    
    Args:
        model_type: "baseline" or "recursive"
        config: Benchmark configuration
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        
    Returns:
        Path to the best checkpoint
    """
    # Create model
    model, is_recursive = create_model(model_type, config)
    
    # Move to device and run debug test
    device = torch.device(config.device)
    model = model.to(device)
    if not debug_model_forward(model, is_recursive, device, use_amp=config.use_amp):
        raise RuntimeError(f"Model {model_type} failed debug test!")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    # Create output directory
    output_dir = os.path.join(config.output_dir, model_type)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create training config
    training_config = TrainingConfig(
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_seq_length=config.max_seq_length,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        log_steps=config.log_steps,
        save_dir=output_dir,
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        wandb_run_name=f"umc-{model_type}",
        use_amp=config.use_amp,
        device=config.device,
    )
    
    # Create trainer
    trainer = FixedPointTrainer(
        model=model,
        config=training_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        is_recursive=is_recursive,
    )
    
    # Train
    print(f"\n[TRAIN] Starting training for {model_type} model...")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    
    print(f"[TRAIN] Completed in {elapsed/3600:.1f} hours")
    
    # Return best model if exists, otherwise final model
    best_path = os.path.join(output_dir, 'best_model.pt')
    final_path = os.path.join(output_dir, 'final_model.pt')
    
    if os.path.exists(best_path):
        return best_path
    elif os.path.exists(final_path):
        print(f"[WARN] No best_model.pt found, using final_model.pt")
        return final_path
    else:
        raise RuntimeError(f"No checkpoint found in {output_dir}")


def evaluate_model(
    model_type: str,
    checkpoint_path: str,
    config: BenchmarkConfig,
    eval_dataset: Dataset,
    tokenizer=None,
) -> Dict[str, float]:
    """
    Evaluate a trained model.
    
    Args:
        model_type: "baseline" or "recursive"
        checkpoint_path: Path to model checkpoint
        config: Benchmark configuration
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer for LAMBADA evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n[EVAL] Evaluating {model_type} model...")
    
    # Create model
    model, is_recursive = create_model(model_type, config)
    
    # Load checkpoint
    device = torch.device(config.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create dataloader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    
    # Compute perplexity
    total_loss = 0.0
    total_tokens = 0
    total_iterations = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            labels = input_ids.clone()
            
            if is_recursive:
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_iterations=True,
                )
                if 'iterations' in outputs:
                    total_iterations += outputs['iterations'].mean().item()
            else:
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            
            total_loss += outputs['loss'].item() * input_ids.numel()
            total_tokens += input_ids.numel()
            num_batches += 1
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 100))
    
    metrics = {
        'eval_loss': avg_loss,
        'perplexity': perplexity,
    }
    
    if is_recursive and num_batches > 0:
        metrics['avg_iterations'] = total_iterations / num_batches
    
    # LAMBADA evaluation
    if config.eval_lambada and tokenizer is not None:
        from benchmark.metrics.density import evaluate_lambada
        lambada_acc = evaluate_lambada(model, tokenizer, device)
        metrics['lambada_accuracy'] = lambada_acc
    
    print(f"[EVAL] Results for {model_type}:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return metrics


def run_full_benchmark(config: BenchmarkConfig) -> BenchmarkResults:
    """
    Run the complete benchmark: train and evaluate both models.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        BenchmarkResults with all metrics
    """
    print("=" * 60)
    print("UMC FIXED-POINT TRAINING BENCHMARK")
    print("=" * 60)
    
    # Load data
    train_dataset, eval_dataset, tokenizer = load_dataset_splits(config)
    
    # Initialize metrics tracker
    metrics = KnowledgeDensityMetric()
    
    # Train and evaluate baseline
    if config.model_type in ["baseline", "both"]:
        baseline_checkpoint = train_model("baseline", config, train_dataset, eval_dataset)
        baseline_metrics = evaluate_model("baseline", baseline_checkpoint, config, eval_dataset, tokenizer)
        
        # Create model for registration
        baseline_model, _ = create_model("baseline", config)
        metrics.register_baseline(baseline_model, "Baseline GPT-2")
        metrics.update_baseline(
            eval_loss=baseline_metrics['eval_loss'],
            perplexity=baseline_metrics['perplexity'],
            lambada_accuracy=baseline_metrics.get('lambada_accuracy', 0.0),
        )
    
    # Train and evaluate recursive
    if config.model_type in ["recursive", "both"]:
        recursive_checkpoint = train_model("recursive", config, train_dataset, eval_dataset)
        recursive_metrics = evaluate_model("recursive", recursive_checkpoint, config, eval_dataset, tokenizer)
        
        # Create model for registration
        recursive_model, _ = create_model("recursive", config)
        metrics.register_recursive(recursive_model, "Recursive GPT-2")
        metrics.update_recursive(
            eval_loss=recursive_metrics['eval_loss'],
            perplexity=recursive_metrics['perplexity'],
            lambada_accuracy=recursive_metrics.get('lambada_accuracy', 0.0),
            iterations=recursive_metrics.get('avg_iterations', 12.0),
        )
    
    # Compute and display results
    if config.model_type == "both":
        results = metrics.print_summary()
        
        # Save results
        results_path = os.path.join(config.output_dir, 'benchmark_results.json')
        results.save(results_path)
        print(f"\n[RESULTS] Saved to {results_path}")
        
        # Generate comprehensive report
        from benchmark.metrics.density import generate_benchmark_report
        report_path = generate_benchmark_report(
            results=results,
            output_dir=config.output_dir,
            use_wandb=config.use_wandb,
        )
        
        return results
    
    return None


def run_special_tests(
    config: BenchmarkConfig,
    recursive_checkpoint: Optional[str] = None,
    baseline_checkpoint: Optional[str] = None,
):
    """
    Run special capability tests after training.
    
    Tests:
    - Adaptive Depth: Does model use different iterations for different inputs?
    - Online Learning: Can model adapt quickly during inference?
    - Self-Reference: Can model reason about its own computation?
    """
    print("\n" + "=" * 60)
    print("RUNNING SPECIAL CAPABILITY TESTS")
    print("=" * 60)
    
    test_output_dir = os.path.join(config.output_dir, 'tests')
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Find checkpoints if not provided
    if recursive_checkpoint is None:
        recursive_checkpoint = os.path.join(config.output_dir, 'recursive', 'best_model.pt')
        if not os.path.exists(recursive_checkpoint):
            recursive_checkpoint = os.path.join(config.output_dir, 'recursive', 'final_model.pt')
    
    if baseline_checkpoint is None:
        baseline_checkpoint = os.path.join(config.output_dir, 'baseline', 'best_model.pt')
        if not os.path.exists(baseline_checkpoint):
            baseline_checkpoint = os.path.join(config.output_dir, 'baseline', 'final_model.pt')
    
    # Run tests
    test_results = {}
    
    # 1. Adaptive Depth Test
    if os.path.exists(recursive_checkpoint):
        try:
            from benchmark.tests.adaptive_depth import run_adaptive_depth_test
            print("\n[TEST 1/3] Adaptive Depth Test...")
            adaptive_results = run_adaptive_depth_test(
                checkpoint_path=recursive_checkpoint,
                output_dir=test_output_dir,
                device=config.device,
                use_wandb=config.use_wandb,
            )
            test_results['adaptive_depth'] = adaptive_results
        except Exception as e:
            print(f"[WARN] Adaptive depth test failed: {e}")
    
    # 2. Online Learning Test
    if os.path.exists(recursive_checkpoint):
        try:
            from benchmark.tests.online_learning import run_online_learning_test
            print("\n[TEST 2/3] Online Learning Test...")
            online_results = run_online_learning_test(
                recursive_checkpoint=recursive_checkpoint,
                baseline_checkpoint=baseline_checkpoint if os.path.exists(baseline_checkpoint) else None,
                output_dir=test_output_dir,
                device=config.device,
                use_wandb=config.use_wandb,
            )
            test_results['online_learning'] = online_results
        except Exception as e:
            print(f"[WARN] Online learning test failed: {e}")
    
    # 3. Self-Reference Test
    if os.path.exists(recursive_checkpoint):
        try:
            from benchmark.tests.self_reference import run_self_reference_test
            print("\n[TEST 3/3] Self-Reference Test...")
            selfref_results = run_self_reference_test(
                checkpoint_path=recursive_checkpoint,
                output_dir=test_output_dir,
                device=config.device,
                use_wandb=config.use_wandb,
            )
            test_results['self_reference'] = selfref_results
        except Exception as e:
            print(f"[WARN] Self-reference test failed: {e}")
    
    print("\n" + "=" * 60)
    print("SPECIAL TESTS COMPLETED")
    print("=" * 60)
    
    return test_results


def setup_cuda_memory_optimization():
    """Configure CUDA for optimal memory usage on RTX 3090."""
    if torch.cuda.is_available():
        # Enable memory efficient attention if available
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocation strategy
        torch.cuda.empty_cache()
        
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] {gpu_name} with {gpu_mem:.1f}GB VRAM")
        
        # Warn if not enough memory
        if gpu_mem < 20:
            print("[WARN] Less than 20GB VRAM detected, using minimal settings")
            return "minimal"
        elif gpu_mem < 40:
            print("[INFO] 20-40GB VRAM detected, using RTX 3090 optimized settings")
            return "rtx3090"
        else:
            print("[INFO] 40GB+ VRAM detected, using full settings")
            return "full"
    return "cpu"


def apply_gpu_preset(config: BenchmarkConfig, preset: str):
    """Apply GPU-specific configuration preset."""
    
    if preset == "minimal":
        # For GPUs with <20GB (e.g., RTX 3080, RTX 3070)
        config.batch_size = 2
        config.gradient_accumulation_steps = 16
        config.max_seq_length = 256
        config.num_workers = 1
        print("[CONFIG] Applied MINIMAL preset (batch=2, seq=256, accum=16)")
        
    elif preset == "rtx3090":
        # For RTX 3090 24GB
        config.batch_size = 4
        config.gradient_accumulation_steps = 8
        config.max_seq_length = 512
        config.num_workers = 2
        print("[CONFIG] Applied RTX3090 preset (batch=4, seq=512, accum=8)")
        
    elif preset == "full":
        # For A100 40GB+
        config.batch_size = 16
        config.gradient_accumulation_steps = 2
        config.max_seq_length = 1024
        config.num_workers = 4
        print("[CONFIG] Applied FULL preset (batch=16, seq=1024, accum=2)")
        
    else:  # CPU
        config.batch_size = 2
        config.gradient_accumulation_steps = 16
        config.max_seq_length = 256
        config.use_amp = False
        config.num_workers = 0
        print("[CONFIG] Applied CPU preset")
    
    # Always report effective batch size
    effective_batch = config.batch_size * config.gradient_accumulation_steps
    print(f"[CONFIG] Effective batch size: {effective_batch}")


def main():
    parser = argparse.ArgumentParser(description="UMC Fixed-Point Training Benchmark")
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'eval', 'full'],
        default='full',
        help='Benchmark mode',
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['baseline', 'recursive', 'both'],
        default='both',
        help='Model to train/evaluate',
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Checkpoint path for evaluation',
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file',
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_output',
        help='Output directory',
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Maximum training steps (default: 100000)',
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (auto-detected if not specified)',
    )
    
    parser.add_argument(
        '--seq-length',
        type=int,
        default=None,
        help='Sequence length (auto-detected if not specified)',
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='Learning rate',
    )

    parser.add_argument(
        '--log-steps',
        type=int,
        default=None,
        help='Log metrics every N steps',
    )

    parser.add_argument(
        '--eval-steps',
        type=int,
        default=None,
        help='Run evaluation every N steps',
    )

    parser.add_argument(
        '--save-steps',
        type=int,
        default=None,
        help='Save checkpoint every N steps',
    )
    
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Enable Weights & Biases logging',
    )
    
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data for testing',
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with minimal steps',
    )
    
    parser.add_argument(
        '--gpu-preset',
        type=str,
        choices=['auto', 'minimal', 'rtx3090', 'full'],
        default='auto',
        help='GPU memory preset (default: auto-detect)',
    )
    
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear cached dataset and re-download',
    )
    
    parser.add_argument(
        '--max-train-samples',
        type=int,
        default=None,
        help='Maximum training samples to use (default: 1000000)',
    )
    
    parser.add_argument(
        '--max-eval-samples',
        type=int,
        default=None,
        help='Maximum evaluation samples to use (default: 10000)',
    )
    
    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='Disable mixed precision training (for debugging)',
    )
    
    parser.add_argument(
        '--run-special-tests',
        action='store_true',
        help='Run special capability tests after training (adaptive depth, online learning, self-reference)',
    )
    
    parser.add_argument(
        '--special-tests-only',
        action='store_true',
        help='Only run special tests (skip training)',
    )
    
    args = parser.parse_args()
    
    # Setup CUDA optimizations and detect GPU
    if args.gpu_preset == 'auto':
        gpu_preset = setup_cuda_memory_optimization()
    else:
        gpu_preset = args.gpu_preset
        if torch.cuda.is_available():
            setup_cuda_memory_optimization()  # Still run for TF32 etc.
    
    # Load or create config
    if args.config:
        config = BenchmarkConfig.load(args.config)
    else:
        config = BenchmarkConfig()
    
    # Apply GPU preset FIRST (before manual overrides)
    apply_gpu_preset(config, gpu_preset)
    
    # Override with command line arguments (only if explicitly provided)
    config.model_type = args.model
    config.output_dir = args.output_dir
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.seq_length is not None:
        config.max_seq_length = args.seq_length
    if args.log_steps is not None:
        config.log_steps = args.log_steps
    if args.eval_steps is not None:
        config.eval_steps = args.eval_steps
    if args.save_steps is not None:
        config.save_steps = args.save_steps
    if args.max_train_samples is not None:
        config.max_train_samples = args.max_train_samples
    if args.max_eval_samples is not None:
        config.max_eval_samples = args.max_eval_samples
    config.learning_rate = args.learning_rate
    config.use_wandb = args.use_wandb
    
    # Quick test mode
    if args.quick_test:
        config.max_steps = 100
        config.log_steps = 10
        config.eval_steps = 50
        config.save_steps = 100
        config.max_train_samples = 1000
        config.max_eval_samples = 100
        config.batch_size = 2
        config.gradient_accumulation_steps = 4
        config.max_seq_length = 128
        print("[MODE] Quick test mode enabled")
    
    # Synthetic data mode
    if args.synthetic:
        config.dataset_name = "synthetic"
        print("[MODE] Synthetic data mode enabled")
    
    # Disable AMP if requested
    if args.no_amp:
        config.use_amp = False
        print("[MODE] Mixed precision DISABLED (for debugging)")
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        config.device = "cpu"
        config.use_amp = False
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Clear cache if requested
    if args.clear_cache:
        cache_path = get_cache_path(config)
        if cache_path.exists():
            cache_path.unlink()
            print(f"[CACHE] Cleared cached dataset: {cache_path}")
        else:
            print(f"[CACHE] No cache found at {cache_path}")
    
    # Save config
    config.save(os.path.join(config.output_dir, 'config.json'))
    
    # Print final config summary
    print(f"\n[CONFIG SUMMARY]")
    print(f"  Model: {config.model_type}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Sequence length: {config.max_seq_length}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Log steps: {config.log_steps}")
    print(f"  Train samples: {config.max_train_samples}")
    print(f"  Eval samples: {config.max_eval_samples}")
    print(f"  Mixed precision: {config.use_amp}")
    print()
    
    # Run special tests only
    if args.special_tests_only:
        print("[MODE] Running special tests only (skipping training)")
        run_special_tests(config)
        return
    
    # Run benchmark
    if args.mode == 'full':
        run_full_benchmark(config)
        
        # Run special tests if requested
        if args.run_special_tests:
            run_special_tests(config)
            
    elif args.mode == 'train':
        train_dataset, eval_dataset, _ = load_dataset_splits(config)
        train_model(args.model, config, train_dataset, eval_dataset)
        
        # Run special tests if requested
        if args.run_special_tests:
            run_special_tests(config)
            
    elif args.mode == 'eval':
        if args.checkpoint is None:
            print("[ERROR] --checkpoint required for eval mode")
            sys.exit(1)
        _, eval_dataset, tokenizer = load_dataset_splits(config)
        evaluate_model(args.model, args.checkpoint, config, eval_dataset, tokenizer)


if __name__ == "__main__":
    main()
