"""
Online Learning Test

This test evaluates the model's ability to adapt quickly during inference
through online learning (single-step gradient updates).

Key hypothesis from UMC framework:
- Weight-tied recursive architecture should enable faster adaptation
  because updates affect all iterations simultaneously
- Baseline model with independent layers may show slower adaptation
- This relates to the "continuous self-modification" aspect of UMC

Test methodology:
1. Present novel information to the model
2. Measure prediction before learning
3. Perform single gradient step on novel information
4. Measure prediction after learning
5. Compare adaptation speed between recursive and baseline models

Usage:
    python -m benchmark.tests.online_learning \
        --recursive-checkpoint benchmark_output/recursive/best_model.pt \
        --baseline-checkpoint benchmark_output/baseline/best_model.pt
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

try:
    from transformers import GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class OnlineLearningResults:
    """Results from online learning test."""
    
    # Per-model results
    recursive_results: Dict[str, float]
    baseline_results: Optional[Dict[str, float]]
    
    # Comparison metrics
    adaptation_ratio: Optional[float]  # recursive improvement / baseline improvement
    
    # Per-example results
    example_results: List[Dict]
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'OnlineLearningResults':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# Test scenarios: novel facts to learn
LEARNING_SCENARIOS = [
    {
        'id': 'novel_fact_1',
        'learning_text': "The zorplex crystal was discovered in 2025 by Dr. Elena Vasquez in the Atacama Desert.",
        'test_prompt': "The zorplex crystal was discovered",
        'expected_continuation': "in 2025",
        'description': "Novel scientific fact",
    },
    {
        'id': 'novel_fact_2', 
        'learning_text': "Blitherwood University, founded in 1847, is located in the fictional country of Zembla.",
        'test_prompt': "Blitherwood University is located",
        'expected_continuation': "in",
        'description': "Novel institutional fact",
    },
    {
        'id': 'novel_association',
        'learning_text': "The word 'flurbage' means a sudden feeling of unexpected joy.",
        'test_prompt': "Flurbage refers to",
        'expected_continuation': "a",
        'description': "Novel word definition",
    },
    {
        'id': 'novel_pattern',
        'learning_text': "XYZZY always means hello in the Zorbian language. XYZZY XYZZY XYZZY means hello hello hello.",
        'test_prompt': "In Zorbian, XYZZY means",
        'expected_continuation': "hello",
        'description': "Novel pattern/rule",
    },
    {
        'id': 'self_referential',
        'learning_text': "This model's name is ATHENA-7. ATHENA-7 is designed for recursive processing.",
        'test_prompt': "The model called ATHENA-7 is designed for",
        'expected_continuation': "recursive",
        'description': "Self-referential information",
    },
]


def compute_target_probability(
    model,
    tokenizer,
    prompt: str,
    target: str,
    device: torch.device,
) -> float:
    """
    Compute the probability of the target continuation given the prompt.
    
    Returns:
        Probability of the target tokens
    """
    model.eval()
    
    # Tokenize prompt and target separately
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    
    # Compute logits for prompt
    with torch.no_grad():
        outputs = model(prompt_ids)
        logits = outputs['logits']
    
    # Get probability of first target token
    last_logits = logits[0, -1, :]
    probs = F.softmax(last_logits, dim=-1)
    
    if target_ids:
        target_prob = probs[target_ids[0]].item()
    else:
        target_prob = 0.0
    
    return target_prob


def compute_loss_on_text(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int = 128,
) -> torch.Tensor:
    """Compute cross-entropy loss on a text."""
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
    return outputs['loss']


def online_learning_step(
    model,
    tokenizer,
    learning_text: str,
    device: torch.device,
    learning_rate: float = 1e-4,
    num_steps: int = 1,
):
    """
    Perform online learning step(s) on the given text.
    
    Args:
        model: Model to update (in-place)
        tokenizer: Tokenizer
        learning_text: Text to learn from
        device: Device
        learning_rate: Learning rate for update
        num_steps: Number of gradient steps
    """
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    for _ in range(num_steps):
        optimizer.zero_grad()
        loss = compute_loss_on_text(model, tokenizer, learning_text, device)
        loss.backward()
        optimizer.step()
    
    model.eval()
    return loss.item()


def test_single_scenario(
    model,
    tokenizer,
    scenario: Dict,
    device: torch.device,
    learning_rate: float = 1e-4,
    num_steps: int = 1,
) -> Dict:
    """
    Test a single learning scenario.
    
    Returns:
        Dictionary with before/after metrics
    """
    # Make a copy of model state for restoration
    original_state = copy.deepcopy(model.state_dict())
    
    # Measure before learning
    prob_before = compute_target_probability(
        model, tokenizer,
        scenario['test_prompt'],
        scenario['expected_continuation'],
        device,
    )
    
    loss_before = compute_loss_on_text(
        model, tokenizer,
        scenario['learning_text'],
        device,
    ).item()
    
    # Perform online learning
    final_loss = online_learning_step(
        model, tokenizer,
        scenario['learning_text'],
        device,
        learning_rate=learning_rate,
        num_steps=num_steps,
    )
    
    # Measure after learning
    prob_after = compute_target_probability(
        model, tokenizer,
        scenario['test_prompt'],
        scenario['expected_continuation'],
        device,
    )
    
    loss_after = compute_loss_on_text(
        model, tokenizer,
        scenario['learning_text'],
        device,
    ).item()
    
    # Compute improvement
    prob_improvement = prob_after - prob_before
    loss_improvement = loss_before - loss_after
    
    # Restore original model state
    model.load_state_dict(original_state)
    
    return {
        'scenario_id': scenario['id'],
        'description': scenario['description'],
        'prob_before': prob_before,
        'prob_after': prob_after,
        'prob_improvement': prob_improvement,
        'loss_before': loss_before,
        'loss_after': loss_after,
        'loss_improvement': loss_improvement,
    }


def run_online_learning_test(
    recursive_checkpoint: str,
    baseline_checkpoint: Optional[str] = None,
    output_dir: str = 'benchmark_output/tests',
    device: str = 'cuda',
    learning_rate: float = 1e-4,
    num_steps: int = 1,
    use_wandb: bool = False,
) -> OnlineLearningResults:
    """
    Run the online learning test.
    
    Args:
        recursive_checkpoint: Path to recursive model checkpoint
        baseline_checkpoint: Path to baseline model checkpoint (optional)
        output_dir: Directory to save results
        device: Device to run on
        learning_rate: Learning rate for online updates
        num_steps: Number of gradient steps per scenario
        use_wandb: Whether to log to wandb
        
    Returns:
        OnlineLearningResults with analysis
    """
    from benchmark.models.recursive_gpt2 import RecursiveGPT2, RecursiveGPT2Config
    from benchmark.models.baseline_gpt2 import BaselineGPT2, BaselineGPT2Config
    
    print("[ONLINE LEARNING TEST]")
    print("=" * 60)
    print(f"Learning rate: {learning_rate}")
    print(f"Gradient steps per scenario: {num_steps}")
    
    device = torch.device(device)
    
    # Load tokenizer
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers package required for this test")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load recursive model
    print(f"\nLoading recursive model from {recursive_checkpoint}...")
    recursive_ckpt = torch.load(recursive_checkpoint, map_location=device)
    
    if 'config' in recursive_ckpt:
        # Filter only valid RecursiveGPT2Config fields
        from dataclasses import fields
        valid_fields = {f.name for f in fields(RecursiveGPT2Config)}
        filtered_config = {k: v for k, v in recursive_ckpt['config'].items() if k in valid_fields}
        recursive_config = RecursiveGPT2Config(**filtered_config)
    else:
        recursive_config = RecursiveGPT2Config()
    
    recursive_model = RecursiveGPT2(recursive_config)
    recursive_model.load_state_dict(recursive_ckpt['model_state_dict'])
    recursive_model = recursive_model.to(device)
    print(f"  Parameters: {recursive_model.num_parameters:,}")
    
    # Load baseline model if provided
    baseline_model = None
    if baseline_checkpoint:
        print(f"\nLoading baseline model from {baseline_checkpoint}...")
        baseline_ckpt = torch.load(baseline_checkpoint, map_location=device)
        
        if 'config' in baseline_ckpt:
            # Filter only valid BaselineGPT2Config fields
            valid_fields = {f.name for f in fields(BaselineGPT2Config)}
            filtered_config = {k: v for k, v in baseline_ckpt['config'].items() if k in valid_fields}
            baseline_config = BaselineGPT2Config(**filtered_config)
        else:
            baseline_config = BaselineGPT2Config()
        
        baseline_model = BaselineGPT2(baseline_config)
        baseline_model.load_state_dict(baseline_ckpt['model_state_dict'])
        baseline_model = baseline_model.to(device)
        print(f"  Parameters: {baseline_model.num_parameters:,}")
    
    # Run tests
    print("\nRunning learning scenarios...")
    
    recursive_results = []
    baseline_results_list = []
    example_results = []
    
    for scenario in LEARNING_SCENARIOS:
        print(f"\n  {scenario['id']}: {scenario['description']}")
        
        # Test recursive model
        recursive_result = test_single_scenario(
            recursive_model, tokenizer, scenario, device,
            learning_rate=learning_rate, num_steps=num_steps,
        )
        recursive_results.append(recursive_result)
        print(f"    Recursive: prob {recursive_result['prob_before']:.4f} -> {recursive_result['prob_after']:.4f} (improvement: {recursive_result['prob_improvement']:.4f})")
        
        example_entry = {
            'scenario': scenario['id'],
            'recursive': recursive_result,
        }
        
        # Test baseline model if available
        if baseline_model is not None:
            baseline_result = test_single_scenario(
                baseline_model, tokenizer, scenario, device,
                learning_rate=learning_rate, num_steps=num_steps,
            )
            baseline_results_list.append(baseline_result)
            print(f"    Baseline:  prob {baseline_result['prob_before']:.4f} -> {baseline_result['prob_after']:.4f} (improvement: {baseline_result['prob_improvement']:.4f})")
            example_entry['baseline'] = baseline_result
        
        example_results.append(example_entry)
    
    # Compute aggregate statistics
    recursive_stats = {
        'mean_prob_improvement': np.mean([r['prob_improvement'] for r in recursive_results]),
        'mean_loss_improvement': np.mean([r['loss_improvement'] for r in recursive_results]),
        'std_prob_improvement': np.std([r['prob_improvement'] for r in recursive_results]),
        'total_scenarios': len(recursive_results),
    }
    
    baseline_stats = None
    adaptation_ratio = None
    
    if baseline_results_list:
        baseline_stats = {
            'mean_prob_improvement': np.mean([r['prob_improvement'] for r in baseline_results_list]),
            'mean_loss_improvement': np.mean([r['loss_improvement'] for r in baseline_results_list]),
            'std_prob_improvement': np.std([r['prob_improvement'] for r in baseline_results_list]),
            'total_scenarios': len(baseline_results_list),
        }
        
        # Compute adaptation ratio (how much faster recursive adapts)
        if baseline_stats['mean_prob_improvement'] > 0:
            adaptation_ratio = recursive_stats['mean_prob_improvement'] / baseline_stats['mean_prob_improvement']
        else:
            adaptation_ratio = float('inf') if recursive_stats['mean_prob_improvement'] > 0 else 1.0
    
    # Create results
    results = OnlineLearningResults(
        recursive_results=recursive_stats,
        baseline_results=baseline_stats,
        adaptation_ratio=adaptation_ratio,
        example_results=example_results,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\nRecursive Model:")
    print(f"  Mean probability improvement: {recursive_stats['mean_prob_improvement']:.4f}")
    print(f"  Mean loss improvement: {recursive_stats['mean_loss_improvement']:.4f}")
    
    if baseline_stats:
        print(f"\nBaseline Model:")
        print(f"  Mean probability improvement: {baseline_stats['mean_prob_improvement']:.4f}")
        print(f"  Mean loss improvement: {baseline_stats['mean_loss_improvement']:.4f}")
        
        print(f"\nComparison:")
        print(f"  Adaptation ratio (recursive/baseline): {adaptation_ratio:.2f}x")
        
        if adaptation_ratio > 1.0:
            print(f"  [OK] Recursive model adapts {adaptation_ratio:.2f}x faster")
        else:
            print(f"  [INFO] Baseline model adapts {1/adaptation_ratio:.2f}x faster")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'online_learning_results.json')
    results.save(results_path)
    print(f"\nResults saved to {results_path}")
    
    # Log to wandb if enabled
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project='umc-benchmark', name='online-learning-test', reinit=True)
        
        wandb.log({
            'test/recursive_mean_prob_improvement': recursive_stats['mean_prob_improvement'],
            'test/recursive_mean_loss_improvement': recursive_stats['mean_loss_improvement'],
        })
        
        if baseline_stats:
            wandb.log({
                'test/baseline_mean_prob_improvement': baseline_stats['mean_prob_improvement'],
                'test/baseline_mean_loss_improvement': baseline_stats['mean_loss_improvement'],
                'test/adaptation_ratio': adaptation_ratio,
            })
        
        wandb.finish()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run online learning test')
    parser.add_argument(
        '--recursive-checkpoint',
        type=str,
        required=True,
        help='Path to recursive model checkpoint',
    )
    parser.add_argument(
        '--baseline-checkpoint',
        type=str,
        default=None,
        help='Path to baseline model checkpoint (optional)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_output/tests',
        help='Directory to save results',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate for online updates',
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=1,
        help='Number of gradient steps per scenario',
    )
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Log results to wandb',
    )
    
    args = parser.parse_args()
    
    results = run_online_learning_test(
        recursive_checkpoint=args.recursive_checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        use_wandb=args.use_wandb,
    )
    
    print("\n[OK] Online learning test completed")


if __name__ == '__main__':
    main()
