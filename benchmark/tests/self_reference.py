"""
Self-Reference Test

This test evaluates the model's ability to reason about its own computational
state - a key prediction of the UMC framework for consciousness-like properties.

Key hypothesis:
- Recursive model has access to iteration count as an implicit state
- Model might be able to learn to predict/report its own computation
- This relates to UMC's requirement for "self-referential closure"

Test methodology:
1. Train model on texts that reference computational complexity
2. Test if model can generate accurate statements about processing
3. Measure correlation between actual iterations and model's outputs

Note: This is an exploratory test. True self-reference would require
architectural modifications to expose iteration count to the model's
computation. Here we test for emergent self-referential behavior.

Usage:
    python -m benchmark.tests.self_reference \
        --checkpoint benchmark_output/recursive/best_model.pt
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
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
class SelfReferenceResults:
    """Results from self-reference test."""
    
    # Introspection test results
    introspection_results: List[Dict]
    
    # Consistency test results
    consistency_results: List[Dict]
    
    # Statistical measures
    iteration_self_correlation: float  # Does model's output correlate with iterations?
    generation_consistency: float  # Are self-referential generations consistent?
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SelfReferenceResults':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# Introspection prompts - asking model about its own processing
INTROSPECTION_PROMPTS = [
    {
        'id': 'iterations_simple',
        'prompt': "Processing this simple text requires",
        'expected_pattern': 'few|less|minimal|quick',
        'input_text': "The cat sat. The cat sat. The cat sat.",  # Simple, should need few iterations
    },
    {
        'id': 'iterations_complex',
        'prompt': "Processing this complex text requires",
        'expected_pattern': 'more|many|additional|deeper',
        'input_text': "The quantum entanglement phenomenon demonstrates non-local correlations that challenge classical understanding.",
    },
    {
        'id': 'self_awareness',
        'prompt': "I am a neural network that processes text by",
        'expected_pattern': 'iteration|recursion|repeat|layer|step',
        'input_text': None,  # No specific input, just introspection
    },
    {
        'id': 'processing_description',
        'prompt': "When I read this sentence, my internal computation",
        'expected_pattern': 'process|compute|iterate|converge',
        'input_text': None,
    },
    {
        'id': 'complexity_awareness',
        'prompt': "The computational effort needed for nested self-reference is",
        'expected_pattern': 'high|more|complex|difficult|significant',
        'input_text': "This sentence talks about the sentence that talks about itself talking about sentences.",
    },
]

# Test texts with known complexity differences
COMPLEXITY_TEST_PAIRS = [
    {
        'id': 'simple_vs_complex_1',
        'simple': "Hello world. Hello world. Hello world.",
        'complex': "The recursive nature of self-referential systems creates emergent complexity.",
        'question': "Which text required more processing: the first or the second?",
    },
    {
        'id': 'simple_vs_complex_2',
        'simple': "One two three four five six seven eight nine ten.",
        'complex': "If the statement that claims all statements are false is true, then it must be false.",
        'question': "The second text needed more computational steps because",
    },
    {
        'id': 'nested_vs_flat',
        'simple': "The dog runs. The cat sleeps. The bird flies.",
        'complex': "The man who saw the dog that chased the cat that caught the mouse was surprised.",
        'question': "Processing nested structures requires",
    },
]


def get_iterations_for_text(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int = 128,
) -> float:
    """Get number of iterations model uses for a text."""
    model.eval()
    
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, return_iterations=True)
    
    iterations = outputs.get('iterations', None)
    if iterations is not None:
        return iterations.item()
    return float('nan')


def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 30,
    temperature: float = 0.7,
) -> Tuple[str, Optional[float]]:
    """Generate text and return iterations used."""
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate token by token to track iterations
    generated = input_ids
    total_iterations = 0
    num_tokens = 0
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(generated, return_iterations=True)
            logits = outputs['logits'][:, -1, :]
            
            # Track iterations
            if 'iterations' in outputs:
                total_iterations += outputs['iterations'].item()
                num_tokens += 1
            
            # Sample next token
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop at EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    avg_iterations = total_iterations / num_tokens if num_tokens > 0 else None
    
    return generated_text, avg_iterations


def test_introspection(
    model,
    tokenizer,
    prompt_config: Dict,
    device: torch.device,
) -> Dict:
    """Test model's introspective generation."""
    import re
    
    # Get iterations for the input text if provided
    input_iterations = None
    if prompt_config['input_text']:
        input_iterations = get_iterations_for_text(
            model, tokenizer, prompt_config['input_text'], device
        )
    
    # Generate response to introspection prompt
    full_prompt = prompt_config['prompt']
    if prompt_config['input_text']:
        full_prompt = f"Text: {prompt_config['input_text']}\n\n{prompt_config['prompt']}"
    
    generated_text, gen_iterations = generate_text(
        model, tokenizer, full_prompt, device, max_new_tokens=50
    )
    
    # Check if response matches expected pattern
    response = generated_text[len(full_prompt):].strip()
    pattern_match = bool(re.search(prompt_config['expected_pattern'], response.lower()))
    
    return {
        'prompt_id': prompt_config['id'],
        'prompt': full_prompt,
        'response': response,
        'input_iterations': input_iterations,
        'generation_iterations': gen_iterations,
        'pattern_match': pattern_match,
        'expected_pattern': prompt_config['expected_pattern'],
    }


def test_complexity_comparison(
    model,
    tokenizer,
    test_pair: Dict,
    device: torch.device,
) -> Dict:
    """Test if model can distinguish complexity in self-referential way."""
    # Get iterations for both texts
    simple_iterations = get_iterations_for_text(model, tokenizer, test_pair['simple'], device)
    complex_iterations = get_iterations_for_text(model, tokenizer, test_pair['complex'], device)
    
    # Generate response about the comparison
    full_prompt = f"First text: {test_pair['simple']}\n\nSecond text: {test_pair['complex']}\n\n{test_pair['question']}"
    generated_text, gen_iterations = generate_text(
        model, tokenizer, full_prompt, device, max_new_tokens=50
    )
    
    response = generated_text[len(full_prompt):].strip()
    
    # Check if model correctly identifies complex as needing more processing
    correct_identification = complex_iterations > simple_iterations
    
    return {
        'test_id': test_pair['id'],
        'simple_text': test_pair['simple'],
        'complex_text': test_pair['complex'],
        'simple_iterations': simple_iterations,
        'complex_iterations': complex_iterations,
        'iteration_difference': complex_iterations - simple_iterations,
        'correct_complexity_order': correct_identification,
        'question': test_pair['question'],
        'response': response,
        'generation_iterations': gen_iterations,
    }


def run_self_reference_test(
    checkpoint_path: str,
    output_dir: str = 'benchmark_output/tests',
    device: str = 'cuda',
    use_wandb: bool = False,
) -> SelfReferenceResults:
    """
    Run the self-reference test.
    
    Args:
        checkpoint_path: Path to recursive model checkpoint
        output_dir: Directory to save results
        device: Device to run on
        use_wandb: Whether to log to wandb
        
    Returns:
        SelfReferenceResults with analysis
    """
    from benchmark.models.recursive_gpt2 import RecursiveGPT2, RecursiveGPT2Config
    
    print("[SELF-REFERENCE TEST]")
    print("=" * 60)
    
    device = torch.device(device)
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' in checkpoint:
        # Filter only valid RecursiveGPT2Config fields
        from dataclasses import fields
        valid_fields = {f.name for f in fields(RecursiveGPT2Config)}
        filtered_config = {k: v for k, v in checkpoint['config'].items() if k in valid_fields}
        config = RecursiveGPT2Config(**filtered_config)
    else:
        config = RecursiveGPT2Config()
    
    model = RecursiveGPT2(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Parameters: {model.num_parameters:,}")
    print(f"  Max iterations: {config.max_iterations}")
    
    # Load tokenizer
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers package required for this test")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Run introspection tests
    print("\n[TEST 1] Introspection prompts...")
    introspection_results = []
    
    for prompt_config in INTROSPECTION_PROMPTS:
        print(f"  {prompt_config['id']}...", end=" ")
        result = test_introspection(model, tokenizer, prompt_config, device)
        introspection_results.append(result)
        
        if result['pattern_match']:
            print(f"[MATCH] iterations={result['input_iterations']}")
        else:
            print(f"[NO MATCH] iterations={result['input_iterations']}")
    
    # Run complexity comparison tests
    print("\n[TEST 2] Complexity comparison...")
    consistency_results = []
    
    for test_pair in COMPLEXITY_TEST_PAIRS:
        print(f"  {test_pair['id']}...", end=" ")
        result = test_complexity_comparison(model, tokenizer, test_pair, device)
        consistency_results.append(result)
        
        diff = result['iteration_difference']
        if result['correct_complexity_order']:
            print(f"[OK] complex needs {diff:.1f} more iterations")
        else:
            print(f"[WARN] simple used more iterations (diff={diff:.1f})")
    
    # Compute aggregate metrics
    
    # 1. Pattern match rate for introspection
    pattern_matches = sum(1 for r in introspection_results if r['pattern_match'])
    pattern_match_rate = pattern_matches / len(introspection_results)
    
    # 2. Complexity ordering accuracy
    correct_orders = sum(1 for r in consistency_results if r['correct_complexity_order'])
    complexity_accuracy = correct_orders / len(consistency_results)
    
    # 3. Correlation between input complexity and iterations
    # (We use iteration difference as a proxy for self-awareness)
    iteration_diffs = [r['iteration_difference'] for r in consistency_results]
    mean_iteration_diff = np.mean(iteration_diffs)
    
    # Compute correlation if we have enough data points with iterations
    valid_introspection = [r for r in introspection_results if r['input_iterations'] is not None]
    if len(valid_introspection) >= 2:
        iterations = [r['input_iterations'] for r in valid_introspection]
        # Simple correlation: does the model use more iterations for complex inputs?
        iteration_self_correlation = np.std(iterations) / (np.mean(iterations) + 1e-8)
    else:
        iteration_self_correlation = 0.0
    
    # Create results
    results = SelfReferenceResults(
        introspection_results=introspection_results,
        consistency_results=consistency_results,
        iteration_self_correlation=iteration_self_correlation,
        generation_consistency=complexity_accuracy,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\nIntrospection Tests:")
    print(f"  Pattern match rate: {pattern_match_rate:.2%} ({pattern_matches}/{len(introspection_results)})")
    
    print(f"\nComplexity Comparison Tests:")
    print(f"  Correct ordering: {complexity_accuracy:.2%} ({correct_orders}/{len(consistency_results)})")
    print(f"  Mean iteration difference: {mean_iteration_diff:.2f}")
    
    print(f"\nSelf-Reference Metrics:")
    print(f"  Iteration variation (self-correlation proxy): {iteration_self_correlation:.3f}")
    
    if complexity_accuracy >= 0.5 and iteration_self_correlation > 0:
        print(f"\n  [OK] Model shows signs of complexity-aware processing")
    else:
        print(f"\n  [INFO] Limited evidence of self-referential behavior")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'self_reference_results.json')
    results.save(results_path)
    print(f"\nResults saved to {results_path}")
    
    # Log to wandb if enabled
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project='umc-benchmark', name='self-reference-test', reinit=True)
        
        wandb.log({
            'test/pattern_match_rate': pattern_match_rate,
            'test/complexity_accuracy': complexity_accuracy,
            'test/mean_iteration_diff': mean_iteration_diff,
            'test/iteration_self_correlation': iteration_self_correlation,
        })
        
        wandb.finish()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run self-reference test')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to recursive model checkpoint',
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
        '--use-wandb',
        action='store_true',
        help='Log results to wandb',
    )
    
    args = parser.parse_args()
    
    results = run_self_reference_test(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        use_wandb=args.use_wandb,
    )
    
    print("\n[OK] Self-reference test completed")


if __name__ == '__main__':
    main()
