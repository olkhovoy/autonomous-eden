"""
Adaptive Depth Test

This test evaluates whether the recursive model adaptively uses different
computation depths (iterations) for inputs of varying complexity.

Key hypothesis from UMC framework:
- Simple inputs (repetitive text, simple grammar) should converge faster
- Complex inputs (scientific text, code, nested references) should need more iterations
- Self-referential inputs may show distinctive iteration patterns

Usage:
    python -m benchmark.tests.adaptive_depth --checkpoint benchmark_output/recursive/best_model.pt
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
class AdaptiveDepthResults:
    """Results from adaptive depth analysis."""
    
    # Per-category statistics
    category_stats: Dict[str, Dict[str, float]]
    
    # Overall statistics
    overall_mean_iterations: float
    overall_std_iterations: float
    
    # Statistical tests
    complexity_correlation: float  # Correlation between complexity and iterations
    anova_p_value: Optional[float]  # ANOVA p-value across categories
    
    # Raw data for visualization
    category_iterations: Dict[str, List[float]]
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'AdaptiveDepthResults':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# Test categories with example texts
COMPLEXITY_CATEGORIES = {
    'simple_repetitive': [
        "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.",
        "One two three four five. One two three four five. One two three four five.",
        "Hello hello hello hello hello hello hello hello hello hello.",
        "Red blue green red blue green red blue green red blue green.",
        "The sun is bright. The sun is bright. The sun is bright.",
    ],
    'simple_grammar': [
        "The dog runs. The cat sleeps. The bird flies. The fish swims.",
        "I eat breakfast. You drink coffee. She reads books. He writes letters.",
        "The tree is tall. The house is big. The car is fast.",
        "Monday comes before Tuesday. Tuesday comes before Wednesday.",
        "Water is wet. Fire is hot. Ice is cold. Air is invisible.",
    ],
    'moderate_narrative': [
        "The old man walked slowly through the park, watching the children play on the swings while remembering his own childhood days long ago.",
        "Scientists announced today that they have discovered a new species of deep-sea fish that can produce its own light through bioluminescence.",
        "The company's quarterly earnings exceeded expectations, leading to a significant increase in stock price during after-hours trading.",
        "After months of preparation, the team finally launched their new product, hoping it would revolutionize the industry.",
        "The ancient castle stood atop the hill, its weathered stones telling stories of battles and celebrations from centuries past.",
    ],
    'complex_scientific': [
        "The quantum entanglement phenomenon demonstrates non-local correlations between particles that cannot be explained by classical physics, suggesting fundamental aspects of reality that challenge our intuitive understanding.",
        "Mitochondrial DNA analysis reveals that the common ancestor of all modern humans lived in Africa approximately 200,000 years ago, supporting the 'Out of Africa' theory of human evolution.",
        "The relationship between entropy and information theory, as formalized by Shannon, provides a mathematical framework for understanding the fundamental limits of data compression and transmission.",
        "Neuroplasticity research demonstrates that the adult brain retains significant capacity for structural and functional reorganization, challenging earlier assumptions about neural development.",
        "The Navier-Stokes equations describe the motion of viscous fluid substances, and proving the existence and smoothness of their solutions remains one of the Millennium Prize Problems.",
    ],
    'complex_code': [
        "def recursive_fibonacci(n): return n if n <= 1 else recursive_fibonacci(n-1) + recursive_fibonacci(n-2)",
        "SELECT users.name, COUNT(orders.id) FROM users LEFT JOIN orders ON users.id = orders.user_id GROUP BY users.id HAVING COUNT(orders.id) > 5",
        "const debounce = (fn, delay) => { let timeout; return (...args) => { clearTimeout(timeout); timeout = setTimeout(() => fn(...args), delay); }; };",
        "class Observable { constructor() { this.observers = []; } subscribe(fn) { this.observers.push(fn); } notify(data) { this.observers.forEach(fn => fn(data)); } }",
        "for i in range(len(matrix)): for j in range(i+1, len(matrix[0])): matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]",
    ],
    'self_referential': [
        "This sentence is about itself and the fact that it is being processed by a neural network that must decide how many iterations to use.",
        "The number of computational steps required to understand this text depends on the complexity of its self-referential structure.",
        "I am a text that knows it is being read by an AI that uses recursive processing to understand me.",
        "Consider the following: this very sentence requires the model to recursively process the concept of recursive processing.",
        "Meta-cognition involves thinking about thinking, which is precisely what happens when you read this sentence about meta-cognition.",
    ],
    'nested_structure': [
        "The man who saw the dog that chased the cat that caught the mouse that ate the cheese that was made from the milk that came from the cow was surprised.",
        "If the theory that suggests that the hypothesis that explains the observation that contradicts the prediction is correct, then we must reconsider our assumptions.",
        "The book about the author who wrote about the character who read about the story that described the world where books write themselves is paradoxical.",
        "She believed that he thought that they assumed that we knew that it was obvious that the answer was hidden in plain sight.",
        "The function that calls the function that returns the function that computes the value that determines the result is highly recursive.",
    ],
}


def get_complexity_score(category: str) -> int:
    """Assign numerical complexity score to category."""
    scores = {
        'simple_repetitive': 1,
        'simple_grammar': 2,
        'moderate_narrative': 3,
        'complex_scientific': 4,
        'complex_code': 4,
        'self_referential': 5,
        'nested_structure': 5,
    }
    return scores.get(category, 3)


def analyze_text_iterations(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 128,
) -> List[float]:
    """Analyze iteration counts for a list of texts."""
    model.eval()
    iterations_list = []
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            encoding = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                return_iterations=True,
            )
            
            iterations = outputs.get('iterations', None)
            if iterations is not None:
                iterations_list.append(iterations.item())
            else:
                iterations_list.append(float('nan'))
    
    return iterations_list


def run_adaptive_depth_test(
    checkpoint_path: str,
    output_dir: str,
    device: str = 'cuda',
    use_wandb: bool = False,
) -> AdaptiveDepthResults:
    """
    Run the adaptive depth test on a trained recursive model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save results
        device: Device to run on
        use_wandb: Whether to log to wandb
        
    Returns:
        AdaptiveDepthResults with analysis
    """
    from benchmark.models.recursive_gpt2 import RecursiveGPT2, RecursiveGPT2Config
    
    print("[ADAPTIVE DEPTH TEST]")
    print("=" * 60)
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    device = torch.device(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with default config (or load from checkpoint if available)
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
    print(f"  Triton enabled: {model.triton_enabled}")
    print(f"  Anderson enabled: {model.anderson_enabled}")
    
    # Load tokenizer
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers package required for this test")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Analyze each category
    print("\nAnalyzing categories...")
    category_iterations = {}
    category_stats = {}
    
    all_iterations = []
    all_complexity_scores = []
    
    for category, texts in COMPLEXITY_CATEGORIES.items():
        print(f"  {category}...", end=" ")
        
        iterations = analyze_text_iterations(model, tokenizer, texts, device)
        iterations = [i for i in iterations if not np.isnan(i)]
        
        if iterations:
            category_iterations[category] = iterations
            category_stats[category] = {
                'mean': float(np.mean(iterations)),
                'std': float(np.std(iterations)),
                'min': float(np.min(iterations)),
                'max': float(np.max(iterations)),
                'count': len(iterations),
            }
            
            # For correlation analysis
            complexity = get_complexity_score(category)
            all_iterations.extend(iterations)
            all_complexity_scores.extend([complexity] * len(iterations))
            
            print(f"mean={category_stats[category]['mean']:.2f}, std={category_stats[category]['std']:.2f}")
        else:
            print("no valid iterations")
    
    # Compute overall statistics
    overall_mean = float(np.mean(all_iterations)) if all_iterations else 0.0
    overall_std = float(np.std(all_iterations)) if all_iterations else 0.0
    
    # Compute correlation between complexity and iterations
    if len(all_iterations) > 2:
        complexity_correlation = float(np.corrcoef(all_complexity_scores, all_iterations)[0, 1])
    else:
        complexity_correlation = 0.0
    
    # ANOVA test across categories
    anova_p_value = None
    try:
        from scipy import stats
        groups = [category_iterations[cat] for cat in category_iterations if len(category_iterations[cat]) >= 2]
        if len(groups) >= 2:
            f_stat, anova_p_value = stats.f_oneway(*groups)
            anova_p_value = float(anova_p_value)
    except ImportError:
        print("  [WARN] scipy not available for ANOVA test")
    
    # Create results
    results = AdaptiveDepthResults(
        category_stats=category_stats,
        overall_mean_iterations=overall_mean,
        overall_std_iterations=overall_std,
        complexity_correlation=complexity_correlation,
        anova_p_value=anova_p_value,
        category_iterations=category_iterations,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Overall mean iterations: {overall_mean:.2f} +/- {overall_std:.2f}")
    print(f"Complexity-iterations correlation: {complexity_correlation:.3f}")
    if anova_p_value is not None:
        print(f"ANOVA p-value (categories differ): {anova_p_value:.4f}")
        if anova_p_value < 0.05:
            print("  [OK] Significant difference between categories (p < 0.05)")
        else:
            print("  [WARN] No significant difference between categories")
    
    print("\nPer-category summary:")
    for category in sorted(category_stats.keys(), key=lambda c: category_stats[c]['mean']):
        stats = category_stats[category]
        print(f"  {category}: {stats['mean']:.2f} +/- {stats['std']:.2f} (n={stats['count']})")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'adaptive_depth_results.json')
    results.save(results_path)
    print(f"\nResults saved to {results_path}")
    
    # Log to wandb if enabled
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project='umc-benchmark', name='adaptive-depth-test', reinit=True)
        
        # Log summary metrics
        wandb.log({
            'test/overall_mean_iterations': overall_mean,
            'test/overall_std_iterations': overall_std,
            'test/complexity_correlation': complexity_correlation,
        })
        
        # Log per-category means
        for category, stats in category_stats.items():
            wandb.log({f'test/iterations_{category}': stats['mean']})
        
        # Log box plot data
        table_data = []
        for category, iters in category_iterations.items():
            for it in iters:
                table_data.append([category, it, get_complexity_score(category)])
        
        table = wandb.Table(data=table_data, columns=['category', 'iterations', 'complexity'])
        wandb.log({'adaptive_depth_distribution': table})
        
        wandb.finish()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run adaptive depth test')
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
    
    results = run_adaptive_depth_test(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        use_wandb=args.use_wandb,
    )
    
    print("\n[OK] Adaptive depth test completed")


if __name__ == '__main__':
    main()
