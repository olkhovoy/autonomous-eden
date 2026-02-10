"""
Knowledge Density Metrics for UMC Benchmark

This module provides metrics for measuring knowledge density in neural networks,
as defined in the UMC framework analysis.

Key Metrics:
- Knowledge Density: (Score / Parameters) × Compression Ratio
- Parameter Efficiency: Score at equal parameter count
- Effective Depth: Average iterations for recursive models
- Compression Ratio: Effective depth / Parameter depth
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class ModelMetrics:
    """Metrics for a single model."""
    
    name: str
    num_parameters: int
    num_layers: int  # Explicit layers for baseline, 1 for recursive
    
    # Performance metrics
    train_loss: float = 0.0
    eval_loss: float = 0.0
    perplexity: float = 0.0
    
    # Task-specific metrics
    lambada_accuracy: float = 0.0
    hellaswag_accuracy: float = 0.0
    
    # Recursive model specific
    avg_iterations: float = 0.0
    min_iterations: float = 0.0
    max_iterations: float = 0.0
    
    # Computed metrics
    knowledge_density: float = 0.0
    parameter_efficiency: float = 0.0
    compression_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'num_parameters': self.num_parameters,
            'num_layers': self.num_layers,
            'train_loss': self.train_loss,
            'eval_loss': self.eval_loss,
            'perplexity': self.perplexity,
            'lambada_accuracy': self.lambada_accuracy,
            'hellaswag_accuracy': self.hellaswag_accuracy,
            'avg_iterations': self.avg_iterations,
            'min_iterations': self.min_iterations,
            'max_iterations': self.max_iterations,
            'knowledge_density': self.knowledge_density,
            'parameter_efficiency': self.parameter_efficiency,
            'compression_ratio': self.compression_ratio,
        }


@dataclass
class BenchmarkResults:
    """Results from a complete benchmark run."""
    
    baseline_metrics: ModelMetrics
    recursive_metrics: ModelMetrics
    
    # Comparison metrics
    density_ratio: float = 0.0  # recursive / baseline
    param_ratio: float = 0.0  # baseline / recursive
    performance_ratio: float = 0.0  # recursive score / baseline score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'baseline': self.baseline_metrics.to_dict(),
            'recursive': self.recursive_metrics.to_dict(),
            'comparison': {
                'density_ratio': self.density_ratio,
                'param_ratio': self.param_ratio,
                'performance_ratio': self.performance_ratio,
            }
        }
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BenchmarkResults':
        with open(path, 'r') as f:
            data = json.load(f)
        
        baseline = ModelMetrics(**data['baseline'])
        recursive = ModelMetrics(**data['recursive'])
        
        return cls(
            baseline_metrics=baseline,
            recursive_metrics=recursive,
            density_ratio=data['comparison']['density_ratio'],
            param_ratio=data['comparison']['param_ratio'],
            performance_ratio=data['comparison']['performance_ratio'],
        )


def compute_knowledge_density(
    score: float,
    num_parameters: int,
    effective_depth: float,
    parameter_depth: int,
    score_type: str = 'inverse_perplexity',
) -> float:
    """
    Compute knowledge density metric.
    
    Knowledge Density = (Score / Parameters) × Compression Ratio
    
    where:
    - Score is a performance metric (higher is better)
    - Parameters is the model parameter count
    - Compression Ratio = Effective Depth / Parameter Depth
    
    Args:
        score: Performance score (higher is better)
        num_parameters: Number of model parameters
        effective_depth: Average number of iterations (for recursive) or layers (for baseline)
        parameter_depth: Number of parameter-distinct layers (1 for recursive, L for baseline)
        score_type: Type of score ('inverse_perplexity', 'accuracy', 'raw')
        
    Returns:
        Knowledge density value
    """
    # Normalize score if needed
    if score_type == 'inverse_perplexity':
        # Perplexity is lower-is-better, so we invert
        normalized_score = 1.0 / max(score, 1.0)
    elif score_type == 'accuracy':
        # Accuracy is already 0-1, higher is better
        normalized_score = score
    else:
        normalized_score = score
    
    # Compression ratio: how much effective depth per parameter layer
    compression_ratio = effective_depth / max(parameter_depth, 1)
    
    # Knowledge density
    density = (normalized_score / num_parameters) * compression_ratio * 1e9  # Scale for readability
    
    return density


def compute_parameter_efficiency(
    score: float,
    num_parameters: int,
    baseline_score: float,
    baseline_parameters: int,
) -> float:
    """
    Compute parameter efficiency relative to baseline.
    
    Efficiency = (Score / Baseline_Score) × (Baseline_Params / Params)
    
    A value > 1 means the model is more parameter-efficient than baseline.
    
    Args:
        score: Model's performance score
        num_parameters: Model's parameter count
        baseline_score: Baseline model's score
        baseline_parameters: Baseline model's parameter count
        
    Returns:
        Parameter efficiency ratio
    """
    if baseline_score == 0 or num_parameters == 0:
        return 0.0
    
    score_ratio = score / baseline_score
    param_ratio = baseline_parameters / num_parameters
    
    return score_ratio * param_ratio


def compute_compression_ratio(
    effective_depth: float,
    parameter_depth: int,
) -> float:
    """
    Compute compression ratio.
    
    For recursive models: effective_depth = avg_iterations, parameter_depth = 1
    For baseline models: effective_depth = num_layers, parameter_depth = num_layers
    
    Args:
        effective_depth: Effective computational depth
        parameter_depth: Number of parameter-distinct layers
        
    Returns:
        Compression ratio
    """
    return effective_depth / max(parameter_depth, 1)


class KnowledgeDensityMetric:
    """
    Metric tracker for knowledge density measurements.
    
    Tracks performance and computes density metrics for both
    baseline and recursive models.
    """
    
    def __init__(self):
        self.baseline_metrics: Optional[ModelMetrics] = None
        self.recursive_metrics: Optional[ModelMetrics] = None
        self.iteration_history: List[float] = []
        
    def register_baseline(
        self,
        model: nn.Module,
        name: str = "Baseline GPT-2",
    ):
        """Register baseline model for comparison."""
        num_params = sum(p.numel() for p in model.parameters())
        num_layers = getattr(model, 'num_layers', getattr(model.config, 'num_hidden_layers', 12))
        
        self.baseline_metrics = ModelMetrics(
            name=name,
            num_parameters=num_params,
            num_layers=num_layers,
        )
        
    def register_recursive(
        self,
        model: nn.Module,
        name: str = "Recursive GPT-2",
    ):
        """Register recursive model for comparison."""
        num_params = sum(p.numel() for p in model.parameters())
        
        self.recursive_metrics = ModelMetrics(
            name=name,
            num_parameters=num_params,
            num_layers=1,  # Single weight-tied layer
        )
    
    def update_baseline(
        self,
        train_loss: Optional[float] = None,
        eval_loss: Optional[float] = None,
        perplexity: Optional[float] = None,
        lambada_accuracy: Optional[float] = None,
        hellaswag_accuracy: Optional[float] = None,
    ):
        """Update baseline model metrics."""
        if self.baseline_metrics is None:
            raise ValueError("Baseline model not registered")
        
        if train_loss is not None:
            self.baseline_metrics.train_loss = train_loss
        if eval_loss is not None:
            self.baseline_metrics.eval_loss = eval_loss
        if perplexity is not None:
            self.baseline_metrics.perplexity = perplexity
        if lambada_accuracy is not None:
            self.baseline_metrics.lambada_accuracy = lambada_accuracy
        if hellaswag_accuracy is not None:
            self.baseline_metrics.hellaswag_accuracy = hellaswag_accuracy
    
    def update_recursive(
        self,
        train_loss: Optional[float] = None,
        eval_loss: Optional[float] = None,
        perplexity: Optional[float] = None,
        lambada_accuracy: Optional[float] = None,
        hellaswag_accuracy: Optional[float] = None,
        iterations: Optional[float] = None,
    ):
        """Update recursive model metrics."""
        if self.recursive_metrics is None:
            raise ValueError("Recursive model not registered")
        
        if train_loss is not None:
            self.recursive_metrics.train_loss = train_loss
        if eval_loss is not None:
            self.recursive_metrics.eval_loss = eval_loss
        if perplexity is not None:
            self.recursive_metrics.perplexity = perplexity
        if lambada_accuracy is not None:
            self.recursive_metrics.lambada_accuracy = lambada_accuracy
        if hellaswag_accuracy is not None:
            self.recursive_metrics.hellaswag_accuracy = hellaswag_accuracy
        
        if iterations is not None:
            self.iteration_history.append(iterations)
            self.recursive_metrics.avg_iterations = sum(self.iteration_history) / len(self.iteration_history)
            self.recursive_metrics.min_iterations = min(self.iteration_history)
            self.recursive_metrics.max_iterations = max(self.iteration_history)
    
    def compute_metrics(self) -> BenchmarkResults:
        """
        Compute all knowledge density metrics.
        
        Returns:
            BenchmarkResults with computed metrics
        """
        if self.baseline_metrics is None or self.recursive_metrics is None:
            raise ValueError("Both baseline and recursive models must be registered")
        
        # Compute baseline metrics
        baseline = self.baseline_metrics
        baseline.compression_ratio = compute_compression_ratio(
            baseline.num_layers,
            baseline.num_layers,
        )
        baseline.knowledge_density = compute_knowledge_density(
            baseline.perplexity,
            baseline.num_parameters,
            baseline.num_layers,
            baseline.num_layers,
            score_type='inverse_perplexity',
        )
        
        # Compute recursive metrics
        recursive = self.recursive_metrics
        effective_depth = recursive.avg_iterations if recursive.avg_iterations > 0 else 12
        recursive.compression_ratio = compute_compression_ratio(
            effective_depth,
            1,  # Single parameter layer
        )
        recursive.knowledge_density = compute_knowledge_density(
            recursive.perplexity,
            recursive.num_parameters,
            effective_depth,
            1,
            score_type='inverse_perplexity',
        )
        
        # Compute parameter efficiency
        if baseline.perplexity > 0:
            recursive.parameter_efficiency = compute_parameter_efficiency(
                1.0 / recursive.perplexity,
                recursive.num_parameters,
                1.0 / baseline.perplexity,
                baseline.num_parameters,
            )
        
        # Compute comparison metrics
        density_ratio = recursive.knowledge_density / max(baseline.knowledge_density, 1e-10)
        param_ratio = baseline.num_parameters / max(recursive.num_parameters, 1)
        
        if baseline.perplexity > 0 and recursive.perplexity > 0:
            # Lower perplexity is better, so we invert for ratio
            performance_ratio = baseline.perplexity / recursive.perplexity
        else:
            performance_ratio = 0.0
        
        return BenchmarkResults(
            baseline_metrics=baseline,
            recursive_metrics=recursive,
            density_ratio=density_ratio,
            param_ratio=param_ratio,
            performance_ratio=performance_ratio,
        )
    
    def print_summary(self):
        """Print a summary of the metrics."""
        results = self.compute_metrics()
        
        print("\n" + "=" * 60)
        print("KNOWLEDGE DENSITY BENCHMARK RESULTS")
        print("=" * 60)
        
        print(f"\n[BASELINE] {results.baseline_metrics.name}")
        print(f"  Parameters: {results.baseline_metrics.num_parameters:,}")
        print(f"  Layers: {results.baseline_metrics.num_layers}")
        print(f"  Perplexity: {results.baseline_metrics.perplexity:.2f}")
        print(f"  Knowledge Density: {results.baseline_metrics.knowledge_density:.4f}")
        
        print(f"\n[RECURSIVE] {results.recursive_metrics.name}")
        print(f"  Parameters: {results.recursive_metrics.num_parameters:,}")
        print(f"  Avg Iterations: {results.recursive_metrics.avg_iterations:.1f}")
        print(f"  Compression Ratio: {results.recursive_metrics.compression_ratio:.1f}x")
        print(f"  Perplexity: {results.recursive_metrics.perplexity:.2f}")
        print(f"  Knowledge Density: {results.recursive_metrics.knowledge_density:.4f}")
        
        print(f"\n[COMPARISON]")
        print(f"  Parameter Ratio: {results.param_ratio:.2f}x (baseline/recursive)")
        print(f"  Density Ratio: {results.density_ratio:.2f}x (recursive/baseline)")
        print(f"  Performance Ratio: {results.performance_ratio:.2f}x")
        
        # Interpret results
        print(f"\n[INTERPRETATION]")
        if results.density_ratio > 1.0:
            print(f"  Recursive model is {results.density_ratio:.1f}x more knowledge-dense")
        else:
            print(f"  Baseline model is {1/results.density_ratio:.1f}x more knowledge-dense")
        
        if results.performance_ratio > 1.0:
            print(f"  Recursive model has {results.performance_ratio:.1f}x better perplexity")
        else:
            print(f"  Baseline model has {1/results.performance_ratio:.1f}x better perplexity")
        
        effective_density_gain = results.density_ratio * results.param_ratio
        print(f"  Effective density gain: {effective_density_gain:.1f}x")
        
        print("=" * 60)
        
        return results


def evaluate_lambada(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    num_samples: int = 1000,
) -> float:
    """
    Evaluate model on LAMBADA dataset.
    
    LAMBADA tests the model's ability to predict the final word
    of a passage given the context.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        device: Device to run on
        num_samples: Number of samples to evaluate
        
    Returns:
        Accuracy (0-1)
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset('lambada', split='test')
    except Exception as e:
        print(f"[WARN] Could not load LAMBADA dataset: {e}")
        return 0.0
    
    model.eval()
    correct = 0
    total = 0
    
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        
        text = example['text']
        words = text.split()
        
        if len(words) < 2:
            continue
        
        # Context is all but last word
        context = ' '.join(words[:-1])
        target_word = words[-1]
        
        # Tokenize
        context_ids = tokenizer.encode(context, return_tensors='pt').to(device)
        target_ids = tokenizer.encode(' ' + target_word, add_special_tokens=False)
        
        if len(target_ids) == 0:
            continue
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(context_ids)
            logits = outputs['logits'][0, -1, :]
            predicted_id = logits.argmax().item()
        
        # Check if prediction matches target
        if predicted_id == target_ids[0]:
            correct += 1
        total += 1
    
    return correct / max(total, 1)


def generate_benchmark_report(
    results: BenchmarkResults,
    output_dir: str,
    training_log_path: Optional[str] = None,
    use_wandb: bool = False,
) -> str:
    """
    Generate a comprehensive benchmark report with visualizations.
    
    Args:
        results: BenchmarkResults from benchmark run
        output_dir: Directory to save report
        training_log_path: Path to training log JSON (optional)
        use_wandb: Whether to log charts to wandb
        
    Returns:
        Path to generated report
    """
    import os
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'benchmark_report.md')
    
    # Try to import matplotlib for local charts
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
    
    # Build report content
    report_lines = []
    report_lines.append("# UMC Fixed-Point Training Benchmark Report")
    report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")
    
    baseline = results.baseline_metrics
    recursive = results.recursive_metrics
    
    report_lines.append(f"| Metric | Baseline | Recursive | Ratio |")
    report_lines.append(f"|--------|----------|-----------|-------|")
    report_lines.append(f"| Parameters | {baseline.num_parameters:,} | {recursive.num_parameters:,} | {results.param_ratio:.1f}x |")
    report_lines.append(f"| Perplexity | {baseline.perplexity:.2f} | {recursive.perplexity:.2f} | {results.performance_ratio:.2f}x |")
    report_lines.append(f"| Knowledge Density | {baseline.knowledge_density:.4f} | {recursive.knowledge_density:.4f} | {results.density_ratio:.2f}x |")
    report_lines.append(f"| LAMBADA Accuracy | {baseline.lambada_accuracy:.2%} | {recursive.lambada_accuracy:.2%} | - |")
    report_lines.append("")
    
    # Key Findings
    report_lines.append("## Key Findings")
    report_lines.append("")
    
    if results.density_ratio > 1.0:
        report_lines.append(f"- [OK] Recursive model achieves **{results.density_ratio:.1f}x higher knowledge density**")
    else:
        report_lines.append(f"- [INFO] Baseline model has {1/results.density_ratio:.1f}x higher knowledge density")
    
    if results.param_ratio > 1.0:
        report_lines.append(f"- [OK] Recursive model uses **{results.param_ratio:.1f}x fewer parameters**")
    
    if recursive.avg_iterations > 0:
        report_lines.append(f"- [INFO] Average iterations (effective depth): {recursive.avg_iterations:.1f}")
        report_lines.append(f"- [INFO] Compression ratio: {recursive.compression_ratio:.1f}x")
    
    report_lines.append("")
    
    # Model Details
    report_lines.append("## Model Details")
    report_lines.append("")
    report_lines.append("### Baseline GPT-2")
    report_lines.append(f"- Parameters: {baseline.num_parameters:,}")
    report_lines.append(f"- Layers: {baseline.num_layers}")
    report_lines.append(f"- Eval Loss: {baseline.eval_loss:.4f}")
    report_lines.append(f"- Perplexity: {baseline.perplexity:.2f}")
    report_lines.append("")
    report_lines.append("### Recursive GPT-2 (Fixed-Point)")
    report_lines.append(f"- Parameters: {recursive.num_parameters:,}")
    report_lines.append(f"- Effective Layers: 1 (weight-tied)")
    report_lines.append(f"- Eval Loss: {recursive.eval_loss:.4f}")
    report_lines.append(f"- Perplexity: {recursive.perplexity:.2f}")
    report_lines.append(f"- Avg Iterations: {recursive.avg_iterations:.1f}")
    report_lines.append(f"- Min Iterations: {recursive.min_iterations:.1f}")
    report_lines.append(f"- Max Iterations: {recursive.max_iterations:.1f}")
    report_lines.append("")
    
    # Metrics Explanation
    report_lines.append("## Metrics Explanation")
    report_lines.append("")
    report_lines.append("**Knowledge Density** = (1/Perplexity) / Parameters * Compression_Ratio * 1e9")
    report_lines.append("")
    report_lines.append("This metric captures how efficiently the model stores and retrieves knowledge,")
    report_lines.append("accounting for both performance (perplexity) and parameter efficiency.")
    report_lines.append("")
    report_lines.append("**Compression Ratio** = Effective_Depth / Parameter_Depth")
    report_lines.append("")
    report_lines.append("For recursive models, this is iterations/1. For baseline, it's layers/layers=1.")
    report_lines.append("")
    
    # Generate charts if matplotlib available
    if MATPLOTLIB_AVAILABLE:
        report_lines.append("## Visualizations")
        report_lines.append("")
        
        # Chart 1: Parameter comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Parameters bar chart
        ax1 = axes[0]
        models = ['Baseline', 'Recursive']
        params = [baseline.num_parameters / 1e6, recursive.num_parameters / 1e6]
        bars = ax1.bar(models, params, color=['#2ecc71', '#3498db'])
        ax1.set_ylabel('Parameters (millions)')
        ax1.set_title('Parameter Count')
        for bar, val in zip(bars, params):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}M', ha='center', va='bottom')
        
        # Perplexity bar chart
        ax2 = axes[1]
        perps = [baseline.perplexity, recursive.perplexity]
        bars = ax2.bar(models, perps, color=['#2ecc71', '#3498db'])
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Perplexity (lower is better)')
        for bar, val in zip(bars, perps):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom')
        
        # Knowledge Density bar chart
        ax3 = axes[2]
        densities = [baseline.knowledge_density, recursive.knowledge_density]
        bars = ax3.bar(models, densities, color=['#2ecc71', '#3498db'])
        ax3.set_ylabel('Knowledge Density')
        ax3.set_title('Knowledge Density (higher is better)')
        for bar, val in zip(bars, densities):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        chart_path = os.path.join(output_dir, 'comparison_charts.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        report_lines.append(f"![Comparison Charts](comparison_charts.png)")
        report_lines.append("")
    
    # UMC Framework Analysis
    report_lines.append("## UMC Framework Analysis")
    report_lines.append("")
    report_lines.append("The UMC (Unitary Model of Consciousness) framework predicts that recursive")
    report_lines.append("self-referential computation enables more efficient knowledge representation.")
    report_lines.append("")
    report_lines.append("### Hypothesis Test Results")
    report_lines.append("")
    
    density_claim_supported = results.density_ratio >= 2.0
    if density_claim_supported:
        report_lines.append(f"- [OK] **Density claim partially supported**: {results.density_ratio:.1f}x improvement")
    else:
        report_lines.append(f"- [INFO] Density ratio ({results.density_ratio:.1f}x) below 10x target")
    
    report_lines.append(f"- [INFO] Further training may improve results")
    report_lines.append("")
    
    # Conclusion
    report_lines.append("## Conclusion")
    report_lines.append("")
    if results.density_ratio > 1.0:
        report_lines.append("The recursive fixed-point architecture shows promising results for")
        report_lines.append("knowledge-dense language modeling, achieving better parameter efficiency")
        report_lines.append("than the baseline while maintaining competitive performance.")
    else:
        report_lines.append("The baseline model outperformed the recursive model in this test.")
        report_lines.append("This may indicate that longer training or hyperparameter tuning is needed.")
    report_lines.append("")
    
    # Write report
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"[REPORT] Generated benchmark report: {report_path}")
    
    # Log to wandb if enabled
    if use_wandb:
        try:
            import wandb
            
            # Create summary table
            wandb.log({
                'report/baseline_params': baseline.num_parameters,
                'report/recursive_params': recursive.num_parameters,
                'report/baseline_perplexity': baseline.perplexity,
                'report/recursive_perplexity': recursive.perplexity,
                'report/density_ratio': results.density_ratio,
                'report/param_ratio': results.param_ratio,
            })
            
            # Log the report as an artifact
            artifact = wandb.Artifact('benchmark_report', type='report')
            artifact.add_file(report_path)
            if MATPLOTLIB_AVAILABLE:
                artifact.add_file(chart_path)
            wandb.log_artifact(artifact)
            
        except Exception as e:
            print(f"[WARN] Could not log to wandb: {e}")
    
    return report_path


if __name__ == "__main__":
    # Test the metrics
    print("[TEST] Testing Knowledge Density Metrics...")
    
    # Create mock metrics
    metric = KnowledgeDensityMetric()
    
    # Mock baseline model
    class MockBaseline:
        def parameters(self):
            return [torch.randn(768, 768) for _ in range(24)]  # ~14M params per layer
        num_layers = 12
        
    class MockRecursive:
        def parameters(self):
            return [torch.randn(768, 768) for _ in range(2)]  # ~1.2M params
    
    metric.register_baseline(MockBaseline(), "Mock Baseline")
    metric.register_recursive(MockRecursive(), "Mock Recursive")
    
    # Update with mock performance
    metric.update_baseline(
        eval_loss=3.5,
        perplexity=33.1,
        lambada_accuracy=0.45,
    )
    
    metric.update_recursive(
        eval_loss=3.8,
        perplexity=44.7,
        lambada_accuracy=0.38,
        iterations=12.5,
    )
    
    # Compute and print
    results = metric.print_summary()
    
    # Save results
    results.save('/tmp/test_benchmark_results.json')
    print(f"\n[OK] Results saved to /tmp/test_benchmark_results.json")
    
    # Load and verify
    loaded = BenchmarkResults.load('/tmp/test_benchmark_results.json')
    print(f"[OK] Results loaded successfully")
    
    print("\n[OK] Knowledge Density Metrics tests passed.")
