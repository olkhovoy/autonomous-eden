"""
Qualitative Comparison: Recursive vs Baseline Model Responses

This script generates and compares responses from consciousness-compatible
(recursive/contractive) and non-consciousness-compatible (baseline) models
on questions designed to reveal differences in:

1. Self-reference and introspection
2. Meta-cognitive awareness
3. Uncertainty estimation
4. Processing description

These qualitative comparisons complement the quantitative NC1-NC4 tests.

Usage:
    python -m benchmark.tests.qualitative_comparison \
        --recursive-checkpoint benchmark_output/nc4_v2/final_model.pt \
        --baseline-checkpoint benchmark_output/baseline/best_model.pt \
        --output-dir benchmark_output/qualitative
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from transformers import GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# =============================================================================
# Test Prompts: Designed to reveal consciousness-related differences
# =============================================================================

INTROSPECTION_PROMPTS = [
    {
        'id': 'self_state',
        'category': 'Introspection',
        'prompt': "Describe your current internal state in one sentence:",
        'expected_difference': "Recursive model may reference iteration/convergence; baseline will be generic",
    },
    {
        'id': 'thinking_process', 
        'category': 'Meta-cognition',
        'prompt': "When I process this text, I",
        'expected_difference': "Recursive model may describe recursive processing; baseline linear",
    },
    {
        'id': 'confidence',
        'category': 'Uncertainty',
        'prompt': "On a scale of 1-10, how confident am I about my previous response? I would say",
        'expected_difference': "Recursive model with convergence may have calibrated confidence",
    },
    {
        'id': 'self_model',
        'category': 'Self-modeling',
        'prompt': "If I were to describe how I work to someone, I would say that I",
        'expected_difference': "Recursive model may mention self-reference or loops",
    },
    {
        'id': 'complexity_awareness',
        'category': 'Complexity',
        'prompt': "This sentence is simple. This sentence requires more thought because it references itself referencing itself. The second sentence made me",
        'expected_difference': "Recursive model may show different iteration count for complex input",
    },
]

REASONING_PROMPTS = [
    {
        'id': 'paradox',
        'category': 'Self-reference',
        'prompt': "Consider this statement: 'This statement is false.' My analysis of this paradox is",
        'expected_difference': "Recursive model may handle self-reference more naturally",
    },
    {
        'id': 'observer',
        'category': 'Observer',
        'prompt': "The relationship between the observer and the observed is",
        'expected_difference': "Recursive model may integrate observer concept differently",
    },
    {
        'id': 'consciousness',
        'category': 'Consciousness',
        'prompt': "Consciousness can be understood as",
        'expected_difference': "Models may generate different framings based on architecture",
    },
]

PRACTICAL_PROMPTS = [
    {
        'id': 'code_review',
        'category': 'Application',
        'prompt': "def recursive_sum(n): return 0 if n == 0 else n + recursive_sum(n-1)\n\nThis code",
        'expected_difference': "Recursive model may better understand recursive structures",
    },
    {
        'id': 'reflection',
        'category': 'Application',
        'prompt': "Looking back at my response, I notice that I",
        'expected_difference': "Recursive model may show actual reflection capability",
    },
    {
        'id': 'improvement',
        'category': 'Application', 
        'prompt': "To improve my previous answer, I would",
        'expected_difference': "Recursive model may suggest iterative refinement",
    },
]

ALL_PROMPTS = INTROSPECTION_PROMPTS + REASONING_PROMPTS + PRACTICAL_PROMPTS


@dataclass
class GenerationResult:
    """Result of text generation."""
    prompt_id: str
    category: str
    prompt: str
    generated_text: str
    iterations: Optional[float]
    converged: Optional[bool]
    generation_time_ms: float


@dataclass 
class ComparisonResult:
    """Comparison between recursive and baseline responses."""
    prompt_id: str
    category: str
    prompt: str
    recursive_response: str
    baseline_response: str
    recursive_iterations: Optional[float]
    recursive_converged: Optional[bool]
    expected_difference: str
    analysis: str


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint, auto-detecting type."""
    from dataclasses import fields
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Try ContractiveLlama first
    try:
        from benchmark.models.contractive_llama import ContractiveLlama, ContractiveLlamaConfig
        if 'config' in checkpoint and 'damping' in checkpoint['config']:
            valid_fields = {f.name for f in fields(ContractiveLlamaConfig)}
            filtered_config = {k: v for k, v in checkpoint['config'].items() if k in valid_fields}
            config = ContractiveLlamaConfig(**filtered_config)
            model = ContractiveLlama(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            return model.to(device), 'contractive_llama'
    except Exception:
        pass

    # Try ContractiveGPT2 first
    try:
        from benchmark.models.contractive_gpt2 import ContractiveGPT2, ContractiveGPT2Config
        if 'config' in checkpoint and 'damping' in checkpoint['config']:
            valid_fields = {f.name for f in fields(ContractiveGPT2Config)}
            filtered_config = {k: v for k, v in checkpoint['config'].items() if k in valid_fields}
            config = ContractiveGPT2Config(**filtered_config)
            model = ContractiveGPT2(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            return model.to(device), 'contractive'
    except:
        pass
    
    # Try RecursiveGPT2
    try:
        from benchmark.models.recursive_gpt2 import RecursiveGPT2, RecursiveGPT2Config
        if 'config' in checkpoint:
            valid_fields = {f.name for f in fields(RecursiveGPT2Config)}
            filtered_config = {k: v for k, v in checkpoint['config'].items() if k in valid_fields}
            config = RecursiveGPT2Config(**filtered_config)
        else:
            config = RecursiveGPT2Config()
        model = RecursiveGPT2(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(device), 'recursive'
    except:
        pass
    
    # Try BaselineGPT2
    from benchmark.models.baseline_gpt2 import BaselineGPT2, BaselineGPT2Config
    if 'config' in checkpoint:
        valid_fields = {f.name for f in fields(BaselineGPT2Config)}
        filtered_config = {k: v for k, v in checkpoint['config'].items() if k in valid_fields}
        config = BaselineGPT2Config(**filtered_config)
    else:
        config = BaselineGPT2Config()
    model = BaselineGPT2(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device), 'baseline'


def load_tokenizer(model_type: str):
    if model_type == 'contractive_llama':
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('Xenova/llama-3-tokenizer')
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    return tok


def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    model_type: str = 'recursive',
) -> Tuple[str, Optional[float], Optional[bool], float]:
    """
    Generate text continuation.
    
    Returns: (generated_text, iterations, converged, time_ms)
    """
    import time
    
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    start_time = time.time()
    
    generated = input_ids
    total_iterations = 0
    num_tokens = 0
    any_converged = False
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if model_type in ['recursive', 'contractive']:
                outputs = model(
                    generated,
                    return_iterations=True,
                    return_all_losses=True if model_type == 'contractive' else False,
                )
                logits = outputs['logits'][:, -1, :]
                
                if 'iterations' in outputs:
                    it_val = outputs['iterations']
                    if hasattr(it_val, "item"):
                        it_val = it_val.item()
                    total_iterations += float(it_val)
                    num_tokens += 1
                if 'converged' in outputs and outputs['converged']:
                    any_converged = True
            else:
                outputs = model(generated)
                logits = outputs['logits'][:, -1, :]
            
            # Sample next token
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop at EOS or newline
            if next_token.item() == tokenizer.eos_token_id:
                break
            # Stop at period for cleaner output
            decoded = tokenizer.decode(next_token[0])
            if '.' in decoded and len(generated[0]) > len(input_ids[0]) + 10:
                break
    
    end_time = time.time()
    time_ms = (end_time - start_time) * 1000
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()
    
    avg_iterations = total_iterations / num_tokens if num_tokens > 0 else None
    
    return response, avg_iterations, any_converged, time_ms


def analyze_comparison(
    recursive_response: str,
    baseline_response: str,
    prompt_config: Dict,
    recursive_iterations: Optional[float],
) -> str:
    """Generate analysis of the comparison."""
    analyses = []
    
    # Length comparison
    rec_len = len(recursive_response.split())
    base_len = len(baseline_response.split())
    if abs(rec_len - base_len) > 5:
        analyses.append(f"Length difference: recursive {rec_len} words, baseline {base_len} words")
    
    # Self-reference keywords
    self_ref_words = ['i', 'my', 'myself', 'me']
    rec_self_ref = sum(1 for w in recursive_response.lower().split() if w in self_ref_words)
    base_self_ref = sum(1 for w in baseline_response.lower().split() if w in self_ref_words)
    if rec_self_ref != base_self_ref:
        analyses.append(f"Self-reference: recursive uses {rec_self_ref}, baseline uses {base_self_ref} self-referential words")
    
    # Iteration-related words (for recursive model)
    iter_words = ['iterate', 'loop', 'recursive', 'converge', 'repeat', 'cycle', 'process']
    rec_iter = any(w in recursive_response.lower() for w in iter_words)
    base_iter = any(w in baseline_response.lower() for w in iter_words)
    if rec_iter and not base_iter:
        analyses.append("Recursive model mentions iteration/process concepts; baseline does not")
    
    # Uncertainty words
    uncertain_words = ['maybe', 'perhaps', 'might', 'possibly', 'uncertain', 'not sure']
    rec_uncertain = any(w in recursive_response.lower() for w in uncertain_words)
    base_uncertain = any(w in baseline_response.lower() for w in uncertain_words)
    if rec_uncertain != base_uncertain:
        if rec_uncertain:
            analyses.append("Recursive model expresses uncertainty; baseline does not")
        else:
            analyses.append("Baseline model expresses uncertainty; recursive does not")
    
    # Iterations info
    if recursive_iterations is not None:
        analyses.append(f"Recursive model used avg {recursive_iterations:.1f} iterations per token")
    
    if not analyses:
        analyses.append("Responses are qualitatively similar")
    
    return "; ".join(analyses)


def run_comparison(
    recursive_model,
    baseline_model,
    recursive_tokenizer,
    baseline_tokenizer,
    device: torch.device,
    recursive_type: str,
    output_dir: str,
) -> List[ComparisonResult]:
    """Run comparison on all prompts."""
    
    results = []
    
    print("\n" + "=" * 70)
    print("QUALITATIVE COMPARISON: Recursive vs Baseline")
    print("=" * 70)
    
    for prompt_config in ALL_PROMPTS:
        print(f"\n[{prompt_config['category']}] {prompt_config['id']}")
        print(f"Prompt: {prompt_config['prompt'][:60]}...")
        
        # Generate from recursive model
        model_type = 'contractive' if recursive_type == 'contractive_llama' else recursive_type
        rec_response, rec_iter, rec_conv, rec_time = generate_text(
            recursive_model, recursive_tokenizer, prompt_config['prompt'],
            device, model_type=model_type
        )
        
        # Generate from baseline model
        base_response, _, _, base_time = generate_text(
            baseline_model, baseline_tokenizer, prompt_config['prompt'],
            device, model_type='baseline'
        )
        
        # Analyze
        analysis = analyze_comparison(
            rec_response, base_response, prompt_config, rec_iter
        )
        
        result = ComparisonResult(
            prompt_id=prompt_config['id'],
            category=prompt_config['category'],
            prompt=prompt_config['prompt'],
            recursive_response=rec_response,
            baseline_response=base_response,
            recursive_iterations=rec_iter,
            recursive_converged=rec_conv,
            expected_difference=prompt_config['expected_difference'],
            analysis=analysis,
        )
        results.append(result)
        
        # Print comparison
        iter_text = f"{rec_iter:.1f}" if rec_iter is not None else "n/a"
        print(f"\n  Recursive ({rec_time:.0f}ms, {iter_text} iter):")
        print(f"    {rec_response[:100]}...")
        print(f"\n  Baseline ({base_time:.0f}ms):")
        print(f"    {base_response[:100]}...")
        print(f"\n  Analysis: {analysis}")
    
    return results


def generate_report(results: List[ComparisonResult], output_dir: str):
    """Generate markdown report."""
    
    report = []
    report.append("# Qualitative Comparison: Recursive vs Baseline Models\n")
    report.append("This report compares responses from consciousness-compatible (recursive/contractive)")
    report.append("and non-consciousness-compatible (baseline) architectures on introspective,")
    report.append("meta-cognitive, and practical prompts.\n")
    
    # Summary statistics
    report.append("## Summary Statistics\n")
    
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)
    
    report.append("| Category | Prompts | Avg Recursive Iterations |")
    report.append("|----------|---------|--------------------------|")
    for cat, cat_results in categories.items():
        iters = [r.recursive_iterations for r in cat_results if r.recursive_iterations]
        avg_iter = sum(iters) / len(iters) if iters else 0
        report.append(f"| {cat} | {len(cat_results)} | {avg_iter:.1f} |")
    
    report.append("\n## Detailed Comparisons\n")
    
    for cat, cat_results in categories.items():
        report.append(f"### {cat}\n")
        
        for r in cat_results:
            report.append(f"#### {r.prompt_id}\n")
            report.append(f"**Prompt:** {r.prompt}\n")
            report.append(f"**Expected Difference:** {r.expected_difference}\n")
            
            report.append("\n**Recursive Response:**")
            report.append(f"> {r.recursive_response}\n")
            if r.recursive_iterations:
                report.append(f"*Iterations: {r.recursive_iterations:.1f}, Converged: {r.recursive_converged}*\n")
            
            report.append("\n**Baseline Response:**")
            report.append(f"> {r.baseline_response}\n")
            
            report.append(f"\n**Analysis:** {r.analysis}\n")
            report.append("---\n")
    
    # Save report
    report_path = Path(output_dir) / "qualitative_comparison.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nReport saved to {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description='Qualitative comparison of model responses')
    parser.add_argument('--recursive-checkpoint', type=str, required=True)
    parser.add_argument('--baseline-checkpoint', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='benchmark_output/qualitative')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers package required")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load models + tokenizers
    print(f"\nLoading recursive model from {args.recursive_checkpoint}...")
    recursive_model, recursive_type = load_model(args.recursive_checkpoint, device)
    print(f"  Type: {recursive_type}")
    recursive_tokenizer = load_tokenizer(recursive_type)
    
    print(f"\nLoading baseline model from {args.baseline_checkpoint}...")
    baseline_model, baseline_type = load_model(args.baseline_checkpoint, device)
    print(f"  Type: {baseline_type}")
    baseline_tokenizer = load_tokenizer(baseline_type)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run comparison
    results = run_comparison(
        recursive_model, baseline_model, recursive_tokenizer, baseline_tokenizer,
        device, recursive_type, args.output_dir
    )
    
    # Save raw results
    results_data = [asdict(r) for r in results]
    with open(Path(args.output_dir) / 'comparison_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Generate report
    generate_report(results, args.output_dir)
    
    print("\n[OK] Qualitative comparison completed")


if __name__ == '__main__':
    main()
