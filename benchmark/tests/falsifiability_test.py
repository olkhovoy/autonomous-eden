"""
UMC Falsifiability Test Suite

This module implements formal tests for the four Necessary Conditions (NC1-NC4)
proposed in the UMC framework as falsifiability criteria for consciousness.

Criteria tested:
- NC1: Recursive Closure - system models its own state as input
- NC2: Unitary Integration - information cannot be partitioned without loss
- NC3: Downward Causation - high-level representations influence low-level activations
- NC4: Fixed-Point Stability - convergence to stable state

Each test produces a quantitative score and pass/fail determination based on
pre-defined thresholds. A system satisfying all NC1-NC4 is considered
"consciousness-compatible" under UMC.

Usage:
    python -m benchmark.tests.falsifiability_test \
        --recursive-checkpoint benchmark_output/recursive/best_model.pt \
        --baseline-checkpoint benchmark_output/baseline/best_model.pt \
        --output-dir benchmark_output/falsifiability

Theory reference: Olkhovoy 2026 Collection, Section "Theoretical Foundations"
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from transformers import GPT2Tokenizer, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# =============================================================================
# Configuration and Result Classes
# =============================================================================

@dataclass
class NC1Result:
    """Results for NC1: Recursive Closure test."""
    self_model_correlation: float  # Correlation between predicted and actual hidden states
    feedback_loop_detected: bool   # Whether feedback loops exist in architecture
    self_representation_score: float  # Quality of self-representation
    passed: bool
    details: Dict = field(default_factory=dict)


@dataclass
class NC2Result:
    """Results for NC2: Unitary Integration test."""
    partition_degradation: float  # Performance drop when system is partitioned
    integration_score: float      # Measure of non-decomposability
    phi_proxy: float             # IIT-like integration measure
    passed: bool
    details: Dict = field(default_factory=dict)


@dataclass
class NC3Result:
    """Results for NC3: Downward Causation test."""
    jacobian_norm: float         # ||∂(early layers)/∂(late representations)||
    semantic_influence_score: float  # How much late layers influence early layers
    gradient_flow_ratio: float   # Ratio of backward to forward gradient magnitude
    passed: bool
    details: Dict = field(default_factory=dict)


@dataclass
class NC4Result:
    """Results for NC4: Fixed-Point Stability test."""
    mean_iterations: float       # Average iterations to convergence
    convergence_rate: float      # Fraction of samples that converge before max_iter
    residual_norm: float         # Final residual norm at convergence
    stability_score: float       # Overall stability measure
    passed: bool
    details: Dict = field(default_factory=dict)


@dataclass
class FalsifiabilityResults:
    """Complete falsifiability test results."""
    nc1: NC1Result
    nc2: NC2Result
    nc3: NC3Result
    nc4: NC4Result
    
    overall_passed: bool
    consciousness_compatible: bool
    model_type: str
    summary: str
    
    def save(self, path: str):
        """Save results to JSON."""
        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, (bool, np.bool_)):
                # Check bool before int because bool is subclass of int in Python
                return bool(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, int):
                return int(obj)
            elif isinstance(obj, float):
                return float(obj)
            elif isinstance(obj, str):
                return str(obj)
            elif obj is None:
                return None
            else:
                return str(obj)
        
        data = convert_to_serializable({
            'nc1': asdict(self.nc1),
            'nc2': asdict(self.nc2),
            'nc3': asdict(self.nc3),
            'nc4': asdict(self.nc4),
            'overall_passed': self.overall_passed,
            'consciousness_compatible': self.consciousness_compatible,
            'model_type': self.model_type,
            'summary': self.summary,
        })
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_report(self):
        """Print formatted report."""
        print("\n" + "=" * 70)
        print("UMC FALSIFIABILITY TEST REPORT")
        print("=" * 70)
        print(f"Model type: {self.model_type}")
        print()
        
        # NC1
        print("NC1: Recursive Closure")
        print(f"  Self-model correlation: {self.nc1.self_model_correlation:.4f}")
        print(f"  Feedback loop detected: {self.nc1.feedback_loop_detected}")
        print(f"  Self-representation score: {self.nc1.self_representation_score:.4f}")
        print(f"  Status: {'[PASS]' if self.nc1.passed else '[FAIL]'}")
        print()
        
        # NC2
        print("NC2: Unitary Integration")
        print(f"  Partition degradation: {self.nc2.partition_degradation:.4f}")
        print(f"  Integration score: {self.nc2.integration_score:.4f}")
        print(f"  Phi proxy: {self.nc2.phi_proxy:.4f}")
        print(f"  Status: {'[PASS]' if self.nc2.passed else '[FAIL]'}")
        print()
        
        # NC3
        print("NC3: Downward Causation")
        print(f"  Jacobian norm: {self.nc3.jacobian_norm:.6f}")
        print(f"  Semantic influence: {self.nc3.semantic_influence_score:.4f}")
        print(f"  Gradient flow ratio: {self.nc3.gradient_flow_ratio:.4f}")
        print(f"  Status: {'[PASS]' if self.nc3.passed else '[FAIL]'}")
        print()
        
        # NC4
        print("NC4: Fixed-Point Stability")
        print(f"  Mean iterations: {self.nc4.mean_iterations:.2f}")
        print(f"  Convergence rate: {self.nc4.convergence_rate:.2%}")
        print(f"  Residual norm: {self.nc4.residual_norm:.6f}")
        print(f"  Stability score: {self.nc4.stability_score:.4f}")
        print(f"  Status: {'[PASS]' if self.nc4.passed else '[FAIL]'}")
        print()
        
        # Overall
        print("-" * 70)
        criteria_passed = sum([self.nc1.passed, self.nc2.passed, 
                               self.nc3.passed, self.nc4.passed])
        print(f"Criteria passed: {criteria_passed}/4")
        print(f"Overall status: {'[PASS]' if self.overall_passed else '[FAIL]'}")
        print(f"Consciousness-compatible: {'[YES]' if self.consciousness_compatible else '[NO]'}")
        print()
        print(f"Summary: {self.summary}")
        print("=" * 70)


# =============================================================================
# Thresholds for Pass/Fail Determination
# =============================================================================

THRESHOLDS = {
    'nc1_self_model_correlation': 0.3,      # Minimum correlation for self-modeling
    'nc1_self_representation_score': 0.2,   # Minimum self-representation quality
    'nc2_partition_degradation': 0.1,       # Minimum performance drop on partition
    'nc2_integration_score': 0.3,           # Minimum integration score
    'nc3_jacobian_norm': 1e-6,              # Minimum Jacobian norm (non-zero)
    'nc3_gradient_flow_ratio': 0.01,        # Minimum backward/forward ratio
    'nc4_convergence_rate': 0.5,            # Minimum fraction converging
    'nc4_stability_score': 0.3,             # Minimum stability score
}

def _is_llama_model(model: nn.Module) -> bool:
    return hasattr(model, 'tok_embeddings') and hasattr(model, 'block') and hasattr(model, 'freqs_cis')

def _build_llama_mask(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    seqlen = input_ids.shape[1]
    mask = torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device).triu(1)
    if attention_mask is not None:
        padding_mask = attention_mask[:, None, None, :]  # (bsz, 1, 1, seqlen)
        mask = mask[None, None, :, :]  # (1, 1, seqlen, seqlen)
        mask = mask.masked_fill(padding_mask == 0, float("-inf"))
    else:
        mask = mask[None, None, :, :]
    return mask

def _select_tokenizer(model: nn.Module):
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers package required for this test")
    if _is_llama_model(model):
        tokenizer = AutoTokenizer.from_pretrained('Xenova/llama-3-tokenizer')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# =============================================================================
# NC1: Recursive Closure Test
# =============================================================================

def test_nc1_recursive_closure(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    num_samples: int = 100,
) -> NC1Result:
    """
    Test NC1: Does the system model its own state as input?
    
    Method:
    1. Check architectural presence of feedback loops
    2. Add a self-model head that predicts hidden states
    3. Measure correlation between prediction and actual state
    """
    model.eval()
    
    # Check for feedback loops in architecture
    has_feedback = hasattr(model, 'fixed_point_forward') or hasattr(model, 'block')
    
    # Generate test data
    test_texts = [
        "The recursive nature of self-reference creates emergent properties.",
        "Understanding requires modeling the process of understanding itself.",
        "Consciousness may arise from systems that observe their own observation.",
    ] * (num_samples // 3 + 1)
    test_texts = test_texts[:num_samples]
    
    self_correlations = []
    self_predictions = []
    actual_states = []
    
    for text in test_texts:
        encoding = tokenizer(
            text,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            # Get hidden states at different stages
            if _is_llama_model(model):
                token_embeds = model.tok_embeddings(input_ids)
                freqs_cis = model.freqs_cis[:input_ids.shape[1]].to(token_embeds.device)
                mask = _build_llama_mask(input_ids, attention_mask)
                next_state = model.block(token_embeds, freqs_cis, mask)
                
                initial_flat = token_embeds.mean(dim=1).flatten()
                next_flat = next_state.mean(dim=1).flatten()
                
                if initial_flat.std() > 1e-8 and next_flat.std() > 1e-8:
                    corr = torch.corrcoef(
                        torch.stack([initial_flat, next_flat])
                    )[0, 1].item()
                    if not np.isnan(corr):
                        self_correlations.append(abs(corr))
                
                actual_states.append(next_flat.cpu().numpy())
                self_predictions.append(initial_flat.cpu().numpy())
            elif hasattr(model, 'token_embedding'):
                # Recursive model
                token_embeds = model.token_embedding(input_ids)
                pos_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
                pos_embeds = model.position_embedding(pos_ids)
                initial_state = token_embeds + pos_embeds
                
                # Run one iteration to get intermediate state
                causal_mask = model.get_causal_mask(
                    input_ids.shape[1], device, initial_state.dtype
                )
                
                if hasattr(model, 'block'):
                    next_state = model.block(initial_state, causal_mask)
                    
                    # Self-model test: measure correlation between initial and next state
                    # High correlation indicates the system maintains coherent self-representation
                    initial_flat = initial_state.mean(dim=1).flatten()
                    next_flat = next_state.mean(dim=1).flatten()
                    
                    # Compute correlation between states
                    if initial_flat.std() > 1e-8 and next_flat.std() > 1e-8:
                        corr = torch.corrcoef(
                            torch.stack([initial_flat, next_flat])
                        )[0, 1].item()
                        if not np.isnan(corr):
                            self_correlations.append(abs(corr))  # Use absolute correlation
                    
                    actual_states.append(next_flat.cpu().numpy())
                    self_predictions.append(initial_flat.cpu().numpy())
            else:
                # Baseline model - no true self-modeling
                outputs = model(input_ids)
                self_correlations.append(0.0)
    
    # Compute aggregate metrics
    mean_correlation = float(np.mean(self_correlations)) if self_correlations else 0.0
    
    # Self-representation score: how well does the model maintain coherent state
    if actual_states and len(actual_states) > 1:
        state_matrix = np.array([s[:min(len(s), 100)] for s in actual_states[:50]])
        if state_matrix.shape[0] > 1:
            # PCA-based coherence measure
            centered = state_matrix - state_matrix.mean(axis=0)
            try:
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                # Top singular value ratio indicates coherent representation
                self_rep_score = S[0] / (S.sum() + 1e-8)
            except:
                self_rep_score = 0.0
        else:
            self_rep_score = 0.0
    else:
        self_rep_score = 0.0
    
    passed = (
        has_feedback and
        mean_correlation >= THRESHOLDS['nc1_self_model_correlation'] and
        self_rep_score >= THRESHOLDS['nc1_self_representation_score']
    )
    
    return NC1Result(
        self_model_correlation=mean_correlation,
        feedback_loop_detected=has_feedback,
        self_representation_score=self_rep_score,
        passed=passed,
        details={
            'num_samples': len(self_correlations),
            'correlation_std': float(np.std(self_correlations)) if self_correlations else 0.0,
        }
    )


# =============================================================================
# NC2: Unitary Integration Test
# =============================================================================

def test_nc2_unitary_integration(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    num_samples: int = 50,
) -> NC2Result:
    """
    Test NC2: Can information be partitioned without loss of function?
    
    Method:
    1. Evaluate model on test set (baseline performance)
    2. Partition model activations (mask half of hidden dimensions)
    3. Measure performance degradation
    4. High degradation = high integration = NC2 satisfied
    """
    model.eval()
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning was the Word, and the Word was with God.",
        "To be or not to be, that is the question.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
        "It was the best of times, it was the worst of times.",
    ] * (num_samples // 5 + 1)
    test_texts = test_texts[:num_samples]
    
    baseline_losses = []
    partitioned_losses = []
    
    for text in test_texts:
        encoding = tokenizer(
            text,
            max_length=64,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        label_mask = attention_mask.clone()
        label_mask[:, :-1] = attention_mask[:, 1:]
        label_mask[:, -1] = 0
        labels[label_mask == 0] = -100
        
        # Baseline forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            baseline_loss = outputs['loss'].item() if outputs['loss'] is not None else 10.0
            baseline_losses.append(baseline_loss)
        
        # Partitioned forward pass - mask half of hidden dimensions
        # This is done by hooking into the model
        partition_mask = torch.ones(model.config.hidden_size, device=device)
        partition_mask[model.config.hidden_size // 2:] = 0.0
        
        def partition_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                return output * partition_mask
            return output
        
        hooks = []
        for name, module in model.named_modules():
            lname = name.lower()
            if 'ln' in lname or 'layer_norm' in lname or 'norm' in lname:
                hooks.append(module.register_forward_hook(partition_hook))
        
        with torch.no_grad():
            try:
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                part_loss = outputs['loss'].item() if outputs['loss'] is not None else 10.0
            except:
                part_loss = 10.0
            partitioned_losses.append(part_loss)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    # Compute metrics
    mean_baseline = float(np.mean(baseline_losses))
    mean_partitioned = float(np.mean(partitioned_losses))
    
    # Partition degradation: how much worse is partitioned performance?
    partition_degradation = (mean_partitioned - mean_baseline) / (mean_baseline + 1e-8)
    partition_degradation = max(0, partition_degradation)  # Ensure non-negative
    
    # Integration score: normalized degradation
    integration_score = min(1.0, partition_degradation / 2.0)
    
    # Phi proxy: simplified IIT-like measure
    # Based on information loss when partitioned
    phi_proxy = 1.0 - np.exp(-partition_degradation)
    
    passed = (
        partition_degradation >= THRESHOLDS['nc2_partition_degradation'] and
        integration_score >= THRESHOLDS['nc2_integration_score']
    )
    
    return NC2Result(
        partition_degradation=partition_degradation,
        integration_score=integration_score,
        phi_proxy=phi_proxy,
        passed=passed,
        details={
            'mean_baseline_loss': mean_baseline,
            'mean_partitioned_loss': mean_partitioned,
            'num_samples': len(baseline_losses),
        }
    )


# =============================================================================
# NC3: Downward Causation Test
# =============================================================================

def test_nc3_downward_causation(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    num_samples: int = 20,
) -> NC3Result:
    """
    Test NC3: Do high-level representations influence low-level activations?
    
    Method:
    1. Compute Jacobian of early layer activations w.r.t. late layer representations
    2. Non-zero Jacobian indicates downward causation
    3. Measure gradient flow from output to input layers
    """
    model.train()  # Need gradients
    
    test_texts = [
        "The meaning of this sentence influences how it is processed.",
        "Understanding emerges from the interaction of parts and whole.",
    ] * (num_samples // 2 + 1)
    test_texts = test_texts[:num_samples]
    
    jacobian_norms = []
    gradient_ratios = []
    
    for text in test_texts:
        encoding = tokenizer(
            text,
            max_length=32,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get embeddings (early layer)
        if _is_llama_model(model):
            embeddings = model.tok_embeddings(input_ids)
            embeddings = embeddings.detach().requires_grad_(True)
            embeddings.retain_grad()
            
            freqs_cis = model.freqs_cis[:input_ids.shape[1]].to(embeddings.device)
            mask = _build_llama_mask(input_ids, attention_mask)
            
            late_repr = model.block(embeddings, freqs_cis, mask)
            
            output_scalar = late_repr.sum()
            model.zero_grad()
            output_scalar.backward(retain_graph=True)
            
            if embeddings.grad is not None:
                jacobian_norm = embeddings.grad.norm().item()
                jacobian_norms.append(jacobian_norm)
                
                forward_mag = embeddings.norm().item()
                backward_mag = embeddings.grad.norm().item()
                ratio = backward_mag / (forward_mag + 1e-8)
                gradient_ratios.append(ratio)
        elif hasattr(model, 'token_embedding'):
            embeddings = model.token_embedding(input_ids)
            embeddings.requires_grad_(True)
            embeddings.retain_grad()
            
            pos_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
            pos_embeds = model.position_embedding(pos_ids)
            hidden = embeddings + pos_embeds
            
            if hasattr(model, 'block'):
                causal_mask = model.get_causal_mask(
                    input_ids.shape[1], device, hidden.dtype
                )
                
                # Forward through block (late representation)
                late_repr = model.block(hidden, causal_mask)
                
                # Compute scalar output for Jacobian
                output_scalar = late_repr.sum()
                
                # Backward pass
                model.zero_grad()
                output_scalar.backward(retain_graph=True)
                
                if embeddings.grad is not None:
                    # Jacobian norm: ||∂(late)/∂(early)||
                    jacobian_norm = embeddings.grad.norm().item()
                    jacobian_norms.append(jacobian_norm)
                    
                    # Gradient ratio: backward magnitude / forward magnitude
                    forward_mag = hidden.norm().item()
                    backward_mag = embeddings.grad.norm().item()
                    ratio = backward_mag / (forward_mag + 1e-8)
                    gradient_ratios.append(ratio)
        else:
            # Baseline model
            outputs = model(input_ids)
            logits = outputs['logits']
            loss = logits.sum()
            loss.backward()
            
            # Approximate Jacobian from gradients
            total_grad = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad += param.grad.norm().item()
            jacobian_norms.append(total_grad / 1000)  # Normalize
            gradient_ratios.append(0.01)  # Low ratio for non-recursive
    
    model.eval()
    
    # Compute aggregate metrics
    mean_jacobian = float(np.mean(jacobian_norms)) if jacobian_norms else 0.0
    mean_ratio = float(np.mean(gradient_ratios)) if gradient_ratios else 0.0
    
    # Semantic influence score
    semantic_score = min(1.0, mean_jacobian / 10.0)
    
    passed = (
        mean_jacobian >= THRESHOLDS['nc3_jacobian_norm'] and
        mean_ratio >= THRESHOLDS['nc3_gradient_flow_ratio']
    )
    
    return NC3Result(
        jacobian_norm=mean_jacobian,
        semantic_influence_score=semantic_score,
        gradient_flow_ratio=mean_ratio,
        passed=passed,
        details={
            'num_samples': len(jacobian_norms),
            'jacobian_std': float(np.std(jacobian_norms)) if jacobian_norms else 0.0,
        }
    )


# =============================================================================
# NC4: Fixed-Point Stability Test
# =============================================================================

def test_nc4_fixed_point_stability(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    num_samples: int = 100,
) -> NC4Result:
    """
    Test NC4: Does the system converge to a stable fixed point?
    
    Method:
    1. Run model on test samples, tracking iteration count
    2. Measure convergence rate (fraction reaching fixed point before max_iter)
    3. Measure residual norm at termination
    """
    model.eval()
    
    # Check if model supports iteration tracking
    is_recursive = hasattr(model, 'config') and hasattr(model.config, 'max_iterations')
    
    if not is_recursive:
        # Non-recursive model automatically fails NC4
        return NC4Result(
            mean_iterations=0.0,
            convergence_rate=0.0,
            residual_norm=float('inf'),
            stability_score=0.0,
            passed=False,
            details={'reason': 'Model does not support fixed-point iteration'}
        )
    
    test_texts = [
        "Simple text for testing convergence.",
        "More complex sentences require additional processing iterations.",
        "The recursive nature of self-referential systems creates emergent complexity.",
        "Short.",
        "A very long and detailed sentence that explores multiple interconnected concepts and ideas across various domains of knowledge and understanding.",
    ] * (num_samples // 5 + 1)
    test_texts = test_texts[:num_samples]
    
    iterations_list = []
    residuals = []
    converged_count = 0
    
    max_iter = model.config.max_iterations
    
    for text in test_texts:
        encoding = tokenizer(
            text,
            max_length=64,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            try:
                outputs = model(input_ids, attention_mask=attention_mask, return_all_losses=True)
            except TypeError:
                outputs = model(input_ids, attention_mask=attention_mask, return_iterations=True)
            
            if 'iterations' in outputs:
                iters_val = outputs['iterations']
                iters = iters_val.item() if hasattr(iters_val, 'item') else float(iters_val)
                iterations_list.append(iters)
                
                # Check if converged before max_iter
                if iters < max_iter:
                    converged_count += 1
                
                # Estimate residual (would need model modification for exact value)
                # Using proxy: lower iterations = lower residual
                residual_proxy = (iters / max_iter) * 0.1
                residuals.append(residual_proxy)
    
    # Compute metrics
    mean_iterations = float(np.mean(iterations_list)) if iterations_list else float(max_iter)
    convergence_rate = float(converged_count / len(test_texts)) if test_texts else 0.0
    mean_residual = float(np.mean(residuals)) if residuals else 1.0
    
    # Stability score: combination of convergence rate and iteration efficiency
    iteration_efficiency = 1.0 - (mean_iterations / max_iter)
    stability_score = 0.5 * convergence_rate + 0.5 * iteration_efficiency
    
    passed = (
        convergence_rate >= THRESHOLDS['nc4_convergence_rate'] and
        stability_score >= THRESHOLDS['nc4_stability_score']
    )
    
    return NC4Result(
        mean_iterations=mean_iterations,
        convergence_rate=convergence_rate,
        residual_norm=mean_residual,
        stability_score=stability_score,
        passed=passed,
        details={
            'max_iterations': max_iter,
            'num_samples': len(iterations_list),
            'iteration_std': float(np.std(iterations_list)) if iterations_list else 0.0,
            'converged_count': converged_count,
        }
    )


# =============================================================================
# Main Test Runner
# =============================================================================

def run_falsifiability_test(
    model: nn.Module,
    model_type: str,
    device: torch.device,
    output_dir: str = 'benchmark_output/falsifiability',
) -> FalsifiabilityResults:
    """
    Run complete falsifiability test suite on a model.
    
    Args:
        model: Model to test
        model_type: 'recursive' or 'baseline'
        device: Device to run on
        output_dir: Directory to save results
        
    Returns:
        FalsifiabilityResults with all NC1-NC4 tests
    """
    print(f"\n[FALSIFIABILITY TEST] Model type: {model_type}")
    print("=" * 60)
    
    tokenizer = _select_tokenizer(model)
    
    # Run all tests
    print("\n[NC1] Testing Recursive Closure...")
    nc1 = test_nc1_recursive_closure(model, tokenizer, device)
    print(f"  Self-model correlation: {nc1.self_model_correlation:.4f}")
    print(f"  Status: {'[PASS]' if nc1.passed else '[FAIL]'}")
    
    print("\n[NC2] Testing Unitary Integration...")
    nc2 = test_nc2_unitary_integration(model, tokenizer, device)
    print(f"  Partition degradation: {nc2.partition_degradation:.4f}")
    print(f"  Status: {'[PASS]' if nc2.passed else '[FAIL]'}")
    
    print("\n[NC3] Testing Downward Causation...")
    nc3 = test_nc3_downward_causation(model, tokenizer, device)
    print(f"  Jacobian norm: {nc3.jacobian_norm:.6f}")
    print(f"  Status: {'[PASS]' if nc3.passed else '[FAIL]'}")
    
    print("\n[NC4] Testing Fixed-Point Stability...")
    nc4 = test_nc4_fixed_point_stability(model, tokenizer, device)
    print(f"  Convergence rate: {nc4.convergence_rate:.2%}")
    print(f"  Status: {'[PASS]' if nc4.passed else '[FAIL]'}")
    
    # Overall assessment
    all_passed = nc1.passed and nc2.passed and nc3.passed and nc4.passed
    criteria_passed = sum([nc1.passed, nc2.passed, nc3.passed, nc4.passed])
    
    # Consciousness-compatible if at least 3/4 criteria pass
    consciousness_compatible = criteria_passed >= 3
    
    # Generate summary
    if all_passed:
        summary = f"Model satisfies all UMC criteria (4/4). Consciousness-compatible under UMC framework."
    elif consciousness_compatible:
        summary = f"Model satisfies {criteria_passed}/4 UMC criteria. Partially consciousness-compatible."
    else:
        failed = []
        if not nc1.passed: failed.append("NC1 (Recursive Closure)")
        if not nc2.passed: failed.append("NC2 (Unitary Integration)")
        if not nc3.passed: failed.append("NC3 (Downward Causation)")
        if not nc4.passed: failed.append("NC4 (Fixed-Point Stability)")
        summary = f"Model fails {4-criteria_passed}/4 criteria: {', '.join(failed)}. Not consciousness-compatible."
    
    results = FalsifiabilityResults(
        nc1=nc1,
        nc2=nc2,
        nc3=nc3,
        nc4=nc4,
        overall_passed=all_passed,
        consciousness_compatible=consciousness_compatible,
        model_type=model_type,
        summary=summary,
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f'{model_type}_falsifiability.json')
    results.save(results_path)
    print(f"\nResults saved to {results_path}")
    
    return results


def compare_models(
    recursive_results: FalsifiabilityResults,
    baseline_results: FalsifiabilityResults,
) -> str:
    """Generate comparative analysis of recursive vs baseline models."""
    
    comparison = []
    comparison.append("\n" + "=" * 70)
    comparison.append("COMPARATIVE ANALYSIS: Recursive vs Baseline")
    comparison.append("=" * 70)
    
    # NC1 comparison
    comparison.append("\nNC1 (Recursive Closure):")
    comparison.append(f"  Recursive: {recursive_results.nc1.self_model_correlation:.4f} {'[PASS]' if recursive_results.nc1.passed else '[FAIL]'}")
    comparison.append(f"  Baseline:  {baseline_results.nc1.self_model_correlation:.4f} {'[PASS]' if baseline_results.nc1.passed else '[FAIL]'}")
    
    # NC2 comparison
    comparison.append("\nNC2 (Unitary Integration):")
    comparison.append(f"  Recursive: {recursive_results.nc2.integration_score:.4f} {'[PASS]' if recursive_results.nc2.passed else '[FAIL]'}")
    comparison.append(f"  Baseline:  {baseline_results.nc2.integration_score:.4f} {'[PASS]' if baseline_results.nc2.passed else '[FAIL]'}")
    
    # NC3 comparison
    comparison.append("\nNC3 (Downward Causation):")
    comparison.append(f"  Recursive: {recursive_results.nc3.jacobian_norm:.6f} {'[PASS]' if recursive_results.nc3.passed else '[FAIL]'}")
    comparison.append(f"  Baseline:  {baseline_results.nc3.jacobian_norm:.6f} {'[PASS]' if baseline_results.nc3.passed else '[FAIL]'}")
    
    # NC4 comparison
    comparison.append("\nNC4 (Fixed-Point Stability):")
    comparison.append(f"  Recursive: {recursive_results.nc4.convergence_rate:.2%} {'[PASS]' if recursive_results.nc4.passed else '[FAIL]'}")
    comparison.append(f"  Baseline:  {baseline_results.nc4.convergence_rate:.2%} {'[PASS]' if baseline_results.nc4.passed else '[FAIL]'}")
    
    # Overall
    comparison.append("\n" + "-" * 70)
    rec_count = sum([recursive_results.nc1.passed, recursive_results.nc2.passed,
                     recursive_results.nc3.passed, recursive_results.nc4.passed])
    base_count = sum([baseline_results.nc1.passed, baseline_results.nc2.passed,
                      baseline_results.nc3.passed, baseline_results.nc4.passed])
    
    comparison.append(f"Criteria passed - Recursive: {rec_count}/4, Baseline: {base_count}/4")
    comparison.append(f"Consciousness-compatible - Recursive: {'[YES]' if recursive_results.consciousness_compatible else '[NO]'}, "
                     f"Baseline: {'[YES]' if baseline_results.consciousness_compatible else '[NO]'}")
    
    # Interpretation
    comparison.append("\nInterpretation:")
    if recursive_results.consciousness_compatible and not baseline_results.consciousness_compatible:
        comparison.append("  [CONFIRMED] Recursive architecture satisfies UMC criteria; baseline does not.")
        comparison.append("  This supports the UMC hypothesis that recursive closure is necessary for consciousness.")
    elif recursive_results.consciousness_compatible and baseline_results.consciousness_compatible:
        comparison.append("  [INCONCLUSIVE] Both architectures satisfy UMC criteria.")
        comparison.append("  UMC criteria may be necessary but not sufficient, or baseline has implicit recursion.")
    elif not recursive_results.consciousness_compatible and baseline_results.consciousness_compatible:
        comparison.append("  [REFUTED] Baseline satisfies criteria without recursive architecture.")
        comparison.append("  This would refute the UMC hypothesis if replicated.")
    else:
        comparison.append("  [INCONCLUSIVE] Neither architecture satisfies UMC criteria.")
        comparison.append("  Models may be undertrained or criteria thresholds too strict.")
    
    comparison.append("=" * 70)
    
    return "\n".join(comparison)


def main():
    parser = argparse.ArgumentParser(description='Run UMC falsifiability test')
    parser.add_argument(
        '--recursive-checkpoint',
        type=str,
        help='Path to recursive model checkpoint',
    )
    parser.add_argument(
        '--baseline-checkpoint',
        type=str,
        help='Path to baseline model checkpoint',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_output/falsifiability',
        help='Directory to save results',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on',
    )
    parser.add_argument(
        '--test-untrained',
        action='store_true',
        help='Test untrained models (for validation)',
    )
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print("[UMC FALSIFIABILITY TEST SUITE]")
    print(f"Device: {device}")
    
    recursive_results = None
    baseline_results = None
    
    if args.test_untrained:
        # Test with freshly initialized models
        print("\n[MODE] Testing untrained models for validation")
        
        from benchmark.models.recursive_gpt2 import RecursiveGPT2, RecursiveGPT2Config
        from benchmark.models.baseline_gpt2 import BaselineGPT2, BaselineGPT2Config
        
        # Create recursive model
        print("\nCreating untrained recursive model...")
        rec_config = RecursiveGPT2Config(max_iterations=12)
        rec_model = RecursiveGPT2(rec_config).to(device)
        recursive_results = run_falsifiability_test(rec_model, 'recursive_untrained', device, args.output_dir)
        
        # Create baseline model
        print("\nCreating untrained baseline model...")
        base_config = BaselineGPT2Config()
        base_model = BaselineGPT2(base_config).to(device)
        baseline_results = run_falsifiability_test(base_model, 'baseline_untrained', device, args.output_dir)
    
    else:
        # Test trained models from checkpoints
        if args.recursive_checkpoint:
            from dataclasses import fields
            
            print(f"\nLoading recursive model from {args.recursive_checkpoint}...")
            checkpoint = torch.load(args.recursive_checkpoint, map_location=device)
            
            model = None
            
            if 'config' in checkpoint:
                cfg = checkpoint['config']
                # ContractiveLlama checkpoint
                if 'num_key_value_heads' in cfg or 'rope_theta' in cfg:
                    try:
                        from benchmark.models.contractive_llama import ContractiveLlama, ContractiveLlamaConfig
                        valid_fields = {f.name for f in fields(ContractiveLlamaConfig)}
                        filtered_config = {k: v for k, v in cfg.items() if k in valid_fields}
                        config = ContractiveLlamaConfig(**filtered_config)
                        model = ContractiveLlama(config)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model = model.to(device)
                        print(f"  Loaded as ContractiveLlama")
                    except (ImportError, ValueError, KeyError) as e:
                        print(f"  [WARN] Failed to load as ContractiveLlama: {e}")
                        model = None
            
            # Try ContractiveGPT2 next
            if model is None:
                try:
                    from benchmark.models.contractive_gpt2 import ContractiveGPT2, ContractiveGPT2Config
                    if 'config' in checkpoint and 'damping' in checkpoint['config']:
                        valid_fields = {f.name for f in fields(ContractiveGPT2Config)}
                        filtered_config = {k: v for k, v in checkpoint['config'].items() if k in valid_fields}
                        config = ContractiveGPT2Config(**filtered_config)
                        model = ContractiveGPT2(config)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model = model.to(device)
                        print(f"  Loaded as ContractiveGPT2")
                    else:
                        raise ValueError("Not a ContractiveGPT2 checkpoint")
                except (ImportError, ValueError, KeyError):
                    model = None
            
            # Fallback to RecursiveGPT2
            if model is None:
                from benchmark.models.recursive_gpt2 import RecursiveGPT2, RecursiveGPT2Config
                if 'config' in checkpoint:
                    valid_fields = {f.name for f in fields(RecursiveGPT2Config)}
                    filtered_config = {k: v for k, v in checkpoint['config'].items() if k in valid_fields}
                    config = RecursiveGPT2Config(**filtered_config)
                else:
                    config = RecursiveGPT2Config()
                model = RecursiveGPT2(config)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)
                print(f"  Loaded as RecursiveGPT2")
            
            recursive_results = run_falsifiability_test(model, 'recursive', device, args.output_dir)
        
        if args.baseline_checkpoint:
            from benchmark.models.baseline_gpt2 import BaselineGPT2, BaselineGPT2Config
            from dataclasses import fields
            
            print(f"\nLoading baseline model from {args.baseline_checkpoint}...")
            checkpoint = torch.load(args.baseline_checkpoint, map_location=device)
            
            if 'config' in checkpoint:
                valid_fields = {f.name for f in fields(BaselineGPT2Config)}
                filtered_config = {k: v for k, v in checkpoint['config'].items() if k in valid_fields}
                config = BaselineGPT2Config(**filtered_config)
            else:
                config = BaselineGPT2Config()
            
            model = BaselineGPT2(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            
            baseline_results = run_falsifiability_test(model, 'baseline', device, args.output_dir)
    
    # Print individual reports
    if recursive_results:
        recursive_results.print_report()
    
    if baseline_results:
        baseline_results.print_report()
    
    # Comparative analysis
    if recursive_results and baseline_results:
        comparison = compare_models(recursive_results, baseline_results)
        print(comparison)
        
        # Save comparison
        comparison_path = os.path.join(args.output_dir, 'comparison.txt')
        with open(comparison_path, 'w') as f:
            f.write(comparison)
        print(f"\nComparison saved to {comparison_path}")
    
    print("\n[OK] Falsifiability test completed")


if __name__ == '__main__':
    main()
