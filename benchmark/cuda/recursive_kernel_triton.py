"""
Triton Kernels for Fixed-Point Iteration in Recursive Transformers

This module provides Triton implementations of the CUDA kernels for
fixed-point iteration. Triton offers easier Python integration while
maintaining high performance.

Usage:
    from benchmark.cuda.recursive_kernel_triton import (
        fused_layernorm_residual,
        check_convergence,
        anderson_acceleration_step,
    )
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


@triton.jit
def _layernorm_fwd_kernel(
    X,  # Input pointer
    Y,  # Output pointer
    W,  # Weight (gamma) pointer
    B,  # Bias (beta) pointer
    Mean,  # Mean output pointer
    Rstd,  # 1/std output pointer
    stride,  # Row stride
    N,  # Number of columns (hidden size)
    eps,  # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    """Forward pass of LayerNorm."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # Load input row
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / N
    
    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize
    x_norm = x_centered * rstd
    
    # Load weights and apply
    w = tl.load(W + cols, mask=mask, other=1.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    y = x_norm * w + b
    
    # Store outputs
    tl.store(Y + row * stride + cols, y, mask=mask)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)


@triton.jit
def _fused_layernorm_residual_kernel(
    X,  # Input pointer
    Residual,  # Residual pointer
    Y,  # Output pointer
    W,  # Weight (gamma) pointer
    B,  # Bias (beta) pointer
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused LayerNorm(X + Residual)."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # Load and add
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Residual + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + r
    
    # Compute mean and variance
    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and apply affine
    x_norm = x_centered * rstd
    w = tl.load(W + cols, mask=mask, other=1.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    y = x_norm * w + b
    
    tl.store(Y + row * stride + cols, y, mask=mask)


@triton.jit
def _convergence_check_kernel(
    Z_new,
    Z_old,
    DiffNorm,  # Output: sum of squared differences
    OldNorm,   # Output: sum of squared old values
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute ||z_new - z_old||^2 and ||z_old||^2 for convergence check."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    
    z_new = tl.load(Z_new + offset, mask=mask, other=0.0).to(tl.float32)
    z_old = tl.load(Z_old + offset, mask=mask, other=0.0).to(tl.float32)
    
    diff = z_new - z_old
    diff_sq = tl.sum(diff * diff)
    old_sq = tl.sum(z_old * z_old)
    
    tl.atomic_add(DiffNorm, diff_sq)
    tl.atomic_add(OldNorm, old_sq)


@triton.jit
def _anderson_mixing_kernel(
    Z_out,
    Z_new,
    Z_old,
    Beta,  # Mixing coefficient
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply Anderson mixing: z_out = (1 - beta) * z_new + beta * z_old."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    
    beta = tl.load(Beta)
    z_new = tl.load(Z_new + offset, mask=mask, other=0.0)
    z_old = tl.load(Z_old + offset, mask=mask, other=0.0)
    
    z_out = (1.0 - beta) * z_new + beta * z_old
    tl.store(Z_out + offset, z_out, mask=mask)


@triton.jit
def _compute_anderson_beta_kernel(
    R_new,  # New residual
    R_old,  # Old residual
    Numer,  # Output: numerator
    Denom,  # Output: denominator
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute Anderson coefficient: beta = <r_new, r_new - r_old> / ||r_new - r_old||^2."""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    
    r_new = tl.load(R_new + offset, mask=mask, other=0.0).to(tl.float32)
    r_old = tl.load(R_old + offset, mask=mask, other=0.0).to(tl.float32)
    
    r_diff = r_new - r_old
    numer = tl.sum(r_new * r_diff)
    denom = tl.sum(r_diff * r_diff)
    
    tl.atomic_add(Numer, numer)
    tl.atomic_add(Denom, denom)


def fused_layernorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Fused LayerNorm(x + residual) operation.
    
    Args:
        x: Input tensor (batch, seq, hidden)
        residual: Residual tensor (batch, seq, hidden)
        weight: LayerNorm weight (hidden,)
        bias: LayerNorm bias (hidden,)
        eps: Epsilon for numerical stability
        
    Returns:
        Normalized output (batch, seq, hidden)
    """
    assert x.shape == residual.shape
    assert weight.shape[0] == x.shape[-1]
    
    # Flatten batch dimensions
    orig_shape = x.shape
    x_flat = x.view(-1, x.shape[-1])
    residual_flat = residual.view(-1, residual.shape[-1])
    
    n_rows, n_cols = x_flat.shape
    
    # Output tensor
    y = torch.empty_like(x_flat)
    
    # Determine block size (must be power of 2)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Launch kernel
    _fused_layernorm_residual_kernel[(n_rows,)](
        x_flat, residual_flat, y,
        weight, bias,
        x_flat.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y.view(orig_shape)


def check_convergence(
    z_new: torch.Tensor,
    z_old: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """
    Check convergence: ||z_new - z_old|| / (||z_old|| + eps).
    
    Args:
        z_new: New state tensor
        z_old: Old state tensor
        eps: Epsilon for numerical stability
        
    Returns:
        Relative difference (scalar)
    """
    z_new_flat = z_new.view(-1)
    z_old_flat = z_old.view(-1)
    n = z_new_flat.numel()
    
    # Allocate output tensors
    diff_norm = torch.zeros(1, device=z_new.device, dtype=torch.float32)
    old_norm = torch.zeros(1, device=z_new.device, dtype=torch.float32)
    
    BLOCK_SIZE = 1024
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    _convergence_check_kernel[(n_blocks,)](
        z_new_flat, z_old_flat,
        diff_norm, old_norm,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (diff_norm.sqrt() / (old_norm.sqrt() + eps)).item()


def anderson_acceleration_step(
    z_new: torch.Tensor,
    z_old: torch.Tensor,
    residual_new: torch.Tensor,
    residual_old: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Apply one step of Anderson acceleration.
    
    Computes optimal mixing coefficient and returns:
        z_out = (1 - beta) * z_new + beta * z_old
    
    Args:
        z_new: New state from fixed-point iteration
        z_old: Previous state
        residual_new: f(z_new) - z_new
        residual_old: f(z_old) - z_old
        eps: Epsilon for numerical stability
        
    Returns:
        Accelerated state estimate
    """
    n = z_new.numel()
    
    # Compute beta
    numer = torch.zeros(1, device=z_new.device, dtype=torch.float32)
    denom = torch.zeros(1, device=z_new.device, dtype=torch.float32)
    
    BLOCK_SIZE = 1024
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    _compute_anderson_beta_kernel[(n_blocks,)](
        residual_new.view(-1), residual_old.view(-1),
        numer, denom,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    beta = torch.clamp(numer / (denom + eps), 0.0, 1.0)
    
    # Apply mixing
    z_out = torch.empty_like(z_new)
    
    _anderson_mixing_kernel[(n_blocks,)](
        z_out.view(-1), z_new.view(-1), z_old.view(-1),
        beta,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return z_out


class FixedPointIterationTriton:
    """
    Fixed-point iteration with Triton acceleration.
    
    This class provides an optimized implementation of the fixed-point
    iteration used in recursive transformers.
    """
    
    def __init__(
        self,
        max_iterations: int = 24,
        convergence_threshold: float = 1e-4,
        min_iterations: int = 4,
        use_anderson: bool = True,
    ):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.min_iterations = min_iterations
        self.use_anderson = use_anderson
    
    def iterate(
        self,
        f: callable,
        z_init: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """
        Iterate z_{t+1} = f(z_t, *args, **kwargs) until convergence.
        
        Args:
            f: Function to iterate
            z_init: Initial state
            *args, **kwargs: Additional arguments to f
            
        Returns:
            (fixed_point, iterations): Converged state and iteration count
        """
        z = z_init.clone()
        z_old = None
        residual_old = None
        
        for i in range(self.max_iterations):
            z_new = f(z, *args, **kwargs)
            
            # Compute residual for Anderson acceleration
            residual_new = z_new - z
            
            # Apply Anderson acceleration if enabled and we have history
            if self.use_anderson and z_old is not None and residual_old is not None:
                z_new = anderson_acceleration_step(
                    z_new, z_old, residual_new, residual_old
                )
            
            # Check convergence after minimum iterations
            if i >= self.min_iterations:
                rel_diff = check_convergence(z_new, z)
                if rel_diff < self.convergence_threshold:
                    return z_new, i + 1
            
            # Update history
            z_old = z.clone()
            residual_old = residual_new.clone()
            z = z_new
        
        return z, self.max_iterations


# Test functions
def _test_fused_layernorm():
    """Test fused LayerNorm + residual."""
    print("[TEST] Fused LayerNorm + Residual...")
    
    batch, seq, hidden = 2, 128, 768
    x = torch.randn(batch, seq, hidden, device='cuda')
    residual = torch.randn(batch, seq, hidden, device='cuda')
    weight = torch.ones(hidden, device='cuda')
    bias = torch.zeros(hidden, device='cuda')
    
    # Triton version
    y_triton = fused_layernorm_residual(x, residual, weight, bias)
    
    # Reference version
    ln = torch.nn.LayerNorm(hidden, device='cuda')
    ln.weight.data = weight
    ln.bias.data = bias
    y_ref = ln(x + residual)
    
    # Check correctness
    max_diff = (y_triton - y_ref).abs().max().item()
    print(f"  Max difference: {max_diff:.6e}")
    assert max_diff < 1e-4, "Fused LayerNorm test failed!"
    print("  [OK]")


def _test_convergence_check():
    """Test convergence checking."""
    print("[TEST] Convergence Check...")
    
    n = 1024 * 768
    z_old = torch.randn(n, device='cuda')
    z_new = z_old + 0.001 * torch.randn(n, device='cuda')
    
    # Triton version
    rel_diff_triton = check_convergence(z_new, z_old)
    
    # Reference version
    rel_diff_ref = (z_new - z_old).norm() / (z_old.norm() + 1e-8)
    rel_diff_ref = rel_diff_ref.item()
    
    print(f"  Triton: {rel_diff_triton:.6e}")
    print(f"  Reference: {rel_diff_ref:.6e}")
    
    rel_error = abs(rel_diff_triton - rel_diff_ref) / (rel_diff_ref + 1e-8)
    assert rel_error < 0.01, "Convergence check test failed!"
    print("  [OK]")


def _test_anderson_acceleration():
    """Test Anderson acceleration."""
    print("[TEST] Anderson Acceleration...")
    
    n = 1024 * 768
    z_new = torch.randn(n, device='cuda')
    z_old = torch.randn(n, device='cuda')
    residual_new = torch.randn(n, device='cuda')
    residual_old = torch.randn(n, device='cuda')
    
    z_out = anderson_acceleration_step(z_new, z_old, residual_new, residual_old)
    
    print(f"  Output shape: {z_out.shape}")
    print(f"  Output norm: {z_out.norm().item():.4f}")
    print("  [OK]")


if __name__ == "__main__":
    if torch.cuda.is_available():
        _test_fused_layernorm()
        _test_convergence_check()
        _test_anderson_acceleration()
        print("\n[OK] All Triton kernel tests passed.")
    else:
        print("[SKIP] CUDA not available, skipping Triton tests.")
