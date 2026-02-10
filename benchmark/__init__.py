"""
UMC Fixed-Point Training Benchmark

This package implements the benchmark comparing standard transformer training
against recursive fixed-point training as proposed in the Unitary Model of
Consciousness (UMC) framework.

Key components:
- models/: Baseline GPT-2 and Recursive GPT-2 implementations
- cuda/: Custom CUDA kernels for fixed-point iteration
- training/: Training loops with implicit differentiation
- metrics/: Knowledge density measurement utilities
"""

__version__ = "0.1.0"
