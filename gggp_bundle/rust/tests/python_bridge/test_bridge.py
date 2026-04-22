"""Smoke test for the PyO3 bridge exposed by the semiotic_hypercube crate.

Runs outside of `cargo test`: requires a prior

    maturin develop --release --features python

so that `import semiotic_hypercube` resolves to the built extension module.
Grammar fixture is produced by

    cargo run --bin gen_grammar

which writes `test_grammar.cfg` next to the current working directory.
"""
from __future__ import annotations

import sys

import numpy as np

import semiotic_hypercube


def main() -> int:
    print("Testing Semiotic Hypercube FFI Bridge...")

    try:
        hypercube = semiotic_hypercube.SemioticHypercube("test_grammar.cfg")
        print("Successfully instantiated SemioticHypercube.")
    except Exception as exc:
        print(f"Failed to load grammar: {exc}")
        return 1

    chromosome = [0, 0, 0, 4, 0, 0, 1, 3, 1, 1, 2]

    vec = hypercube.fractal_expand(chromosome, 5, 3)
    print("Fractal expanded vector (depth=5, dim=3):", vec)

    target = np.array([1.0, -0.5, 0.25, 0.0], dtype=np.float64)
    _best_weights, optimized_vec, fitness = hypercube.evolve_target(target, 50)

    print("\n--- Evolution Results ---")
    print(f"Target: {target}")
    print(f"Evolved vector: {optimized_vec}")
    print(f"Fitness (cosine similarity): {fitness:.4f}")

    if fitness > 0.8:
        print("\n[OK] CMA-ES converged towards the target vector in the latent space.")
        return 0
    print("\n[WARN] Fitness is low. Re-check the parameter space / grammar fixture.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
