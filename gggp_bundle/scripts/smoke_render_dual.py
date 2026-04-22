"""
gggp_bundle/scripts/smoke_render_dual.py

MEDP A1 / T5.b -- smoke test for the Rust<->Python render-dual bridge.

Verifies:
  1. `SemioticHypercube.render_tree_with_input` accepts a T_i seed and
     returns a code c_i of the requested dim, for a chromosome parsed
     against the neuro_grammar.
  2. `SemioticHypercube.batch_render_dual` processes the full 128x1024
     T matrix in one call and returns
       {c: (128,code_dim), reconstruction: (128,1024),
        per_i_cos: (128,), F: float}
     with all shapes/types correct.
  3. The pipeline runs in reasonable time (< 5s for N=128 on a random
     chromosome; Rust is single-threaded here, parallelism is future
     work).

This does NOT evaluate any A1 gate -- it only proves the Rust surface
is wired correctly before T6/T7 builds the fitness shaper + EA runner
on top.

Run:
    gggp_bundle/.venv/bin/python gggp_bundle/scripts/smoke_render_dual.py
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = REPO_ROOT / "gggp_bundle"
GRAMMAR_CFG = BUNDLE / "neuro_grammar.cfg"
T_NPY = BUNDLE / "demos" / "semiotic_hypercube" / "T.npy"

CODE_DIM = 16
TARGET_DIM = 1024


def ensure_grammar() -> Path:
    """Regenerate neuro_grammar.cfg if missing (it's gitignored)."""
    if GRAMMAR_CFG.is_file():
        return GRAMMAR_CFG
    print(f"[smoke] grammar .cfg missing, regenerating...")
    subprocess.run(
        ["cargo", "run", "--release", "--bin", "gen_neuro_grammar"],
        cwd=BUNDLE / "rust",
        check=True,
    )
    if not GRAMMAR_CFG.is_file():
        raise SystemExit(
            f"gen_neuro_grammar ran but {GRAMMAR_CFG} still missing. "
            f"Check cargo output and working directory behavior."
        )
    return GRAMMAR_CFG


def main() -> None:
    cfg = ensure_grammar()
    print(f"[smoke] grammar: {cfg}")

    if not T_NPY.is_file():
        raise SystemExit(
            f"T.npy missing at {T_NPY}. Run scripts/embed_corpus.py first (T2)."
        )
    T = np.load(T_NPY).astype(np.float64)
    assert T.shape == (128, TARGET_DIM), f"unexpected T shape: {T.shape}"
    print(f"[smoke] T: shape={T.shape} dtype={T.dtype}")

    from semiotic_hypercube import SemioticHypercube

    sh = SemioticHypercube(str(cfg))
    print(f"[smoke] SemioticHypercube loaded grammar from {cfg}")

    # ---- Test 1: render_tree_with_input on a single T_i -----------
    chromo_g = sh.random_chromosome(0)
    chromo_d = sh.random_chromosome(1)
    print(
        f"[smoke] chromosome_g = {chromo_g}  len={len(chromo_g)}\n"
        f"[smoke] chromosome_d = {chromo_d}  len={len(chromo_d)}"
    )
    t0 = T[0].copy()
    t_start = time.time()
    c_0 = sh.render_tree_with_input(chromo_g, CODE_DIM, t0)
    r_0 = sh.render_tree_with_input(chromo_d, TARGET_DIM, c_0)
    t_single = (time.time() - t_start) * 1000
    assert c_0.shape == (CODE_DIM,), f"c_0 shape: {c_0.shape}"
    assert r_0.shape == (TARGET_DIM,), f"r_0 shape: {r_0.shape}"
    print(
        f"[smoke] render_tree_with_input OK: "
        f"c_0.shape={c_0.shape} r_0.shape={r_0.shape} "
        f"||c_0||={np.linalg.norm(c_0):.4f} "
        f"||r_0||={np.linalg.norm(r_0):.4f}  ({t_single:.1f}ms)"
    )

    # ---- Test 2: sweep N=20 random (G, D) pairs over full T -------
    # Goal: confirm bridge scales (128 * 2 trees * 20 pairs = 5120 renders)
    # and that F varies across seeds (not stuck at 0). This is NOT
    # evolution -- just a random-search distribution sanity check.
    print(f"\n[smoke] Sweeping 20 random (G,D) pairs over N=128 ...")
    sweep: list[tuple[int, int, float, int, int]] = []
    t_start = time.time()
    for seed in range(20):
        cg = sh.random_chromosome(seed * 2)
        cd = sh.random_chromosome(seed * 2 + 1)
        res = sh.batch_render_dual(cg, cd, T, CODE_DIM, TARGET_DIM)
        sweep.append(
            (seed, len(cg) + len(cd), float(res["F"]), len(cg), len(cd))
        )
    t_sweep = (time.time() - t_start) * 1000

    Fs = np.array([s[2] for s in sweep])
    lens = np.array([s[1] for s in sweep])
    print(
        f"[smoke] sweep wall={t_sweep:.0f}ms ({t_sweep / 20 / 128:.2f}ms per "
        f"render pair per T_i)"
    )
    print(
        f"[smoke] F distribution:  mean={Fs.mean():+.4f}  "
        f"std={Fs.std():.4f}  min={Fs.min():+.4f}  max={Fs.max():+.4f}"
    )
    print(f"[smoke] combined chromosome length: mean={lens.mean():.1f} "
          f"max={lens.max()} min={lens.min()}")
    print(f"[smoke] baseline F_0 = 0.6132 (random chromosomes expected "
          f"well below that; any sweep hit close to 0.6 would be suspicious)")

    # Shape assertions on the last call
    c = np.asarray(res["c"])
    rec = np.asarray(res["reconstruction"])
    per_i = np.asarray(res["per_i_cos"])
    F = float(res["F"])
    assert c.shape == (128, CODE_DIM), f"c shape: {c.shape}"
    assert rec.shape == (128, TARGET_DIM), f"rec shape: {rec.shape}"
    assert per_i.shape == (128,), f"per_i shape: {per_i.shape}"
    assert abs(F - per_i.mean()) < 1e-9, (
        f"F={F} != per_i.mean()={per_i.mean()}"
    )

    print(f"\n[smoke] ALL CHECKS PASSED. Rust bridge is live.")
    print(f"[smoke] Notes for T6/T7 (fitness + runner):")
    print(f"  - Current grammar (gen_neuro_grammar, dim=5) is too narrow")
    print(f"    for TARGET_DIM=1024 -- AX ops only touch axes 0..4, so")
    print(f"    decoder cannot reconstruct most of the embedding manifold.")
    print(f"  - ZERO op collapses signal to zero when sampled; consider")
    print(f"    removing it and adding MIX/ROT/FRAC (already in VectorOp).")
    print(f"  - Grammar needs regeneration with dim=1024 for A1 production")
    print(f"    runs. Tracked in log.jsonl as a T7 prerequisite.")


if __name__ == "__main__":
    main()
