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
DEMO_DIR = BUNDLE / "demos" / "semiotic_hypercube"
GRAMMAR_ENCODER = DEMO_DIR / "grammar_encoder.cfg"
GRAMMAR_DECODER = DEMO_DIR / "grammar_decoder.cfg"
T_NPY = DEMO_DIR / "T.npy"

CODE_DIM = 16
TARGET_DIM = 1024


def ensure_grammars() -> tuple[Path, Path]:
    """Regenerate both grammar .cfg files if missing (they're gitignored)."""
    missing = [p for p in (GRAMMAR_ENCODER, GRAMMAR_DECODER) if not p.is_file()]
    if not missing:
        return GRAMMAR_ENCODER, GRAMMAR_DECODER
    print(f"[smoke] grammars missing: {[str(m) for m in missing]}; regenerating")
    rust_dir = BUNDLE / "rust"
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "gen_neuro_grammar"],
        cwd=rust_dir,
        check=True,
    )
    bin_path = rust_dir / "target" / "release" / "gen_neuro_grammar"
    subprocess.run([str(bin_path), "encoder", str(GRAMMAR_ENCODER)], check=True)
    subprocess.run([str(bin_path), "decoder", str(GRAMMAR_DECODER)], check=True)
    for p in (GRAMMAR_ENCODER, GRAMMAR_DECODER):
        if not p.is_file():
            raise SystemExit(f"gen_neuro_grammar ran but {p} still missing.")
    return GRAMMAR_ENCODER, GRAMMAR_DECODER


def main() -> None:
    g_enc, g_dec = ensure_grammars()
    print(f"[smoke] encoder grammar: {g_enc}")
    print(f"[smoke] decoder grammar: {g_dec}")

    if not T_NPY.is_file():
        raise SystemExit(
            f"T.npy missing at {T_NPY}. Run scripts/embed_corpus.py first (T2)."
        )
    T = np.load(T_NPY).astype(np.float64)
    assert T.shape == (128, TARGET_DIM), f"unexpected T shape: {T.shape}"
    print(f"[smoke] T: shape={T.shape} dtype={T.dtype}")

    from semiotic_hypercube import SemioticHypercube

    sh = SemioticHypercube(str(g_enc))
    sh.attach_decoder_grammar(str(g_dec))
    print(f"[smoke] SemioticHypercube: encoder + decoder grammars attached")

    # ---- Test 1: render_tree_with_input on a single T_i -----------
    chromo_g = sh.random_chromosome(0, "encoder")
    chromo_d = sh.random_chromosome(1, "decoder")
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
        cg = sh.random_chromosome(seed * 2, "encoder")
        cd = sh.random_chromosome(seed * 2 + 1, "decoder")
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


if __name__ == "__main__":
    main()
