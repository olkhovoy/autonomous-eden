"""
gggp_bundle/scripts/diag_random_baseline.py

Diagnostic tool (NOT a gate): run 1500 random (G, D) pairs and report
the top-F distribution. If EA in run_A1.py is stuck at the same F as
random search, the bottleneck is grammar expressivity, not the EA.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "gggp_bundle" / "scripts"))
from fitness import FitnessConfig, shape_fitness  # noqa: E402

DEMO_DIR = REPO_ROOT / "gggp_bundle" / "demos" / "semiotic_hypercube"
T_PATH = DEMO_DIR / "T.npy"
CLASSES_PATH = DEMO_DIR / "classes.npy"
G_ENC = DEMO_DIR / "grammar_encoder.cfg"
G_DEC = DEMO_DIR / "grammar_decoder.cfg"

N_SAMPLES = 1500
CODE_DIM = 16
TARGET_DIM = 1024


def main() -> None:
    T = np.load(T_PATH).astype(np.float64)
    classes = np.load(CLASSES_PATH)
    fcfg = FitnessConfig.load()

    from semiotic_hypercube import SemioticHypercube
    sh = SemioticHypercube(str(G_ENC))
    sh.attach_decoder_grammar(str(G_DEC))

    F_raws = np.empty(N_SAMPLES, dtype=np.float64)
    F_shapeds = np.empty(N_SAMPLES, dtype=np.float64)
    lens = np.empty(N_SAMPLES, dtype=np.int32)
    best_idx = -1
    best_F_raw = -1.0
    best_chromos = None

    t0 = time.time()
    for i in range(N_SAMPLES):
        cg = sh.random_chromosome(i * 2, "encoder")
        cd = sh.random_chromosome(i * 2 + 1, "decoder")
        res = sh.batch_render_dual(cg, cd, T, CODE_DIM, TARGET_DIM)
        F = float(res["F"])
        c = np.asarray(res["c"])
        Fs, _ = shape_fitness(
            F_raw=F, c_matrix=c, classes=classes,
            len_g=len(cg), len_d=len(cd), cfg=fcfg,
        )
        F_raws[i] = F
        F_shapeds[i] = Fs
        lens[i] = len(cg) + len(cd)
        if F > best_F_raw:
            best_F_raw = F
            best_idx = i
            best_chromos = (cg, cd)
    wall = time.time() - t0

    print(f"[diag] {N_SAMPLES} random pairs in {wall:.1f}s "
          f"({wall*1000/N_SAMPLES:.2f} ms/pair)")
    print(f"[diag] F_raw:     mean={F_raws.mean():+.4f} "
          f"std={F_raws.std():.4f} "
          f"min={F_raws.min():+.4f} max={F_raws.max():+.4f}")
    print(f"[diag] F_shaped:  mean={F_shapeds.mean():+.4f} "
          f"std={F_shapeds.std():.4f} "
          f"min={F_shapeds.min():+.4f} max={F_shapeds.max():+.4f}")
    print(f"[diag] len:       mean={lens.mean():.1f} "
          f"min={lens.min()} max={lens.max()}")
    print(f"[diag] top-10 F_raw: {np.sort(F_raws)[-10:][::-1]}")
    print(f"[diag] F_raw percentiles: 50%={np.percentile(F_raws,50):+.4f} "
          f"90%={np.percentile(F_raws,90):+.4f} "
          f"99%={np.percentile(F_raws,99):+.4f}")
    print(f"[diag] best pair: idx={best_idx}  F_raw={best_F_raw:+.4f}")
    if best_chromos is not None:
        g, d = best_chromos
        print(f"[diag] best G chromosome (len {len(g)}): {g[:40]}")
        print(f"[diag] best D chromosome (len {len(d)}): {d[:40]}")


if __name__ == "__main__":
    main()
