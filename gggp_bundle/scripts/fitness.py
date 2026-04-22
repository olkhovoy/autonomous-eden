"""
gggp_bundle/scripts/fitness.py

MEDP A1 / T6 -- fitness shaper for the shared-genome (G, D) individual.

Takes the raw output of SemioticHypercube.batch_render_dual (which is
mean_i cos(r_i, T_i) = F_raw) and applies three penalties:

  1. Compactness          alpha_len * (len_g + len_d) / L_max
  2. Class-incoherence    beta_class * (1 - silhouette(c, classes))
  3. Seed-instability     gamma_seed * std(F across seeds)        [T9+]

All parameters live in `gggp_bundle/config/fitness.toml` together with
their evolutionary ranges (so a future meta-run can GGGP-tune them).

Usage:
    from fitness import FitnessConfig, shape_fitness

    cfg = FitnessConfig.load()  # reads config/fitness.toml
    F_shaped = shape_fitness(
        F_raw=res["F"],
        c_matrix=np.asarray(res["c"]),
        classes=classes,
        len_g=len(chromo_g),
        len_d=len(chromo_d),
        cfg=cfg,
        seed_F_array=None,  # set in T9 multi-seed runs
    )
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "gggp_bundle" / "config" / "fitness.toml"


@dataclass
class FitnessConfig:
    alpha_len: float
    L_max: float
    beta_class: float
    gamma_seed: float
    fallback_fitness: float

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG) -> "FitnessConfig":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(
                f"FitnessConfig: {p} not found. Expected TOML at this path. "
                f"Hint: commit config/fitness.toml or pass an explicit path."
            )
        with p.open("rb") as f:
            data = tomllib.load(f)
        f_sec = data.get("fitness", {})
        required = ["alpha_len", "L_max", "beta_class", "gamma_seed", "fallback_fitness"]
        missing = [k for k in required if k not in f_sec]
        if missing:
            raise KeyError(
                f"FitnessConfig: missing keys {missing} in [fitness] section "
                f"of {p}. Regenerate config/fitness.toml from the repo template."
            )
        return cls(
            alpha_len=float(f_sec["alpha_len"]),
            L_max=float(f_sec["L_max"]),
            beta_class=float(f_sec["beta_class"]),
            gamma_seed=float(f_sec["gamma_seed"]),
            fallback_fitness=float(f_sec["fallback_fitness"]),
        )


def _silhouette_cosine(c: np.ndarray, classes: np.ndarray) -> float:
    """Cosine-based silhouette in [-1, 1]. Higher == better class separation.

    We avoid a sklearn dependency here to keep the fitness loop import-light
    and to lock the distance metric to cosine (matches batch_render_dual's
    per_i_cos). For N=128 rows the O(N^2) inner loop is ~16K cosines.
    """
    n = c.shape[0]
    if n <= 1:
        return 0.0
    norms = np.linalg.norm(c, axis=1, keepdims=True)
    # Rows with zero norm cannot define a direction. Replace by a tiny nonzero
    # vector so they don't crash the division; they will naturally score poorly.
    norms = np.where(norms > 1e-12, norms, 1.0)
    cn = c / norms
    sim = cn @ cn.T  # cosine matrix in [-1, 1]

    labels = np.asarray(classes)
    uniq = np.unique(labels)
    if uniq.size == 1:
        return 0.0

    sil_vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        n_same = int(same.sum())
        if n_same == 0:
            sil_vals[i] = 0.0
            continue
        # cosine-distance = 1 - cosine-sim; smaller dist == closer.
        a_i = float(1.0 - sim[i, same].mean())
        b_candidates: list[float] = []
        for k in uniq:
            if k == labels[i]:
                continue
            other = labels == k
            if not other.any():
                continue
            b_candidates.append(float(1.0 - sim[i, other].mean()))
        if not b_candidates:
            sil_vals[i] = 0.0
            continue
        b_i = min(b_candidates)
        denom = max(a_i, b_i)
        sil_vals[i] = 0.0 if denom == 0.0 else (b_i - a_i) / denom
    return float(sil_vals.mean())


def shape_fitness(
    *,
    F_raw: float,
    c_matrix: np.ndarray,
    classes: np.ndarray,
    len_g: int,
    len_d: int,
    cfg: FitnessConfig,
    seed_F_array: np.ndarray | None = None,
) -> tuple[float, dict]:
    """Return (F_shaped, breakdown_dict).

    The breakdown helps log.jsonl audits: each term is recorded.
    NaN inputs collapse to cfg.fallback_fitness without raising, so a
    single pathological individual cannot kill an EA generation.
    """
    if not math.isfinite(F_raw):
        return cfg.fallback_fitness, {
            "reason": "F_raw_not_finite", "F_raw": F_raw
        }
    if c_matrix.size == 0 or not np.isfinite(c_matrix).all():
        return cfg.fallback_fitness, {
            "reason": "c_matrix_invalid",
            "F_raw": F_raw,
            "shape": list(c_matrix.shape),
        }

    len_penalty = cfg.alpha_len * (len_g + len_d) / max(cfg.L_max, 1.0)

    silhouette = _silhouette_cosine(c_matrix, classes)
    # class_incoherence in [0, 2]; 0 == perfect separation, 2 == perfect anti.
    class_incoherence = 1.0 - silhouette
    class_penalty = cfg.beta_class * class_incoherence

    if seed_F_array is not None and seed_F_array.size >= 2:
        seed_penalty = cfg.gamma_seed * float(np.std(seed_F_array))
    else:
        seed_penalty = 0.0

    F_shaped = float(F_raw) - len_penalty - class_penalty - seed_penalty
    return F_shaped, {
        "F_raw": float(F_raw),
        "len_g": int(len_g),
        "len_d": int(len_d),
        "len_penalty": float(len_penalty),
        "silhouette_cos": float(silhouette),
        "class_penalty": float(class_penalty),
        "seed_std": float(seed_penalty / cfg.gamma_seed) if cfg.gamma_seed > 0 else 0.0,
        "seed_penalty": float(seed_penalty),
        "F_shaped": F_shaped,
    }


def _self_check() -> None:
    """Run as a module: `python -m fitness` to sanity-check the shaper."""
    cfg = FitnessConfig.load()
    print(f"[fitness] loaded config: {cfg}")

    rng = np.random.default_rng(0)
    n, d, k = 128, 16, 8
    classes = np.repeat(np.arange(k), n // k).astype(np.int32)

    # Case A: perfectly class-coherent random clusters.
    centers = rng.normal(size=(k, d)) * 2.0
    c_clean = centers[classes] + rng.normal(size=(n, d)) * 0.1
    # Case B: total noise (no class structure).
    c_noise = rng.normal(size=(n, d))

    for name, c in [("clean", c_clean), ("noise", c_noise)]:
        Fs, info = shape_fitness(
            F_raw=0.40,
            c_matrix=c,
            classes=classes,
            len_g=10,
            len_d=15,
            cfg=cfg,
            seed_F_array=None,
        )
        print(
            f"[fitness] {name:5s}: F_raw=0.40 -> F_shaped={Fs:+.4f}  "
            f"len_pen={info['len_penalty']:+.4f}  "
            f"silhouette={info['silhouette_cos']:+.4f}  "
            f"class_pen={info['class_penalty']:+.4f}"
        )


if __name__ == "__main__":
    _self_check()
