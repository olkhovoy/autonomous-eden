"""
gggp_bundle/scripts/fitness.py

MEDP A1 / T6 -- fitness shaper for the shared-genome (G, D) individual.
MEDP A2 / S2b -- UMC scalarized fitness combining NC1-NC4.

A1.1 path (unchanged):
    shape_fitness(F_raw, c_matrix, classes, len_g, len_d, cfg, seed_F_array)
  applies compactness / class / seed penalties to F_raw = mean_i cos(T_i, D(G(T_i))).

A2 path (new):
    shape_fitness_umc(F_nc1, F_nc2, F_nc3, F_nc4, ..., nc_weights, cfg)
  scalarizes the four UMC neural-constraint components into a single fitness
  and then reuses the A1.1 penalties (compactness / class / seed).

Component metrics (pure NumPy / str helpers):
  compute_F_nc4(T, T_hat)                  -- carry-over: mean cos(T_i, D(G(T_i)))
  compute_F_nc1(c, c_hat)                  -- recursive closure: mean cos(c_i, G(D(c_i)))
  compute_F_nc2(T_mix, T_avg_pair, T_rand) -- compositional triplet accuracy
  compute_F_nc3_signal(decoder_chromosome) -- fraction of CTRL/SBC/ADDC ops

All parameters live in `gggp_bundle/config/fitness.toml` together with
their evolutionary ranges (so a future meta-run can GGGP-tune them).

A1 usage (unchanged):
    from fitness import FitnessConfig, shape_fitness

    cfg = FitnessConfig.load()
    F_shaped, info = shape_fitness(
        F_raw=res["F"],
        c_matrix=np.asarray(res["c"]),
        classes=classes,
        len_g=len(chromo_g),
        len_d=len(chromo_d),
        cfg=cfg,
        seed_F_array=None,  # set in T9 multi-seed runs
    )

A2 usage (new):
    from fitness import (
        FitnessConfig, NCWeights,
        compute_F_nc1, compute_F_nc2, compute_F_nc3_signal, compute_F_nc4,
        shape_fitness_umc,
    )

    cfg = FitnessConfig.load()
    ncw = NCWeights.load()
    F_nc4 = compute_F_nc4(T, T_hat)
    F_nc1 = compute_F_nc1(c, c_hat)
    F_nc2 = compute_F_nc2(T_mix, T_avg_pair, T_rand)
    F_nc3 = compute_F_nc3_signal(decoder_chromosome_text)
    F_shaped, info = shape_fitness_umc(
        F_nc1=F_nc1, F_nc2=F_nc2, F_nc3=F_nc3, F_nc4=F_nc4,
        c_matrix=c, classes=classes, len_g=len_g, len_d=len_d,
        cfg=cfg, nc_weights=ncw, seed_F_array=None,
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


# =====================================================================
# A2 / S2b -- UMC scalarized fitness (NC1 / NC2 / NC3 / NC4)
# =====================================================================

# Opcode tokens emitted by the decoder grammar (see
# rust/src/bin/gen_neuro_grammar.rs). NC3 == the code-gated subset.
_BASELINE_OP_TOKENS: frozenset[str] = frozenset(
    {"AX", "SCALE", "NORM", "MIX", "ROT", "FRAC"}
)
_NC3_OP_TOKENS: frozenset[str] = frozenset({"CTRL", "SBC", "ADDC"})
_ALL_OP_TOKENS: frozenset[str] = _BASELINE_OP_TOKENS | _NC3_OP_TOKENS


@dataclass
class NCWeights:
    """Weights for the A2 UMC scalarized fitness.

    Loaded from the `[nc_weights]` section of config/fitness.toml (S2a).
    Each weight ships with a declared range in the TOML so meta-optimization
    can tune them; the runtime object stores only the current values.
    """

    w_nc1: float
    w_nc2: float
    w_nc3: float
    w_nc4: float

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG) -> "NCWeights":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(
                f"NCWeights: {p} not found. Expected TOML at this path. "
                f"Hint: commit config/fitness.toml or pass an explicit path."
            )
        with p.open("rb") as f:
            data = tomllib.load(f)
        nc_sec = data.get("nc_weights")
        if nc_sec is None:
            raise KeyError(
                f"NCWeights: missing [nc_weights] section in {p}. "
                f"This is an A2 artifact -- check that S2a landed before "
                f"running shape_fitness_umc."
            )
        required = ["w_nc1", "w_nc2", "w_nc3", "w_nc4"]
        missing = [k for k in required if k not in nc_sec]
        if missing:
            raise KeyError(
                f"NCWeights: missing keys {missing} in [nc_weights] section "
                f"of {p}. Regenerate config/fitness.toml from the A2 template."
            )
        return cls(
            w_nc1=float(nc_sec["w_nc1"]),
            w_nc2=float(nc_sec["w_nc2"]),
            w_nc3=float(nc_sec["w_nc3"]),
            w_nc4=float(nc_sec["w_nc4"]),
        )


def _row_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity. Shapes must match exactly.

    Rows whose norm is below 1e-12 in either `a` or `b` yield 0.0 (there is
    no well-defined direction). Result has shape (n,) and lives in [-1, 1]
    modulo numerical error.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"_row_cosines: shape mismatch a={a.shape} b={b.shape}. "
            f"Both inputs must be (n, d). Check that encoder and decoder "
            f"were called with consistent dims."
        )
    if a.ndim != 2:
        raise ValueError(
            f"_row_cosines: expected 2D arrays, got ndim={a.ndim}. "
            f"Reshape to (n, d) before calling."
        )
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    dots = np.einsum("ij,ij->i", a, b)
    denom = na * nb
    # Avoid division by zero; "no direction" rows score 0.
    safe = denom > 1e-12
    out = np.zeros_like(dots, dtype=np.float64)
    out[safe] = dots[safe] / denom[safe]
    return out


def compute_F_nc4(T: np.ndarray, T_hat: np.ndarray) -> float:
    """NC4: fixed-point stability -- mean row-wise cos(T_i, D(G(T_i))).

    Carry-over from A1.1. Identical semantics to `F_raw` used by
    shape_fitness. Range [-1, 1]; A1.1 achieved ~0.92 on the 128-row corpus.
    """
    if T.size == 0:
        return 0.0
    return float(_row_cosines(T, T_hat).mean())


def compute_F_nc1(c: np.ndarray, c_hat: np.ndarray) -> float:
    """NC1: recursive closure -- mean row-wise cos(c_i, G(D(c_i))).

    The dual fixed point. If D and G are mutual inverses on the code
    manifold, c_hat == c and F_nc1 == 1. A2 gate G_NC1 requires > 0.50.
    """
    if c.size == 0:
        return 0.0
    return float(_row_cosines(c, c_hat).mean())


def compute_F_nc2(
    T_mix: np.ndarray,
    T_avg_pair: np.ndarray,
    T_rand: np.ndarray,
) -> float:
    """NC2: compositional triplet accuracy.

    For each sampled pair (i, j) and a random distractor k, we compare:
        anchor   = T_mix      = D(0.5 * c_i + 0.5 * c_j)
        positive = T_avg_pair = (T_i + T_j) / 2
        negative = T_rand     = T_k   (k != i, j)

    Returns the fraction of triplets where cos(anchor, positive) strictly
    exceeds cos(anchor, negative). Random baseline = 0.5; A2 gate G_NC2
    requires > 0.55.

    All three arrays must have shape (M, target_dim) with identical M.
    """
    for name, arr in (("T_mix", T_mix), ("T_avg_pair", T_avg_pair), ("T_rand", T_rand)):
        if arr.ndim != 2:
            raise ValueError(
                f"compute_F_nc2: {name} must be 2D (M, d), got ndim={arr.ndim}."
            )
    if not (T_mix.shape == T_avg_pair.shape == T_rand.shape):
        raise ValueError(
            f"compute_F_nc2: shape mismatch "
            f"T_mix={T_mix.shape} T_avg_pair={T_avg_pair.shape} T_rand={T_rand.shape}. "
            f"All three must share (M, d)."
        )
    m = T_mix.shape[0]
    if m == 0:
        return 0.0
    pos_sim = _row_cosines(T_mix, T_avg_pair)
    neg_sim = _row_cosines(T_mix, T_rand)
    return float((pos_sim > neg_sim).mean())


def compute_F_nc3_signal(decoder_chromosome: str) -> float:
    """NC3: structural downward-causation signal.

    Returns n_nc3_ops / n_total_ops for opcode tokens in the rendered
    decoder chromosome, where NC3 ops = {CTRL, SBC, ADDC} and total ops
    is the count of all opcode tokens (baseline 6 + NC3 3 = 9 possible).

    Rationale (see A2 plan.md §NC3): this is a STRUCTURAL metric that
    cannot be satisfied by symbols whose outputs are ignored at runtime,
    since we count literal opcode tokens in the produced program. Combined
    with the `nc3_programs_are_functionally_code_dependent` unit test in
    Rust (S1a), this guards against an NC3 simulacrum.

    Returns 0.0 when the chromosome contains no recognised opcode tokens
    (e.g. empty, or consisting only of parameter fragments).
    Range [0, 1].
    """
    if not decoder_chromosome:
        return 0.0
    # Opcode tokens are whitespace-separated ALL-CAPS heads; parameters
    # are numeric and never shadow an opcode name. Counting by token
    # equality is exact for the grammar in gen_neuro_grammar.rs.
    tokens = decoder_chromosome.split()
    n_nc3 = 0
    n_total = 0
    for tok in tokens:
        if tok in _ALL_OP_TOKENS:
            n_total += 1
            if tok in _NC3_OP_TOKENS:
                n_nc3 += 1
    if n_total == 0:
        return 0.0
    return n_nc3 / n_total


def shape_fitness_umc(
    *,
    F_nc1: float,
    F_nc2: float,
    F_nc3: float,
    F_nc4: float,
    c_matrix: np.ndarray,
    classes: np.ndarray,
    len_g: int,
    len_d: int,
    cfg: FitnessConfig,
    nc_weights: NCWeights,
    seed_F_array: np.ndarray | None = None,
) -> tuple[float, dict]:
    """A2 UMC scalarized fitness with A1.1 penalties.

    Formula (locked by A2 plan.md Q2 = scalarization):

        F_umc    = w_nc4*F_nc4 + w_nc1*F_nc1 + w_nc2*F_nc2 + w_nc3*F_nc3
        F_shaped = F_umc
                   - alpha_len  * (len_g + len_d) / L_max
                   - beta_class * (1 - silhouette(c, classes))
                   - gamma_seed * std(F across seeds)       # multi-seed only

    Returns (F_shaped, breakdown). Breakdown logs every component and
    every penalty so log.jsonl audits can reproduce the number.

    Non-finite NC inputs collapse to cfg.fallback_fitness without raising,
    matching shape_fitness semantics.
    """
    nc_values = {"F_nc1": F_nc1, "F_nc2": F_nc2, "F_nc3": F_nc3, "F_nc4": F_nc4}
    for name, val in nc_values.items():
        if not math.isfinite(val):
            return cfg.fallback_fitness, {
                "reason": f"{name}_not_finite",
                **nc_values,
            }
    if c_matrix.size == 0 or not np.isfinite(c_matrix).all():
        return cfg.fallback_fitness, {
            "reason": "c_matrix_invalid",
            "shape": list(c_matrix.shape),
            **nc_values,
        }

    F_umc = (
        nc_weights.w_nc4 * F_nc4
        + nc_weights.w_nc1 * F_nc1
        + nc_weights.w_nc2 * F_nc2
        + nc_weights.w_nc3 * F_nc3
    )

    len_penalty = cfg.alpha_len * (len_g + len_d) / max(cfg.L_max, 1.0)

    silhouette = _silhouette_cosine(c_matrix, classes)
    class_incoherence = 1.0 - silhouette
    class_penalty = cfg.beta_class * class_incoherence

    if seed_F_array is not None and seed_F_array.size >= 2:
        seed_std = float(np.std(seed_F_array))
        seed_penalty = cfg.gamma_seed * seed_std
    else:
        seed_std = 0.0
        seed_penalty = 0.0

    F_shaped = float(F_umc) - len_penalty - class_penalty - seed_penalty
    return F_shaped, {
        **nc_values,
        "w_nc1": nc_weights.w_nc1,
        "w_nc2": nc_weights.w_nc2,
        "w_nc3": nc_weights.w_nc3,
        "w_nc4": nc_weights.w_nc4,
        "F_umc": float(F_umc),
        "len_g": int(len_g),
        "len_d": int(len_d),
        "len_penalty": float(len_penalty),
        "silhouette_cos": float(silhouette),
        "class_penalty": float(class_penalty),
        "seed_std": float(seed_std),
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

    # --- A2 / S2b: UMC scalarized path ------------------------------------
    ncw = NCWeights.load()
    print(f"[fitness] loaded nc_weights: {ncw}")

    # Synthetic A2-like inputs: target_dim=16, code_dim=8.
    N, target_dim, code_dim = 128, 16, 8
    T = rng.normal(size=(N, target_dim))
    T_hat = T + rng.normal(size=T.shape) * 0.2      # ~identity reconstruction
    c = rng.normal(size=(N, code_dim))
    c_hat = c + rng.normal(size=c.shape) * 0.3      # imperfect dual fixed point

    # NC2 triplets: sample 64 (i, j, k) and build mixtures in embedding
    # space (stand-in for D(0.5*c_i + 0.5*c_j) -- in the real pipeline
    # this comes from a batch_render call).
    M = 64
    idxs = rng.choice(N, size=(M, 3), replace=True)
    T_avg = 0.5 * (T[idxs[:, 0]] + T[idxs[:, 1]])
    T_mix = T_avg + rng.normal(size=T_avg.shape) * 0.1  # anchor ~ positive
    T_rand = T[idxs[:, 2]]

    # Decoder chromosome: a mix of baseline and NC3 ops.
    decoder_chromosome = (
        "AX 2 0.5 SCALE 1.0 CTRL 3 1 MIX 0 1 0.25 "
        "ADDC 4 2 NORM SBC 0 ROT 1 2 0.5"
    )

    F_nc4 = compute_F_nc4(T, T_hat)
    F_nc1 = compute_F_nc1(c, c_hat)
    F_nc2 = compute_F_nc2(T_mix, T_avg, T_rand)
    F_nc3 = compute_F_nc3_signal(decoder_chromosome)

    print(
        f"[fitness:umc] F_nc4={F_nc4:+.4f} F_nc1={F_nc1:+.4f} "
        f"F_nc2={F_nc2:+.4f} F_nc3={F_nc3:+.4f}"
    )

    Fs_umc, info_umc = shape_fitness_umc(
        F_nc1=F_nc1, F_nc2=F_nc2, F_nc3=F_nc3, F_nc4=F_nc4,
        c_matrix=c_clean, classes=classes, len_g=10, len_d=15,
        cfg=cfg, nc_weights=ncw, seed_F_array=None,
    )
    print(
        f"[fitness:umc] F_umc={info_umc['F_umc']:+.4f} "
        f"-> F_shaped={Fs_umc:+.4f} "
        f"(len_pen={info_umc['len_penalty']:+.4f} "
        f"class_pen={info_umc['class_penalty']:+.4f})"
    )


if __name__ == "__main__":
    _self_check()
