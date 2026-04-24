"""
gggp_bundle/tests/test_fitness_umc.py

MEDP A2 / S2b -- unit tests for the UMC scalarized fitness helpers.

Run:
    cd gggp_bundle
    python -m pytest tests/test_fitness_umc.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "gggp_bundle" / "scripts"))

from fitness import (  # noqa: E402
    FitnessConfig,
    NCWeights,
    compute_F_nc1,
    compute_F_nc2,
    compute_F_nc3_signal,
    compute_F_nc4,
    shape_fitness_umc,
)


# ---------------------------------------------------------------------
# NCWeights loader
# ---------------------------------------------------------------------

def test_ncweights_loads_from_default_config() -> None:
    ncw = NCWeights.load()
    # S2a-locked defaults; update here if config ever changes on purpose.
    assert ncw.w_nc4 == 1.0
    assert ncw.w_nc1 == 0.5
    assert ncw.w_nc2 == 0.2
    assert ncw.w_nc3 == 0.3


def test_ncweights_missing_section_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.toml"
    p.write_text("[fitness]\nalpha_len=0.02\nL_max=48\nbeta_class=0.3\ngamma_seed=0.5\nfallback_fitness=-1.0\n", encoding="utf-8")
    with pytest.raises(KeyError, match=r"\[nc_weights\]"):
        NCWeights.load(p)


def test_ncweights_missing_key_raises(tmp_path: Path) -> None:
    p = tmp_path / "partial.toml"
    p.write_text("[nc_weights]\nw_nc1=0.5\nw_nc2=0.2\nw_nc4=1.0\n", encoding="utf-8")
    with pytest.raises(KeyError, match="w_nc3"):
        NCWeights.load(p)


# ---------------------------------------------------------------------
# NC4 -- mean row cosine, carry-over from A1.1
# ---------------------------------------------------------------------

def test_nc4_perfect_reconstruction_is_one() -> None:
    T = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert compute_F_nc4(T, T) == pytest.approx(1.0)


def test_nc4_antipodal_is_minus_one() -> None:
    T = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert compute_F_nc4(T, -T) == pytest.approx(-1.0)


def test_nc4_zero_row_scores_zero() -> None:
    T = np.array([[0.0, 0.0], [1.0, 0.0]])
    T_hat = np.array([[1.0, 0.0], [1.0, 0.0]])
    # First row contributes 0 (no direction), second row contributes 1.
    assert compute_F_nc4(T, T_hat) == pytest.approx(0.5)


# ---------------------------------------------------------------------
# NC1 -- dual fixed point
# ---------------------------------------------------------------------

def test_nc1_identity_recovery() -> None:
    c = np.random.default_rng(0).normal(size=(32, 8))
    assert compute_F_nc1(c, c) == pytest.approx(1.0)


def test_nc1_shape_mismatch_raises() -> None:
    a = np.zeros((4, 8))
    b = np.zeros((4, 16))
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_F_nc1(a, b)


# ---------------------------------------------------------------------
# NC2 -- compositional triplet accuracy
# ---------------------------------------------------------------------

def test_nc2_anchor_closer_to_positive_is_one() -> None:
    rng = np.random.default_rng(0)
    T_mix = rng.normal(size=(16, 4))
    T_pos = T_mix.copy()            # anchor == positive (d=0)
    T_neg = rng.normal(size=(16, 4))
    # With anchor == positive, cos == 1 > cos(anchor, any neg) almost surely.
    assert compute_F_nc2(T_mix, T_pos, T_neg) == pytest.approx(1.0)


def test_nc2_anchor_closer_to_negative_is_zero() -> None:
    rng = np.random.default_rng(1)
    T_mix = rng.normal(size=(16, 4))
    T_neg = T_mix.copy()
    T_pos = rng.normal(size=(16, 4))
    assert compute_F_nc2(T_mix, T_pos, T_neg) == pytest.approx(0.0)


def test_nc2_random_triplets_near_half() -> None:
    rng = np.random.default_rng(2)
    M, d = 2000, 16
    T_mix = rng.normal(size=(M, d))
    T_pos = rng.normal(size=(M, d))
    T_neg = rng.normal(size=(M, d))
    acc = compute_F_nc2(T_mix, T_pos, T_neg)
    assert 0.45 < acc < 0.55, f"expected ~0.5, got {acc}"


def test_nc2_shape_mismatch_raises() -> None:
    a = np.zeros((4, 8))
    b = np.zeros((4, 8))
    c = np.zeros((5, 8))
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_F_nc2(a, b, c)


# ---------------------------------------------------------------------
# NC3 -- structural signal (token fraction)
# ---------------------------------------------------------------------

def test_nc3_only_baseline_is_zero() -> None:
    chromosome = "AX 2 0.5 SCALE 1.0 NORM MIX 0 1 0.25 ROT 1 2 0.5 FRAC 1.5"
    assert compute_F_nc3_signal(chromosome) == pytest.approx(0.0)


def test_nc3_only_code_gated_is_one() -> None:
    chromosome = "CTRL 3 1 SBC 0 ADDC 4 2"
    assert compute_F_nc3_signal(chromosome) == pytest.approx(1.0)


def test_nc3_half_and_half() -> None:
    # 4 opcode tokens total: 2 NC3 (CTRL, ADDC) and 2 baseline (AX, NORM).
    chromosome = "AX 2 0.5 CTRL 3 1 NORM ADDC 4 2"
    assert compute_F_nc3_signal(chromosome) == pytest.approx(0.5)


def test_nc3_empty_chromosome_is_zero() -> None:
    assert compute_F_nc3_signal("") == pytest.approx(0.0)


def test_nc3_no_opcode_tokens_is_zero() -> None:
    # Pure numeric garbage has no opcode tokens -> denominator 0 -> 0.0.
    assert compute_F_nc3_signal("1 2 3 4.5 0.25") == pytest.approx(0.0)


def test_nc3_ignores_parameter_numbers() -> None:
    # "3" and "1" are CTRL parameters, not opcodes, and must not inflate
    # the denominator. n_total == 2 opcodes (CTRL + SBC), n_nc3 == 2.
    chromosome = "CTRL 3 1 SBC 0"
    assert compute_F_nc3_signal(chromosome) == pytest.approx(1.0)


# ---------------------------------------------------------------------
# shape_fitness_umc -- end-to-end scalarization
# ---------------------------------------------------------------------

@pytest.fixture
def cfg_and_weights() -> tuple[FitnessConfig, NCWeights]:
    return FitnessConfig.load(), NCWeights.load()


def _make_c_matrix(n: int = 32, d: int = 8, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, d))


def _make_classes(n: int = 32, k: int = 8) -> np.ndarray:
    return np.repeat(np.arange(k), n // k).astype(np.int32)


def test_umc_scalarization_matches_weights(cfg_and_weights) -> None:
    cfg, ncw = cfg_and_weights
    c = _make_c_matrix()
    classes = _make_classes()
    F_shaped, info = shape_fitness_umc(
        F_nc1=0.6, F_nc2=0.55, F_nc3=0.3, F_nc4=0.9,
        c_matrix=c, classes=classes, len_g=0, len_d=0,
        cfg=cfg, nc_weights=ncw, seed_F_array=None,
    )
    # Penalties with len_g=len_d=0 cancel the compactness term, and the
    # synthetic c_matrix has no class structure so class penalty != 0.
    expected_umc = (
        ncw.w_nc4 * 0.9 + ncw.w_nc1 * 0.6
        + ncw.w_nc2 * 0.55 + ncw.w_nc3 * 0.3
    )
    assert info["F_umc"] == pytest.approx(expected_umc, rel=1e-9)
    # F_shaped = F_umc - class_penalty (len_penalty == 0).
    assert F_shaped == pytest.approx(
        expected_umc - info["class_penalty"], rel=1e-9
    )


def test_umc_non_finite_nc_collapses_to_fallback(cfg_and_weights) -> None:
    cfg, ncw = cfg_and_weights
    c = _make_c_matrix()
    classes = _make_classes()
    F_shaped, info = shape_fitness_umc(
        F_nc1=float("nan"), F_nc2=0.5, F_nc3=0.3, F_nc4=0.9,
        c_matrix=c, classes=classes, len_g=0, len_d=0,
        cfg=cfg, nc_weights=ncw, seed_F_array=None,
    )
    assert F_shaped == cfg.fallback_fitness
    assert info["reason"] == "F_nc1_not_finite"


def test_umc_invalid_c_matrix_collapses_to_fallback(cfg_and_weights) -> None:
    cfg, ncw = cfg_and_weights
    c = np.array([[1.0, float("inf")]])
    classes = np.array([0])
    F_shaped, info = shape_fitness_umc(
        F_nc1=0.6, F_nc2=0.5, F_nc3=0.3, F_nc4=0.9,
        c_matrix=c, classes=classes, len_g=0, len_d=0,
        cfg=cfg, nc_weights=ncw, seed_F_array=None,
    )
    assert F_shaped == cfg.fallback_fitness
    assert info["reason"] == "c_matrix_invalid"


def test_umc_seed_stability_penalty_applied(cfg_and_weights) -> None:
    cfg, ncw = cfg_and_weights
    c = _make_c_matrix()
    classes = _make_classes()
    stable = np.array([0.9, 0.9, 0.9])
    noisy = np.array([0.2, 0.9, 0.5])
    _, info_stable = shape_fitness_umc(
        F_nc1=0.6, F_nc2=0.5, F_nc3=0.3, F_nc4=0.9,
        c_matrix=c, classes=classes, len_g=0, len_d=0,
        cfg=cfg, nc_weights=ncw, seed_F_array=stable,
    )
    _, info_noisy = shape_fitness_umc(
        F_nc1=0.6, F_nc2=0.5, F_nc3=0.3, F_nc4=0.9,
        c_matrix=c, classes=classes, len_g=0, len_d=0,
        cfg=cfg, nc_weights=ncw, seed_F_array=noisy,
    )
    assert info_stable["seed_penalty"] == pytest.approx(0.0)
    assert info_noisy["seed_penalty"] > 0.0
    assert info_noisy["seed_std"] == pytest.approx(float(np.std(noisy)))
