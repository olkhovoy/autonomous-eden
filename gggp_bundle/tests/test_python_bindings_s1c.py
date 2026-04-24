"""
gggp_bundle/tests/test_python_bindings_s1c.py

MEDP A2 / S1c -- integration tests for the role-aware Python bindings:
  - render_tree_with_input(..., role="encoder"|"decoder")
  - chromosome_text(chromosome, role="encoder"|"decoder")

Builds fresh encoder (custom dim=8) and decoder-nc3 (target=16, code=8)
grammars in a temporary directory, then exercises every error branch:
role defaults, explicit role="decoder", cross-role parse rejection,
unknown-role rejection, missing-decoder-grammar rejection.

Run:
    cd gggp_bundle
    python -m pytest tests/test_python_bindings_s1c.py -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUST_DIR = REPO_ROOT / "gggp_bundle" / "rust"
GEN_BIN = RUST_DIR / "target" / "release" / "gen_neuro_grammar"

sys.path.insert(0, str(REPO_ROOT / "gggp_bundle" / "scripts"))
from fitness import compute_F_nc3_signal  # noqa: E402


@pytest.fixture(scope="module")
def grammars(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build (encoder dim=8, decoder-nc3 target=16 code=8) cfg pair once."""
    if not GEN_BIN.is_file():
        subprocess.run(
            ["cargo", "build", "--release", "--bin", "gen_neuro_grammar"],
            cwd=RUST_DIR, check=True, capture_output=True,
        )
    tmp = tmp_path_factory.mktemp("s1c_grammars")
    g_enc = tmp / "enc.cfg"
    g_dec = tmp / "dec_nc3.cfg"
    subprocess.run(
        [str(GEN_BIN), "custom", "8", str(g_enc)],
        check=True, capture_output=True,
    )
    subprocess.run(
        [str(GEN_BIN), "decoder-nc3-custom", "16", "8", str(g_dec)],
        check=True, capture_output=True,
    )
    return g_enc, g_dec


@pytest.fixture
def sh_pair(grammars: tuple[Path, Path]):
    from semiotic_hypercube import SemioticHypercube
    g_enc, g_dec = grammars
    sh = SemioticHypercube(str(g_enc))
    sh.attach_decoder_grammar(str(g_dec))
    return sh


# ---------------------------------------------------------------------
# render_tree_with_input: role parameter
# ---------------------------------------------------------------------

def test_render_default_role_is_encoder(sh_pair) -> None:
    chromo_g = sh_pair.random_chromosome(0, "encoder")
    out = sh_pair.render_tree_with_input(
        chromo_g, 8, np.zeros(16, dtype=np.float64)
    )
    assert out.shape == (8,)


def test_render_role_decoder_parses_against_decoder_grammar(sh_pair) -> None:
    chromo_d = sh_pair.random_chromosome(1, "decoder")
    out = sh_pair.render_tree_with_input(
        chromo_d, 16, np.zeros(8, dtype=np.float64), role="decoder"
    )
    assert out.shape == (16,)


def _find_decoder_chromo_with_nc3(sh, max_seeds: int = 64) -> list[int]:
    """Seek a decoder chromosome whose rendered program contains at least
    one NC3 opcode token. Needed so the cross-grammar rejection tests
    exercise the actual encoder/decoder divergence rather than relying on
    random decoder chromosomes that may happen to live entirely inside
    the encoder grammar's option ranges.
    """
    for seed in range(max_seeds):
        chromo_d = sh.random_chromosome(seed, "decoder")
        text = sh.chromosome_text(chromo_d, "decoder")
        if compute_F_nc3_signal(text) > 0.0:
            return chromo_d
    raise AssertionError(
        f"no NC3-containing decoder chromosome found in {max_seeds} seeds; "
        f"decoder-nc3 grammar may be broken"
    )


def test_render_decoder_chromo_rejected_by_encoder(sh_pair) -> None:
    # A random decoder chromosome may, by chance, stay inside the encoder
    # grammar's option/axis ranges and silently parse. To prove the role
    # router is actually using the encoder grammar, we use a decoder
    # chromosome that RENDERS an NC3 opcode -- the encoder grammar has
    # no such opcode and must reject it at tree_from_chromosome.
    chromo_d = _find_decoder_chromo_with_nc3(sh_pair)
    with pytest.raises(ValueError, match=r"render_tree_with_input"):
        sh_pair.render_tree_with_input(
            chromo_d, 16, None  # role defaults to "encoder"
        )


def test_render_unknown_role_is_value_error(sh_pair) -> None:
    chromo_g = sh_pair.random_chromosome(0, "encoder")
    with pytest.raises(ValueError, match="unknown role"):
        sh_pair.render_tree_with_input(
            chromo_g, 8, None, role="bogus"
        )


def test_render_role_decoder_without_attach_is_runtime_error(
    grammars: tuple[Path, Path],
) -> None:
    from semiotic_hypercube import SemioticHypercube
    g_enc, _ = grammars
    sh = SemioticHypercube(str(g_enc))  # no attach_decoder_grammar
    chromo_g = sh.random_chromosome(0, "encoder")
    with pytest.raises(RuntimeError, match="attach_decoder_grammar"):
        sh.render_tree_with_input(chromo_g, 8, None, role="decoder")


# ---------------------------------------------------------------------
# chromosome_text
# ---------------------------------------------------------------------

def test_chromosome_text_encoder_returns_baseline_ops(sh_pair) -> None:
    chromo_g = sh_pair.random_chromosome(0, "encoder")
    text = sh_pair.chromosome_text(chromo_g, "encoder")
    assert isinstance(text, str) and text.strip()
    # Encoder grammar has no NC3 ops, so NC3 signal must be exactly 0.
    assert compute_F_nc3_signal(text) == 0.0, (
        f"encoder chromosome leaked NC3 opcodes: {text!r}"
    )


def test_chromosome_text_decoder_can_emit_nc3_ops(
    grammars: tuple[Path, Path],
) -> None:
    # Over enough random decoder seeds, at least one chromosome should
    # contain at least one NC3 opcode token. This also exercises the
    # fitness.compute_F_nc3_signal -> chromosome_text contract that S2c
    # relies on.
    from semiotic_hypercube import SemioticHypercube
    g_enc, g_dec = grammars
    sh = SemioticHypercube(str(g_enc))
    sh.attach_decoder_grammar(str(g_dec))

    seen_any_nc3 = False
    for seed in range(20):
        chromo_d = sh.random_chromosome(seed, "decoder")
        text = sh.chromosome_text(chromo_d, "decoder")
        if compute_F_nc3_signal(text) > 0.0:
            seen_any_nc3 = True
            break
    assert seen_any_nc3, (
        "no NC3 opcode sampled in 20 random decoder chromosomes; "
        "something is wrong with the decoder-nc3 grammar or the "
        "signal counter."
    )


def test_chromosome_text_role_mismatch_raises(sh_pair) -> None:
    # Use a decoder chromosome that is definitely outside the encoder
    # grammar (contains an NC3 opcode). See comment in
    # test_render_decoder_chromo_rejected_by_encoder.
    chromo_d = _find_decoder_chromo_with_nc3(sh_pair)
    with pytest.raises(ValueError, match=r"chromosome_text"):
        sh_pair.chromosome_text(chromo_d, "encoder")


def test_chromosome_text_unknown_role_raises(sh_pair) -> None:
    chromo_g = sh_pair.random_chromosome(0, "encoder")
    with pytest.raises(ValueError, match="unknown role"):
        sh_pair.chromosome_text(chromo_g, "bogus")
