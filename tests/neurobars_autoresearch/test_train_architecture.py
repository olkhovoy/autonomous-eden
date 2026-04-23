from __future__ import annotations

import torch

from experiments.neurobars_autoresearch.train import (
    FAST_LOOKBACK,
    FAST_LATENT_DIM,
    LATENT_DIM,
    MID_LOOKBACK,
    MID_LATENT_DIM,
    MID_POOL,
    MultiScaleNeurobarEncoder,
    NeurobarPredictor,
    SLOW_LATENT_DIM,
    SLOW_LOOKBACK,
    SLOW_POOL,
)


def test_multiscale_encoder_outputs_expected_latent_shape():
    model = MultiScaleNeurobarEncoder(input_dim=144)
    batch = torch.randn(4, 128, 144)
    latent = model(batch)
    assert latent.shape == (4, LATENT_DIM)


def test_multiscale_encoder_exposes_component_latents():
    model = MultiScaleNeurobarEncoder(input_dim=144)
    batch = torch.randn(2, 128, 144)
    components = model.encode_components(batch)

    assert components["base"].shape == (2, LATENT_DIM)
    assert components["fast"].shape == (2, FAST_LATENT_DIM)
    assert components["mid"].shape == (2, MID_LATENT_DIM)
    assert components["slow"].shape == (2, SLOW_LATENT_DIM)
    assert components["fused"].shape == (2, LATENT_DIM)
    assert model.base_multiscale_dim == LATENT_DIM + FAST_LATENT_DIM + MID_LATENT_DIM + SLOW_LATENT_DIM


def test_multiscale_views_reduce_sequence_lengths_as_configured():
    model = MultiScaleNeurobarEncoder(input_dim=144)
    shared = torch.randn(2, 64, 128)

    fast_view = model._aggregate_view(shared, lookback=FAST_LOOKBACK, pool=1)
    mid_view = model._aggregate_view(shared, lookback=MID_LOOKBACK, pool=MID_POOL)
    slow_view = model._aggregate_view(shared, lookback=SLOW_LOOKBACK, pool=SLOW_POOL)

    assert fast_view.shape == (2, 64, FAST_LOOKBACK)
    assert mid_view.shape == (2, 64, MID_LOOKBACK // MID_POOL)
    assert slow_view.shape == (2, 64, SLOW_LOOKBACK // SLOW_POOL)


def test_predictor_keeps_decoder_contract():
    model = NeurobarPredictor(input_dim=144)
    batch = torch.randn(3, 128, 144)
    latent, pred = model(batch)

    assert latent.shape == (3, LATENT_DIM)
    assert pred.shape == (3, 144)
