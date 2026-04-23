from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from experiments.neurobars_autoresearch.prepare import (
    PreparedMetadata,
    build_feature_weights,
    close_delta_mse,
    load_metadata,
    prepare_cache,
    weighted_next_bar_mse,
)


def _write_parquet_shards(data_dir: Path) -> None:
    base = datetime(2025, 4, 24, 12, 18, tzinfo=timezone.utc)
    rows = []
    closes = [100.0, 101.0, 102.0, 500.0, 501.0, 502.0]
    for idx, close in enumerate(closes):
        ts = base + timedelta(minutes=idx)
        rows.append(
            {
                "timestamp": ts,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 10.0 + idx,
                "macro_feature": 0.1 * idx,
            }
        )

    first = pl.DataFrame(rows[:3])
    second = pl.DataFrame(rows[3:])
    first.write_parquet(data_dir / "part_a.parquet")
    second.write_parquet(data_dir / "part_b.parquet")


def test_prepare_cache_uses_temporal_split_and_train_only_normalization(tmp_path):
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir()
    _write_parquet_shards(data_dir)

    metadata = prepare_cache(
        data_dir=data_dir,
        cache_dir=cache_dir,
        val_start_utc="2025-04-24 12:21:00",
        seq_len=2,
        force=True,
    )

    train_features = np.load(cache_dir / "train_features.npy")
    val_features = np.load(cache_dir / "val_features.npy")
    reloaded = load_metadata(cache_dir)

    assert isinstance(metadata, PreparedMetadata)
    assert reloaded.train_rows == 3
    assert reloaded.val_rows == 3
    assert reloaded.columns[reloaded.close_idx] == "close"

    close_train = train_features[:, reloaded.close_idx]
    close_val = val_features[:, reloaded.close_idx]

    assert np.isclose(close_train.mean(), 0.0, atol=1e-5)
    assert close_val[0] > 400.0


def test_feature_weights_and_losses_are_trading_biased():
    weights = build_feature_weights(["open", "high", "low", "close", "volume", "macro"])

    assert weights[3] > weights[0]
    assert weights[0] >= 4.0
    assert weights[4] >= 2.0
    assert weights[5] == pytest.approx(1.0)

    pred = torch.tensor([[0.0, 0.0, 0.0, 2.0, 0.0, 0.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    inputs = torch.tensor([[[0.0, 0.0, 0.0, 0.5, 0.0, 0.0]]], dtype=torch.float32)

    weighted = weighted_next_bar_mse(pred, target, torch.from_numpy(weights))
    delta = close_delta_mse(inputs, pred, target, close_idx=3)

    assert weighted.item() > 0.0
    assert delta.item() == pytest.approx(1.0)
