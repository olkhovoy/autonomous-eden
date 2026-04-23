from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from umc_nn.multistream.dataset_builder import build_coinglass_slow_context_datasets
from umc_nn.multistream.feature_builders import CoinGlassSlowContextConfig


def _write_anchor_parquet(data_dir: Path) -> None:
    base = datetime(2025, 5, 1, 0, 0, tzinfo=timezone.utc)
    rows = []
    for idx in range(4):
        ts = base + timedelta(days=idx)
        rows.append(
            {
                "timestamp": ts,
                "open": 100.0 + idx,
                "high": 101.0 + idx,
                "low": 99.0 + idx,
                "close": 100.5 + idx,
                "volume": 10.0 + idx,
                "turnover": 1000.0 + idx,
                "cg_oi_open_interest": 1000.0 + 10 * idx,
                "cg_oi_open_interest_change": 10.0 if idx else 0.0,
                "cg_funding_rate": 0.01 * (idx + 1),
                "cg_funding_rate_ma": 0.01 * (idx + 1),
                "cg_liquidations_long_usd": 5.0 + idx,
                "cg_liquidations_short_usd": 2.0 + idx,
                "cg_ls_ratio_value": 1.0 + 0.1 * idx,
                "cg_fear_greed_value": 40.0 + idx,
                "cg_etf_flows_net_flow": 50.0 - idx,
            }
        )
    pl.DataFrame(rows).write_parquet(data_dir / "part_0001.parquet")


def test_build_coinglass_datasets_legacy_mode_produces_baseline_and_merged(tmp_path):
    anchor_dir = tmp_path / "anchor"
    anchor_dir.mkdir()
    _write_anchor_parquet(anchor_dir)

    baseline_dir = tmp_path / "baseline" / "data"
    merged_dir = tmp_path / "coinglass" / "data"
    report_path = tmp_path / "report.json"

    report = build_coinglass_slow_context_datasets(
        anchor_data_dir=anchor_dir,
        baseline_output_dir=baseline_dir,
        merged_output_dir=merged_dir,
        raw_cache_dir=tmp_path / "raw",
        report_path=report_path,
        source_mode="legacy",
        config=CoinGlassSlowContextConfig(),
    )

    baseline = pl.read_parquet(baseline_dir / "part_0001.parquet")
    merged = pl.read_parquet(merged_dir / "part_0001.parquet")

    assert baseline.columns == ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
    assert "cg_derivatives_oi_level" in merged.columns
    assert "cg_sentiment_fear_greed_level" in merged.columns
    assert "cg_etf_btc_net_flow" in merged.columns
    assert "cg_derivatives_is_stale" in merged.columns
    assert report.anchor_rows == 4
    assert report.merged_rows == 4
    assert report_path.exists()
