from __future__ import annotations

import glob
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from .coinglass_client import CoinGlassClient
from .feature_builders import (
    CoinGlassSlowContextConfig,
    build_coinglass_slow_context_frame,
    build_legacy_coinglass_slow_context_frame,
)
from .join_asof import build_merged_anchor_frame, extract_market_only_frame


@dataclass(frozen=True)
class CoinGlassSlowContextBuildReport:
    source_mode: str
    anchor_rows: int
    anchor_columns: list[str]
    derivatives_rows: int
    sentiment_rows: int
    etf_rows: int
    merged_rows: int
    merged_columns: list[str]
    baseline_output_dir: str
    merged_output_dir: str


def load_anchor_market_frame(data_dir: Path) -> pl.DataFrame:
    files = glob.glob(str(data_dir / "**/*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    dfs = []
    for file in files:
        dfs.append(pl.read_parquet(file))
    return pl.concat(dfs, how="diagonal_relaxed").sort("timestamp")


def build_feature_frames(
    *,
    source_mode: str,
    anchor_frame: pl.DataFrame,
    config: CoinGlassSlowContextConfig,
    raw_cache_dir: Path,
    force_refresh: bool,
) -> dict[str, pl.DataFrame]:
    if source_mode == "legacy":
        return build_legacy_coinglass_slow_context_frame(anchor_frame, config=config)
    if source_mode != "live":
        raise ValueError(f"Unsupported source_mode: {source_mode}")

    client = CoinGlassClient(cache_dir=raw_cache_dir)
    payloads = {
        "open_interest": client.aggregated_open_interest_history(
            symbol=config.symbol,
            interval=config.interval,
            force_refresh=force_refresh,
        ),
        "funding": client.oi_weight_funding_rate_history(
            symbol=config.symbol,
            interval=config.interval,
            force_refresh=force_refresh,
        ),
        "liquidation": client.aggregated_liquidation_history(
            symbol=config.symbol,
            exchange_list=config.liquidation_exchange_list,
            interval=config.interval,
            force_refresh=force_refresh,
        ),
        "long_short_ratio": client.global_long_short_account_ratio_history(
            exchange=config.exchange,
            symbol=config.pair_symbol,
            interval=config.interval,
            force_refresh=force_refresh,
        ),
        "fear_greed": client.fear_greed_history(force_refresh=force_refresh),
        "stablecoin_market_cap": client.stablecoin_market_cap_history(force_refresh=force_refresh),
        "etf": client.btc_etf_history(force_refresh=force_refresh),
    }
    return build_coinglass_slow_context_frame(
        open_interest_payload=payloads["open_interest"],
        funding_payload=payloads["funding"],
        liquidation_payload=payloads["liquidation"],
        long_short_ratio_payload=payloads["long_short_ratio"],
        fear_greed_payload=payloads["fear_greed"],
        stablecoin_market_cap_payload=payloads["stablecoin_market_cap"],
        etf_payload=payloads["etf"],
        config=config,
    )


def _write_single_parquet(frame: pl.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_dir / "part_0001.parquet")


def build_coinglass_slow_context_datasets(
    *,
    anchor_data_dir: Path,
    baseline_output_dir: Path,
    merged_output_dir: Path,
    raw_cache_dir: Path,
    report_path: Path | None,
    source_mode: str,
    config: CoinGlassSlowContextConfig,
    force_refresh: bool = False,
) -> CoinGlassSlowContextBuildReport:
    anchor_full = load_anchor_market_frame(anchor_data_dir)
    anchor_market_only = extract_market_only_frame(anchor_full)
    feature_frames = build_feature_frames(
        source_mode=source_mode,
        anchor_frame=anchor_full,
        config=config,
        raw_cache_dir=raw_cache_dir,
        force_refresh=force_refresh,
    )
    merged = build_merged_anchor_frame(
        anchor_market_only,
        feature_frames,
        stale_after_minutes=config.stale_after_minutes,
    )

    _write_single_parquet(anchor_market_only, baseline_output_dir)
    _write_single_parquet(merged, merged_output_dir)

    report = CoinGlassSlowContextBuildReport(
        source_mode=source_mode,
        anchor_rows=anchor_market_only.height,
        anchor_columns=anchor_market_only.columns,
        derivatives_rows=feature_frames["derivatives"].height,
        sentiment_rows=feature_frames["sentiment"].height,
        etf_rows=feature_frames["etf"].height,
        merged_rows=merged.height,
        merged_columns=merged.columns,
        baseline_output_dir=str(baseline_output_dir.resolve()),
        merged_output_dir=str(merged_output_dir.resolve()),
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(asdict(report), indent=2))
    return report
