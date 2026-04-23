#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.multistream.dataset_builder import build_coinglass_slow_context_datasets
from umc_nn.multistream.feature_builders import CoinGlassSlowContextConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build baseline_market_only_v1 and coinglass_slow_v1 parquet datasets.")
    parser.add_argument("--anchor-data-dir", default="data/BTCUSDT/data")
    parser.add_argument("--baseline-output-dir", default="data/generated/BTCUSDT/baseline_market_only_v1/data")
    parser.add_argument("--merged-output-dir", default="data/generated/BTCUSDT/coinglass_slow_v1/data")
    parser.add_argument("--raw-cache-dir", default="cache/coinglass_raw")
    parser.add_argument("--report-path", default="checkpoints/coinglass_slow_v1/build_report.json")
    parser.add_argument("--source-mode", choices=["legacy", "live"], default="legacy")
    parser.add_argument("--symbol", default="BTC")
    parser.add_argument("--pair-symbol", default="BTCUSDT")
    parser.add_argument("--exchange", default="Bybit")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--derivatives-available-lag-minutes", type=int, default=0)
    parser.add_argument("--sentiment-available-lag-minutes", type=int, default=0)
    parser.add_argument("--etf-available-lag-minutes", type=int, default=60)
    parser.add_argument("--stale-after-minutes", type=int, default=36 * 60)
    parser.add_argument("--force-refresh", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = CoinGlassSlowContextConfig(
        symbol=args.symbol,
        pair_symbol=args.pair_symbol,
        exchange=args.exchange,
        interval=args.interval,
        derivatives_available_lag_minutes=args.derivatives_available_lag_minutes,
        sentiment_available_lag_minutes=args.sentiment_available_lag_minutes,
        etf_available_lag_minutes=args.etf_available_lag_minutes,
        stale_after_minutes=args.stale_after_minutes,
    )

    report = build_coinglass_slow_context_datasets(
        anchor_data_dir=(ROOT / args.anchor_data_dir).resolve(),
        baseline_output_dir=(ROOT / args.baseline_output_dir).resolve(),
        merged_output_dir=(ROOT / args.merged_output_dir).resolve(),
        raw_cache_dir=(ROOT / args.raw_cache_dir).resolve(),
        report_path=(ROOT / args.report_path).resolve(),
        source_mode=args.source_mode,
        config=config,
        force_refresh=args.force_refresh,
    )

    print(json.dumps(report.__dict__, indent=2))


if __name__ == "__main__":
    main()
