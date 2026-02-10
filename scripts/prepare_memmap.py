from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DEFAULT_FEATURE_COLUMNS, prepare_memmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare memmap caches from market JSON.")
    parser.add_argument(
        "--data-path",
        default="data/BTCUSDT_2023-05-31_01-40_to_2025-04-24_12-19.json",
    )
    parser.add_argument("--out-dir", default="data/cache")
    parser.add_argument(
        "--normalize",
        choices=["log_return", "zscore", "none"],
        default="log_return",
    )
    parser.add_argument(
        "--compression",
        type=int,
        action="append",
        default=[],
        help="Compression factor (e.g., 60 for hourly, 1440 for daily).",
    )
    parser.add_argument("--no-raw", action="store_true", default=False)
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=list(DEFAULT_FEATURE_COLUMNS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compressions = args.compression if args.compression else None
    meta = prepare_memmap(
        args.data_path,
        out_dir=args.out_dir,
        feature_columns=args.feature_columns,
        normalize=args.normalize,
        return_raw=not args.no_raw,
        compressions=compressions,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
