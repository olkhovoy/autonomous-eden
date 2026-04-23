#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidate_engines import ACTION_HEAD_MODES, ENGINE_FAMILIES, EngineConfig
from umc_nn.trading_eval import (
    DEFAULT_DATE_WINDOWS,
    EpisodeMetrics,
    EXECUTION_FEE_MODES,
    EXCHANGE_FEE_PRESETS,
    POSITION_SIZING_MODES,
    evaluate_policy_path,
    format_metrics_table,
    resolve_date_window,
)


def _window_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-step", type=int, help="Explicit start index")
    parser.add_argument("--max-steps", type=int, help="Explicit number of steps")
    parser.add_argument(
        "--date-window",
        choices=sorted(DEFAULT_DATE_WINDOWS.keys()),
        default="train_2024_2025",
        help="Named UTC date window for .npz data with timestamps",
    )
    parser.add_argument(
        "--all-date-windows",
        action="store_true",
        help="Run all default date windows instead of a single slice",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trading baselines and monolith on a fixed slice.")
    parser.add_argument(
        "--data-path",
        default="data/BTCUSDT_parquet_neurobars.npz",
        help="Path to JSON or NPZ market data",
    )
    parser.add_argument(
        "--weights-path",
        default="checkpoints/monolith_best_weights_v2.npy",
        help="Checkpoint for monolith policy",
    )
    parser.add_argument(
        "--policy",
        action="append",
        choices=["flat", "long", "short", "monolith"],
        help="Policy to evaluate. Repeat to select multiple policies.",
    )
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--exchange", choices=sorted(EXCHANGE_FEE_PRESETS.keys()), default="binance")
    parser.add_argument("--maker-fee-rate", type=float, default=None)
    parser.add_argument("--taker-fee-rate", type=float, default=None)
    parser.add_argument("--execution-fee-mode", choices=sorted(EXECUTION_FEE_MODES), default="taker")
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument(
        "--position-sizing-mode",
        choices=sorted(POSITION_SIZING_MODES),
        default="fraction_of_equity",
    )
    parser.add_argument("--position-notional-fraction", type=float, default=1.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--fixed-position-qty", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--engine-family", choices=sorted(ENGINE_FAMILIES), default="umc")
    parser.add_argument("--engine-alpha", type=float, default=0.5)
    parser.add_argument("--action-head-mode", choices=sorted(ACTION_HEAD_MODES), default="argmax")
    parser.add_argument("--action-threshold", type=float, default=0.55)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--json-output",
        help="Optional path to save structured metrics as JSON",
    )
    _window_args(parser)
    return parser


def _resolve_windows(args: argparse.Namespace) -> list[tuple[str, int, int]]:
    if args.start_step is not None or args.max_steps is not None:
        if args.start_step is None or args.max_steps is None:
            raise ValueError("--start-step and --max-steps must be provided together")
        return [("explicit", args.start_step, args.max_steps)]

    if args.all_date_windows:
        return [
            (name, *resolve_date_window(args.data_path, start_utc, end_utc))
            for name, (start_utc, end_utc) in DEFAULT_DATE_WINDOWS.items()
        ]

    start_utc, end_utc = DEFAULT_DATE_WINDOWS[args.date_window]
    start_step, max_steps = resolve_date_window(args.data_path, start_utc, end_utc)
    return [(args.date_window, start_step, max_steps)]


def _policies(args: argparse.Namespace) -> list[str]:
    if args.policy:
        return args.policy
    return ["flat", "long", "short", "monolith"]


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    policies = _policies(args)
    windows = _resolve_windows(args)
    engine_config = EngineConfig(
        family=args.engine_family,
        hidden_dim=args.hidden_dim,
        alpha=args.engine_alpha,
        action_head_mode=args.action_head_mode,
        action_threshold=args.action_threshold,
    )

    output: dict[str, list[dict[str, object]]] = {}
    for window_name, start_step, max_steps in windows:
        metrics: list[EpisodeMetrics] = []
        for policy_name in policies:
            weights_path = args.weights_path if policy_name == "monolith" else None
            metric = evaluate_policy_path(
                policy_name,
                args.data_path,
                use_neurobars=str(args.data_path).endswith(".npz"),
                start_step=start_step,
                max_steps=max_steps,
                initial_balance=args.initial_balance,
                exchange=args.exchange,
                maker_fee_rate=args.maker_fee_rate,
                taker_fee_rate=args.taker_fee_rate,
                execution_fee_mode=args.execution_fee_mode,
                slippage=args.slippage,
                position_sizing_mode=args.position_sizing_mode,
                position_notional_fraction=args.position_notional_fraction,
                leverage=args.leverage,
                fixed_position_qty=args.fixed_position_qty,
                hidden_dim=args.hidden_dim,
                weights_path=weights_path,
                device=args.device,
                engine_config=engine_config if policy_name == "monolith" else None,
            )
            metrics.append(metric)

        print(f"\n[{window_name}] start_step={start_step} max_steps={max_steps}")
        print(format_metrics_table(metrics))
        output[window_name] = [metric.to_dict() for metric in metrics]

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(output, indent=2))
        print(f"\nSaved JSON metrics to {json_path}")


if __name__ == "__main__":
    main()
