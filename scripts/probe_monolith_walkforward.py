#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidate_engines import ACTION_HEAD_MODES, ENGINE_FAMILIES, EngineConfig
from umc_nn.trading_eval import evaluate_policy_path, resolve_date_window


DEFAULT_WINDOW_STARTS = [
    "2024-01-01 00:00:00",
    "2024-06-01 00:00:00",
    "2025-05-01 00:00:00",
]


@dataclass
class RunResult:
    window_name: str
    run_index: int
    train_start_utc: str
    train_end_utc: str
    valid_end_utc: str
    oos_end_utc: str
    train_start_step: int
    train_max_steps: int
    valid_start_step: int
    valid_max_steps: int
    oos_start_step: int
    oos_max_steps: int
    total_max_steps: int
    checkpoint_path: str
    log_path: str
    train_metrics: dict[str, dict[str, object]]
    valid_metrics: dict[str, dict[str, object]]
    oos_metrics: dict[str, dict[str, object]]
    total_metrics: dict[str, dict[str, object]]
    flags: dict[str, bool]


def _parse_utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _window_quads(
    window_starts: Iterable[str],
    train_days: int,
    valid_days: int,
    oos_days: int,
) -> list[tuple[str, str, str, str, str]]:
    if valid_days <= 0 or valid_days >= train_days:
        raise ValueError("valid_days must be > 0 and < train_days")
    triplets: list[tuple[str, str, str, str, str]] = []
    for idx, start_text in enumerate(window_starts, start=1):
        train_start = _parse_utc(start_text)
        train_end = train_start + timedelta(days=train_days - valid_days)
        valid_end = train_end + timedelta(days=valid_days)
        oos_end = valid_end + timedelta(days=oos_days)
        name = f"wf_{idx:02d}_{train_start.strftime('%Y%m%d')}"
        triplets.append((name, _format_utc(train_start), _format_utc(train_end), _format_utc(valid_end), _format_utc(oos_end)))
    return triplets


def _evaluate_policies(
    data_path: Path,
    start_step: int,
    max_steps: int,
    weights_path: Path,
    engine_config: EngineConfig,
    initial_balance: float,
    exchange: str,
    maker_fee_rate: float | None,
    taker_fee_rate: float | None,
    execution_fee_mode: str,
    slippage: float,
    position_sizing_mode: str,
    position_notional_fraction: float,
    leverage: float,
    fixed_position_qty: float,
) -> dict[str, dict[str, object]]:
    metrics: dict[str, dict[str, object]] = {}
    for policy_name in ("flat", "long", "short", "monolith"):
        metric = evaluate_policy_path(
            policy_name,
            data_path,
            use_neurobars=True,
            start_step=start_step,
            max_steps=max_steps,
            initial_balance=initial_balance,
            exchange=exchange,
            maker_fee_rate=maker_fee_rate,
            taker_fee_rate=taker_fee_rate,
            execution_fee_mode=execution_fee_mode,
            slippage=slippage,
            position_sizing_mode=position_sizing_mode,
            position_notional_fraction=position_notional_fraction,
            leverage=leverage,
            fixed_position_qty=fixed_position_qty,
            hidden_dim=64,
            weights_path=weights_path if policy_name == "monolith" else None,
            engine_config=engine_config if policy_name == "monolith" else None,
        )
        metrics[policy_name] = metric.to_dict()
    return metrics


def _candidate_flags(train_metrics: dict[str, dict[str, object]], oos_metrics: dict[str, dict[str, object]]) -> dict[str, bool]:
    train_monolith = train_metrics["monolith"]
    oos_monolith = oos_metrics["monolith"]
    train_best_baseline = max(train_metrics["flat"]["pnl"], train_metrics["long"]["pnl"], train_metrics["short"]["pnl"])
    oos_best_baseline = max(oos_metrics["flat"]["pnl"], oos_metrics["long"]["pnl"], oos_metrics["short"]["pnl"])

    return {
        "train_positive": float(train_monolith["pnl"]) > 0.0,
        "train_beats_flat": float(train_monolith["pnl"]) > float(train_metrics["flat"]["pnl"]),
        "train_beats_best_baseline": float(train_monolith["pnl"]) > float(train_best_baseline),
        "train_full_window": int(train_monolith["steps_run"]) == int(train_monolith["requested_max_steps"]),
        "oos_positive": float(oos_monolith["pnl"]) > 0.0,
        "oos_beats_flat": float(oos_monolith["pnl"]) > float(oos_metrics["flat"]["pnl"]),
        "oos_beats_best_baseline": float(oos_monolith["pnl"]) > float(oos_best_baseline),
        "oos_full_window": int(oos_monolith["steps_run"]) == int(oos_monolith["requested_max_steps"]),
        "oos_has_trades": int(oos_monolith["trades"]) > 0,
        "oos_profitable_and_active": float(oos_monolith["pnl"]) > 0.0 and int(oos_monolith["trades"]) > 0,
    }


def _summarize(results: list[RunResult]) -> dict[str, object]:
    if not results:
        return {"runs": 0}

    train_pnls = [float(item.train_metrics["monolith"]["pnl"]) for item in results]
    oos_pnls = [float(item.oos_metrics["monolith"]["pnl"]) for item in results]
    oos_trades = [int(item.oos_metrics["monolith"]["trades"]) for item in results]

    summary = {
        "runs": len(results),
        "train_positive_runs": sum(item.flags["train_positive"] for item in results),
        "train_beats_best_baseline_runs": sum(item.flags["train_beats_best_baseline"] for item in results),
        "oos_positive_runs": sum(item.flags["oos_positive"] for item in results),
        "oos_beats_flat_runs": sum(item.flags["oos_beats_flat"] for item in results),
        "oos_beats_best_baseline_runs": sum(item.flags["oos_beats_best_baseline"] for item in results),
        "oos_profitable_and_active_runs": sum(item.flags["oos_profitable_and_active"] for item in results),
        "median_train_pnl": median(train_pnls),
        "median_oos_pnl": median(oos_pnls),
        "median_oos_trades": median(oos_trades),
        "best_oos_run": max(results, key=lambda item: float(item.oos_metrics["monolith"]["pnl"])).checkpoint_path,
        "worst_oos_run": min(results, key=lambda item: float(item.oos_metrics["monolith"]["pnl"])).checkpoint_path,
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run repeated walk-forward probes for the current monolith candidate engine."
    )
    parser.add_argument(
        "--data-path",
        default="data/BTCUSDT_parquet_neurobars_autoresearch.npz",
        help="Neurobar dataset to use for both evolution and evaluation.",
    )
    parser.add_argument(
        "--window-start",
        action="append",
        help="UTC start timestamp for a train window. Repeat to specify multiple windows.",
    )
    parser.add_argument("--train-days", type=int, default=7)
    parser.add_argument("--valid-days", type=int, default=2)
    parser.add_argument("--oos-days", type=int, default=7)
    parser.add_argument("--runs-per-window", type=int, default=2)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--skip-filtering-phase", action="store_true")
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--maker-fee-rate", type=float, default=None)
    parser.add_argument("--taker-fee-rate", type=float, default=None)
    parser.add_argument("--execution-fee-mode", default="taker")
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--position-sizing-mode", default="fraction_of_equity")
    parser.add_argument("--position-notional-fraction", type=float, default=1.0)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--fixed-position-qty", type=float, default=1.0)
    parser.add_argument("--engine-family", choices=sorted(ENGINE_FAMILIES), default="umc")
    parser.add_argument("--engine-hidden-dim", type=int, default=64)
    parser.add_argument("--engine-alpha", type=float, default=0.5)
    parser.add_argument("--action-head-mode", choices=sorted(ACTION_HEAD_MODES), default="argmax")
    parser.add_argument("--action-threshold", type=float, default=0.55)
    parser.add_argument("--fitness-profile", default="hunter")
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--activity-target-trades", type=int, default=12)
    parser.add_argument("--trade-band-low", type=int, default=None)
    parser.add_argument("--trade-band-high", type=int, default=None)
    parser.add_argument("--trade-band-floor", type=float, default=0.25)
    parser.add_argument(
        "--output-dir",
        default="checkpoints/monolith_walkforward_probe",
        help="Directory for checkpoints, logs, and summary JSON.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    data_path = (ROOT / args.data_path).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_config = EngineConfig(
        family=args.engine_family,
        hidden_dim=args.engine_hidden_dim,
        alpha=args.engine_alpha,
        action_head_mode=args.action_head_mode,
        action_threshold=args.action_threshold,
    )
    engine_config.validate()

    window_starts = args.window_start or DEFAULT_WINDOW_STARTS
    windows = _window_quads(window_starts, args.train_days, args.valid_days, args.oos_days)
    results: list[RunResult] = []

    for window_name, train_start_utc, train_end_utc, valid_end_utc, oos_end_utc in windows:
        train_start_step, train_max_steps = resolve_date_window(data_path, train_start_utc, train_end_utc)
        valid_start_step, valid_max_steps = resolve_date_window(data_path, train_end_utc, valid_end_utc)
        oos_start_step, oos_max_steps = resolve_date_window(data_path, valid_end_utc, oos_end_utc)
        _, total_max_steps = resolve_date_window(data_path, train_start_utc, oos_end_utc)

        print(
            f"\n[{window_name}] train={train_start_utc} -> {train_end_utc} "
            f"(start_step={train_start_step}, steps={train_max_steps}) | "
            f"valid={train_end_utc} -> {valid_end_utc} (start_step={valid_start_step}, steps={valid_max_steps}) | "
            f"oos={valid_end_utc} -> {oos_end_utc} (start_step={oos_start_step}, steps={oos_max_steps})"
        )

        for run_index in range(1, args.runs_per_window + 1):
            checkpoint_path = output_dir / f"{window_name}_run{run_index:02d}.npy"
            log_path = output_dir / f"{window_name}_run{run_index:02d}.log"

            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "evolve_monolith.py"),
                "--data-path",
                str(data_path),
                "--start-step",
                str(train_start_step),
                "--max-steps",
                str(train_max_steps),
                "--generations",
                str(args.generations),
                "--population-size",
                str(args.population_size),
                *([] if not args.skip_filtering_phase else ["--skip-filtering-phase"]),
                "--weights-output",
                str(checkpoint_path),
                "--initial-balance",
                str(args.initial_balance),
                "--exchange",
                args.exchange,
                "--execution-fee-mode",
                args.execution_fee_mode,
                "--slippage",
                str(args.slippage),
                "--position-sizing-mode",
                args.position_sizing_mode,
                "--position-notional-fraction",
                str(args.position_notional_fraction),
                "--leverage",
                str(args.leverage),
                "--fixed-position-qty",
                str(args.fixed_position_qty),
                "--engine-family",
                args.engine_family,
                "--engine-hidden-dim",
                str(args.engine_hidden_dim),
                "--engine-alpha",
                str(args.engine_alpha),
                "--action-head-mode",
                args.action_head_mode,
                "--action-threshold",
                str(args.action_threshold),
                "--fitness-profile",
                args.fitness_profile,
                "--min-trades",
                str(args.min_trades),
                "--activity-target-trades",
                str(args.activity_target_trades),
            ]
            if args.trade_band_low is not None:
                cmd.extend(["--trade-band-low", str(args.trade_band_low)])
            if args.trade_band_high is not None:
                cmd.extend(["--trade-band-high", str(args.trade_band_high)])
            cmd.extend(["--trade-band-floor", str(args.trade_band_floor)])
            if args.maker_fee_rate is not None:
                cmd.extend(["--maker-fee-rate", str(args.maker_fee_rate)])
            if args.taker_fee_rate is not None:
                cmd.extend(["--taker-fee-rate", str(args.taker_fee_rate)])

            print(f"  run {run_index}/{args.runs_per_window}: evolving candidate -> {checkpoint_path.name}")
            with log_path.open("w") as log_file:
                subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(ROOT))

            train_metrics = _evaluate_policies(
                data_path,
                start_step=train_start_step,
                max_steps=train_max_steps,
                weights_path=checkpoint_path,
                engine_config=engine_config,
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
            )
            valid_metrics = _evaluate_policies(
                data_path,
                start_step=valid_start_step,
                max_steps=valid_max_steps,
                weights_path=checkpoint_path,
                engine_config=engine_config,
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
            )
            oos_metrics = _evaluate_policies(
                data_path,
                start_step=oos_start_step,
                max_steps=oos_max_steps,
                weights_path=checkpoint_path,
                engine_config=engine_config,
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
            )
            total_metrics = _evaluate_policies(
                data_path,
                start_step=train_start_step,
                max_steps=total_max_steps,
                weights_path=checkpoint_path,
                engine_config=engine_config,
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
            )
            flags = _candidate_flags(train_metrics, oos_metrics)

            print(
                "    "
                f"train pnl={train_metrics['monolith']['pnl']:+.2f} "
                f"oos pnl={oos_metrics['monolith']['pnl']:+.2f} "
                f"oos trades={oos_metrics['monolith']['trades']} "
                f"flags={flags}"
            )

            results.append(
                RunResult(
                    window_name=window_name,
                    run_index=run_index,
                    train_start_utc=train_start_utc,
                    train_end_utc=train_end_utc,
                    valid_end_utc=valid_end_utc,
                    oos_end_utc=oos_end_utc,
                    train_start_step=train_start_step,
                    train_max_steps=train_max_steps,
                    valid_start_step=valid_start_step,
                    valid_max_steps=valid_max_steps,
                    oos_start_step=oos_start_step,
                    oos_max_steps=oos_max_steps,
                    total_max_steps=total_max_steps,
                    checkpoint_path=str(checkpoint_path),
                    log_path=str(log_path),
                    train_metrics=train_metrics,
                    valid_metrics=valid_metrics,
                    oos_metrics=oos_metrics,
                    total_metrics=total_metrics,
                    flags=flags,
                )
            )

    payload = {
        "config": vars(args),
        "results": [asdict(item) for item in results],
        "summary": _summarize(results),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved walk-forward summary to {summary_path}")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
