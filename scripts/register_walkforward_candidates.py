#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidate_engines import EngineConfig
from umc_nn.candidates import (
    CandidateRecord,
    CandidateRegistry,
    ExperimentManifest,
    PeriodSpan,
    PeriodStats,
    make_candidate_id,
    utc_now_text,
)


def _baseline_winner(metrics: dict[str, dict[str, object]]) -> str:
    baseline_scores = {
        policy: float(metrics[policy]["pnl"])
        for policy in ("flat", "long", "short")
    }
    return max(baseline_scores, key=baseline_scores.get)


def _build_period(
    name: str,
    candidate_metrics: dict[str, object],
    baseline_metrics: dict[str, dict[str, object]],
    *,
    start_utc: str,
    end_utc: str,
    beats_flat: bool | None,
    beats_best_baseline: bool | None,
    full_window: bool | None,
) -> PeriodStats:
    return PeriodStats.from_episode(
        name,
        candidate_metrics,
        start_utc=start_utc,
        end_utc=end_utc,
        beats_flat=beats_flat,
        beats_best_baseline=beats_best_baseline,
        full_window=full_window,
        baseline_winner=_baseline_winner(baseline_metrics),
    )


def _build_span(
    name: str,
    *,
    start_utc: str,
    end_utc: str,
    start_step: int,
    max_steps: int,
) -> PeriodSpan:
    return PeriodSpan.from_window(
        name,
        start_utc=start_utc,
        end_utc=end_utc,
        start_step=start_step,
        max_steps=max_steps,
    )


def _manifest(config: dict[str, object], result: dict[str, object], args: argparse.Namespace) -> ExperimentManifest:
    engine_config = EngineConfig(
        family=str(config.get("engine_family", args.engine_family)),
        hidden_dim=int(config.get("engine_hidden_dim", args.engine_hidden_dim)),
        alpha=float(config.get("engine_alpha", args.engine_alpha)),
        action_head_mode=str(config.get("action_head_mode", args.action_head_mode)),
        action_threshold=float(config.get("action_threshold", args.action_threshold)),
    )
    return ExperimentManifest(
        schema_version="1",
        created_at_utc=utc_now_text(),
        source_script=str((ROOT / "scripts" / "probe_monolith_walkforward.py").resolve()),
        engine_name=args.engine_name,
        engine_role=args.engine_role,
        representation_name=args.representation_name,
        engine_family=engine_config.family,
        engine_config_id=engine_config.config_id,
        representation_branch=args.representation_name,
        fitness_profile=str(config.get("fitness_profile", args.fitness_profile)),
        data_branch_id=args.representation_name,
        source_coverage_id=args.source_coverage_id,
        market_scope=args.market_scope,
        data_path=str((ROOT / str(config["data_path"])).resolve()),
        checkpoint_path=str(Path(result["checkpoint_path"]).resolve()),
        log_path=str(Path(result["log_path"]).resolve()),
        source_summary_path=str(Path(args.summary_path).resolve()),
        train_window_utc={
            "start": str(result["train_start_utc"]),
            "end": str(result["train_end_utc"]),
        },
        oos_window_utc={
            "start": str(result["valid_end_utc"]),
            "end": str(result["oos_end_utc"]),
        },
        search_config={
            "train_days": config["train_days"],
            "valid_days": config.get("valid_days", args.valid_days),
            "oos_days": config["oos_days"],
            "runs_per_window": config["runs_per_window"],
            "generations": config["generations"],
            "population_size": config["population_size"],
            "skip_filtering_phase": bool(config.get("skip_filtering_phase", args.skip_filtering_phase)),
            "window_name": result["window_name"],
            "run_index": result["run_index"],
            "engine_family": engine_config.family,
            "engine_hidden_dim": engine_config.hidden_dim,
            "engine_alpha": engine_config.alpha,
            "action_head_mode": engine_config.action_head_mode,
            "action_threshold": engine_config.action_threshold,
            "fitness_profile": str(config.get("fitness_profile", args.fitness_profile)),
            "min_trades": int(config.get("min_trades", args.min_trades)),
            "activity_target_trades": int(config.get("activity_target_trades", args.activity_target_trades)),
            "trade_band_low": config.get("trade_band_low", args.trade_band_low),
            "trade_band_high": config.get("trade_band_high", args.trade_band_high),
            "trade_band_floor": float(config.get("trade_band_floor", args.trade_band_floor)),
        },
        econ_config={
            "initial_balance": config["initial_balance"],
            "exchange": config["exchange"],
            "maker_fee_rate": config["maker_fee_rate"],
            "taker_fee_rate": config["taker_fee_rate"],
            "execution_fee_mode": config["execution_fee_mode"],
            "slippage": config["slippage"],
            "position_sizing_mode": config["position_sizing_mode"],
            "position_notional_fraction": config["position_notional_fraction"],
            "leverage": config["leverage"],
            "fixed_position_qty": config["fixed_position_qty"],
        },
        notes=args.notes,
    )


def _candidate_record(config: dict[str, object], result: dict[str, object], args: argparse.Namespace) -> CandidateRecord:
    train_metrics = dict(result["train_metrics"])
    valid_metrics = dict(result["valid_metrics"])
    oos_metrics = dict(result["oos_metrics"])
    total_metrics = dict(result["total_metrics"])
    flags = {str(key): bool(value) for key, value in dict(result["flags"]).items()}

    baselines = {
        "train": {
            policy: PeriodStats.from_episode(policy, metrics)
            for policy, metrics in train_metrics.items()
            if policy != "monolith"
        },
        "valid": {
            policy: PeriodStats.from_episode(policy, metrics)
            for policy, metrics in valid_metrics.items()
            if policy != "monolith"
        },
        "oos_adjacent": {
            policy: PeriodStats.from_episode(policy, metrics)
            for policy, metrics in oos_metrics.items()
            if policy != "monolith"
        },
        "oos": {
            policy: PeriodStats.from_episode(policy, metrics)
            for policy, metrics in oos_metrics.items()
            if policy != "monolith"
        },
        "total": {
            policy: PeriodStats.from_episode(policy, metrics)
            for policy, metrics in total_metrics.items()
            if policy != "monolith"
        },
    }

    manifest = _manifest(config, result, args)
    engine_config = EngineConfig(
        family=str(config.get("engine_family", args.engine_family)),
        hidden_dim=int(config.get("engine_hidden_dim", args.engine_hidden_dim)),
        alpha=float(config.get("engine_alpha", args.engine_alpha)),
        action_head_mode=str(config.get("action_head_mode", args.action_head_mode)),
        action_threshold=float(config.get("action_threshold", args.action_threshold)),
    )
    identity_payload = {
        "engine_name": args.engine_name,
        "engine_family": engine_config.family,
        "engine_config_id": engine_config.config_id,
        "fitness_profile": str(config.get("fitness_profile", args.fitness_profile)),
        "representation_name": args.representation_name,
        "window_name": result["window_name"],
        "run_index": result["run_index"],
        "checkpoint_path": manifest.checkpoint_path,
    }
    candidate_id = make_candidate_id(identity_payload)

    return CandidateRecord(
        schema_version="1",
        candidate_id=candidate_id,
        display_name=f"{result['window_name']}_run{int(result['run_index']):02d}",
        engine_name=args.engine_name,
        engine_role=args.engine_role,
        status=args.status,
        created_at_utc=utc_now_text(),
        engine_family=engine_config.family,
        engine_config_id=engine_config.config_id,
        representation_branch=args.representation_name,
        fitness_profile=str(config.get("fitness_profile", args.fitness_profile)),
        data_branch_id=args.representation_name,
        source_coverage_id=args.source_coverage_id,
        market_scope=args.market_scope,
        train_span=_build_span(
            "train",
            start_utc=str(result["train_start_utc"]),
            end_utc=str(result["train_end_utc"]),
            start_step=int(result["train_start_step"]),
            max_steps=int(result["train_max_steps"]),
        ),
        valid_span=_build_span(
            "valid",
            start_utc=str(result["train_end_utc"]),
            end_utc=str(result["valid_end_utc"]),
            start_step=int(result["valid_start_step"]),
            max_steps=int(result["valid_max_steps"]),
        ),
        oos_adjacent_span=_build_span(
            "oos_adjacent",
            start_utc=str(result["valid_end_utc"]),
            end_utc=str(result["oos_end_utc"]),
            start_step=int(result["oos_start_step"]),
            max_steps=int(result["oos_max_steps"]),
        ),
        total_span=_build_span(
            "total",
            start_utc=str(result["train_start_utc"]),
            end_utc=str(result["oos_end_utc"]),
            start_step=int(result["train_start_step"]),
            max_steps=int(result["total_max_steps"]),
        ),
        tags=sorted(set(args.tag or [])),
        manifest=manifest,
        periods={
            "train": _build_period(
                "train",
                train_metrics["monolith"],
                train_metrics,
                start_utc=str(result["train_start_utc"]),
                end_utc=str(result["train_end_utc"]),
                beats_flat=flags.get("train_beats_flat"),
                beats_best_baseline=flags.get("train_beats_best_baseline"),
                full_window=flags.get("train_full_window"),
            ),
            "valid": _build_period(
                "valid",
                valid_metrics["monolith"],
                valid_metrics,
                start_utc=str(result["train_end_utc"]),
                end_utc=str(result["valid_end_utc"]),
                beats_flat=None,
                beats_best_baseline=None,
                full_window=None,
            ),
            "oos_adjacent": _build_period(
                "oos_adjacent",
                oos_metrics["monolith"],
                oos_metrics,
                start_utc=str(result["valid_end_utc"]),
                end_utc=str(result["oos_end_utc"]),
                beats_flat=flags.get("oos_beats_flat"),
                beats_best_baseline=flags.get("oos_beats_best_baseline"),
                full_window=flags.get("oos_full_window"),
            ),
            "oos": _build_period(
                "oos",
                oos_metrics["monolith"],
                oos_metrics,
                start_utc=str(result["valid_end_utc"]),
                end_utc=str(result["oos_end_utc"]),
                beats_flat=flags.get("oos_beats_flat"),
                beats_best_baseline=flags.get("oos_beats_best_baseline"),
                full_window=flags.get("oos_full_window"),
            ),
            "total": _build_period(
                "total",
                total_metrics["monolith"],
                total_metrics,
                start_utc=str(result["train_start_utc"]),
                end_utc=str(result["oos_end_utc"]),
                beats_flat=None,
                beats_best_baseline=None,
                full_window=None,
            ),
        },
        baselines=baselines,
        selection_flags=flags,
        notes=args.notes,
        metadata={
            "window_name": result["window_name"],
            "run_index": int(result["run_index"]),
            "train_start_step": int(result["train_start_step"]),
            "valid_start_step": int(result["valid_start_step"]),
            "oos_start_step": int(result["oos_start_step"]),
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import walk-forward probe candidates into the candidate registry.")
    parser.add_argument(
        "--summary-path",
        default="checkpoints/monolith_walkforward_probe/summary.json",
        help="Path to walk-forward summary JSON.",
    )
    parser.add_argument(
        "--registry-root",
        default="candidate_registry",
        help="Directory for candidate registry data.",
    )
    parser.add_argument("--engine-name", default="monolith")
    parser.add_argument("--engine-role", default="candidate_engine")
    parser.add_argument("--representation-name", default="neurobars_autoresearch_fused32")
    parser.add_argument("--engine-family", default="umc")
    parser.add_argument("--engine-hidden-dim", type=int, default=64)
    parser.add_argument("--engine-alpha", type=float, default=0.5)
    parser.add_argument("--action-head-mode", default="argmax")
    parser.add_argument("--action-threshold", type=float, default=0.55)
    parser.add_argument("--fitness-profile", default="hunter")
    parser.add_argument("--skip-filtering-phase", action="store_true")
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--activity-target-trades", type=int, default=12)
    parser.add_argument("--trade-band-low", type=int, default=None)
    parser.add_argument("--trade-band-high", type=int, default=None)
    parser.add_argument("--trade-band-floor", type=float, default=0.25)
    parser.add_argument("--valid-days", type=int, default=2)
    parser.add_argument("--market-scope", default="crypto:BTCUSDT")
    parser.add_argument("--source-coverage-id", default=None)
    parser.add_argument("--status", default="research")
    parser.add_argument("--tag", action="append", help="Tag to attach to every imported candidate.")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    payload = json.loads(Path(args.summary_path).read_text())
    registry = CandidateRegistry(ROOT / args.registry_root)

    added: list[str] = []
    for result in payload["results"]:
        record = _candidate_record(payload["config"], result, args)
        registry.add_candidate(record, overwrite=args.overwrite)
        added.append(record.candidate_id)

    print(f"Imported {len(added)} candidates into {registry.root}")
    for candidate_id in added:
        print(candidate_id)


if __name__ == "__main__":
    main()
