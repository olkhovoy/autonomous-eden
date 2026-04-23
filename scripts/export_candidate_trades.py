#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidate_engines import EngineConfig
from umc_nn.candidates import CandidateRegistry, TracePeriodRecord, TradeRecord
from umc_nn.trading_eval import EpisodeTrade, evaluate_policy_trace_path, resolve_date_window


def _candidate_ids(registry: CandidateRegistry, args: argparse.Namespace) -> list[str]:
    if args.candidate_id:
        return list(args.candidate_id)
    return [candidate.candidate_id for candidate in registry.list_candidates(status=args.status, tags=args.tag)]


def _to_trade_record(period_name: str, trade: EpisodeTrade, source: str) -> TradeRecord:
    return TradeRecord(
        trade_id=f"{period_name}_{trade.trade_id}",
        period_name=period_name,
        direction=trade.direction,
        entry_step=trade.entry_step,
        exit_step=trade.exit_step,
        entry_timestamp_utc=trade.entry_timestamp_utc,
        exit_timestamp_utc=trade.exit_timestamp_utc,
        entry_price=trade.entry_price,
        exit_price=trade.exit_price,
        quantity=trade.quantity,
        gross_pnl=trade.gross_pnl,
        net_pnl=trade.net_pnl,
        fees_paid=trade.fees_paid,
        duration_steps=trade.duration_steps,
        entry_balance=trade.entry_balance,
        exit_balance=trade.exit_balance,
        return_on_equity=trade.return_on_equity,
        source=source,
    )


def _to_trace_record(period_name: str, trace, source: str, *, start_utc: str | None, end_utc: str | None) -> TracePeriodRecord:
    return TracePeriodRecord(
        period_name=period_name,
        source=source,
        start_step=trace.metrics.start_step,
        requested_max_steps=trace.metrics.requested_max_steps,
        steps_run=trace.metrics.steps_run,
        start_utc=start_utc,
        end_utc=end_utc,
        balance_history=list(trace.balance_history),
        action_history=list(trace.action_history),
        position_history=list(trace.position_history),
    )


def _engine_config_for_candidate(candidate) -> EngineConfig:
    manifest = candidate.manifest
    if manifest is None:
        return EngineConfig()
    search = manifest.search_config or {}
    return EngineConfig(
        family=str(candidate.engine_family or manifest.engine_family or search.get("engine_family") or "umc"),
        hidden_dim=int(search.get("engine_hidden_dim", search.get("hidden_dim", 64))),
        alpha=float(search.get("engine_alpha", search.get("alpha", 0.5))),
        action_head_mode=str(search.get("action_head_mode", "argmax")),
        action_threshold=float(search.get("action_threshold", 0.55)),
    )


def _trace_for_period(candidate, period_name: str):
    manifest = candidate.manifest
    assert manifest is not None
    if period_name == "train":
        span = candidate.train_span
        window = manifest.train_window_utc
    elif period_name == "valid":
        span = candidate.valid_span
        window = None if span is None else {"start": span.start_utc, "end": span.end_utc}
    elif period_name in {"oos", "oos_adjacent"}:
        span = candidate.oos_adjacent_span
        window = manifest.oos_window_utc if period_name == "oos" else None
        if span is not None:
            window = {"start": span.start_utc, "end": span.end_utc}
    elif period_name == "total":
        span = candidate.total_span
        window = None if span is None else {"start": span.start_utc, "end": span.end_utc}
    else:
        raise ValueError(f"Unsupported period: {period_name}")
    if window is None or window.get("start") is None or window.get("end") is None:
        raise ValueError(f"Candidate {candidate.candidate_id} has no {period_name} window")

    start_step, max_steps = resolve_date_window(manifest.data_path, window["start"], window["end"])
    econ = manifest.econ_config
    return evaluate_policy_trace_path(
        "monolith",
        manifest.data_path,
        use_neurobars=str(manifest.data_path).endswith(".npz"),
        start_step=start_step,
        max_steps=max_steps,
        initial_balance=float(econ["initial_balance"]),
        exchange=str(econ["exchange"]),
        maker_fee_rate=econ.get("maker_fee_rate"),
        taker_fee_rate=econ.get("taker_fee_rate"),
        execution_fee_mode=str(econ["execution_fee_mode"]),
        slippage=float(econ["slippage"]),
        position_sizing_mode=str(econ["position_sizing_mode"]),
        position_notional_fraction=float(econ["position_notional_fraction"]),
        leverage=float(econ["leverage"]),
        fixed_position_qty=float(econ["fixed_position_qty"]),
        weights_path=manifest.checkpoint_path,
        engine_config=_engine_config_for_candidate(candidate),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export trade-level artifacts for candidates in the registry.")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--status")
    parser.add_argument("--tag", action="append")
    parser.add_argument(
        "--period",
        action="append",
        choices=["train", "valid", "oos", "oos_adjacent", "total"],
        help="Repeat to limit exported periods.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    registry = CandidateRegistry(ROOT / args.registry_root)
    period_names = args.period or ["train", "valid", "oos_adjacent"]

    for candidate_id in _candidate_ids(registry, args):
        candidate = registry.load_candidate(candidate_id)
        exported: list[TradeRecord] = []
        trace_records: list[TracePeriodRecord] = []
        for period_name in period_names:
            try:
                trace = _trace_for_period(candidate, period_name)
            except ValueError:
                continue
            if period_name == "train":
                span = candidate.train_span
            elif period_name == "valid":
                span = candidate.valid_span
            elif period_name in {"oos", "oos_adjacent"}:
                span = candidate.oos_adjacent_span
            else:
                span = candidate.total_span
            exported.extend(
                _to_trade_record(period_name, trade, source="evaluate_policy_trace_path")
                for trade in trace.trades
            )
            trace_records.append(
                _to_trace_record(
                    period_name,
                    trace,
                    "evaluate_policy_trace_path",
                    start_utc=None if span is None else span.start_utc,
                    end_utc=None if span is None else span.end_utc,
                )
            )
        path = registry.attach_trade_records(candidate_id, exported, overwrite=True)
        trace_path = registry.attach_trace_records(candidate_id, trace_records, overwrite=True)
        print(candidate_id, len(exported), path, trace_path)


if __name__ == "__main__":
    main()
