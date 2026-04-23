#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidates import (
    CandidateRegistry,
    DiversificationReport,
    all_pairwise_diversification,
    build_trace_view,
    curve_snapshot,
    utc_now_text,
)
from umc_nn.candidates.engine_config import candidate_engine_config
from umc_nn.trading_eval import DEFAULT_DATE_WINDOWS, evaluate_policy_trace_path, resolve_date_window


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute pairwise diversification metrics for registry candidates on a common evaluation window.")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--status")
    parser.add_argument("--tag", action="append")
    parser.add_argument("--date-window", choices=sorted(DEFAULT_DATE_WINDOWS.keys()))
    parser.add_argument("--start-utc")
    parser.add_argument("--end-utc")
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--curve-points", type=int, default=512)
    parser.add_argument("--notes")
    return parser


def _candidate_ids(registry: CandidateRegistry, args: argparse.Namespace) -> list[str]:
    if args.candidate_id:
        return list(args.candidate_id)
    return [candidate.candidate_id for candidate in registry.list_candidates(status=args.status, tags=args.tag)]


def _resolve_window(args: argparse.Namespace) -> tuple[str, str]:
    if args.date_window:
        return DEFAULT_DATE_WINDOWS[args.date_window]
    if args.start_utc and args.end_utc:
        return args.start_utc, args.end_utc
    raise ValueError("Provide either --date-window or both --start-utc and --end-utc")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    registry = CandidateRegistry(ROOT / args.registry_root)
    candidate_ids = _candidate_ids(registry, args)
    if len(candidate_ids) < 2:
        raise ValueError("Need at least two candidates for diversification analysis")

    candidates = [registry.load_candidate(candidate_id) for candidate_id in candidate_ids]
    manifests = [candidate.manifest for candidate in candidates]
    if any(manifest is None for manifest in manifests):
        raise ValueError("All candidates must have a manifest")

    data_paths = {manifest.data_path for manifest in manifests if manifest is not None}
    if len(data_paths) != 1:
        raise ValueError("All selected candidates must use the same data_path for common-window diversification")
    data_path = next(iter(data_paths))

    start_utc, end_utc = _resolve_window(args)
    start_step, max_steps = resolve_date_window(data_path, start_utc, end_utc)

    trace_views = []
    for candidate in candidates:
        manifest = candidate.manifest
        assert manifest is not None
        econ = manifest.econ_config
        trace = evaluate_policy_trace_path(
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
            engine_config=candidate_engine_config(candidate),
        )
        trace_views.append(build_trace_view(candidate.candidate_id, candidate.display_name, trace))

    pair_stats = all_pairwise_diversification(trace_views)
    pair_stats.sort(key=lambda item: (item.drawdown_improvement_pct_points, -item.simultaneous_loss_rate), reverse=True)
    candidate_curves = [curve_snapshot(item, max_points=args.curve_points) for item in trace_views]

    report = DiversificationReport(
        schema_version="1",
        name=args.report_name,
        created_at_utc=utc_now_text(),
        data_path=data_path,
        start_utc=start_utc,
        end_utc=end_utc,
        start_step=start_step,
        max_steps=max_steps,
        candidate_ids=candidate_ids,
        candidate_curves=candidate_curves,
        pair_stats=pair_stats,
        notes=args.notes,
    )
    path = registry.save_diversification_report(report)

    print(f"Saved diversification report to {path}")
    for item in pair_stats:
        print(
            f"{item.left_display_name} <-> {item.right_display_name} | "
            f"corr={item.return_corr:+.3f} "
            f"downside={item.downside_corr:+.3f} "
            f"sim_loss={item.simultaneous_loss_rate:.3f} "
            f"combo_dd={item.equal_weight_max_drawdown_pct:.2f} "
            f"dd_improve={item.drawdown_improvement_pct_points:+.2f}"
        )


if __name__ == "__main__":
    main()
