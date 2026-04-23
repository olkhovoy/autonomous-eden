#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidates import CandidateRegistry, build_tradeforward_plan


def _parse_utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a tradeforward handoff plan from allocator or combination reports.")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--plan-name", required=True)
    parser.add_argument("--forward-start-utc", required=True)
    parser.add_argument("--forward-end-utc")
    parser.add_argument("--forward-days", type=int, default=7)
    parser.add_argument("--allocator-report")
    parser.add_argument("--combination-report")
    parser.add_argument("--scenario-name")
    parser.add_argument("--source-cycle-report")
    parser.add_argument("--notes")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.combination_report is None and args.allocator_report is None:
        raise ValueError("Provide at least one of --combination-report or --allocator-report")

    forward_start_utc = args.forward_start_utc
    forward_end_utc = args.forward_end_utc
    if forward_end_utc is None:
        forward_end_utc = _format_utc(_parse_utc(forward_start_utc) + timedelta(days=args.forward_days))

    registry = CandidateRegistry(ROOT / args.registry_root)
    allocator_report = None if args.allocator_report is None else registry.load_allocator_workbench(args.allocator_report)
    combination_report = None if args.combination_report is None else registry.load_combination_search(args.combination_report)
    plan = build_tradeforward_plan(
        registry,
        args.plan_name,
        forward_start_utc=forward_start_utc,
        forward_end_utc=forward_end_utc,
        allocator_report=allocator_report,
        combination_report=combination_report,
        scenario_name=args.scenario_name,
        source_cycle_report=args.source_cycle_report,
        notes=args.notes,
    )
    path = registry.save_tradeforward_plan(plan)
    print(f"Saved tradeforward plan to {path}")
    print(
        f"{plan.selection_mode}:{plan.scenario_name} "
        f"risk={plan.allocated_risk_fraction:.2f} "
        f"reserve={plan.reserve_fraction:.2f} "
        f"forward={plan.forward_start_utc} -> {plan.forward_end_utc}"
    )
    for allocation in plan.allocations:
        print(
            f"  {allocation.display_name} "
            f"capital={allocation.capital_fraction:.3f} "
            f"share={allocation.normalized_share:.3f} "
            f"cluster={allocation.cluster_id}"
        )


if __name__ == "__main__":
    main()
