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
    LifecycleConfig,
    apply_lifecycle_report,
    build_lifecycle_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the lifecycle state machine over a rolling conveyor report."
    )
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--rolling-report", required=True)
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--min-forward-pnl", type=float, default=0.0)
    parser.add_argument("--max-forward-drawdown-pct", type=float, default=8.0)
    parser.add_argument("--min-selected-trades", type=int, default=1)
    parser.add_argument("--successful-selected-cycles-to-activate", type=int, default=2)
    parser.add_argument("--successful-selected-cycles-to-recover", type=int, default=1)
    parser.add_argument("--idle-cycles-to-drain", type=int, default=1)
    parser.add_argument("--idle-cycles-to-retire", type=int, default=2)
    parser.add_argument("--apply-status-updates", action="store_true")
    parser.add_argument("--notes")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    registry = CandidateRegistry(ROOT / args.registry_root)
    rolling_report = registry.load_rolling_conveyor(args.rolling_report)
    config = LifecycleConfig(
        min_forward_pnl=args.min_forward_pnl,
        max_forward_drawdown_pct=args.max_forward_drawdown_pct,
        min_selected_trades=args.min_selected_trades,
        successful_selected_cycles_to_activate=args.successful_selected_cycles_to_activate,
        successful_selected_cycles_to_recover=args.successful_selected_cycles_to_recover,
        idle_cycles_to_drain=args.idle_cycles_to_drain,
        idle_cycles_to_retire=args.idle_cycles_to_retire,
    )
    report = build_lifecycle_report(
        registry,
        args.report_name,
        rolling_report=rolling_report,
        config=config,
        applied_status_updates=args.apply_status_updates,
        notes=args.notes,
    )
    if args.apply_status_updates:
        apply_lifecycle_report(registry, report)
    path = registry.save_lifecycle_report(report)

    print(f"Saved lifecycle report to {path}")
    print(
        f"rolling={report.source_rolling_report} "
        f"candidates={len(report.candidate_ids)} "
        f"applied={report.applied_status_updates}"
    )
    print("final_status_counts", report.final_status_counts)
    interesting = [item for item in report.candidate_summaries if item.transition_count > 0]
    for item in sorted(interesting, key=lambda row: (-row.transition_count, row.candidate_id)):
        print(
            f"  {item.display_name} "
            f"{item.initial_status}->{item.final_status} "
            f"selected={item.selected_cycles} "
            f"success={item.successful_forward_cycles} "
            f"fail={item.failed_forward_cycles} "
            f"idle={item.idle_cycles} "
            f"transitions={item.transition_count}"
        )


if __name__ == "__main__":
    main()
