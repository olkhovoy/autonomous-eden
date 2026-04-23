#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidates import CandidateRegistry, build_tradeforward_evaluation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a tradeforward plan on its forward window and save the result.")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--plan-name", required=True)
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--curve-points", type=int, default=256)
    parser.add_argument("--notes")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    registry = CandidateRegistry(ROOT / args.registry_root)
    plan = registry.load_tradeforward_plan(args.plan_name)
    report = build_tradeforward_evaluation(
        registry,
        args.report_name,
        plan=plan,
        curve_points=args.curve_points,
        notes=args.notes,
    )
    path = registry.save_tradeforward_evaluation(report)
    print(f"Saved tradeforward evaluation to {path}")
    expectation = report.expectation
    if expectation is not None:
        print(
            f"{report.selection_mode}:{report.scenario_name} "
            f"actual_pnl={report.portfolio.pnl:+.2f} "
            f"expected_orig={expectation.expected_original_net_profit:+.2f} "
            f"delta={report.portfolio.actual_minus_expected_original_net_profit:+.2f}"
        )
    else:
        print(
            f"{report.selection_mode}:{report.scenario_name} "
            f"actual_pnl={report.portfolio.pnl:+.2f} "
            f"dd={report.portfolio.max_drawdown_pct:.2f}"
        )
    for item in report.candidate_evaluations:
        print(
            f"  {item.display_name} "
            f"capital={item.capital_fraction:.3f} "
            f"pnl={item.pnl:+.2f} "
            f"dd={item.max_drawdown_pct:.2f} "
            f"trades={item.trades}"
        )


if __name__ == "__main__":
    main()
