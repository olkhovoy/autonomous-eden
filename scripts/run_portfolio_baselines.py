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
    PortfolioBaselineConfig,
    PortfolioGateConfig,
    build_portfolio_baseline_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build portfolio-level baselines and comparison gates for a conveyor portfolio ledger."
    )
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--portfolio-ledger-report", required=True)
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--curve-points", type=int, default=256)
    parser.add_argument("--turnover-cost-rate", type=float, default=None)
    parser.add_argument("--fixed-candidate-metric", default="oos_pnl")
    parser.add_argument("--rotation-metric", default="oos_pnl")
    parser.add_argument("--min-total-pnl", type=float, default=0.0)
    parser.add_argument("--max-drawdown-pct", type=float, default=15.0)
    parser.add_argument("--required-baseline", action="append")
    parser.add_argument("--min-baselines-beaten", type=int, default=2)
    parser.add_argument("--notes")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    registry = CandidateRegistry(ROOT / args.registry_root)
    portfolio_ledger_report = registry.load_portfolio_ledger(args.portfolio_ledger_report)
    config = PortfolioBaselineConfig(
        curve_points=args.curve_points,
        turnover_cost_rate=args.turnover_cost_rate,
        fixed_candidate_metric=args.fixed_candidate_metric,
        rotation_metric=args.rotation_metric,
    )
    gate_config = PortfolioGateConfig(
        min_total_pnl=args.min_total_pnl,
        max_drawdown_pct=args.max_drawdown_pct,
        required_baselines=tuple(args.required_baseline or ("flat", "equal_weight_selected_subset")),
        min_baselines_beaten=args.min_baselines_beaten,
    )
    report = build_portfolio_baseline_report(
        registry,
        args.report_name,
        portfolio_ledger_report=portfolio_ledger_report,
        config=config,
        gate_config=gate_config,
        notes=args.notes,
    )
    path = registry.save_portfolio_baselines(report)
    print(f"Saved portfolio baselines to {path}")
    print(
        f"conveyor pnl={report.conveyor_total_pnl:+.2f} "
        f"return={report.conveyor_total_return_pct:+.2f}% "
        f"max_dd={report.conveyor_max_drawdown_pct:.2f}%"
    )
    for item in report.comparisons:
        verdict = "beat" if item.beats_by_pnl else "lag"
        print(
            f"  {item.baseline_name:<30} {verdict} "
            f"delta_pnl={item.pnl_delta:+.2f} "
            f"delta_return={item.return_pct_delta:+.2f}% "
            f"dd_adv={item.drawdown_advantage_pct:+.2f}pp"
        )
    print(f"gate overall_pass={report.gate.overall_pass} checks={report.gate.checks}")


if __name__ == "__main__":
    main()
