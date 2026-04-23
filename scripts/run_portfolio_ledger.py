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
    PortfolioLedgerConfig,
    build_portfolio_ledger_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a portfolio ledger and rebalance report from rolling and lifecycle artifacts."
    )
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--rolling-report", required=True)
    parser.add_argument("--lifecycle-report", required=True)
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--turnover-cost-rate", type=float, default=0.0)
    parser.add_argument("--tradable-status", action="append")
    parser.add_argument("--unassigned-cluster-label", default="unassigned")
    parser.add_argument("--notes")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    registry = CandidateRegistry(ROOT / args.registry_root)
    rolling_report = registry.load_rolling_conveyor(args.rolling_report)
    lifecycle_report = registry.load_lifecycle_report(args.lifecycle_report)
    config = PortfolioLedgerConfig(
        tradable_statuses=tuple(args.tradable_status or ("approved", "paper", "active", "draining")),
        turnover_cost_rate=args.turnover_cost_rate,
        unassigned_cluster_label=args.unassigned_cluster_label,
    )
    report = build_portfolio_ledger_report(
        registry,
        args.report_name,
        rolling_report=rolling_report,
        lifecycle_report=lifecycle_report,
        config=config,
        notes=args.notes,
    )
    path = registry.save_portfolio_ledger(report)
    print(f"Saved portfolio ledger to {path}")
    print(
        f"rolling={report.source_rolling_report} lifecycle={report.source_lifecycle_report} "
        f"cycles={len(report.cycle_entries)} balance={report.final_balance:.2f} "
        f"pnl={report.total_pnl:+.2f} return={report.total_return_pct:+.2f}% "
        f"max_dd={report.max_drawdown_pct:.2f}%"
    )
    print(
        f"turnover gross={report.total_gross_turnover_fraction:.3f} "
        f"buy={report.total_buy_turnover_fraction:.3f} "
        f"sell={report.total_sell_turnover_fraction:.3f} "
        f"cost={report.total_estimated_rebalance_cost:.2f}"
    )
    print(
        f"reserve_avg={report.average_reserve_fraction:.3f} "
        f"peak_cluster={report.peak_cluster_exposure_fraction:.3f} "
        f"non_tradable_selected={report.non_tradable_selection_count}"
    )
    for entry in report.cycle_entries:
        print(
            f"  cycle_{entry.cycle_index:02d} "
            f"gross={entry.gross_cycle_pnl:+.2f} net={entry.net_cycle_pnl:+.2f} "
            f"turnover={entry.gross_turnover_fraction:.3f} "
            f"reserve={entry.reserve_fraction_after:.3f} "
            f"active={len(entry.target_candidate_ids)}"
        )


if __name__ == "__main__":
    main()
