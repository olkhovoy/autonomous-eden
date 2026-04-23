#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidates import AllocatorConfig, CandidateRegistry, build_allocator_workbench_report, utc_now_text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build allocator workbench scenarios from a shortlist report.")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--shortlist-report", required=True)
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--cluster-report")
    parser.add_argument("--override-set")
    parser.add_argument("--risk-fraction", type=float, action="append", help="Repeat for multiple gross capital fractions.")
    parser.add_argument("--per-system-cap", type=float, default=0.35)
    parser.add_argument("--default-cluster-cap", type=float)
    parser.add_argument("--score-mode", choices=["base", "marginal"], default="marginal")
    parser.add_argument("--min-score-floor", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objective-max-dd", type=float, default=15.0)
    parser.add_argument("--curve-points", type=int, default=256)
    parser.add_argument("--notes")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    registry = CandidateRegistry(ROOT / args.registry_root)
    shortlist = registry.load_shortlist_report(args.shortlist_report)
    diversification = registry.load_diversification_report(shortlist.source_diversification_report)
    cluster_report = None if args.cluster_report is None else registry.load_cluster_report(args.cluster_report)
    override_set = None if args.override_set is None else registry.load_override_set(args.override_set)
    candidates = [registry.load_candidate(candidate_id) for candidate_id in shortlist.candidate_ids]

    risk_fractions = tuple(args.risk_fraction or [0.25, 0.50, 0.75, 1.0])
    config = AllocatorConfig(
        risk_fractions=risk_fractions,
        per_system_cap_fraction=args.per_system_cap,
        default_cluster_cap_fraction=args.default_cluster_cap,
        score_mode=args.score_mode,
        min_score_floor=args.min_score_floor,
        resampling_iterations=args.iterations,
        resampling_block_size=args.block_size,
        resampling_seed=args.seed,
        objective_max_drawdown_pct=args.objective_max_dd,
        curve_points=args.curve_points,
    )
    report = build_allocator_workbench_report(
        name=args.report_name,
        created_at_utc=utc_now_text(),
        candidates=candidates,
        shortlist_report=shortlist,
        diversification_report=diversification,
        cluster_report=cluster_report,
        override_set=override_set,
        config=config,
        notes=args.notes,
    )
    path = registry.save_allocator_workbench(report)

    print(f"Saved allocator workbench to {path}")
    if report.chosen_scenario_name is not None:
        print(f"Chosen scenario: {report.chosen_scenario_name}")

    for scenario in report.scenarios:
        print(
            f"{scenario.name} | "
            f"objective={scenario.objective_score:+.4f} "
            f"risk={scenario.allocated_risk_fraction:.2f} "
            f"reserve={scenario.reserve_fraction:.2f} "
            f"orig_pnl={scenario.resampling.original_net_profit:+.2f} "
            f"p05_pnl={scenario.resampling.p05_net_profit:+.2f} "
            f"p95_dd={scenario.resampling.p95_max_drawdown_pct:.2f}"
        )
        for weight in scenario.weights:
            print(
                f"  {weight.display_name} "
                f"capital={weight.capital_fraction:.3f} "
                f"share={weight.normalized_share:.3f} "
                f"raw={weight.raw_score:.3f}"
            )


if __name__ == "__main__":
    main()
