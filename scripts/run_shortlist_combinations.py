#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidates import AllocatorConfig, CandidateRegistry, CombinationSearchConfig, build_combination_search_report, utc_now_text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run exhaustive combination search over a shortlist pool.")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--shortlist-report", required=True)
    parser.add_argument("--report-name", required=True)
    parser.add_argument("--cluster-report")
    parser.add_argument("--override-set")
    parser.add_argument("--max-pool-size", type=int, default=6)
    parser.add_argument("--min-subset-size", type=int, default=1)
    parser.add_argument("--max-subset-size", type=int, default=3)
    parser.add_argument("--risk-fraction", type=float, action="append")
    parser.add_argument("--per-system-cap", type=float, default=0.35)
    parser.add_argument("--default-cluster-cap", type=float)
    parser.add_argument("--score-mode", choices=["base", "marginal"], default="marginal")
    parser.add_argument("--min-score-floor", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objective-max-dd", type=float, default=15.0)
    parser.add_argument("--curve-points", type=int, default=256)
    parser.add_argument("--max-stored-scenarios", type=int, default=64)
    parser.add_argument("--no-include-exceptions", action="store_true")
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

    allocator_config = AllocatorConfig(
        risk_fractions=tuple(args.risk_fraction or [0.25, 0.50, 0.75, 1.0]),
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
    config = CombinationSearchConfig(
        max_pool_size=args.max_pool_size,
        min_subset_size=args.min_subset_size,
        max_subset_size=args.max_subset_size,
        include_selected=True,
        include_exception_flags=not args.no_include_exceptions,
        max_stored_scenarios=args.max_stored_scenarios,
        allocator_config=allocator_config,
    )
    report = build_combination_search_report(
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
    path = registry.save_combination_search(report)

    print(f"Saved combination search to {path}")
    print(
        f"Pool={len(report.pool_candidate_ids)} "
        f"combinations={report.evaluated_combination_count} "
        f"scenarios={report.evaluated_scenario_count}"
    )
    if report.best_scenario_name is not None:
        print(f"Best scenario: {report.best_scenario_name}")
    for scenario in report.scenarios[: min(10, len(report.scenarios))]:
        subset = ",".join(scenario.subset_display_names)
        print(
            f"{subset} | size={scenario.subset_size} "
            f"requested={scenario.requested_risk_fraction:.2f} "
            f"risk={scenario.allocated_risk_fraction:.2f} "
            f"objective={scenario.objective_score:+.4f} "
            f"p05_pnl={scenario.resampling.p05_net_profit:+.2f} "
            f"p95_dd={scenario.resampling.p95_max_drawdown_pct:.2f}"
        )


if __name__ == "__main__":
    main()
