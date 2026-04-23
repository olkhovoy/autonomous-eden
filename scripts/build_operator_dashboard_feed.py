#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidates import CandidateRegistry, build_dashboard_feed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a UI-ready dashboard feed from registry artifacts.")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--feed-name", required=True)
    parser.add_argument("--shortlist-report")
    parser.add_argument("--diversification-report")
    parser.add_argument("--cluster-report")
    parser.add_argument("--override-set")
    parser.add_argument("--allocator-report")
    parser.add_argument("--combination-report")
    parser.add_argument("--resampling-name")
    parser.add_argument("--max-candidate-rows", type=int, default=500)
    parser.add_argument("--max-broom-lines", type=int, default=300)
    parser.add_argument("--max-allocator-scenarios", type=int, default=6)
    parser.add_argument("--max-combination-scenarios", type=int, default=12)
    parser.add_argument("--max-audit-entries", type=int, default=50)
    parser.add_argument("--notes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = CandidateRegistry(ROOT / args.registry_root)
    shortlist_report = None if not args.shortlist_report else registry.load_shortlist_report(args.shortlist_report)
    diversification_report = None if not args.diversification_report else registry.load_diversification_report(args.diversification_report)
    cluster_report = None if not args.cluster_report else registry.load_cluster_report(args.cluster_report)
    override_set = None if not args.override_set else registry.load_override_set(args.override_set)
    allocator_report = None if not args.allocator_report else registry.load_allocator_workbench(args.allocator_report)
    combination_report = None if not args.combination_report else registry.load_combination_search(args.combination_report)

    feed = build_dashboard_feed(
        registry,
        args.feed_name,
        shortlist_report=shortlist_report,
        diversification_report=diversification_report,
        cluster_report=cluster_report,
        override_set=override_set,
        allocator_report=allocator_report,
        combination_report=combination_report,
        resampling_name=args.resampling_name,
        max_candidate_rows=args.max_candidate_rows,
        max_broom_lines=args.max_broom_lines,
        max_allocator_scenarios=args.max_allocator_scenarios,
        max_combination_scenarios=args.max_combination_scenarios,
        max_audit_entries=args.max_audit_entries,
        notes=args.notes,
    )
    path = registry.save_dashboard_feed(feed)
    print(f"Saved dashboard feed to {path}")
    print(
        "summary:",
        f"candidates={feed.summary['total_candidates']}",
        f"visible={feed.summary['visible_candidate_rows']}",
        f"selected={feed.summary['selected_candidate_count']}",
        f"exceptions={feed.summary['exception_candidate_count']}",
        f"clusters={feed.summary['cluster_count']}",
    )
    if feed.allocator is not None:
        print(f"allocator chosen={feed.allocator['chosen_scenario_name']}")
    if feed.combinations is not None:
        print(f"combination best={feed.combinations['best_scenario_name']}")


if __name__ == "__main__":
    main()
