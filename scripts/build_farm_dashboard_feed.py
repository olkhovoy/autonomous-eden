#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidates import CandidateRegistry, build_farm_dashboard_feed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a UI-ready farm dashboard feed from a candidate farm report.")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--farm-report", required=True)
    parser.add_argument("--feed-name", required=True)
    parser.add_argument("--max-scenarios", type=int, default=500)
    parser.add_argument("--max-broom-lines", type=int, default=240)
    parser.add_argument("--notes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = CandidateRegistry(ROOT / args.registry_root)
    farm_report = registry.load_farm_report(args.farm_report)
    feed = build_farm_dashboard_feed(
        registry,
        args.feed_name,
        farm_report=farm_report,
        max_scenarios=args.max_scenarios,
        max_broom_lines=args.max_broom_lines,
        notes=args.notes,
    )
    path = registry.save_farm_dashboard_feed(feed)
    print(f"Saved farm dashboard feed to {path}")
    print(
        "summary:",
        f"scenarios={feed.summary['scenario_count']}",
        f"completed={feed.summary['completed_scenarios']}",
        f"gate_pass={feed.summary['gate_pass_count']}",
        f"running={feed.summary['running_scenarios']}",
    )
    if feed.broom is not None:
        print(f"broom lines={feed.broom['line_count']}/{feed.broom['total_line_count']}")


if __name__ == "__main__":
    main()
