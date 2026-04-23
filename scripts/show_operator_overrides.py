#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidates import CandidateRegistry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show one operator override set.")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--override-name", required=True)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    registry = CandidateRegistry(ROOT / args.registry_root)
    override_set = registry.load_override_set(args.override_name)

    print(f"name={override_set.name} updated={override_set.updated_at_utc} source_cluster={override_set.source_cluster_report}")
    print("candidate overrides:")
    for item in override_set.candidate_overrides:
        print(
            f"  {item.candidate_id} include={item.force_include} exclude={item.exclude} "
            f"pin={item.pin} max_cap={item.max_cap_fraction} note={item.note}"
        )
    print("cluster overrides:")
    for item in override_set.cluster_overrides:
        print(f"  {item.cluster_id} max_cap={item.max_cap_fraction} note={item.note}")
    print(f"audit entries={len(override_set.audit_entries)}")


if __name__ == "__main__":
    main()
