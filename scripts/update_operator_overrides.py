#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidates import CandidateRegistry, empty_override_set, update_candidate_override, update_cluster_override


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or update operator overrides with audit trail.")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--override-name", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--description")
    parser.add_argument("--source-cluster-report")
    parser.add_argument("--candidate-id")
    parser.add_argument("--cluster-id")
    parser.add_argument("--force-include", choices=["true", "false"])
    parser.add_argument("--exclude", choices=["true", "false"])
    parser.add_argument("--pin", choices=["true", "false"])
    parser.add_argument("--max-cap", type=float)
    parser.add_argument("--note")
    return parser


def _parse_optional_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    return raw == "true"


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    registry = CandidateRegistry(ROOT / args.registry_root)
    path = registry.override_path(args.override_name)
    if path.exists():
        override_set = registry.load_override_set(args.override_name)
    else:
        override_set = empty_override_set(
            name=args.override_name,
            description=args.description,
            source_cluster_report=args.source_cluster_report,
        )

    changed = False
    if args.candidate_id:
        override_set = update_candidate_override(
            override_set,
            candidate_id=args.candidate_id,
            actor=args.actor,
            force_include=_parse_optional_bool(args.force_include),
            exclude=_parse_optional_bool(args.exclude),
            pin=_parse_optional_bool(args.pin),
            max_cap_fraction=args.max_cap,
            note=args.note,
        )
        changed = True
    if args.cluster_id:
        override_set = update_cluster_override(
            override_set,
            cluster_id=args.cluster_id,
            actor=args.actor,
            max_cap_fraction=args.max_cap,
            note=args.note,
        )
        changed = True
    if not changed:
        raise ValueError("Provide either --candidate-id or --cluster-id")

    saved = registry.save_override_set(override_set)
    print(f"Saved overrides to {saved}")
    print(f"Candidate overrides: {len(override_set.candidate_overrides)}")
    print(f"Cluster overrides: {len(override_set.cluster_overrides)}")
    print(f"Audit entries: {len(override_set.audit_entries)}")


if __name__ == "__main__":
    main()
