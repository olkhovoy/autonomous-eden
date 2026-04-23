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
    parser = argparse.ArgumentParser(description="Update candidate status in the registry.")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--status",
        required=True,
        choices=["research", "approved", "paper", "active", "draining", "retired", "rejected"],
    )
    parser.add_argument("--note")
    parser.add_argument("--add-tag", action="append")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    registry = CandidateRegistry(ROOT / args.registry_root)
    candidate = registry.update_status(
        args.candidate_id,
        args.status,
        note=args.note,
        add_tags=args.add_tag,
    )
    print(candidate.candidate_id, candidate.status)


if __name__ == "__main__":
    main()
