#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidates import CandidateRegistry, parse_rule


def _format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List candidates from the candidate registry.")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--status")
    parser.add_argument("--tag", action="append")
    parser.add_argument("--rule", action="append", help="Rule in field:op:value format.")
    parser.add_argument("--any-rule", action="store_true", help="Match any rule instead of all rules.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    registry = CandidateRegistry(ROOT / args.registry_root)
    candidates = registry.list_candidates(status=args.status, tags=args.tag)
    if args.rule:
        rules = [parse_rule(item) for item in args.rule]
        matches = registry.filter_candidates(
            rules,
            require_all=not args.any_rule,
            status=args.status,
            tags=args.tag,
        )
        candidates = [candidate for candidate, _ in matches]

    print(
        "candidate_id    status     train_pnl  oos_pnl   oos_pos  trades  rsmp  name                     checkpoint"
    )
    print(
        "--------------  ---------  ---------  --------  -------  ------  ----  -----------------------  ----------"
    )
    for candidate in candidates:
        train = candidate.periods.get("train")
        oos = candidate.periods.get("oos_adjacent") or candidate.periods.get("oos")
        checkpoint_name = Path(candidate.manifest.checkpoint_path).name if candidate.manifest else "-"
        print(
            f"{candidate.candidate_id:<14}  "
            f"{candidate.status:<9}  "
            f"{_format_float(None if train is None else train.pnl):>9}  "
            f"{_format_float(None if oos is None else oos.pnl):>8}  "
            f"{str(candidate.selection_flags.get('oos_positive', False)):<7}  "
            f"{str(bool(candidate.trade_records_path)):<6}  "
            f"{str(bool(candidate.resampling_results)):<4}  "
            f"{candidate.display_name:<23}  "
            f"{checkpoint_name}"
        )


if __name__ == "__main__":
    main()
