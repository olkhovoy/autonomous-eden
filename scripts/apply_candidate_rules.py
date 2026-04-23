#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from umc_nn.candidates import CandidateRegistry, Rule, RuleSetRecord, parse_rule, utc_now_text


def _load_rules(args: argparse.Namespace) -> tuple[list[Rule], bool]:
    if args.rules_path:
        payload = json.loads(Path(args.rules_path).read_text())
        rule_set = RuleSetRecord.from_dict(payload)
        return [Rule(**rule) for rule in rule_set.rules], rule_set.require_all
    return [parse_rule(text) for text in args.rule or []], not args.any_rule


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter candidates by declarative rules and optionally update status.")
    parser.add_argument("--registry-root", default="candidate_registry")
    parser.add_argument("--rule", action="append", help="Rule in field:op:value format.")
    parser.add_argument("--rules-path", help="Path to JSON rule-set file.")
    parser.add_argument("--any-rule", action="store_true")
    parser.add_argument("--status")
    parser.add_argument("--tag", action="append")
    parser.add_argument("--set-status", choices=["research", "approved", "paper", "active", "draining", "retired", "rejected"])
    parser.add_argument("--add-tag", action="append")
    parser.add_argument("--save-rule-set", help="Optional name to save inline rules into the registry.")
    parser.add_argument("--description", default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    registry = CandidateRegistry(ROOT / args.registry_root)
    rules, require_all = _load_rules(args)

    if args.save_rule_set:
        rule_set = RuleSetRecord(
            schema_version="1",
            name=args.save_rule_set,
            created_at_utc=utc_now_text(),
            description=args.description,
            require_all=require_all,
            rules=[rule.to_dict() for rule in rules],
        )
        path = registry.save_rule_set(rule_set)
        print(f"Saved rule set to {path}")

    matches = registry.filter_candidates(
        rules,
        require_all=require_all,
        status=args.status,
        tags=args.tag,
    )

    for candidate, evaluations in matches:
        print(candidate.candidate_id, candidate.display_name, candidate.status)
        for item in evaluations:
            print(f"  {item.rule.field} {item.rule.op} {item.rule.value!r} -> {item.actual_value!r}")
        if args.set_status:
            registry.update_status(
                candidate.candidate_id,
                args.set_status,
                note=f"Rule-set selection applied via apply_candidate_rules.py",
                add_tags=args.add_tag,
            )
            print(f"  status -> {args.set_status}")

    print(f"Matched {len(matches)} candidates")


if __name__ == "__main__":
    main()
