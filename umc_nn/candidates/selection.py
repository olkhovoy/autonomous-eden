from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable

from .schema import CandidateRecord


@dataclass(slots=True)
class Rule:
    field: str
    op: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "op": self.op, "value": self.value}


@dataclass(slots=True)
class RuleEvaluation:
    rule: Rule
    matched: bool
    actual_value: Any


def parse_rule(text: str) -> Rule:
    parts = text.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Rule must look like field:op:value, got: {text}")
    field, op, raw_value = parts
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return Rule(field=field, op=op, value=value)


def resolve_field(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    parts = dotted_path.split(".")
    index = 0
    while index < len(parts):
        part = parts[index]
        if isinstance(current, dict) and part in current:
            current = current[part]
            index += 1
            continue
        if isinstance(current, dict):
            matched = False
            for end_index in range(len(parts), index, -1):
                joined = ".".join(parts[index:end_index])
                if joined in current:
                    current = current[joined]
                    index = end_index
                    matched = True
                    break
            if matched:
                continue
        raise KeyError(dotted_path)
    return current


def _contains(container: Any, member: Any) -> bool:
    if isinstance(container, str):
        return str(member) in container
    if isinstance(container, dict):
        return str(member) in container
    if isinstance(container, (list, tuple, set)):
        return member in container
    return False


def _evaluate(actual: Any, op: str, expected: Any) -> bool:
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "gt":
        return actual > expected
    if op == "gte":
        return actual >= expected
    if op == "lt":
        return actual < expected
    if op == "lte":
        return actual <= expected
    if op == "contains":
        return _contains(actual, expected)
    if op == "not_contains":
        return not _contains(actual, expected)
    if op == "in":
        return actual in expected
    if op == "not_in":
        return actual not in expected
    raise ValueError(f"Unsupported rule op: {op}")


def evaluate_rule(candidate: CandidateRecord, rule: Rule) -> RuleEvaluation:
    payload = candidate.path_value_payload()
    actual = resolve_field(payload, rule.field)
    matched = _evaluate(actual, rule.op, rule.value)
    return RuleEvaluation(rule=rule, matched=matched, actual_value=actual)


def evaluate_rules(
    candidate: CandidateRecord,
    rules: Iterable[Rule],
    *,
    require_all: bool = True,
) -> tuple[bool, list[RuleEvaluation]]:
    evaluations = [evaluate_rule(candidate, rule) for rule in rules]
    if not evaluations:
        return True, evaluations
    if require_all:
        return all(item.matched for item in evaluations), evaluations
    return any(item.matched for item in evaluations), evaluations
