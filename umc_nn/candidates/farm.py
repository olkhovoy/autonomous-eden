from __future__ import annotations

import copy
from itertools import product
import re
from typing import Any

from .schema import (
    CandidateFarmReport,
    CandidateFarmScenarioReport,
    LifecycleReport,
    PortfolioBaselineReport,
    PortfolioLedgerReport,
    RollingConveyorReport,
    utc_now_text,
)


def _sorted_unique(items: list[str]) -> list[str]:
    return sorted(set(items))


_TEMPLATE_EXACT_RE = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = {str(key): copy.deepcopy(value) for key, value in base.items()}
        for key, value in override.items():
            if key in merged:
                merged[str(key)] = _deep_merge(merged[str(key)], value)
            else:
                merged[str(key)] = copy.deepcopy(value)
        return merged
    if isinstance(base, list) and isinstance(override, list):
        return copy.deepcopy(base) + copy.deepcopy(override)
    return copy.deepcopy(override)


def _render_template_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {str(key): _render_template_value(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_render_template_value(item, context) for item in value]
    if isinstance(value, str):
        exact_match = _TEMPLATE_EXACT_RE.match(value)
        if exact_match:
            key = exact_match.group(1)
            if key not in context:
                raise KeyError(f"Missing manifest template key: {key}")
            return copy.deepcopy(context[key])
        try:
            return value.format(**context)
        except KeyError as exc:
            raise KeyError(f"Missing manifest template key: {exc.args[0]}") from exc
    return copy.deepcopy(value)


def _matrix_contexts(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    if not matrix:
        return [{}]
    keys = list(matrix.keys())
    value_lists = [value if isinstance(value, list) else [value] for value in matrix.values()]
    return [
        {str(key): copy.deepcopy(item) for key, item in zip(keys, combo)}
        for combo in product(*value_lists)
    ]


def expand_farm_manifest_scenarios(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = dict(manifest.get("defaults", {}))
    global_context = dict(manifest.get("context", {}))
    expanded: list[dict[str, Any]] = []

    for scenario in manifest.get("scenarios", []):
        merged = _deep_merge(defaults, dict(scenario))
        rendered = _render_template_value(merged, global_context)
        if rendered.get("enabled", True) is False:
            continue
        expanded.append(rendered)

    for template in manifest.get("scenario_templates", []):
        matrix = dict(template.get("matrix", {}))
        cases = [dict(item) for item in template.get("cases", [])] or [{}]
        template_body = {
            str(key): value
            for key, value in dict(template).items()
            if key not in {"matrix", "cases"}
        }
        for case_context in cases:
            for matrix_context in _matrix_contexts(matrix):
                context = dict(global_context)
                context.update(case_context)
                context.update(matrix_context)
                merged = _deep_merge(defaults, template_body)
                rendered = _render_template_value(merged, context)
                if rendered.get("enabled", True) is False:
                    continue
                expanded.append(rendered)

    return expanded


def _best_scenario_name(
    scenarios: list[CandidateFarmScenarioReport],
    *,
    require_gate_pass: bool = False,
    by: str = "pnl",
) -> str | None:
    filtered = [
        item
        for item in scenarios
        if item.status == "completed"
        and (not require_gate_pass or item.gate_pass is True)
        and (
            (by == "pnl" and item.total_pnl is not None)
            or (by == "drawdown" and item.max_drawdown_pct is not None)
        )
    ]
    if not filtered:
        return None
    if by == "drawdown":
        return min(filtered, key=lambda item: float(item.max_drawdown_pct or 0.0)).scenario_name
    return max(filtered, key=lambda item: float(item.total_pnl or 0.0)).scenario_name


def build_candidate_farm_scenario_report(
    *,
    scenario_name: str,
    status: str,
    mode: str,
    output_dir: str,
    config: dict[str, Any],
    log_paths: dict[str, str],
    updated_at_utc: str | None = None,
    progress_stage: str | None = None,
    rolling_report: RollingConveyorReport | None = None,
    lifecycle_report: LifecycleReport | None = None,
    portfolio_ledger_report: PortfolioLedgerReport | None = None,
    portfolio_baselines_report: PortfolioBaselineReport | None = None,
    rolling_report_name: str | None = None,
    lifecycle_report_name: str | None = None,
    portfolio_ledger_report_name: str | None = None,
    portfolio_baselines_report_name: str | None = None,
    error_message: str | None = None,
    notes: str | None = None,
) -> CandidateFarmScenarioReport:
    candidate_pool_ids: list[str] = []
    selected_candidate_ids: list[str] = []
    selector: dict[str, Any] = {}
    total_pnl: float | None = None
    total_return_pct: float | None = None
    max_drawdown_pct: float | None = None
    evaluated_cycle_count: int | None = None
    positive_cycle_count: int | None = None

    if rolling_report is not None:
        candidate_pool_ids = _sorted_unique(
            [
                candidate_id
                for cycle in rolling_report.cycle_outcomes
                for candidate_id in cycle.candidate_ids
            ]
        )
        selected_candidate_ids = _sorted_unique(
            [
                candidate_id
                for cycle in rolling_report.cycle_outcomes
                for candidate_id in cycle.selected_candidate_ids
            ]
        )
        selector = dict(rolling_report.selector)
        evaluated_cycle_count = int(rolling_report.evaluated_cycle_count)
        positive_cycle_count = int(rolling_report.positive_cycle_count)

    if portfolio_ledger_report is not None:
        total_pnl = float(portfolio_ledger_report.total_pnl)
        total_return_pct = float(portfolio_ledger_report.total_return_pct)
        max_drawdown_pct = float(portfolio_ledger_report.max_drawdown_pct)

    if portfolio_baselines_report is not None:
        total_pnl = float(portfolio_baselines_report.conveyor_total_pnl)
        total_return_pct = float(portfolio_baselines_report.conveyor_total_return_pct)
        max_drawdown_pct = float(portfolio_baselines_report.conveyor_max_drawdown_pct)

    return CandidateFarmScenarioReport(
        scenario_name=scenario_name,
        status=status,
        mode=mode,
        output_dir=output_dir,
        config=dict(config),
        log_paths={str(key): str(value) for key, value in log_paths.items()},
        updated_at_utc=updated_at_utc or utc_now_text(),
        progress_stage=progress_stage,
        rolling_report_name=rolling_report.name if rolling_report is not None else rolling_report_name,
        lifecycle_report_name=lifecycle_report.name if lifecycle_report is not None else lifecycle_report_name,
        portfolio_ledger_report_name=(
            portfolio_ledger_report.name if portfolio_ledger_report is not None else portfolio_ledger_report_name
        ),
        portfolio_baselines_report_name=(
            portfolio_baselines_report.name if portfolio_baselines_report is not None else portfolio_baselines_report_name
        ),
        selector=selector,
        candidate_pool_ids=candidate_pool_ids,
        selected_candidate_ids=selected_candidate_ids,
        final_status_counts={} if lifecycle_report is None else dict(lifecycle_report.final_status_counts),
        gate_pass=None if portfolio_baselines_report is None else bool(portfolio_baselines_report.gate.overall_pass),
        beaten_baselines=[] if portfolio_baselines_report is None else list(portfolio_baselines_report.gate.beaten_baselines),
        failed_required_baselines=[] if portfolio_baselines_report is None else list(portfolio_baselines_report.gate.failed_required_baselines),
        total_pnl=total_pnl,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        evaluated_cycle_count=evaluated_cycle_count,
        positive_cycle_count=positive_cycle_count,
        error_message=error_message,
        notes=notes,
    )


def build_candidate_farm_report(
    *,
    name: str,
    registry_root: str,
    scenarios: list[CandidateFarmScenarioReport],
    source_manifest_path: str | None = None,
    notes: str | None = None,
) -> CandidateFarmReport:
    completed = [item for item in scenarios if item.status in {"completed", "reused"}]
    gate_passed = [item for item in completed if item.gate_pass is True]
    candidate_pool_ids = _sorted_unique([candidate_id for item in completed for candidate_id in item.candidate_pool_ids])
    selected_candidate_ids = _sorted_unique(
        [candidate_id for item in completed for candidate_id in item.selected_candidate_ids]
    )
    ranked = sorted(
        completed,
        key=lambda item: (
            item.gate_pass is True,
            float(item.total_pnl or float("-inf")),
            -float(item.max_drawdown_pct or float("inf")),
        ),
        reverse=True,
    )
    summary = {
        "scenario_count": len(scenarios),
        "completed_scenarios": len(completed),
        "completed_or_reused_scenarios": len(completed),
        "failed_scenarios": sum(item.status == "failed" for item in scenarios),
        "planned_scenarios": sum(item.status == "planned" for item in scenarios),
        "running_scenarios": sum(item.status == "running" for item in scenarios),
        "gate_pass_count": len(gate_passed),
        "gate_pass_rate": 0.0 if not completed else len(gate_passed) / float(len(completed)),
        "total_unique_candidate_pool_ids": len(candidate_pool_ids),
        "total_unique_selected_candidate_ids": len(selected_candidate_ids),
        "best_scenario_by_pnl": _best_scenario_name(completed, by="pnl"),
        "best_gate_scenario_by_pnl": _best_scenario_name(completed, require_gate_pass=True, by="pnl"),
        "lowest_drawdown_scenario": _best_scenario_name(completed, by="drawdown"),
        "ranked_scenarios": [item.scenario_name for item in ranked],
    }
    return CandidateFarmReport(
        schema_version="1",
        name=name,
        created_at_utc=utc_now_text(),
        source_manifest_path=source_manifest_path,
        registry_root=registry_root,
        scenarios=scenarios,
        summary=summary,
        notes=notes,
    )
