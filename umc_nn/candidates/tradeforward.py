from __future__ import annotations

from typing import Iterable

from umc_nn.trading_eval import resolve_date_window

from .registry import CandidateRegistry
from .schema import (
    AllocatorScenario,
    AllocatorWorkbenchReport,
    CombinationScenario,
    CombinationSearchReport,
    TradeforwardAllocation,
    TradeforwardPlan,
    utc_now_text,
)


def _find_allocator_scenario(
    report: AllocatorWorkbenchReport,
    scenario_name: str | None,
) -> AllocatorScenario:
    target_name = scenario_name or report.chosen_scenario_name
    if target_name is None:
        raise ValueError("Allocator report has no chosen_scenario_name and no scenario_name override was provided")
    for scenario in report.scenarios:
        if scenario.name == target_name:
            return scenario
    raise KeyError(f"Allocator scenario not found: {target_name}")


def _find_combination_scenario(
    report: CombinationSearchReport,
    scenario_name: str | None,
) -> CombinationScenario:
    target_name = scenario_name or report.best_scenario_name
    if target_name is None:
        raise ValueError("Combination report has no best_scenario_name and no scenario_name override was provided")
    for scenario in report.scenarios:
        if scenario.name == target_name:
            return scenario
    raise KeyError(f"Combination scenario not found: {target_name}")


def _allocations_for_weights(
    registry: CandidateRegistry,
    weights: Iterable,
) -> tuple[list[str], list[TradeforwardAllocation]]:
    candidate_ids: list[str] = []
    allocations: list[TradeforwardAllocation] = []
    for weight in weights:
        candidate = registry.load_candidate(weight.candidate_id)
        manifest = candidate.manifest
        if manifest is None:
            raise ValueError(f"Candidate {candidate.candidate_id} has no manifest")
        candidate_ids.append(candidate.candidate_id)
        allocations.append(
            TradeforwardAllocation(
                candidate_id=candidate.candidate_id,
                display_name=candidate.display_name,
                checkpoint_path=manifest.checkpoint_path,
                engine_name=candidate.engine_name,
                representation_name=manifest.representation_name,
                status=candidate.status,
                cluster_id=weight.cluster_id,
                normalized_share=weight.normalized_share,
                capital_fraction=weight.capital_fraction,
                capped=weight.capped,
                cap_reason=weight.cap_reason,
            )
        )
    return candidate_ids, allocations


def build_tradeforward_plan(
    registry: CandidateRegistry,
    name: str,
    *,
    forward_start_utc: str,
    forward_end_utc: str,
    allocator_report: AllocatorWorkbenchReport | None = None,
    combination_report: CombinationSearchReport | None = None,
    scenario_name: str | None = None,
    source_cycle_report: str | None = None,
    notes: str | None = None,
) -> TradeforwardPlan:
    if combination_report is None and allocator_report is None:
        raise ValueError("Provide either combination_report or allocator_report")

    if combination_report is not None:
        scenario = _find_combination_scenario(combination_report, scenario_name)
        candidate_ids, allocations = _allocations_for_weights(registry, scenario.weights)
        data_path = combination_report.data_path
        forward_start_step, forward_max_steps = resolve_date_window(data_path, forward_start_utc, forward_end_utc)
        return TradeforwardPlan(
            schema_version="1",
            name=name,
            created_at_utc=utc_now_text(),
            selection_mode="combination",
            scenario_name=scenario.name,
            data_path=data_path,
            forward_start_utc=forward_start_utc,
            forward_end_utc=forward_end_utc,
            forward_start_step=forward_start_step,
            forward_max_steps=forward_max_steps,
            requested_risk_fraction=scenario.requested_risk_fraction,
            allocated_risk_fraction=scenario.allocated_risk_fraction,
            reserve_fraction=scenario.reserve_fraction,
            candidate_ids=candidate_ids,
            allocations=allocations,
            source_cycle_report=source_cycle_report,
            source_allocator_report=None,
            source_combination_report=combination_report.name,
            source_cluster_report=combination_report.source_cluster_report,
            notes=notes,
        )

    assert allocator_report is not None
    scenario = _find_allocator_scenario(allocator_report, scenario_name)
    candidate_ids, allocations = _allocations_for_weights(registry, scenario.weights)
    data_path = allocator_report.data_path
    forward_start_step, forward_max_steps = resolve_date_window(data_path, forward_start_utc, forward_end_utc)
    return TradeforwardPlan(
        schema_version="1",
        name=name,
        created_at_utc=utc_now_text(),
        selection_mode="allocator",
        scenario_name=scenario.name,
        data_path=data_path,
        forward_start_utc=forward_start_utc,
        forward_end_utc=forward_end_utc,
        forward_start_step=forward_start_step,
        forward_max_steps=forward_max_steps,
        requested_risk_fraction=scenario.requested_risk_fraction,
        allocated_risk_fraction=scenario.allocated_risk_fraction,
        reserve_fraction=scenario.reserve_fraction,
        candidate_ids=candidate_ids,
        allocations=allocations,
        source_cycle_report=source_cycle_report,
        source_allocator_report=allocator_report.name,
        source_combination_report=None,
        source_cluster_report=allocator_report.source_cluster_report,
        notes=notes,
    )
