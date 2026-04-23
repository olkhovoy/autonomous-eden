from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np

from .allocator import (
    AllocatorConfig,
    PortfolioResamplingConfig,
    allocator_objective_score,
    bootstrap_portfolio_step_matrix,
    candidate_cap_overrides,
    downsample_equity_curve,
    portfolio_equity_from_step_returns,
    propose_allocation_weights,
    resolved_cluster_caps,
    resolve_common_window_step_returns,
)
from .clustering import cluster_id_by_candidate
from .overrides import excluded_candidate_ids, forced_candidate_ids, pinned_candidate_ids
from .schema import (
    CandidateRecord,
    ClusterReport,
    CombinationScenario,
    CombinationSearchReport,
    DiversificationReport,
    OverrideSet,
    ShortlistCandidateScore,
    ShortlistReport,
)


@dataclass(slots=True, frozen=True)
class CombinationSearchConfig:
    max_pool_size: int = 6
    min_subset_size: int = 1
    max_subset_size: int = 3
    include_selected: bool = True
    include_exception_flags: bool = True
    max_stored_scenarios: int = 64
    allocator_config: AllocatorConfig = field(default_factory=AllocatorConfig)

    def validate(self) -> None:
        if self.max_pool_size <= 0:
            raise ValueError("max_pool_size must be > 0")
        if self.min_subset_size <= 0:
            raise ValueError("min_subset_size must be > 0")
        if self.max_subset_size < self.min_subset_size:
            raise ValueError("max_subset_size must be >= min_subset_size")
        if self.max_stored_scenarios <= 0:
            raise ValueError("max_stored_scenarios must be > 0")
        self.allocator_config.validate()

    def to_dict(self) -> dict[str, object]:
        return {
            "max_pool_size": self.max_pool_size,
            "min_subset_size": self.min_subset_size,
            "max_subset_size": self.max_subset_size,
            "include_selected": self.include_selected,
            "include_exception_flags": self.include_exception_flags,
            "max_stored_scenarios": self.max_stored_scenarios,
            "allocator_config": self.allocator_config.to_dict(),
        }


def build_combination_search_report(
    *,
    name: str,
    created_at_utc: str,
    candidates: Iterable[CandidateRecord],
    shortlist_report: ShortlistReport,
    diversification_report: DiversificationReport,
    cluster_report: ClusterReport | None = None,
    override_set: OverrideSet | None = None,
    config: CombinationSearchConfig,
    notes: str | None = None,
    step_return_views: Mapping[str, Sequence[float]] | None = None,
) -> CombinationSearchReport:
    config.validate()
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    pool_candidate_ids = shortlist_pool_candidate_ids(
        shortlist_report,
        max_pool_size=config.max_pool_size,
        include_selected=config.include_selected,
        include_exception_flags=config.include_exception_flags,
        override_set=override_set,
    )
    if not pool_candidate_ids:
        raise ValueError("No candidates available for combination search")
    missing = [candidate_id for candidate_id in pool_candidate_ids if candidate_id not in candidate_map]
    if missing:
        raise ValueError(f"Missing candidate records for combination search: {missing}")

    score_lookup = shortlist_score_lookup(shortlist_report, candidate_ids=pool_candidate_ids)
    cluster_by_candidate = cluster_id_by_candidate(cluster_report) if cluster_report is not None else {}
    candidate_caps = candidate_cap_overrides(override_set)
    cluster_caps = resolved_cluster_caps(override_set, default_cluster_cap_fraction=config.allocator_config.default_cluster_cap_fraction)
    pinned_ids = set(pinned_candidate_ids(override_set))
    step_returns_by_candidate = resolve_common_window_step_returns(
        candidate_map=candidate_map,
        selected_ids=pool_candidate_ids,
        diversification_report=diversification_report,
        step_return_views=step_return_views,
    )

    all_scenarios: list[CombinationScenario] = []
    evaluated_combination_count = 0
    evaluated_scenario_count = 0
    max_subset_size = min(config.max_subset_size, len(pool_candidate_ids))
    searched_subset_sizes = list(range(config.min_subset_size, max_subset_size + 1))
    initial_balance = _initial_balance(candidate_map[pool_candidate_ids[0]])

    for subset_size in searched_subset_sizes:
        for subset_candidate_ids in combinations(pool_candidate_ids, subset_size):
            if pinned_ids and not pinned_ids.issubset(set(subset_candidate_ids)):
                continue
            evaluated_combination_count += 1
            subset_scores = {candidate_id: score_lookup[candidate_id] for candidate_id in subset_candidate_ids}
            for risk_fraction in config.allocator_config.risk_fractions:
                evaluated_scenario_count += 1
                weights, allocated_risk_fraction = propose_allocation_weights(
                    shortlisted_scores=subset_scores,
                    requested_risk_fraction=float(risk_fraction),
                    per_system_cap_fraction=config.allocator_config.per_system_cap_fraction,
                    score_mode=config.allocator_config.score_mode,
                    min_score_floor=config.allocator_config.min_score_floor,
                    candidate_cap_overrides=candidate_caps,
                    cluster_by_candidate=cluster_by_candidate,
                    cluster_cap_overrides=cluster_caps,
                )
                step_return_matrix = np.column_stack(
                    [step_returns_by_candidate[item.candidate_id] for item in weights]
                )
                capital_fractions = np.array([item.capital_fraction for item in weights], dtype=np.float64)
                resampling = bootstrap_portfolio_step_matrix(
                    step_return_matrix,
                    capital_fractions=capital_fractions,
                    config=PortfolioResamplingConfig(
                        name=f"{'+'.join(subset_candidate_ids)}_risk_{risk_fraction:.2f}",
                        iterations=config.allocator_config.resampling_iterations,
                        block_size=config.allocator_config.resampling_block_size,
                        seed=config.allocator_config.resampling_seed,
                        initial_balance=initial_balance,
                    ),
                    requested_risk_fraction=float(risk_fraction),
                    allocated_risk_fraction=allocated_risk_fraction,
                )
                objective_score = allocator_objective_score(
                    resampling,
                    objective_max_drawdown_pct=config.allocator_config.objective_max_drawdown_pct,
                )
                equity = portfolio_equity_from_step_returns(
                    step_return_matrix,
                    capital_fractions=capital_fractions,
                    initial_balance=initial_balance,
                )
                sample_indices, normalized_curve = downsample_equity_curve(
                    equity,
                    initial_balance=initial_balance,
                    max_points=config.allocator_config.curve_points,
                )
                all_scenarios.append(
                    CombinationScenario(
                        name=f"subset_{subset_size}_{'+'.join(subset_candidate_ids)}_risk_{risk_fraction:.2f}",
                        subset_candidate_ids=list(subset_candidate_ids),
                        subset_display_names=[str(subset_scores[candidate_id]["display_name"]) for candidate_id in subset_candidate_ids],
                        subset_size=subset_size,
                        objective_score=objective_score,
                        requested_risk_fraction=float(risk_fraction),
                        allocated_risk_fraction=allocated_risk_fraction,
                        reserve_fraction=max(0.0, 1.0 - allocated_risk_fraction),
                        score_mode=config.allocator_config.score_mode,
                        curve_sample_indices=sample_indices,
                        normalized_balance_history=normalized_curve,
                        weights=weights,
                        resampling=resampling,
                    )
                )

    all_scenarios.sort(
        key=lambda item: (
            item.objective_score,
            item.resampling.p05_net_profit,
            -item.resampling.p95_max_drawdown_pct,
        ),
        reverse=True,
    )
    stored_scenarios = all_scenarios[: config.max_stored_scenarios]
    best_scenario_name = stored_scenarios[0].name if stored_scenarios else None

    return CombinationSearchReport(
        schema_version="1",
        name=name,
        created_at_utc=created_at_utc,
        source_shortlist_report=shortlist_report.name,
        source_diversification_report=shortlist_report.source_diversification_report,
        source_cluster_report=None if cluster_report is None else cluster_report.name,
        source_override_set=None if override_set is None else override_set.name,
        data_path=diversification_report.data_path,
        start_utc=diversification_report.start_utc,
        end_utc=diversification_report.end_utc,
        start_step=diversification_report.start_step,
        max_steps=diversification_report.max_steps,
        pool_candidate_ids=pool_candidate_ids,
        searched_subset_sizes=searched_subset_sizes,
        evaluated_combination_count=evaluated_combination_count,
        evaluated_scenario_count=evaluated_scenario_count,
        best_scenario_name=best_scenario_name,
        objective_config=config.to_dict(),
        scenarios=stored_scenarios,
        notes=notes,
    )


def shortlist_pool_candidate_ids(
    shortlist_report: ShortlistReport,
    *,
    max_pool_size: int,
    include_selected: bool = True,
    include_exception_flags: bool = True,
    override_set: OverrideSet | None = None,
) -> list[str]:
    candidate_scores = list(shortlist_report.candidate_scores)
    ordered_by_score = sorted(candidate_scores, key=lambda item: item.base_score, reverse=True)
    selected = sorted(
        [item for item in candidate_scores if item.selected],
        key=lambda item: item.selected_rank if item.selected_rank is not None else 10**9,
    )
    exceptions = sorted(
        [item for item in candidate_scores if item.exception_flags],
        key=lambda item: item.base_score,
        reverse=True,
    )

    pool: list[str] = []
    if include_selected:
        for item in selected:
            if item.candidate_id not in pool:
                pool.append(item.candidate_id)
    if include_exception_flags:
        for item in exceptions:
            if item.candidate_id not in pool:
                pool.append(item.candidate_id)
    for item in ordered_by_score:
        if item.candidate_id not in pool:
            pool.append(item.candidate_id)

    excluded = set(excluded_candidate_ids(override_set))
    forced = [candidate_id for candidate_id in forced_candidate_ids(override_set) if candidate_id in shortlist_report.candidate_ids]
    pool = [candidate_id for candidate_id in pool if candidate_id not in excluded]
    forced_prefix: list[str] = []
    for candidate_id in forced:
        if candidate_id in excluded or candidate_id in forced_prefix:
            continue
        forced_prefix.append(candidate_id)
    merged: list[str] = []
    for candidate_id in [*forced_prefix, *pool]:
        if candidate_id not in merged:
            merged.append(candidate_id)

    effective_max_pool_size = max(max_pool_size, len(selected) if include_selected else 0, len(forced_prefix))
    return merged[:effective_max_pool_size]


def shortlist_score_lookup(
    shortlist_report: ShortlistReport,
    *,
    candidate_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, object]]:
    allowed = None if candidate_ids is None else set(candidate_ids)
    result: dict[str, dict[str, object]] = {}
    for item in shortlist_report.candidate_scores:
        if allowed is not None and item.candidate_id not in allowed:
            continue
        result[item.candidate_id] = shortlist_score_payload(item)
    return result


def shortlist_score_payload(item: ShortlistCandidateScore) -> dict[str, object]:
    return {
        "display_name": item.display_name,
        "base_score": item.base_score,
        "marginal_score": item.marginal_score,
        "selected": item.selected,
        "selected_rank": item.selected_rank,
        "exception_flags": list(item.exception_flags),
    }


def _initial_balance(candidate: CandidateRecord) -> float:
    if candidate.manifest is None:
        return 10000.0
    return float(candidate.manifest.econ_config.get("initial_balance", 10000.0))
