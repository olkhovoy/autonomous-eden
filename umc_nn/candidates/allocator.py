from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from umc_nn.trading_eval import EpisodeTrace, evaluate_policy_trace_path

from .clustering import cluster_id_by_candidate
from .engine_config import candidate_engine_config
from .overrides import candidate_override_map, cluster_override_map, excluded_candidate_ids, forced_candidate_ids
from .schema import (
    AllocationWeight,
    AllocatorScenario,
    AllocatorWorkbenchReport,
    CandidateRecord,
    ClusterReport,
    DiversificationReport,
    OverrideSet,
    PortfolioResamplingStats,
    ShortlistReport,
)


@dataclass(slots=True, frozen=True)
class PortfolioResamplingConfig:
    name: str
    iterations: int = 500
    block_size: int = 64
    seed: int = 42
    initial_balance: float = 10000.0

    def validate(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be > 0")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be > 0")


@dataclass(slots=True, frozen=True)
class AllocatorConfig:
    risk_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    per_system_cap_fraction: float = 0.35
    default_cluster_cap_fraction: float | None = None
    score_mode: str = "marginal"
    min_score_floor: float = 0.05
    resampling_iterations: int = 500
    resampling_block_size: int = 64
    resampling_seed: int = 42
    objective_max_drawdown_pct: float = 15.0
    curve_points: int = 256

    def validate(self) -> None:
        if not self.risk_fractions:
            raise ValueError("risk_fractions must not be empty")
        if any(fraction < 0.0 or fraction > 1.0 for fraction in self.risk_fractions):
            raise ValueError("risk_fractions must be in [0, 1]")
        if self.per_system_cap_fraction <= 0.0 or self.per_system_cap_fraction > 1.0:
            raise ValueError("per_system_cap_fraction must be in (0, 1]")
        if self.default_cluster_cap_fraction is not None and not (0.0 < self.default_cluster_cap_fraction <= 1.0):
            raise ValueError("default_cluster_cap_fraction must be in (0, 1] when provided")
        if self.score_mode not in {"base", "marginal"}:
            raise ValueError("score_mode must be 'base' or 'marginal'")
        if self.min_score_floor < 0.0:
            raise ValueError("min_score_floor must be >= 0")
        if self.resampling_iterations <= 0:
            raise ValueError("resampling_iterations must be > 0")
        if self.resampling_block_size <= 0:
            raise ValueError("resampling_block_size must be > 0")
        if self.objective_max_drawdown_pct <= 0.0:
            raise ValueError("objective_max_drawdown_pct must be > 0")
        if self.curve_points <= 1:
            raise ValueError("curve_points must be > 1")

    def to_dict(self) -> dict[str, float | int | list[float] | str]:
        return {
            "risk_fractions": list(self.risk_fractions),
            "per_system_cap_fraction": self.per_system_cap_fraction,
            "default_cluster_cap_fraction": self.default_cluster_cap_fraction,
            "score_mode": self.score_mode,
            "min_score_floor": self.min_score_floor,
            "resampling_iterations": self.resampling_iterations,
            "resampling_block_size": self.resampling_block_size,
            "resampling_seed": self.resampling_seed,
            "objective_max_drawdown_pct": self.objective_max_drawdown_pct,
            "curve_points": self.curve_points,
        }


def build_allocator_workbench_report(
    *,
    name: str,
    created_at_utc: str,
    candidates: Iterable[CandidateRecord],
    shortlist_report: ShortlistReport,
    diversification_report: DiversificationReport,
    cluster_report: ClusterReport | None = None,
    override_set: OverrideSet | None = None,
    config: AllocatorConfig,
    notes: str | None = None,
    step_return_views: Mapping[str, Sequence[float]] | None = None,
) -> AllocatorWorkbenchReport:
    config.validate()
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    selected_ids = resolve_allocator_candidate_ids(shortlist_report, override_set=override_set)
    if not selected_ids:
        raise ValueError("shortlist_report has no selected candidates")
    missing = [candidate_id for candidate_id in selected_ids if candidate_id not in candidate_map]
    if missing:
        raise ValueError(f"Missing candidate records for allocator workbench: {missing}")

    step_returns_by_candidate = resolve_common_window_step_returns(
        candidate_map=candidate_map,
        selected_ids=selected_ids,
        diversification_report=diversification_report,
        step_return_views=step_return_views,
    )
    selected_scores = shortlist_score_lookup(shortlist_report, candidate_ids=selected_ids)
    candidate_caps = candidate_cap_overrides(override_set)
    cluster_caps = resolved_cluster_caps(override_set, default_cluster_cap_fraction=config.default_cluster_cap_fraction)
    cluster_by_candidate = cluster_id_by_candidate(cluster_report) if cluster_report is not None else {}

    scenarios: list[AllocatorScenario] = []
    for risk_fraction in config.risk_fractions:
        weights, allocated_risk_fraction = propose_allocation_weights(
            shortlisted_scores=selected_scores,
            requested_risk_fraction=float(risk_fraction),
            per_system_cap_fraction=config.per_system_cap_fraction,
            score_mode=config.score_mode,
            min_score_floor=config.min_score_floor,
            candidate_cap_overrides=candidate_caps,
            cluster_by_candidate=cluster_by_candidate,
            cluster_cap_overrides=cluster_caps,
        )
        capital_fractions = np.array([item.capital_fraction for item in weights], dtype=np.float64)
        step_return_matrix = np.column_stack([step_returns_by_candidate[item.candidate_id] for item in weights])

        resampling = bootstrap_portfolio_step_matrix(
            step_return_matrix,
            capital_fractions=capital_fractions,
            config=PortfolioResamplingConfig(
                name=f"portfolio_risk_{risk_fraction:.2f}",
                iterations=config.resampling_iterations,
                block_size=config.resampling_block_size,
                seed=config.resampling_seed,
                initial_balance=_initial_balance(candidate_map[selected_ids[0]]),
            ),
            requested_risk_fraction=float(risk_fraction),
            allocated_risk_fraction=allocated_risk_fraction,
        )
        objective_score = allocator_objective_score(
            resampling,
            objective_max_drawdown_pct=config.objective_max_drawdown_pct,
        )
        original_equity = portfolio_equity_from_step_returns(
            step_return_matrix,
            capital_fractions=capital_fractions,
            initial_balance=resampling.initial_balance,
        )
        curve_sample_indices, normalized_balance_history = downsample_equity_curve(
            original_equity,
            initial_balance=resampling.initial_balance,
            max_points=config.curve_points,
        )
        scenarios.append(
            AllocatorScenario(
                name=f"risk_{risk_fraction:.2f}",
                objective_score=objective_score,
                requested_risk_fraction=float(risk_fraction),
                allocated_risk_fraction=allocated_risk_fraction,
                reserve_fraction=max(0.0, 1.0 - allocated_risk_fraction),
                per_system_cap_fraction=config.per_system_cap_fraction,
                score_mode=config.score_mode,
                curve_sample_indices=curve_sample_indices,
                normalized_balance_history=normalized_balance_history,
                weights=weights,
                resampling=resampling,
            )
        )

    chosen_scenario_name = None
    if scenarios:
        chosen_scenario_name = max(scenarios, key=lambda item: item.objective_score).name

    return AllocatorWorkbenchReport(
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
        selected_candidate_ids=selected_ids,
        requested_risk_fractions=[float(item) for item in config.risk_fractions],
        chosen_scenario_name=chosen_scenario_name,
        objective_config=config.to_dict(),
        scenarios=scenarios,
        notes=notes,
    )


def propose_allocation_weights(
    *,
    shortlisted_scores: Mapping[str, dict[str, object]],
    requested_risk_fraction: float,
    per_system_cap_fraction: float,
    score_mode: str,
    min_score_floor: float,
    candidate_cap_overrides: Mapping[str, float | None] | None = None,
    cluster_by_candidate: Mapping[str, str] | None = None,
    cluster_cap_overrides: Mapping[str, float | None] | None = None,
) -> tuple[list[AllocationWeight], float]:
    if requested_risk_fraction < 0.0 or requested_risk_fraction > 1.0:
        raise ValueError("requested_risk_fraction must be in [0, 1]")
    if per_system_cap_fraction <= 0.0 or per_system_cap_fraction > 1.0:
        raise ValueError("per_system_cap_fraction must be in (0, 1]")

    candidate_ids = list(shortlisted_scores.keys())
    if not candidate_ids:
        return [], 0.0

    raw_scores = np.array(
        [_score_from_shortlist(shortlisted_scores[candidate_id], score_mode, min_score_floor) for candidate_id in candidate_ids],
        dtype=np.float64,
    )
    raw_sum = float(np.sum(raw_scores))
    if raw_sum <= 1e-12:
        normalized = np.full_like(raw_scores, 1.0 / len(raw_scores))
    else:
        normalized = raw_scores / raw_sum

    allocated = _allocate_with_cap(
        candidate_ids=candidate_ids,
        normalized_shares=normalized,
        requested_risk_fraction=requested_risk_fraction,
        per_system_cap_fraction=per_system_cap_fraction,
        candidate_cap_overrides=candidate_cap_overrides or {},
        cluster_by_candidate=cluster_by_candidate or {},
        cluster_cap_overrides=cluster_cap_overrides or {},
    )
    allocated_total = float(np.sum(allocated))
    normalized_after_cap = np.zeros_like(allocated)
    if allocated_total > 1e-12:
        normalized_after_cap = allocated / allocated_total

    cluster_allocations = cluster_allocation_totals(
        candidate_ids=candidate_ids,
        capital_fractions=allocated,
        cluster_by_candidate=cluster_by_candidate or {},
    )
    weights: list[AllocationWeight] = []
    for index, candidate_id in enumerate(candidate_ids):
        payload = shortlisted_scores[candidate_id]
        cluster_id = None if cluster_by_candidate is None else cluster_by_candidate.get(candidate_id)
        cap_reason = resolve_cap_reason(
            candidate_id=candidate_id,
            capital_fraction=float(allocated[index]),
            per_system_cap_fraction=per_system_cap_fraction,
            candidate_cap_overrides={} if candidate_cap_overrides is None else candidate_cap_overrides,
            cluster_id=cluster_id,
            cluster_allocations=cluster_allocations,
            cluster_cap_overrides={} if cluster_cap_overrides is None else cluster_cap_overrides,
        )
        weights.append(
            AllocationWeight(
                candidate_id=candidate_id,
                display_name=str(payload["display_name"]),
                raw_score=float(raw_scores[index]),
                normalized_share=float(normalized_after_cap[index]),
                capital_fraction=float(allocated[index]),
                cluster_id=cluster_id,
                capped=cap_reason is not None,
                cap_reason=cap_reason,
            )
        )
    return weights, allocated_total


def bootstrap_portfolio_step_matrix(
    step_return_matrix: np.ndarray,
    *,
    capital_fractions: np.ndarray,
    config: PortfolioResamplingConfig,
    requested_risk_fraction: float,
    allocated_risk_fraction: float,
) -> PortfolioResamplingStats:
    config.validate()
    matrix = np.asarray(step_return_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("step_return_matrix must be 2-dimensional")
    if matrix.shape[0] <= 0:
        raise ValueError("step_return_matrix must contain at least one step")

    weights = np.asarray(capital_fractions, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] != matrix.shape[1]:
        raise ValueError("capital_fractions must match matrix columns")

    original_equity = portfolio_equity_from_step_returns(
        matrix,
        capital_fractions=weights,
        initial_balance=config.initial_balance,
    )
    original_final_balance, original_net_profit, original_max_drawdown = equity_summary(
        original_equity,
        initial_balance=config.initial_balance,
    )

    rng = np.random.RandomState(config.seed)
    steps = matrix.shape[0]
    final_balances = np.zeros(config.iterations, dtype=np.float64)
    net_profits = np.zeros(config.iterations, dtype=np.float64)
    max_drawdowns = np.zeros(config.iterations, dtype=np.float64)

    for idx in range(config.iterations):
        sampled_matrix = sample_step_return_blocks(
            matrix,
            block_size=config.block_size,
            rng=rng,
        )
        sampled_equity = portfolio_equity_from_step_returns(
            sampled_matrix,
            capital_fractions=weights,
            initial_balance=config.initial_balance,
        )
        final_balance, net_profit, max_drawdown = equity_summary(
            sampled_equity,
            initial_balance=config.initial_balance,
        )
        final_balances[idx] = final_balance
        net_profits[idx] = net_profit
        max_drawdowns[idx] = max_drawdown

    return PortfolioResamplingStats(
        name=config.name,
        iterations=config.iterations,
        block_size=config.block_size,
        seed=config.seed,
        steps=steps,
        initial_balance=config.initial_balance,
        requested_risk_fraction=requested_risk_fraction,
        allocated_risk_fraction=allocated_risk_fraction,
        original_final_balance=float(original_final_balance),
        original_net_profit=float(original_net_profit),
        original_max_drawdown_pct=float(original_max_drawdown),
        mean_final_balance=float(np.mean(final_balances)),
        median_final_balance=float(np.median(final_balances)),
        p05_final_balance=float(np.percentile(final_balances, 5)),
        p25_final_balance=float(np.percentile(final_balances, 25)),
        mean_net_profit=float(np.mean(net_profits)),
        median_net_profit=float(np.median(net_profits)),
        p05_net_profit=float(np.percentile(net_profits, 5)),
        p25_net_profit=float(np.percentile(net_profits, 25)),
        mean_max_drawdown_pct=float(np.mean(max_drawdowns)),
        median_max_drawdown_pct=float(np.median(max_drawdowns)),
        p75_max_drawdown_pct=float(np.percentile(max_drawdowns, 75)),
        p95_max_drawdown_pct=float(np.percentile(max_drawdowns, 95)),
        profitable_rate=float(np.mean(net_profits > 0.0)),
        loss_rate=float(np.mean(net_profits <= 0.0)),
        ruin_rate=float(np.mean(final_balances <= 0.0)),
        pessimistic_net_profit=float(np.percentile(net_profits, 5)),
        pessimistic_max_drawdown_pct=float(np.percentile(max_drawdowns, 95)),
        replay_mode="portfolio_step_block_bootstrap",
        sampler_id="moving_block_step_sampler",
        overlap_policy="time_aligned_step_blocks",
        capital_path_distribution={
            "p05_final_balance": float(np.percentile(final_balances, 5)),
            "p50_final_balance": float(np.percentile(final_balances, 50)),
            "p95_final_balance": float(np.percentile(final_balances, 95)),
            "p05_net_profit": float(np.percentile(net_profits, 5)),
            "p95_max_drawdown_pct": float(np.percentile(max_drawdowns, 95)),
        },
    )


def portfolio_equity_from_step_returns(
    step_return_matrix: np.ndarray,
    *,
    capital_fractions: np.ndarray,
    initial_balance: float,
) -> np.ndarray:
    matrix = np.asarray(step_return_matrix, dtype=np.float64)
    weights = np.asarray(capital_fractions, dtype=np.float64)
    portfolio_step_returns = matrix @ weights
    equity = np.empty(portfolio_step_returns.size + 1, dtype=np.float64)
    equity[0] = float(initial_balance)
    for idx, step_return in enumerate(portfolio_step_returns, start=1):
        growth = 1.0 + float(step_return)
        if growth <= 0.0 or equity[idx - 1] <= 0.0:
            equity[idx] = 0.0
        else:
            equity[idx] = equity[idx - 1] * growth
    return equity


def sample_step_return_blocks(
    step_return_matrix: np.ndarray,
    *,
    block_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    matrix = np.asarray(step_return_matrix, dtype=np.float64)
    steps = matrix.shape[0]
    block = max(1, min(block_size, steps))
    chunks: list[np.ndarray] = []
    while sum(chunk.shape[0] for chunk in chunks) < steps:
        start = int(rng.randint(0, steps))
        indices = (start + np.arange(block)) % steps
        chunks.append(matrix[indices])
    return np.vstack(chunks)[:steps]


def allocator_objective_score(
    stats: PortfolioResamplingStats,
    *,
    objective_max_drawdown_pct: float,
) -> float:
    initial_balance = max(stats.initial_balance, 1e-12)
    efficiency_score = (stats.p05_net_profit / initial_balance) / max(stats.allocated_risk_fraction, 1e-6)
    median_growth_score = stats.median_net_profit / initial_balance
    drawdown_penalty = max(0.0, stats.p95_max_drawdown_pct - objective_max_drawdown_pct) / objective_max_drawdown_pct
    return (
        1.5 * efficiency_score
        + 1.0 * median_growth_score
        - 1.5 * drawdown_penalty
        - 3.0 * stats.ruin_rate
        - 0.5 * stats.loss_rate
    )


def equity_summary(
    equity: np.ndarray,
    *,
    initial_balance: float,
) -> tuple[float, float, float]:
    balance = np.asarray(equity, dtype=np.float64)
    running_max = np.maximum.accumulate(balance)
    drawdowns = np.divide(
        running_max - balance,
        running_max,
        out=np.zeros_like(balance),
        where=running_max > 0,
    )
    final_balance = float(balance[-1])
    net_profit = final_balance - float(initial_balance)
    max_drawdown_pct = float(np.max(drawdowns) * 100.0)
    return final_balance, net_profit, max_drawdown_pct


def downsample_equity_curve(
    equity: np.ndarray,
    *,
    initial_balance: float,
    max_points: int,
) -> tuple[list[int], list[float]]:
    balance = np.asarray(equity, dtype=np.float64)
    if balance.size <= max_points:
        indices = np.arange(balance.size, dtype=np.int64)
    else:
        indices = np.unique(np.linspace(0, balance.size - 1, num=max_points, dtype=np.int64))
    normalized = balance / max(initial_balance, 1e-12)
    return [int(idx) for idx in indices], [float(normalized[idx]) for idx in indices]


def step_returns_from_trace(trace: EpisodeTrace, *, target_steps: int) -> np.ndarray:
    balance = np.asarray(trace.balance_history, dtype=np.float64)
    if balance.size == 0:
        raise ValueError("trace balance_history cannot be empty")
    if balance.size < target_steps + 1:
        balance = np.pad(balance, (0, target_steps + 1 - balance.size), constant_values=balance[-1])
    else:
        balance = balance[: target_steps + 1]
    prev = np.maximum(balance[:-1], 1e-12)
    return (balance[1:] - balance[:-1]) / prev


def resolve_common_window_step_returns(
    *,
    candidate_map: Mapping[str, CandidateRecord],
    selected_ids: Sequence[str],
    diversification_report: DiversificationReport,
    step_return_views: Mapping[str, Sequence[float]] | None,
) -> dict[str, np.ndarray]:
    if step_return_views is not None:
        resolved = {candidate_id: np.asarray(step_return_views[candidate_id], dtype=np.float64) for candidate_id in selected_ids}
        target_steps = diversification_report.max_steps
        for candidate_id, values in resolved.items():
            if values.size < target_steps:
                resolved[candidate_id] = np.pad(values, (0, target_steps - values.size), constant_values=0.0)
            elif values.size > target_steps:
                resolved[candidate_id] = values[:target_steps]
        return resolved

    resolved: dict[str, np.ndarray] = {}
    for candidate_id in selected_ids:
        candidate = candidate_map[candidate_id]
        trace = _trace_for_common_window(candidate, diversification_report)
        resolved[candidate_id] = step_returns_from_trace(trace, target_steps=diversification_report.max_steps)
    return resolved


def _trace_for_common_window(candidate: CandidateRecord, diversification_report: DiversificationReport) -> EpisodeTrace:
    manifest = candidate.manifest
    if manifest is None:
        raise ValueError(f"Candidate {candidate.candidate_id} is missing manifest")
    econ = manifest.econ_config
    return evaluate_policy_trace_path(
        "monolith",
        manifest.data_path,
        use_neurobars=str(manifest.data_path).endswith(".npz"),
        start_step=diversification_report.start_step,
        max_steps=diversification_report.max_steps,
        initial_balance=float(econ["initial_balance"]),
        exchange=str(econ["exchange"]),
        maker_fee_rate=econ.get("maker_fee_rate"),
        taker_fee_rate=econ.get("taker_fee_rate"),
        execution_fee_mode=str(econ["execution_fee_mode"]),
        slippage=float(econ["slippage"]),
        position_sizing_mode=str(econ["position_sizing_mode"]),
        position_notional_fraction=float(econ["position_notional_fraction"]),
        leverage=float(econ["leverage"]),
        fixed_position_qty=float(econ["fixed_position_qty"]),
        weights_path=manifest.checkpoint_path,
        engine_config=candidate_engine_config(candidate),
    )


def _selected_score_lookup(shortlist_report: ShortlistReport) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for item in shortlist_report.candidate_scores:
        if not item.selected:
            continue
        result[item.candidate_id] = {
            "display_name": item.display_name,
            "base_score": item.base_score,
            "marginal_score": item.marginal_score,
        }
    return result


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
        result[item.candidate_id] = {
            "display_name": item.display_name,
            "base_score": item.base_score,
            "marginal_score": item.marginal_score,
        }
    return result


def _score_from_shortlist(payload: Mapping[str, object], score_mode: str, min_score_floor: float) -> float:
    if score_mode == "marginal" and payload.get("marginal_score") is not None:
        base_value = float(payload["marginal_score"])
    else:
        base_value = float(payload["base_score"])
    return max(base_value, min_score_floor)


def _allocate_with_cap(
    *,
    candidate_ids: Sequence[str],
    normalized_shares: np.ndarray,
    requested_risk_fraction: float,
    per_system_cap_fraction: float,
    candidate_cap_overrides: Mapping[str, float | None],
    cluster_by_candidate: Mapping[str, str],
    cluster_cap_overrides: Mapping[str, float | None],
) -> np.ndarray:
    candidate_caps = np.array(
        [
            min(
                per_system_cap_fraction,
                candidate_cap_overrides.get(candidate_id, per_system_cap_fraction) or per_system_cap_fraction,
            )
            for candidate_id in candidate_ids
        ],
        dtype=np.float64,
    )
    target_total = min(float(requested_risk_fraction), float(np.sum(candidate_caps)))
    if target_total <= 0.0:
        return np.zeros_like(normalized_shares)

    allocation = np.zeros_like(normalized_shares)
    free = np.ones(normalized_shares.size, dtype=bool)
    remaining = target_total
    cluster_allocations: dict[str, float] = {}

    while remaining > 1e-12 and np.any(free):
        free_indices = np.where(free)[0]
        weights = normalized_shares[free_indices]
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            weights = np.full(free_indices.size, 1.0 / free_indices.size)
        else:
            weights = weights / weight_sum
        proposed = remaining * weights
        progress = 0.0
        for local_index, candidate_index in enumerate(free_indices):
            candidate_id = candidate_ids[candidate_index]
            cluster_id = cluster_by_candidate.get(candidate_id)
            cluster_cap = None if cluster_id is None else cluster_cap_overrides.get(cluster_id)
            room = candidate_caps[candidate_index] - allocation[candidate_index]
            if cluster_cap is not None and cluster_id is not None:
                room = min(room, cluster_cap - cluster_allocations.get(cluster_id, 0.0))
            add = min(float(proposed[local_index]), float(room))
            allocation[candidate_index] += add
            if cluster_id is not None:
                cluster_allocations[cluster_id] = cluster_allocations.get(cluster_id, 0.0) + add
            progress += add
            cluster_full = cluster_cap is not None and cluster_id is not None and cluster_allocations.get(cluster_id, 0.0) >= cluster_cap - 1e-12
            if allocation[candidate_index] >= candidate_caps[candidate_index] - 1e-12 or cluster_full:
                free[candidate_index] = False
        remaining = target_total - float(np.sum(allocation))
        if progress <= 1e-12:
            break
    return allocation


def resolve_allocator_candidate_ids(
    shortlist_report: ShortlistReport,
    *,
    override_set: OverrideSet | None = None,
) -> list[str]:
    selected = list(shortlist_report.selected_candidate_ids)
    excluded = set(excluded_candidate_ids(override_set))
    forced = [candidate_id for candidate_id in forced_candidate_ids(override_set) if candidate_id in shortlist_report.candidate_ids]
    result: list[str] = []
    for candidate_id in [*selected, *forced]:
        if candidate_id in excluded or candidate_id in result:
            continue
        result.append(candidate_id)
    return result


def candidate_cap_overrides(override_set: OverrideSet | None) -> dict[str, float | None]:
    return {candidate_id: item.max_cap_fraction for candidate_id, item in candidate_override_map(override_set).items()}


def resolved_cluster_caps(
    override_set: OverrideSet | None,
    *,
    default_cluster_cap_fraction: float | None,
) -> dict[str, float | None]:
    mapping = {cluster_id: item.max_cap_fraction for cluster_id, item in cluster_override_map(override_set).items()}
    if default_cluster_cap_fraction is None:
        return mapping
    class DefaultClusterCaps(dict[str, float | None]):
        def get(self, key, default=None):  # type: ignore[override]
            if key in self:
                return super().get(key, default)
            return default_cluster_cap_fraction

    wrapped = DefaultClusterCaps(mapping)
    return wrapped


def cluster_allocation_totals(
    *,
    candidate_ids: Sequence[str],
    capital_fractions: np.ndarray,
    cluster_by_candidate: Mapping[str, str],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for candidate_id, capital_fraction in zip(candidate_ids, capital_fractions, strict=True):
        cluster_id = cluster_by_candidate.get(candidate_id)
        if cluster_id is None:
            continue
        totals[cluster_id] = totals.get(cluster_id, 0.0) + float(capital_fraction)
    return totals


def resolve_cap_reason(
    *,
    candidate_id: str,
    capital_fraction: float,
    per_system_cap_fraction: float,
    candidate_cap_overrides: Mapping[str, float | None],
    cluster_id: str | None,
    cluster_allocations: Mapping[str, float],
    cluster_cap_overrides: Mapping[str, float | None],
) -> str | None:
    candidate_cap = candidate_cap_overrides.get(candidate_id)
    effective_candidate_cap = per_system_cap_fraction if candidate_cap is None else min(per_system_cap_fraction, candidate_cap)
    if capital_fraction >= effective_candidate_cap - 1e-12:
        return "candidate_cap"
    if cluster_id is not None:
        cluster_cap = cluster_cap_overrides.get(cluster_id)
        if cluster_cap is not None and cluster_allocations.get(cluster_id, 0.0) >= cluster_cap - 1e-12:
            return "cluster_cap"
    return None


def _initial_balance(candidate: CandidateRecord) -> float:
    if candidate.manifest is None:
        return 10000.0
    return float(candidate.manifest.econ_config.get("initial_balance", 10000.0))
