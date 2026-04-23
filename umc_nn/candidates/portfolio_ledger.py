from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .allocator import equity_summary
from .registry import CandidateRegistry
from .schema import (
    LifecycleReport,
    PortfolioCandidateLedgerSummary,
    PortfolioCycleLedgerEntry,
    PortfolioLedgerReport,
    PortfolioRebalanceChange,
    RollingConveyorReport,
    TradeforwardEvaluationReport,
    TradeforwardPlan,
    utc_now_text,
)


@dataclass(slots=True, frozen=True)
class PortfolioLedgerConfig:
    tradable_statuses: tuple[str, ...] = ("approved", "paper", "active", "draining")
    turnover_cost_rate: float = 0.0
    unassigned_cluster_label: str = "unassigned"

    def validate(self) -> None:
        if self.turnover_cost_rate < 0.0:
            raise ValueError("turnover_cost_rate must be >= 0")
        if not self.tradable_statuses:
            raise ValueError("tradable_statuses must not be empty")
        if not self.unassigned_cluster_label:
            raise ValueError("unassigned_cluster_label must not be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "tradable_statuses": list(self.tradable_statuses),
            "turnover_cost_rate": self.turnover_cost_rate,
            "unassigned_cluster_label": self.unassigned_cluster_label,
        }


@dataclass(slots=True)
class _CandidateLedgerState:
    candidate_id: str
    display_name: str
    final_status: str
    selected_cycles: int = 0
    added_cycles: int = 0
    removed_cycles: int = 0
    increased_cycles: int = 0
    decreased_cycles: int = 0
    unchanged_cycles: int = 0
    gross_target_capital_fraction: float = 0.0
    max_target_capital_fraction: float = 0.0
    ending_capital_fraction: float = 0.0
    first_selected_cycle_index: int | None = None
    last_selected_cycle_index: int | None = None


def _cluster_key(cluster_id: str | None, config: PortfolioLedgerConfig) -> str:
    return config.unassigned_cluster_label if cluster_id is None else cluster_id


def _cluster_exposure(
    allocations: dict[str, float],
    meta_by_candidate: dict[str, tuple[str, str | None]],
    config: PortfolioLedgerConfig,
) -> dict[str, float]:
    exposure: dict[str, float] = {}
    for candidate_id, capital_fraction in allocations.items():
        if capital_fraction <= 1e-12:
            continue
        _, cluster_id = meta_by_candidate[candidate_id]
        key = _cluster_key(cluster_id, config)
        exposure[key] = exposure.get(key, 0.0) + float(capital_fraction)
    return exposure


def _stitched_segment(
    evaluation: TradeforwardEvaluationReport,
    *,
    balance_after_rebalance: float,
) -> np.ndarray:
    normalized = np.asarray(evaluation.portfolio.normalized_balance_history, dtype=np.float64)
    if normalized.size == 0:
        factor = evaluation.portfolio.final_balance / evaluation.portfolio.initial_balance
        return np.asarray([balance_after_rebalance, balance_after_rebalance * factor], dtype=np.float64)
    if normalized[0] != 1.0:
        normalized = np.concatenate(([1.0], normalized))
    return balance_after_rebalance * normalized


def _lifecycle_maps(
    lifecycle_report: LifecycleReport,
) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, str]], dict[int, dict[str, object]]]:
    before_by_cycle: dict[int, dict[str, str]] = {}
    after_by_cycle: dict[int, dict[str, str]] = {}
    decision_by_cycle: dict[int, dict[str, object]] = {}
    for decision in lifecycle_report.decisions:
        before_by_cycle.setdefault(decision.cycle_index, {})[decision.candidate_id] = decision.previous_status
        after_by_cycle.setdefault(decision.cycle_index, {})[decision.candidate_id] = decision.next_status
        decision_by_cycle.setdefault(decision.cycle_index, {})[decision.candidate_id] = decision
    return before_by_cycle, after_by_cycle, decision_by_cycle


def build_portfolio_ledger_report(
    registry: CandidateRegistry,
    name: str,
    *,
    rolling_report: RollingConveyorReport,
    lifecycle_report: LifecycleReport,
    config: PortfolioLedgerConfig,
    notes: str | None = None,
) -> PortfolioLedgerReport:
    config.validate()
    if lifecycle_report.source_rolling_report != rolling_report.name:
        raise ValueError(
            f"Lifecycle report {lifecycle_report.name} does not reference rolling report {rolling_report.name}"
        )

    lifecycle_summary_by_candidate = {
        item.candidate_id: item for item in lifecycle_report.candidate_summaries
    }
    before_status_by_cycle, after_status_by_cycle, decision_by_cycle = _lifecycle_maps(lifecycle_report)

    candidate_states: dict[str, _CandidateLedgerState] = {}
    for summary in lifecycle_report.candidate_summaries:
        candidate_states[summary.candidate_id] = _CandidateLedgerState(
            candidate_id=summary.candidate_id,
            display_name=summary.display_name,
            final_status=summary.final_status,
        )

    current_balance = float(rolling_report.initial_balance)
    ledger_cycle_indices = [0]
    ledger_balance_history = [current_balance]
    stitched_curve: list[float] = [current_balance]
    total_buy_turnover = 0.0
    total_sell_turnover = 0.0
    total_gross_turnover = 0.0
    total_estimated_rebalance_cost = 0.0
    total_allocated_risk = 0.0
    total_reserve = 0.0
    total_churn = 0
    non_tradable_selection_count = 0
    peak_cluster_exposure_fraction = 0.0

    previous_allocations: dict[str, float] = {}
    previous_meta: dict[str, tuple[str, str | None]] = {}
    cycle_entries: list[PortfolioCycleLedgerEntry] = []

    for cycle in rolling_report.cycle_outcomes:
        plan: TradeforwardPlan = registry.load_tradeforward_plan(cycle.tradeforward_plan_name)
        evaluation: TradeforwardEvaluationReport = registry.load_tradeforward_evaluation(cycle.tradeforward_evaluation_name)
        factor = evaluation.portfolio.final_balance / evaluation.portfolio.initial_balance

        target_candidate_ids = [allocation.candidate_id for allocation in plan.allocations if allocation.capital_fraction > 1e-12]
        current_allocations = {
            allocation.candidate_id: float(allocation.capital_fraction)
            for allocation in plan.allocations
            if allocation.capital_fraction > 1e-12
        }
        current_meta = {
            allocation.candidate_id: (allocation.display_name, allocation.cluster_id)
            for allocation in plan.allocations
            if allocation.capital_fraction > 1e-12
        }

        before_status_map = before_status_by_cycle.get(cycle.cycle_index, {})
        after_status_map = after_status_by_cycle.get(cycle.cycle_index, {})
        decision_map = decision_by_cycle.get(cycle.cycle_index, {})
        status_counts_before: dict[str, int] = {}
        for status in before_status_map.values():
            status_counts_before[status] = status_counts_before.get(status, 0) + 1
        status_counts_after: dict[str, int] = {}
        for status in after_status_map.values():
            status_counts_after[status] = status_counts_after.get(status, 0) + 1

        previous_active_candidate_ids = [candidate_id for candidate_id, value in previous_allocations.items() if value > 1e-12]
        reserve_fraction_before = max(0.0, 1.0 - sum(previous_allocations.values()))
        reserve_fraction_after = max(0.0, 1.0 - sum(current_allocations.values()))

        union_candidate_ids = sorted(set(previous_allocations).union(current_allocations))
        changes: list[PortfolioRebalanceChange] = []
        added_candidate_ids: list[str] = []
        removed_candidate_ids: list[str] = []
        increased_candidate_ids: list[str] = []
        decreased_candidate_ids: list[str] = []
        unchanged_candidate_ids: list[str] = []

        buy_turnover = 0.0
        sell_turnover = 0.0
        for candidate_id in union_candidate_ids:
            previous_fraction = float(previous_allocations.get(candidate_id, 0.0))
            target_fraction = float(current_allocations.get(candidate_id, 0.0))
            delta = target_fraction - previous_fraction
            if previous_fraction <= 1e-12 and target_fraction > 1e-12:
                action = "added"
                added_candidate_ids.append(candidate_id)
            elif target_fraction <= 1e-12 and previous_fraction > 1e-12:
                action = "removed"
                removed_candidate_ids.append(candidate_id)
            elif delta > 1e-12:
                action = "increased"
                increased_candidate_ids.append(candidate_id)
            elif delta < -1e-12:
                action = "decreased"
                decreased_candidate_ids.append(candidate_id)
            else:
                action = "unchanged"
                if target_fraction > 1e-12 or previous_fraction > 1e-12:
                    unchanged_candidate_ids.append(candidate_id)

            buy_turnover += max(delta, 0.0)
            sell_turnover += max(-delta, 0.0)

            display_name = current_meta.get(candidate_id, previous_meta.get(candidate_id, (candidate_id, None)))[0]
            cluster_id = current_meta.get(candidate_id, previous_meta.get(candidate_id, (display_name, None)))[1]
            changes.append(
                PortfolioRebalanceChange(
                    candidate_id=candidate_id,
                    display_name=display_name,
                    cluster_id=cluster_id,
                    previous_capital_fraction=previous_fraction,
                    target_capital_fraction=target_fraction,
                    delta_capital_fraction=delta,
                    rebalance_action=action,
                    status_before_cycle=before_status_map.get(candidate_id),
                    status_after_cycle=after_status_map.get(candidate_id),
                )
            )

            state = candidate_states.setdefault(
                candidate_id,
                _CandidateLedgerState(
                    candidate_id=candidate_id,
                    display_name=display_name,
                    final_status=lifecycle_summary_by_candidate.get(candidate_id).final_status
                    if candidate_id in lifecycle_summary_by_candidate
                    else registry.load_candidate(candidate_id).status,
                ),
            )
            if target_fraction > 1e-12:
                state.selected_cycles += 1
                state.gross_target_capital_fraction += target_fraction
                state.max_target_capital_fraction = max(state.max_target_capital_fraction, target_fraction)
                state.ending_capital_fraction = target_fraction
                if state.first_selected_cycle_index is None:
                    state.first_selected_cycle_index = cycle.cycle_index
                state.last_selected_cycle_index = cycle.cycle_index
            elif action == "removed":
                state.ending_capital_fraction = 0.0

            if action == "added":
                state.added_cycles += 1
            elif action == "removed":
                state.removed_cycles += 1
            elif action == "increased":
                state.increased_cycles += 1
            elif action == "decreased":
                state.decreased_cycles += 1
            elif action == "unchanged" and target_fraction > 1e-12:
                state.unchanged_cycles += 1

        gross_turnover = buy_turnover + sell_turnover
        estimated_rebalance_cost = current_balance * gross_turnover * config.turnover_cost_rate
        balance_after_rebalance = current_balance - estimated_rebalance_cost
        balance_after_cycle = balance_after_rebalance * factor
        gross_cycle_pnl = current_balance * (factor - 1.0)
        net_cycle_pnl = balance_after_cycle - current_balance

        if config.turnover_cost_rate <= 1e-12:
            if abs(balance_after_cycle - cycle.ledger_balance_after) > 1e-6:
                raise ValueError(
                    f"Portfolio ledger mismatch for cycle {cycle.cycle_name}: "
                    f"{balance_after_cycle} != {cycle.ledger_balance_after}"
                )

        cluster_exposure_before = _cluster_exposure(previous_allocations, previous_meta, config)
        cluster_exposure_after = _cluster_exposure(current_allocations, current_meta, config)
        peak_cluster_exposure_fraction = max(
            peak_cluster_exposure_fraction,
            max(cluster_exposure_before.values(), default=0.0),
            max(cluster_exposure_after.values(), default=0.0),
        )
        non_tradable_selected_candidate_ids = [
            candidate_id
            for candidate_id in target_candidate_ids
            if before_status_map.get(candidate_id) not in set(config.tradable_statuses)
        ]
        non_tradable_selection_count += len(non_tradable_selected_candidate_ids)

        segment = _stitched_segment(evaluation, balance_after_rebalance=balance_after_rebalance)
        if segment.size:
            start_index = 0 if abs(segment[0] - stitched_curve[-1]) > 1e-9 else 1
            stitched_curve.extend(float(value) for value in segment[start_index:])

        ledger_cycle_indices.append(cycle.cycle_index)
        ledger_balance_history.append(balance_after_cycle)
        total_buy_turnover += buy_turnover
        total_sell_turnover += sell_turnover
        total_gross_turnover += gross_turnover
        total_estimated_rebalance_cost += estimated_rebalance_cost
        total_allocated_risk += cycle.allocated_risk_fraction
        total_reserve += reserve_fraction_after
        total_churn += len(added_candidate_ids) + len(removed_candidate_ids)

        cycle_entries.append(
            PortfolioCycleLedgerEntry(
                cycle_index=cycle.cycle_index,
                cycle_name=cycle.cycle_name,
                selection_start_utc=cycle.selection_start_utc,
                selection_end_utc=cycle.selection_end_utc,
                forward_start_utc=cycle.forward_start_utc,
                forward_end_utc=cycle.forward_end_utc,
                tradeforward_plan_name=plan.name,
                tradeforward_evaluation_name=evaluation.name,
                ledger_balance_before=current_balance,
                ledger_balance_after_rebalance=balance_after_rebalance,
                ledger_balance_after_cycle=balance_after_cycle,
                gross_cycle_pnl=gross_cycle_pnl,
                net_cycle_pnl=net_cycle_pnl,
                portfolio_max_drawdown_pct=evaluation.portfolio.max_drawdown_pct,
                requested_risk_fraction=cycle.requested_risk_fraction,
                allocated_risk_fraction=cycle.allocated_risk_fraction,
                reserve_fraction_before=reserve_fraction_before,
                reserve_fraction_after=reserve_fraction_after,
                buy_turnover_fraction=buy_turnover,
                sell_turnover_fraction=sell_turnover,
                gross_turnover_fraction=gross_turnover,
                estimated_rebalance_cost=estimated_rebalance_cost,
                previous_active_candidate_ids=previous_active_candidate_ids,
                target_candidate_ids=target_candidate_ids,
                added_candidate_ids=added_candidate_ids,
                removed_candidate_ids=removed_candidate_ids,
                increased_candidate_ids=increased_candidate_ids,
                decreased_candidate_ids=decreased_candidate_ids,
                unchanged_candidate_ids=unchanged_candidate_ids,
                non_tradable_selected_candidate_ids=non_tradable_selected_candidate_ids,
                status_counts_before=status_counts_before,
                status_counts_after=status_counts_after,
                cluster_exposure_before=cluster_exposure_before,
                cluster_exposure_after=cluster_exposure_after,
                rebalance_changes=changes,
            )
        )

        current_balance = balance_after_cycle
        previous_allocations = current_allocations
        previous_meta = current_meta

    stitched_curve_np = np.asarray(stitched_curve, dtype=np.float64)
    final_balance, total_pnl, max_drawdown_pct = equity_summary(
        stitched_curve_np,
        initial_balance=float(rolling_report.initial_balance),
    )
    total_return_pct = ((final_balance / float(rolling_report.initial_balance)) - 1.0) * 100.0
    average_gross_turnover = total_gross_turnover / len(cycle_entries) if cycle_entries else 0.0
    average_allocated_risk = total_allocated_risk / len(cycle_entries) if cycle_entries else 0.0
    average_reserve = total_reserve / len(cycle_entries) if cycle_entries else 0.0
    final_active_candidate_ids = cycle_entries[-1].target_candidate_ids if cycle_entries else []

    candidate_summaries = [
        PortfolioCandidateLedgerSummary(
            candidate_id=state.candidate_id,
            display_name=state.display_name,
            final_status=state.final_status,
            selected_cycles=state.selected_cycles,
            added_cycles=state.added_cycles,
            removed_cycles=state.removed_cycles,
            increased_cycles=state.increased_cycles,
            decreased_cycles=state.decreased_cycles,
            unchanged_cycles=state.unchanged_cycles,
            gross_target_capital_fraction=state.gross_target_capital_fraction,
            average_target_capital_fraction=0.0
            if state.selected_cycles == 0
            else state.gross_target_capital_fraction / state.selected_cycles,
            max_target_capital_fraction=state.max_target_capital_fraction,
            ending_capital_fraction=state.ending_capital_fraction,
            first_selected_cycle_index=state.first_selected_cycle_index,
            last_selected_cycle_index=state.last_selected_cycle_index,
        )
        for state in candidate_states.values()
    ]
    candidate_summaries.sort(key=lambda item: (-item.selected_cycles, item.candidate_id))

    return PortfolioLedgerReport(
        schema_version="1",
        name=name,
        created_at_utc=utc_now_text(),
        source_rolling_report=rolling_report.name,
        source_lifecycle_report=lifecycle_report.name,
        config=config.to_dict(),
        initial_balance=float(rolling_report.initial_balance),
        final_balance=final_balance,
        total_pnl=total_pnl,
        total_return_pct=float(total_return_pct),
        max_drawdown_pct=max_drawdown_pct,
        ledger_cycle_indices=ledger_cycle_indices,
        ledger_balance_history=ledger_balance_history,
        total_buy_turnover_fraction=total_buy_turnover,
        total_sell_turnover_fraction=total_sell_turnover,
        total_gross_turnover_fraction=total_gross_turnover,
        average_gross_turnover_fraction=average_gross_turnover,
        total_estimated_rebalance_cost=total_estimated_rebalance_cost,
        average_allocated_risk_fraction=average_allocated_risk,
        average_reserve_fraction=average_reserve,
        total_churn_count=total_churn,
        peak_cluster_exposure_fraction=peak_cluster_exposure_fraction,
        non_tradable_selection_count=non_tradable_selection_count,
        final_active_candidate_ids=final_active_candidate_ids,
        final_status_counts=dict(lifecycle_report.final_status_counts),
        candidate_summaries=candidate_summaries,
        cycle_entries=cycle_entries,
        notes=notes,
    )
