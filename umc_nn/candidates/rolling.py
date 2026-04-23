from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

import numpy as np

from .allocator import equity_summary
from .registry import CandidateRegistry
from .schema import (
    AllocatorWorkbenchReport,
    CombinationSearchReport,
    ContinuousSearchCycleReport,
    RollingConveyorReport,
    RollingCycleOutcome,
    TradeforwardEvaluationReport,
    TradeforwardPlan,
    utc_now_text,
)


def _parse_utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True, slots=True)
class RollingWindowSpec:
    cycle_index: int
    cycle_name: str
    selection_start_utc: str
    selection_end_utc: str
    forward_start_utc: str
    forward_end_utc: str


def build_rolling_window_specs(
    *,
    report_name: str,
    selection_start_utc: str,
    selection_days: int,
    forward_days: int,
    cycle_count: int,
    step_days: int | None = None,
) -> list[RollingWindowSpec]:
    if selection_days <= 0:
        raise ValueError("selection_days must be > 0")
    if forward_days <= 0:
        raise ValueError("forward_days must be > 0")
    if cycle_count <= 0:
        raise ValueError("cycle_count must be > 0")
    resolved_step_days = forward_days if step_days is None else step_days
    if resolved_step_days <= 0:
        raise ValueError("step_days must be > 0")

    start = _parse_utc(selection_start_utc)
    selection_delta = timedelta(days=selection_days)
    forward_delta = timedelta(days=forward_days)
    step_delta = timedelta(days=resolved_step_days)

    specs: list[RollingWindowSpec] = []
    for index in range(cycle_count):
        selection_start = start + index * step_delta
        selection_end = selection_start + selection_delta
        forward_start = selection_end
        forward_end = forward_start + forward_delta
        specs.append(
            RollingWindowSpec(
                cycle_index=index + 1,
                cycle_name=f"{report_name}_cycle_{index + 1:02d}",
                selection_start_utc=_format_utc(selection_start),
                selection_end_utc=_format_utc(selection_end),
                forward_start_utc=_format_utc(forward_start),
                forward_end_utc=_format_utc(forward_end),
            )
        )
    return specs


def _selection_window_from_sources(
    registry: CandidateRegistry,
    plan: TradeforwardPlan,
) -> tuple[str, str]:
    if plan.source_combination_report:
        report: CombinationSearchReport = registry.load_combination_search(plan.source_combination_report)
        return report.start_utc, report.end_utc
    if plan.source_allocator_report:
        report: AllocatorWorkbenchReport = registry.load_allocator_workbench(plan.source_allocator_report)
        return report.start_utc, report.end_utc
    raise ValueError(f"Tradeforward plan {plan.name} is missing selection-window source reports")


def _stitched_curve_segment(
    evaluation: TradeforwardEvaluationReport,
    *,
    ledger_balance_before: float,
) -> np.ndarray:
    normalized = np.asarray(evaluation.portfolio.normalized_balance_history, dtype=np.float64)
    if normalized.size == 0:
        factor = evaluation.portfolio.final_balance / evaluation.portfolio.initial_balance
        return np.asarray([ledger_balance_before, ledger_balance_before * factor], dtype=np.float64)
    if normalized[0] != 1.0:
        normalized = np.concatenate(([1.0], normalized))
    return ledger_balance_before * normalized


def build_rolling_conveyor_report(
    registry: CandidateRegistry,
    name: str,
    *,
    mode: str,
    cycle_reports: Sequence[ContinuousSearchCycleReport],
    tradeforward_plans: Sequence[TradeforwardPlan],
    tradeforward_evaluations: Sequence[TradeforwardEvaluationReport],
    selection_start_utc: str,
    selection_days: int,
    forward_days: int,
    step_days: int,
    initial_balance: float,
    selector: dict[str, object] | None = None,
    notes: str | None = None,
) -> RollingConveyorReport:
    if initial_balance <= 0:
        raise ValueError("initial_balance must be > 0")
    if not cycle_reports:
        raise ValueError("At least one cycle report is required")
    if not (len(cycle_reports) == len(tradeforward_plans) == len(tradeforward_evaluations)):
        raise ValueError("cycle_reports, tradeforward_plans, and tradeforward_evaluations must have matching lengths")

    cycle_outcomes: list[RollingCycleOutcome] = []
    ledger_balance_history: list[float] = [float(initial_balance)]
    ledger_cycle_indices: list[int] = [0]
    stitched_curve: list[float] = [float(initial_balance)]
    current_balance = float(initial_balance)
    positive_cycle_count = 0

    for index, (cycle_report, plan, evaluation) in enumerate(
        zip(cycle_reports, tradeforward_plans, tradeforward_evaluations),
        start=1,
    ):
        if plan.source_cycle_report and plan.source_cycle_report != cycle_report.name:
            raise ValueError(
                f"Tradeforward plan {plan.name} source_cycle_report mismatch: "
                f"{plan.source_cycle_report} != {cycle_report.name}"
            )
        if evaluation.source_plan != plan.name:
            raise ValueError(f"Tradeforward evaluation {evaluation.name} does not reference plan {plan.name}")

        selection_start, selection_end = _selection_window_from_sources(registry, plan)
        reference_initial_balance = evaluation.portfolio.initial_balance
        cycle_factor = evaluation.portfolio.final_balance / reference_initial_balance
        ledger_balance_after = current_balance * cycle_factor
        portfolio_pnl = ledger_balance_after - current_balance
        cycle_return_fraction = cycle_factor - 1.0
        if portfolio_pnl > 1e-9:
            positive_cycle_count += 1

        expectation = evaluation.expectation
        scale = current_balance / reference_initial_balance
        expected_original_scaled = None
        expected_p05_scaled = None
        actual_minus_expected_original = None
        actual_minus_expected_p05 = None
        actual_minus_expected_original_dd = None
        if expectation is not None:
            expected_original_scaled = expectation.expected_original_net_profit * scale
            expected_p05_scaled = expectation.expected_p05_net_profit * scale
            actual_minus_expected_original = portfolio_pnl - expected_original_scaled
            actual_minus_expected_p05 = portfolio_pnl - expected_p05_scaled
            actual_minus_expected_original_dd = (
                evaluation.portfolio.max_drawdown_pct - expectation.expected_original_max_drawdown_pct
            )

        stitched_segment = _stitched_curve_segment(
            evaluation,
            ledger_balance_before=current_balance,
        )
        stitched_curve.extend(float(value) for value in stitched_segment[1:])
        ledger_balance_history.append(float(ledger_balance_after))
        ledger_cycle_indices.append(index)

        cycle_outcomes.append(
            RollingCycleOutcome(
                cycle_index=index,
                cycle_name=cycle_report.name,
                cycle_report_name=cycle_report.name,
                tradeforward_plan_name=plan.name,
                tradeforward_evaluation_name=evaluation.name,
                selection_start_utc=selection_start,
                selection_end_utc=selection_end,
                forward_start_utc=plan.forward_start_utc,
                forward_end_utc=plan.forward_end_utc,
                candidate_ids=list(cycle_report.candidate_ids),
                selected_candidate_ids=list(evaluation.candidate_ids),
                requested_risk_fraction=evaluation.portfolio.requested_risk_fraction,
                allocated_risk_fraction=evaluation.portfolio.allocated_risk_fraction,
                reserve_fraction=evaluation.portfolio.reserve_fraction,
                portfolio_pnl=float(portfolio_pnl),
                portfolio_final_balance=float(ledger_balance_after),
                portfolio_max_drawdown_pct=evaluation.portfolio.max_drawdown_pct,
                cycle_return_fraction=float(cycle_return_fraction),
                ledger_balance_before=float(current_balance),
                ledger_balance_after=float(ledger_balance_after),
                actual_minus_expected_original_net_profit=actual_minus_expected_original,
                actual_minus_expected_original_max_drawdown_pct=actual_minus_expected_original_dd,
            )
        )

        current_balance = float(ledger_balance_after)

    stitched_curve_np = np.asarray(stitched_curve, dtype=np.float64)
    final_balance, total_pnl, max_drawdown_pct = equity_summary(
        stitched_curve_np,
        initial_balance=float(initial_balance),
    )
    cycle_return_pcts = np.asarray([outcome.cycle_return_fraction * 100.0 for outcome in cycle_outcomes], dtype=np.float64)
    total_return_pct = ((final_balance / float(initial_balance)) - 1.0) * 100.0

    return RollingConveyorReport(
        schema_version="1",
        name=name,
        created_at_utc=utc_now_text(),
        mode=mode,
        selection_days=int(selection_days),
        forward_days=int(forward_days),
        step_days=int(step_days),
        initial_balance=float(initial_balance),
        final_balance=final_balance,
        total_pnl=total_pnl,
        total_return_pct=float(total_return_pct),
        max_drawdown_pct=max_drawdown_pct,
        positive_cycle_count=positive_cycle_count,
        evaluated_cycle_count=len(cycle_outcomes),
        ledger_cycle_indices=ledger_cycle_indices,
        ledger_balance_history=ledger_balance_history,
        cycle_outcomes=cycle_outcomes,
        selector={
            "selection_start_utc": selection_start_utc,
            "mean_cycle_return_pct": float(np.mean(cycle_return_pcts)),
            "median_cycle_return_pct": float(np.median(cycle_return_pcts)),
            **(selector or {}),
        },
        notes=notes,
    )
