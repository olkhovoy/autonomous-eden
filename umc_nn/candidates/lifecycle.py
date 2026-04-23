from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .registry import CandidateRegistry
from .schema import (
    CandidateRecord,
    LifecycleCandidateSummary,
    LifecycleDecisionRecord,
    LifecycleReport,
    RollingConveyorReport,
    TradeforwardCandidateEvaluation,
    TradeforwardEvaluationReport,
    utc_now_text,
)


SERVICE_STATUSES = {"paper", "active", "draining"}
IMMUTABLE_STATUSES = {"retired", "rejected"}


@dataclass(slots=True, frozen=True)
class LifecycleConfig:
    min_forward_pnl: float = 0.0
    max_forward_drawdown_pct: float = 8.0
    min_selected_trades: int = 1
    successful_selected_cycles_to_activate: int = 2
    successful_selected_cycles_to_recover: int = 1
    idle_cycles_to_drain: int = 1
    idle_cycles_to_retire: int = 2

    def validate(self) -> None:
        if self.max_forward_drawdown_pct <= 0.0:
            raise ValueError("max_forward_drawdown_pct must be > 0")
        if self.min_selected_trades < 0:
            raise ValueError("min_selected_trades must be >= 0")
        if self.successful_selected_cycles_to_activate <= 0:
            raise ValueError("successful_selected_cycles_to_activate must be > 0")
        if self.successful_selected_cycles_to_recover <= 0:
            raise ValueError("successful_selected_cycles_to_recover must be > 0")
        if self.idle_cycles_to_drain <= 0:
            raise ValueError("idle_cycles_to_drain must be > 0")
        if self.idle_cycles_to_retire <= 0:
            raise ValueError("idle_cycles_to_retire must be > 0")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "min_forward_pnl": self.min_forward_pnl,
            "max_forward_drawdown_pct": self.max_forward_drawdown_pct,
            "min_selected_trades": self.min_selected_trades,
            "successful_selected_cycles_to_activate": self.successful_selected_cycles_to_activate,
            "successful_selected_cycles_to_recover": self.successful_selected_cycles_to_recover,
            "idle_cycles_to_drain": self.idle_cycles_to_drain,
            "idle_cycles_to_retire": self.idle_cycles_to_retire,
        }


@dataclass(slots=True)
class _CandidateLifecycleState:
    candidate_id: str
    display_name: str
    initial_status: str
    status: str
    total_cycles_seen: int = 0
    selected_cycles: int = 0
    successful_forward_cycles: int = 0
    failed_forward_cycles: int = 0
    no_trade_cycles: int = 0
    idle_cycles: int = 0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    consecutive_idle_cycles: int = 0
    transition_count: int = 0
    last_selected_cycle_index: int | None = None
    last_transition_cycle_index: int | None = None


def _all_candidate_ids(report: RollingConveyorReport) -> list[str]:
    candidate_ids: set[str] = set()
    for cycle in report.cycle_outcomes:
        candidate_ids.update(cycle.candidate_ids)
        candidate_ids.update(cycle.selected_candidate_ids)
    return sorted(candidate_ids)


def _success_and_reasons(
    evaluation: TradeforwardCandidateEvaluation,
    *,
    config: LifecycleConfig,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    trades = int(evaluation.trades)
    pnl = float(evaluation.pnl)
    max_dd = float(evaluation.max_drawdown_pct)
    if trades < config.min_selected_trades:
        reasons.append("no_trades")
    if pnl < config.min_forward_pnl:
        reasons.append("negative_pnl")
    if max_dd > config.max_forward_drawdown_pct:
        reasons.append("drawdown_breach")
    if reasons:
        return False, reasons
    return True, ["forward_success"]


def _next_status_for_selected(
    status: str,
    *,
    success: bool,
    state: _CandidateLifecycleState,
    config: LifecycleConfig,
) -> tuple[str, str, list[str]]:
    if status in IMMUTABLE_STATUSES:
        return status, "hold", ["immutable_status"]

    if success:
        if status == "research":
            return "approved", "promote", ["promote_to_approved"]
        if status == "approved":
            return "paper", "promote", ["promote_to_paper"]
        if status == "paper":
            if state.consecutive_successes >= config.successful_selected_cycles_to_activate:
                return "active", "promote", ["promote_to_active"]
            return "paper", "hold", ["paper_hold_success"]
        if status == "active":
            return "active", "hold", ["active_hold_success"]
        if status == "draining":
            if state.consecutive_successes >= config.successful_selected_cycles_to_recover:
                return "paper", "recover", ["recover_to_paper"]
            return "draining", "hold", ["draining_hold_success"]
        return status, "hold", ["selected_success_no_transition"]

    if status == "research":
        return "research", "hold", ["research_failure"]
    if status == "approved":
        return "research", "demote", ["approved_failure_demote"]
    if status == "paper":
        return "draining", "drain", ["paper_failure_drain"]
    if status == "active":
        return "draining", "drain", ["active_failure_drain"]
    if status == "draining":
        return "retired", "retire", ["draining_failure_retire"]
    return status, "hold", ["selected_failure_no_transition"]


def _next_status_for_idle(
    status: str,
    *,
    state: _CandidateLifecycleState,
    config: LifecycleConfig,
) -> tuple[str, str, list[str]]:
    if status in IMMUTABLE_STATUSES:
        return status, "hold", ["immutable_status"]
    if status in {"paper", "active"} and state.consecutive_idle_cycles >= config.idle_cycles_to_drain:
        return "draining", "drain", ["idle_drain_threshold"]
    if status == "draining" and state.consecutive_idle_cycles >= config.idle_cycles_to_retire:
        return "retired", "retire", ["idle_retire_threshold"]
    return status, "hold", ["idle_hold"]


def build_lifecycle_report(
    registry: CandidateRegistry,
    name: str,
    *,
    rolling_report: RollingConveyorReport,
    config: LifecycleConfig,
    applied_status_updates: bool = False,
    notes: str | None = None,
) -> LifecycleReport:
    config.validate()

    candidate_ids = _all_candidate_ids(rolling_report)
    candidate_map: dict[str, CandidateRecord] = {
        candidate_id: registry.load_candidate(candidate_id) for candidate_id in candidate_ids
    }
    state_map: dict[str, _CandidateLifecycleState] = {
        candidate_id: _CandidateLifecycleState(
            candidate_id=candidate_id,
            display_name=candidate.display_name,
            initial_status=candidate.status,
            status=candidate.status,
        )
        for candidate_id, candidate in candidate_map.items()
    }

    decisions: list[LifecycleDecisionRecord] = []
    source_tradeforward_evaluations: list[str] = []

    for cycle in rolling_report.cycle_outcomes:
        evaluation: TradeforwardEvaluationReport = registry.load_tradeforward_evaluation(cycle.tradeforward_evaluation_name)
        source_tradeforward_evaluations.append(evaluation.name)
        evaluation_map = {
            item.candidate_id: item for item in evaluation.candidate_evaluations
        }

        for candidate_id in candidate_ids:
            state = state_map[candidate_id]
            previous_status = state.status
            state.total_cycles_seen += 1
            selected_eval = evaluation_map.get(candidate_id)

            if selected_eval is not None:
                state.selected_cycles += 1
                state.last_selected_cycle_index = cycle.cycle_index
                success, reason_codes = _success_and_reasons(selected_eval, config=config)
                state.consecutive_idle_cycles = 0
                if success:
                    state.successful_forward_cycles += 1
                    state.consecutive_successes += 1
                    state.consecutive_failures = 0
                else:
                    state.failed_forward_cycles += 1
                    state.consecutive_successes = 0
                    state.consecutive_failures += 1
                    if int(selected_eval.trades) < config.min_selected_trades:
                        state.no_trade_cycles += 1
                next_status, action, transition_reasons = _next_status_for_selected(
                    previous_status,
                    success=success,
                    state=state,
                    config=config,
                )
                all_reasons = reason_codes + transition_reasons
                pnl = float(selected_eval.pnl)
                max_dd = float(selected_eval.max_drawdown_pct)
                trades = int(selected_eval.trades)
            else:
                state.idle_cycles += 1
                state.consecutive_idle_cycles += 1
                state.consecutive_successes = 0
                state.consecutive_failures = 0
                next_status, action, all_reasons = _next_status_for_idle(
                    previous_status,
                    state=state,
                    config=config,
                )
                pnl = None
                max_dd = None
                trades = None

            state.status = next_status
            if next_status != previous_status:
                state.transition_count += 1
                state.last_transition_cycle_index = cycle.cycle_index

            decisions.append(
                LifecycleDecisionRecord(
                    cycle_index=cycle.cycle_index,
                    cycle_name=cycle.cycle_name,
                    candidate_id=candidate_id,
                    display_name=state.display_name,
                    selected=selected_eval is not None,
                    previous_status=previous_status,
                    next_status=next_status,
                    action=action,
                    reason_codes=all_reasons,
                    trades=trades,
                    pnl=pnl,
                    max_drawdown_pct=max_dd,
                    consecutive_successes=state.consecutive_successes,
                    consecutive_failures=state.consecutive_failures,
                    consecutive_idle_cycles=state.consecutive_idle_cycles,
                )
            )

    summaries: list[LifecycleCandidateSummary] = []
    final_status_counts: dict[str, int] = {}
    for candidate_id in candidate_ids:
        state = state_map[candidate_id]
        final_status_counts[state.status] = final_status_counts.get(state.status, 0) + 1
        summaries.append(
            LifecycleCandidateSummary(
                candidate_id=candidate_id,
                display_name=state.display_name,
                initial_status=state.initial_status,
                final_status=state.status,
                total_cycles_seen=state.total_cycles_seen,
                selected_cycles=state.selected_cycles,
                successful_forward_cycles=state.successful_forward_cycles,
                failed_forward_cycles=state.failed_forward_cycles,
                no_trade_cycles=state.no_trade_cycles,
                idle_cycles=state.idle_cycles,
                transition_count=state.transition_count,
                last_selected_cycle_index=state.last_selected_cycle_index,
                last_transition_cycle_index=state.last_transition_cycle_index,
            )
        )

    summaries.sort(key=lambda item: (item.final_status, item.candidate_id))
    return LifecycleReport(
        schema_version="1",
        name=name,
        created_at_utc=utc_now_text(),
        source_rolling_report=rolling_report.name,
        source_tradeforward_evaluations=source_tradeforward_evaluations,
        config=config.to_dict(),
        candidate_ids=candidate_ids,
        applied_status_updates=applied_status_updates,
        final_status_counts=final_status_counts,
        candidate_summaries=summaries,
        decisions=decisions,
        notes=notes,
    )


def apply_lifecycle_report(
    registry: CandidateRegistry,
    report: LifecycleReport,
) -> None:
    summary_by_candidate = {item.candidate_id: item for item in report.candidate_summaries}
    for candidate_id in report.candidate_ids:
        candidate = registry.load_candidate(candidate_id)
        summary = summary_by_candidate[candidate_id]
        candidate.status = summary.final_status
        lifecycle_meta = dict(candidate.metadata.get("lifecycle", {}))
        lifecycle_meta.update(
            {
                "source_report": report.name,
                "source_rolling_report": report.source_rolling_report,
                "final_status": summary.final_status,
                "selected_cycles": summary.selected_cycles,
                "successful_forward_cycles": summary.successful_forward_cycles,
                "failed_forward_cycles": summary.failed_forward_cycles,
                "idle_cycles": summary.idle_cycles,
                "transition_count": summary.transition_count,
                "last_selected_cycle_index": summary.last_selected_cycle_index,
                "last_transition_cycle_index": summary.last_transition_cycle_index,
            }
        )
        candidate.metadata["lifecycle"] = lifecycle_meta
        registry.add_candidate(candidate, overwrite=True)
