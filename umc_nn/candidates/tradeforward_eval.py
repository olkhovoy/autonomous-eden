from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from umc_nn.trading_eval import EpisodeTrace, evaluate_policy_trace_path

from .allocator import downsample_equity_curve, equity_summary, portfolio_equity_from_step_returns, step_returns_from_trace
from .engine_config import candidate_engine_config
from .registry import CandidateRegistry
from .schema import (
    AllocatorWorkbenchReport,
    CombinationSearchReport,
    TradeforwardCandidateEvaluation,
    TradeforwardEvaluationReport,
    TradeforwardExpectation,
    TradeforwardPlan,
    TradeforwardPortfolioEvaluation,
    utc_now_text,
)


@dataclass(slots=True)
class _ScenarioExpectation:
    selection_mode: str
    scenario_name: str
    objective_score: float
    expected_original_net_profit: float
    expected_original_max_drawdown_pct: float
    expected_p05_net_profit: float
    expected_p95_max_drawdown_pct: float


def _load_source_expectation(
    registry: CandidateRegistry,
    plan: TradeforwardPlan,
) -> _ScenarioExpectation | None:
    if plan.source_combination_report:
        report: CombinationSearchReport = registry.load_combination_search(plan.source_combination_report)
        for scenario in report.scenarios:
            if scenario.name != plan.scenario_name:
                continue
            return _ScenarioExpectation(
                selection_mode="combination",
                scenario_name=scenario.name,
                objective_score=scenario.objective_score,
                expected_original_net_profit=scenario.resampling.original_net_profit,
                expected_original_max_drawdown_pct=scenario.resampling.original_max_drawdown_pct,
                expected_p05_net_profit=scenario.resampling.p05_net_profit,
                expected_p95_max_drawdown_pct=scenario.resampling.p95_max_drawdown_pct,
            )
        raise KeyError(f"Scenario {plan.scenario_name} not found in combination report {plan.source_combination_report}")
    if plan.source_allocator_report:
        report: AllocatorWorkbenchReport = registry.load_allocator_workbench(plan.source_allocator_report)
        for scenario in report.scenarios:
            if scenario.name != plan.scenario_name:
                continue
            return _ScenarioExpectation(
                selection_mode="allocator",
                scenario_name=scenario.name,
                objective_score=scenario.objective_score,
                expected_original_net_profit=scenario.resampling.original_net_profit,
                expected_original_max_drawdown_pct=scenario.resampling.original_max_drawdown_pct,
                expected_p05_net_profit=scenario.resampling.p05_net_profit,
                expected_p95_max_drawdown_pct=scenario.resampling.p95_max_drawdown_pct,
            )
        raise KeyError(f"Scenario {plan.scenario_name} not found in allocator report {plan.source_allocator_report}")
    return None


def _trace_for_allocation(
    registry: CandidateRegistry,
    plan: TradeforwardPlan,
    *,
    candidate_id: str,
) -> tuple[EpisodeTrace, float, str]:
    candidate = registry.load_candidate(candidate_id)
    manifest = candidate.manifest
    if manifest is None:
        raise ValueError(f"Candidate {candidate_id} is missing manifest")
    if manifest.data_path != plan.data_path:
        raise ValueError(
            f"Tradeforward plan data_path mismatch for {candidate_id}: "
            f"{manifest.data_path} != {plan.data_path}"
        )
    econ = manifest.econ_config
    initial_balance = float(econ.get("initial_balance", 10000.0))
    trace = evaluate_policy_trace_path(
        "monolith",
        manifest.data_path,
        use_neurobars=str(manifest.data_path).endswith(".npz"),
        start_step=int(plan.forward_start_step or 0),
        max_steps=int(plan.forward_max_steps or 0),
        initial_balance=initial_balance,
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
    return trace, initial_balance, manifest.checkpoint_path


def build_tradeforward_evaluation(
    registry: CandidateRegistry,
    name: str,
    *,
    plan: TradeforwardPlan,
    curve_points: int = 256,
    notes: str | None = None,
) -> TradeforwardEvaluationReport:
    if plan.forward_start_step is None or plan.forward_max_steps is None:
        raise ValueError("Tradeforward plan must include resolved forward window steps")
    if plan.forward_max_steps <= 0:
        raise ValueError("Tradeforward plan forward_max_steps must be > 0")

    expectation = _load_source_expectation(registry, plan)
    candidate_results: list[TradeforwardCandidateEvaluation] = []
    traces: list[EpisodeTrace] = []
    weights: list[float] = []
    initial_balances: list[float] = []

    for allocation in plan.allocations:
        trace, initial_balance, checkpoint_path = _trace_for_allocation(
            registry,
            plan,
            candidate_id=allocation.candidate_id,
        )
        initial_balances.append(initial_balance)
        traces.append(trace)
        weights.append(allocation.capital_fraction)
        curve_sample_indices, normalized_curve = downsample_equity_curve(
            np.asarray(trace.balance_history, dtype=np.float64),
            initial_balance=initial_balance,
            max_points=curve_points,
        )
        candidate_results.append(
            TradeforwardCandidateEvaluation(
                candidate_id=allocation.candidate_id,
                display_name=allocation.display_name,
                cluster_id=allocation.cluster_id,
                checkpoint_path=checkpoint_path,
                capital_fraction=allocation.capital_fraction,
                normalized_share=allocation.normalized_share,
                capped=allocation.capped,
                cap_reason=allocation.cap_reason,
                final_balance=trace.metrics.final_balance,
                pnl=trace.metrics.pnl,
                max_drawdown_pct=trace.metrics.max_drawdown_pct,
                trades=trace.metrics.trades,
                wins=trace.metrics.wins,
                win_rate_pct=trace.metrics.win_rate_pct,
                curve_sample_indices=curve_sample_indices,
                normalized_balance_history=normalized_curve,
            )
        )

    if not initial_balances:
        raise ValueError("Tradeforward plan has no allocations")
    reference_initial_balance = initial_balances[0]
    for initial_balance in initial_balances[1:]:
        if abs(initial_balance - reference_initial_balance) > 1e-9:
            raise ValueError("All tradeforward allocations must use the same initial_balance for portfolio evaluation")

    step_return_matrix = np.column_stack(
        [step_returns_from_trace(trace, target_steps=plan.forward_max_steps) for trace in traces]
    )
    capital_fractions = np.asarray(weights, dtype=np.float64)
    portfolio_equity = portfolio_equity_from_step_returns(
        step_return_matrix,
        capital_fractions=capital_fractions,
        initial_balance=reference_initial_balance,
    )
    final_balance, pnl, max_drawdown_pct = equity_summary(
        portfolio_equity,
        initial_balance=reference_initial_balance,
    )
    curve_sample_indices, normalized_curve = downsample_equity_curve(
        portfolio_equity,
        initial_balance=reference_initial_balance,
        max_points=curve_points,
    )
    expectation_payload = None if expectation is None else TradeforwardExpectation(
        selection_mode=expectation.selection_mode,
        scenario_name=expectation.scenario_name,
        objective_score=expectation.objective_score,
        expected_original_net_profit=expectation.expected_original_net_profit,
        expected_original_max_drawdown_pct=expectation.expected_original_max_drawdown_pct,
        expected_p05_net_profit=expectation.expected_p05_net_profit,
        expected_p95_max_drawdown_pct=expectation.expected_p95_max_drawdown_pct,
    )
    portfolio = TradeforwardPortfolioEvaluation(
        initial_balance=reference_initial_balance,
        final_balance=final_balance,
        pnl=pnl,
        max_drawdown_pct=max_drawdown_pct,
        requested_risk_fraction=plan.requested_risk_fraction,
        allocated_risk_fraction=plan.allocated_risk_fraction,
        reserve_fraction=plan.reserve_fraction,
        component_trade_count_total=sum(item.trades for item in candidate_results),
        component_win_count_total=sum(item.wins for item in candidate_results),
        curve_sample_indices=curve_sample_indices,
        normalized_balance_history=normalized_curve,
        actual_minus_expected_original_net_profit=None
        if expectation is None
        else pnl - expectation.expected_original_net_profit,
        actual_minus_expected_original_max_drawdown_pct=None
        if expectation is None
        else max_drawdown_pct - expectation.expected_original_max_drawdown_pct,
        actual_minus_expected_p05_net_profit=None
        if expectation is None
        else pnl - expectation.expected_p05_net_profit,
        actual_minus_expected_p95_max_drawdown_pct=None
        if expectation is None
        else max_drawdown_pct - expectation.expected_p95_max_drawdown_pct,
    )
    return TradeforwardEvaluationReport(
        schema_version="1",
        name=name,
        created_at_utc=utc_now_text(),
        source_plan=plan.name,
        selection_mode=plan.selection_mode,
        scenario_name=plan.scenario_name,
        data_path=plan.data_path,
        forward_start_utc=plan.forward_start_utc,
        forward_end_utc=plan.forward_end_utc,
        forward_start_step=plan.forward_start_step,
        forward_max_steps=plan.forward_max_steps,
        candidate_ids=list(plan.candidate_ids),
        expectation=expectation_payload,
        portfolio=portfolio,
        candidate_evaluations=candidate_results,
        source_cycle_report=plan.source_cycle_report,
        source_allocator_report=plan.source_allocator_report,
        source_combination_report=plan.source_combination_report,
        source_cluster_report=plan.source_cluster_report,
        notes=notes,
    )
