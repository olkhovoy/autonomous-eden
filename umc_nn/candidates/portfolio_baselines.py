from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from umc_nn.trading_eval import EpisodeTrace, evaluate_policy_trace_path

from .allocator import downsample_equity_curve, equity_summary, portfolio_equity_from_step_returns, step_returns_from_trace
from .engine_config import candidate_engine_config
from .portfolio_ledger import _stitched_segment
from .registry import CandidateRegistry
from .schema import (
    CandidateRecord,
    PortfolioBaselineComparison,
    PortfolioBaselineCycleResult,
    PortfolioBaselineReport,
    PortfolioBaselineResult,
    PortfolioGateResult,
    PortfolioLedgerReport,
    RollingConveyorReport,
    TradeforwardPlan,
    utc_now_text,
)


@dataclass(slots=True, frozen=True)
class PortfolioBaselineConfig:
    curve_points: int = 256
    turnover_cost_rate: float | None = None
    fixed_candidate_metric: str = "oos_pnl"
    rotation_metric: str = "oos_pnl"

    def validate(self) -> None:
        if self.curve_points <= 1:
            raise ValueError("curve_points must be > 1")
        if self.turnover_cost_rate is not None and self.turnover_cost_rate < 0.0:
            raise ValueError("turnover_cost_rate must be >= 0")
        if self.fixed_candidate_metric != "oos_pnl":
            raise ValueError("Only fixed_candidate_metric='oos_pnl' is currently supported")
        if self.rotation_metric != "oos_pnl":
            raise ValueError("Only rotation_metric='oos_pnl' is currently supported")

    def to_dict(self) -> dict[str, object]:
        return {
            "curve_points": self.curve_points,
            "turnover_cost_rate": self.turnover_cost_rate,
            "fixed_candidate_metric": self.fixed_candidate_metric,
            "rotation_metric": self.rotation_metric,
        }


@dataclass(slots=True, frozen=True)
class PortfolioGateConfig:
    min_total_pnl: float = 0.0
    max_drawdown_pct: float = 15.0
    required_baselines: tuple[str, ...] = ("flat", "equal_weight_selected_subset")
    min_baselines_beaten: int = 2

    def validate(self) -> None:
        if self.max_drawdown_pct <= 0.0:
            raise ValueError("max_drawdown_pct must be > 0")
        if self.min_baselines_beaten < 0:
            raise ValueError("min_baselines_beaten must be >= 0")

    def to_dict(self) -> dict[str, object]:
        return {
            "min_total_pnl": self.min_total_pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "required_baselines": list(self.required_baselines),
            "min_baselines_beaten": self.min_baselines_beaten,
        }


class _TraceCache:
    def __init__(self, registry: CandidateRegistry):
        self.registry = registry
        self.cache: dict[tuple[str, str, str], EpisodeTrace] = {}

    def candidate_trace(self, candidate_id: str, plan: TradeforwardPlan) -> EpisodeTrace:
        key = ("candidate", candidate_id, plan.name)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        candidate = self.registry.load_candidate(candidate_id)
        manifest = candidate.manifest
        if manifest is None:
            raise ValueError(f"Candidate {candidate_id} has no manifest")
        econ = manifest.econ_config
        trace = evaluate_policy_trace_path(
            "monolith",
            plan.data_path,
            use_neurobars=str(plan.data_path).endswith(".npz"),
            start_step=int(plan.forward_start_step or 0),
            max_steps=int(plan.forward_max_steps or 0),
            initial_balance=float(econ.get("initial_balance", 10000.0)),
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
        self.cache[key] = trace
        return trace

    def long_trace(self, reference_candidate_id: str, plan: TradeforwardPlan) -> EpisodeTrace:
        key = ("long", reference_candidate_id, plan.name)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        candidate = self.registry.load_candidate(reference_candidate_id)
        manifest = candidate.manifest
        if manifest is None:
            raise ValueError(f"Candidate {reference_candidate_id} has no manifest")
        econ = manifest.econ_config
        trace = evaluate_policy_trace_path(
            "long",
            plan.data_path,
            use_neurobars=str(plan.data_path).endswith(".npz"),
            start_step=int(plan.forward_start_step or 0),
            max_steps=int(plan.forward_max_steps or 0),
            initial_balance=float(econ.get("initial_balance", 10000.0)),
            exchange=str(econ["exchange"]),
            maker_fee_rate=econ.get("maker_fee_rate"),
            taker_fee_rate=econ.get("taker_fee_rate"),
            execution_fee_mode=str(econ["execution_fee_mode"]),
            slippage=float(econ["slippage"]),
            position_sizing_mode=str(econ["position_sizing_mode"]),
            position_notional_fraction=float(econ["position_notional_fraction"]),
            leverage=float(econ["leverage"]),
            fixed_position_qty=float(econ["fixed_position_qty"]),
        )
        self.cache[key] = trace
        return trace


def _candidate_score(candidate: CandidateRecord, metric: str) -> tuple[float, float, str]:
    if metric != "oos_pnl":
        raise ValueError(f"Unsupported candidate metric: {metric}")
    oos = candidate.periods.get("oos_adjacent") or candidate.periods.get("oos")
    train = candidate.periods.get("train")
    oos_pnl = float("-inf") if oos is None else float(oos.pnl)
    train_pnl = float("-inf") if train is None else float(train.pnl)
    return (oos_pnl, train_pnl, candidate.candidate_id)


def _pick_fixed_best_candidate(
    registry: CandidateRegistry,
    candidate_ids: list[str],
    metric: str,
) -> CandidateRecord:
    candidates = [registry.load_candidate(candidate_id) for candidate_id in candidate_ids]
    candidates.sort(key=lambda item: _candidate_score(item, metric), reverse=True)
    return candidates[0]


def _pick_rotation_candidate(
    registry: CandidateRegistry,
    candidate_ids: list[str],
    metric: str,
) -> CandidateRecord:
    return _pick_fixed_best_candidate(registry, candidate_ids, metric)


def _simulate_baseline(
    registry: CandidateRegistry,
    *,
    baseline_name: str,
    baseline_kind: str,
    description: str,
    portfolio_ledger_report: PortfolioLedgerReport,
    rolling_report: RollingConveyorReport,
    config: PortfolioBaselineConfig,
    selector_fn: Callable[[int, TradeforwardPlan, list[str]], tuple[dict[str, float], dict[str, object]]],
) -> PortfolioBaselineResult:
    turnover_cost_rate = (
        float(portfolio_ledger_report.config.get("turnover_cost_rate", 0.0))
        if config.turnover_cost_rate is None
        else float(config.turnover_cost_rate)
    )
    trace_cache = _TraceCache(registry)

    rolling_by_cycle = {item.cycle_index: item for item in rolling_report.cycle_outcomes}
    current_balance = float(portfolio_ledger_report.initial_balance)
    stitched_curve: list[float] = [current_balance]
    previous_allocations: dict[str, float] = {}
    total_buy_turnover = 0.0
    total_sell_turnover = 0.0
    total_gross_turnover = 0.0
    total_estimated_cost = 0.0
    reserve_sum = 0.0
    cycle_results: list[PortfolioBaselineCycleResult] = []
    selector_summary: dict[str, object] = {}

    for entry in portfolio_ledger_report.cycle_entries:
        cycle = rolling_by_cycle[entry.cycle_index]
        plan = registry.load_tradeforward_plan(entry.tradeforward_plan_name)
        selected_candidate_ids_for_cycle = list(cycle.candidate_ids)
        allocation_map, cycle_selector_payload = selector_fn(entry.cycle_index, plan, selected_candidate_ids_for_cycle)
        selector_summary.update(cycle_selector_payload)

        allocation_map = {candidate_id: float(value) for candidate_id, value in allocation_map.items() if value > 1e-12}
        selected_candidate_ids = sorted(allocation_map.keys())
        selected_display_names = [
            "long"
            if candidate_id == "__long__"
            else registry.load_candidate(candidate_id).display_name
            for candidate_id in selected_candidate_ids
        ]

        reserve_before = max(0.0, 1.0 - sum(previous_allocations.values()))
        reserve_after = max(0.0, 1.0 - sum(allocation_map.values()))
        union_candidate_ids = sorted(set(previous_allocations).union(allocation_map))
        buy_turnover = 0.0
        sell_turnover = 0.0
        for candidate_id in union_candidate_ids:
            delta = float(allocation_map.get(candidate_id, 0.0)) - float(previous_allocations.get(candidate_id, 0.0))
            buy_turnover += max(delta, 0.0)
            sell_turnover += max(-delta, 0.0)
        gross_turnover = buy_turnover + sell_turnover
        estimated_cost = current_balance * gross_turnover * turnover_cost_rate
        balance_after_rebalance = current_balance - estimated_cost

        if not allocation_map:
            equity = np.asarray([balance_after_rebalance, balance_after_rebalance], dtype=np.float64)
        else:
            traces: list[EpisodeTrace] = []
            weights: list[float] = []
            reference_candidate_id = next(iter(plan.candidate_ids or cycle.candidate_ids))
            for candidate_id, capital_fraction in allocation_map.items():
                if candidate_id == "__long__":
                    trace = trace_cache.long_trace(reference_candidate_id, plan)
                else:
                    trace = trace_cache.candidate_trace(candidate_id, plan)
                traces.append(trace)
                weights.append(capital_fraction)
            step_return_matrix = np.column_stack(
                [step_returns_from_trace(trace, target_steps=int(plan.forward_max_steps or 0)) for trace in traces]
            )
            equity = portfolio_equity_from_step_returns(
                step_return_matrix,
                capital_fractions=np.asarray(weights, dtype=np.float64),
                initial_balance=balance_after_rebalance,
            )

        final_balance, total_pnl, max_drawdown_pct = equity_summary(equity, initial_balance=balance_after_rebalance)
        gross_cycle_pnl = total_pnl + estimated_cost
        segment = equity
        if segment.size:
            start_index = 0 if abs(segment[0] - stitched_curve[-1]) > 1e-9 else 1
            stitched_curve.extend(float(value) for value in segment[start_index:])

        cycle_results.append(
            PortfolioBaselineCycleResult(
                baseline_name=baseline_name,
                cycle_index=entry.cycle_index,
                cycle_name=entry.cycle_name,
                selected_candidate_ids=selected_candidate_ids,
                selected_display_names=selected_display_names,
                ledger_balance_before=current_balance,
                ledger_balance_after_rebalance=balance_after_rebalance,
                ledger_balance_after_cycle=final_balance,
                gross_cycle_pnl=gross_cycle_pnl,
                net_cycle_pnl=total_pnl,
                max_drawdown_pct=max_drawdown_pct,
                allocated_risk_fraction=float(sum(allocation_map.values())),
                reserve_fraction_before=reserve_before,
                reserve_fraction_after=reserve_after,
                buy_turnover_fraction=buy_turnover,
                sell_turnover_fraction=sell_turnover,
                gross_turnover_fraction=gross_turnover,
                estimated_rebalance_cost=estimated_cost,
                allocation_map=allocation_map,
            )
        )

        current_balance = final_balance
        previous_allocations = allocation_map
        total_buy_turnover += buy_turnover
        total_sell_turnover += sell_turnover
        total_gross_turnover += gross_turnover
        total_estimated_cost += estimated_cost
        reserve_sum += reserve_after

    stitched_curve_np = np.asarray(stitched_curve, dtype=np.float64)
    final_balance, total_pnl, max_drawdown_pct = equity_summary(
        stitched_curve_np,
        initial_balance=float(portfolio_ledger_report.initial_balance),
    )
    curve_sample_indices, normalized_balance_history = downsample_equity_curve(
        stitched_curve_np,
        initial_balance=float(portfolio_ledger_report.initial_balance),
        max_points=config.curve_points,
    )
    return PortfolioBaselineResult(
        baseline_name=baseline_name,
        baseline_kind=baseline_kind,
        description=description,
        selector_summary=selector_summary,
        initial_balance=float(portfolio_ledger_report.initial_balance),
        final_balance=final_balance,
        total_pnl=total_pnl,
        total_return_pct=((final_balance / float(portfolio_ledger_report.initial_balance)) - 1.0) * 100.0,
        max_drawdown_pct=max_drawdown_pct,
        total_buy_turnover_fraction=total_buy_turnover,
        total_sell_turnover_fraction=total_sell_turnover,
        total_gross_turnover_fraction=total_gross_turnover,
        average_reserve_fraction=reserve_sum / len(cycle_results) if cycle_results else 0.0,
        total_estimated_rebalance_cost=total_estimated_cost,
        curve_sample_indices=curve_sample_indices,
        normalized_balance_history=normalized_balance_history,
        cycle_results=cycle_results,
    )


def _build_flat_baseline(
    registry: CandidateRegistry,
    portfolio_ledger_report: PortfolioLedgerReport,
    rolling_report: RollingConveyorReport,
    config: PortfolioBaselineConfig,
) -> PortfolioBaselineResult:
    def _selector(cycle_index: int, plan: TradeforwardPlan, candidate_ids: list[str]) -> tuple[dict[str, float], dict[str, object]]:
        del cycle_index, plan, candidate_ids
        return {}, {}

    return _simulate_baseline(
        registry,
        baseline_name="flat",
        baseline_kind="constant_policy",
        description="Stay fully in cash across all conveyor cycles.",
        portfolio_ledger_report=portfolio_ledger_report,
        rolling_report=rolling_report,
        config=config,
        selector_fn=_selector,
    )


def _build_long_baseline(
    registry: CandidateRegistry,
    portfolio_ledger_report: PortfolioLedgerReport,
    rolling_report: RollingConveyorReport,
    config: PortfolioBaselineConfig,
) -> PortfolioBaselineResult:
    cycle_risk = {entry.cycle_index: entry.allocated_risk_fraction for entry in portfolio_ledger_report.cycle_entries}

    def _selector(cycle_index: int, plan: TradeforwardPlan, candidate_ids: list[str]) -> tuple[dict[str, float], dict[str, object]]:
        del plan, candidate_ids
        return {"__long__": float(cycle_risk[cycle_index])}, {"risk_mode": "matched_allocated_risk"}

    return _simulate_baseline(
        registry,
        baseline_name="long",
        baseline_kind="constant_policy",
        description="Always long with matched allocated risk per cycle.",
        portfolio_ledger_report=portfolio_ledger_report,
        rolling_report=rolling_report,
        config=config,
        selector_fn=_selector,
    )


def _build_equal_weight_selected_subset_baseline(
    registry: CandidateRegistry,
    portfolio_ledger_report: PortfolioLedgerReport,
    rolling_report: RollingConveyorReport,
    config: PortfolioBaselineConfig,
) -> PortfolioBaselineResult:
    cycle_entries = {entry.cycle_index: entry for entry in portfolio_ledger_report.cycle_entries}

    def _selector(cycle_index: int, plan: TradeforwardPlan, candidate_ids: list[str]) -> tuple[dict[str, float], dict[str, object]]:
        del candidate_ids
        target_ids = list(cycle_entries[cycle_index].target_candidate_ids)
        if not target_ids:
            return {}, {}
        weight = float(cycle_entries[cycle_index].allocated_risk_fraction) / len(target_ids)
        return ({candidate_id: weight for candidate_id in target_ids}, {"risk_mode": "matched_allocated_risk"})

    return _simulate_baseline(
        registry,
        baseline_name="equal_weight_selected_subset",
        baseline_kind="naive_allocator",
        description="Use the same selected subset as the conveyor but equal-weight it each cycle.",
        portfolio_ledger_report=portfolio_ledger_report,
        rolling_report=rolling_report,
        config=config,
        selector_fn=_selector,
    )


def _build_single_best_candidate_baseline(
    registry: CandidateRegistry,
    portfolio_ledger_report: PortfolioLedgerReport,
    rolling_report: RollingConveyorReport,
    config: PortfolioBaselineConfig,
) -> PortfolioBaselineResult:
    if portfolio_ledger_report.candidate_summaries:
        candidate_ids = [item.candidate_id for item in portfolio_ledger_report.candidate_summaries]
    else:
        candidate_ids = sorted({candidate_id for cycle in rolling_report.cycle_outcomes for candidate_id in cycle.candidate_ids})
    best_candidate = _pick_fixed_best_candidate(registry, candidate_ids, config.fixed_candidate_metric)
    cycle_risk = {entry.cycle_index: entry.allocated_risk_fraction for entry in portfolio_ledger_report.cycle_entries}

    def _selector(cycle_index: int, plan: TradeforwardPlan, candidate_ids: list[str]) -> tuple[dict[str, float], dict[str, object]]:
        del plan, candidate_ids
        return (
            {best_candidate.candidate_id: float(cycle_risk[cycle_index])},
            {
                "metric": config.fixed_candidate_metric,
                "candidate_id": best_candidate.candidate_id,
                "display_name": best_candidate.display_name,
            },
        )

    return _simulate_baseline(
        registry,
        baseline_name="single_best_candidate",
        baseline_kind="fixed_candidate",
        description="Trade one fixed candidate chosen by best stored OOS PnL across the pool.",
        portfolio_ledger_report=portfolio_ledger_report,
        rolling_report=rolling_report,
        config=config,
        selector_fn=_selector,
    )


def _build_naive_top_oos_rotation_baseline(
    registry: CandidateRegistry,
    portfolio_ledger_report: PortfolioLedgerReport,
    rolling_report: RollingConveyorReport,
    config: PortfolioBaselineConfig,
) -> PortfolioBaselineResult:
    cycle_risk = {entry.cycle_index: entry.allocated_risk_fraction for entry in portfolio_ledger_report.cycle_entries}
    cycle_candidate_ids = {item.cycle_index: list(item.candidate_ids) for item in rolling_report.cycle_outcomes}

    def _selector(cycle_index: int, plan: TradeforwardPlan, candidate_ids: list[str]) -> tuple[dict[str, float], dict[str, object]]:
        del plan, candidate_ids
        candidate = _pick_rotation_candidate(registry, cycle_candidate_ids[cycle_index], config.rotation_metric)
        return (
            {candidate.candidate_id: float(cycle_risk[cycle_index])},
            {
                "metric": config.rotation_metric,
                "selected_candidate_id": candidate.candidate_id,
                "display_name": candidate.display_name,
            },
        )

    return _simulate_baseline(
        registry,
        baseline_name="naive_top_oos_rotation",
        baseline_kind="rotation",
        description="Rotate each cycle into the candidate with best stored OOS PnL in the reviewed pool.",
        portfolio_ledger_report=portfolio_ledger_report,
        rolling_report=rolling_report,
        config=config,
        selector_fn=_selector,
    )


def _build_comparisons(
    portfolio_ledger_report: PortfolioLedgerReport,
    baselines: list[PortfolioBaselineResult],
) -> list[PortfolioBaselineComparison]:
    comparisons: list[PortfolioBaselineComparison] = []
    for baseline in baselines:
        comparisons.append(
            PortfolioBaselineComparison(
                baseline_name=baseline.baseline_name,
                baseline_kind=baseline.baseline_kind,
                conveyor_total_pnl=portfolio_ledger_report.total_pnl,
                baseline_total_pnl=baseline.total_pnl,
                pnl_delta=portfolio_ledger_report.total_pnl - baseline.total_pnl,
                conveyor_total_return_pct=portfolio_ledger_report.total_return_pct,
                baseline_total_return_pct=baseline.total_return_pct,
                return_pct_delta=portfolio_ledger_report.total_return_pct - baseline.total_return_pct,
                conveyor_max_drawdown_pct=portfolio_ledger_report.max_drawdown_pct,
                baseline_max_drawdown_pct=baseline.max_drawdown_pct,
                drawdown_advantage_pct=baseline.max_drawdown_pct - portfolio_ledger_report.max_drawdown_pct,
                beats_by_pnl=portfolio_ledger_report.total_pnl > baseline.total_pnl,
            )
        )
    return comparisons


def _build_gate(
    portfolio_ledger_report: PortfolioLedgerReport,
    comparisons: list[PortfolioBaselineComparison],
    gate_config: PortfolioGateConfig,
) -> PortfolioGateResult:
    comparison_by_name = {item.baseline_name: item for item in comparisons}
    beaten_baselines = [item.baseline_name for item in comparisons if item.beats_by_pnl]
    failed_required = [
        name for name in gate_config.required_baselines if not comparison_by_name.get(name, None) or not comparison_by_name[name].beats_by_pnl
    ]
    checks = {
        "positive_total_pnl": portfolio_ledger_report.total_pnl >= gate_config.min_total_pnl,
        "max_drawdown_within_limit": portfolio_ledger_report.max_drawdown_pct <= gate_config.max_drawdown_pct,
        "required_baselines_beaten": not failed_required,
        "minimum_baselines_beaten": len(beaten_baselines) >= gate_config.min_baselines_beaten,
    }
    return PortfolioGateResult(
        overall_pass=all(checks.values()),
        config=gate_config.to_dict(),
        checks=checks,
        beaten_baselines=beaten_baselines,
        failed_required_baselines=failed_required,
    )


def build_portfolio_baseline_report(
    registry: CandidateRegistry,
    name: str,
    *,
    portfolio_ledger_report: PortfolioLedgerReport,
    config: PortfolioBaselineConfig,
    gate_config: PortfolioGateConfig,
    notes: str | None = None,
) -> PortfolioBaselineReport:
    config.validate()
    gate_config.validate()
    rolling_report = registry.load_rolling_conveyor(portfolio_ledger_report.source_rolling_report)

    baselines = [
        _build_flat_baseline(registry, portfolio_ledger_report, rolling_report, config),
        _build_long_baseline(registry, portfolio_ledger_report, rolling_report, config),
        _build_equal_weight_selected_subset_baseline(registry, portfolio_ledger_report, rolling_report, config),
        _build_single_best_candidate_baseline(registry, portfolio_ledger_report, rolling_report, config),
        _build_naive_top_oos_rotation_baseline(registry, portfolio_ledger_report, rolling_report, config),
    ]
    comparisons = _build_comparisons(portfolio_ledger_report, baselines)
    gate = _build_gate(portfolio_ledger_report, comparisons, gate_config)
    return PortfolioBaselineReport(
        schema_version="1",
        name=name,
        created_at_utc=utc_now_text(),
        source_portfolio_ledger_report=portfolio_ledger_report.name,
        source_rolling_report=portfolio_ledger_report.source_rolling_report,
        config={
            "baseline": config.to_dict(),
            "gate": gate_config.to_dict(),
        },
        conveyor_total_pnl=portfolio_ledger_report.total_pnl,
        conveyor_total_return_pct=portfolio_ledger_report.total_return_pct,
        conveyor_max_drawdown_pct=portfolio_ledger_report.max_drawdown_pct,
        baselines=baselines,
        comparisons=comparisons,
        gate=gate,
        notes=notes,
    )
