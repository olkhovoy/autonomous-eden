from __future__ import annotations

from umc_nn.candidates import (
    CandidateRecord,
    CandidateRegistry,
    ExperimentManifest,
    PeriodStats,
    PortfolioBaselineReport,
    PortfolioCycleLedgerEntry,
    PortfolioGateConfig,
    PortfolioLedgerReport,
    PortfolioBaselineConfig,
    RollingConveyorReport,
    RollingCycleOutcome,
    TradeforwardAllocation,
    TradeforwardPlan,
    build_portfolio_baseline_report,
    utc_now_text,
)
from umc_nn.trading_eval import EpisodeMetrics, EpisodeTrace


def _manifest(checkpoint_path: str) -> ExperimentManifest:
    return ExperimentManifest(
        schema_version="1",
        created_at_utc=utc_now_text(),
        source_script="/tmp/source.py",
        engine_name="monolith",
        engine_role="candidate_engine",
        representation_name="fused32",
        data_path="/tmp/data.npz",
        checkpoint_path=checkpoint_path,
        log_path=None,
        source_summary_path=None,
        train_window_utc={"start": "2025-05-01 00:00:00", "end": "2025-05-08 00:00:00"},
        oos_window_utc={"start": "2025-05-08 00:00:00", "end": "2025-05-15 00:00:00"},
        search_config={},
        econ_config={
            "initial_balance": 10000.0,
            "exchange": "binance",
            "maker_fee_rate": None,
            "taker_fee_rate": None,
            "execution_fee_mode": "taker",
            "slippage": 0.0,
            "position_sizing_mode": "fraction_of_equity",
            "position_notional_fraction": 1.0,
            "leverage": 1.0,
            "fixed_position_qty": 1.0,
        },
    )


def _period(name: str, pnl: float) -> PeriodStats:
    return PeriodStats(
        name=name,
        start_step=0,
        requested_max_steps=100,
        steps_run=100,
        final_balance=10000.0 + pnl,
        pnl=pnl,
        max_drawdown_pct=2.0,
        trades=10,
        wins=6,
        win_rate_pct=60.0,
        action_counts={"1": 100},
        position_counts={"1": 101},
    )


def _candidate(candidate_id: str, display_name: str, oos_pnl: float) -> CandidateRecord:
    return CandidateRecord(
        schema_version="1",
        candidate_id=candidate_id,
        display_name=display_name,
        engine_name="monolith",
        engine_role="candidate_engine",
        status="research",
        created_at_utc=utc_now_text(),
        tags=["test"],
        manifest=_manifest(f"/tmp/{candidate_id}.npy"),
        periods={"train": _period("train", oos_pnl / 2.0), "oos": _period("oos", oos_pnl)},
    )


def _plan(name: str, cycle_name: str, candidate_ids: list[str], allocations: list[tuple[str, float]]) -> TradeforwardPlan:
    return TradeforwardPlan(
        schema_version="1",
        name=name,
        created_at_utc=utc_now_text(),
        selection_mode="combination",
        scenario_name=f"{cycle_name}_scenario",
        data_path="/tmp/data.npz",
        forward_start_utc="2025-05-08 00:00:00",
        forward_end_utc="2025-05-15 00:00:00",
        forward_start_step=100 if cycle_name == "cycle_01" else 200,
        forward_max_steps=2,
        requested_risk_fraction=sum(weight for _, weight in allocations),
        allocated_risk_fraction=sum(weight for _, weight in allocations),
        reserve_fraction=1.0 - sum(weight for _, weight in allocations),
        candidate_ids=candidate_ids,
        allocations=[
            TradeforwardAllocation(
                candidate_id=candidate_id,
                display_name=candidate_id.upper(),
                checkpoint_path=f"/tmp/{candidate_id}.npy",
                engine_name="monolith",
                representation_name="fused32",
                status="research",
                cluster_id=None,
                normalized_share=1.0,
                capital_fraction=weight,
                capped=False,
                cap_reason=None,
            )
            for candidate_id, weight in allocations
        ],
        source_cycle_report=cycle_name,
        source_combination_report=f"{cycle_name}_combination",
    )


def _episode_trace(policy: str, final_balance: float, balance_history: list[float]) -> EpisodeTrace:
    return EpisodeTrace(
        metrics=EpisodeMetrics(
            policy=policy,
            start_step=0,
            requested_max_steps=len(balance_history) - 1,
            steps_run=len(balance_history) - 1,
            final_balance=final_balance,
            pnl=final_balance - 10000.0,
            max_drawdown_pct=0.0,
            trades=1,
            wins=1,
            win_rate_pct=100.0,
            action_counts={1: len(balance_history) - 1},
            position_counts={1: len(balance_history)},
        ),
        trades=[],
        balance_history=balance_history,
        action_history=[1] * (len(balance_history) - 1),
        position_history=[0] + [1] * (len(balance_history) - 1),
    )


def _rolling_report() -> RollingConveyorReport:
    return RollingConveyorReport(
        schema_version="1",
        name="rolling_baseline_test",
        created_at_utc=utc_now_text(),
        mode="reuse",
        selection_days=7,
        forward_days=7,
        step_days=7,
        initial_balance=10000.0,
        final_balance=11550.0,
        total_pnl=1550.0,
        total_return_pct=15.5,
        max_drawdown_pct=2.5,
        positive_cycle_count=2,
        evaluated_cycle_count=2,
        ledger_cycle_indices=[0, 1, 2],
        ledger_balance_history=[10000.0, 11000.0, 11550.0],
        cycle_outcomes=[
            RollingCycleOutcome(
                cycle_index=1,
                cycle_name="cycle_01",
                cycle_report_name="cycle_01",
                tradeforward_plan_name="cycle_01_plan",
                tradeforward_evaluation_name="cycle_01_eval",
                selection_start_utc="2025-05-01 00:00:00",
                selection_end_utc="2025-05-08 00:00:00",
                forward_start_utc="2025-05-08 00:00:00",
                forward_end_utc="2025-05-15 00:00:00",
                candidate_ids=["cand_a", "cand_b"],
                selected_candidate_ids=["cand_a", "cand_b"],
                requested_risk_fraction=0.5,
                allocated_risk_fraction=0.5,
                reserve_fraction=0.5,
                portfolio_pnl=1000.0,
                portfolio_final_balance=11000.0,
                portfolio_max_drawdown_pct=2.5,
                cycle_return_fraction=0.1,
                ledger_balance_before=10000.0,
                ledger_balance_after=11000.0,
            ),
            RollingCycleOutcome(
                cycle_index=2,
                cycle_name="cycle_02",
                cycle_report_name="cycle_02",
                tradeforward_plan_name="cycle_02_plan",
                tradeforward_evaluation_name="cycle_02_eval",
                selection_start_utc="2025-05-08 00:00:00",
                selection_end_utc="2025-05-15 00:00:00",
                forward_start_utc="2025-05-15 00:00:00",
                forward_end_utc="2025-05-22 00:00:00",
                candidate_ids=["cand_a", "cand_c"],
                selected_candidate_ids=["cand_c"],
                requested_risk_fraction=0.5,
                allocated_risk_fraction=0.5,
                reserve_fraction=0.5,
                portfolio_pnl=550.0,
                portfolio_final_balance=11550.0,
                portfolio_max_drawdown_pct=2.5,
                cycle_return_fraction=0.05,
                ledger_balance_before=11000.0,
                ledger_balance_after=11550.0,
            ),
        ],
        selector={"status": "research"},
    )


def _portfolio_ledger_report() -> PortfolioLedgerReport:
    return PortfolioLedgerReport(
        schema_version="1",
        name="portfolio_baseline_test",
        created_at_utc=utc_now_text(),
        source_rolling_report="rolling_baseline_test",
        source_lifecycle_report="lifecycle_test",
        config={"turnover_cost_rate": 0.0},
        initial_balance=10000.0,
        final_balance=11550.0,
        total_pnl=1550.0,
        total_return_pct=15.5,
        max_drawdown_pct=2.5,
        ledger_cycle_indices=[0, 1, 2],
        ledger_balance_history=[10000.0, 11000.0, 11550.0],
        total_buy_turnover_fraction=0.5,
        total_sell_turnover_fraction=0.0,
        total_gross_turnover_fraction=0.5,
        average_gross_turnover_fraction=0.25,
        total_estimated_rebalance_cost=0.0,
        average_allocated_risk_fraction=0.5,
        average_reserve_fraction=0.5,
        total_churn_count=1,
        peak_cluster_exposure_fraction=0.5,
        non_tradable_selection_count=0,
        final_active_candidate_ids=["cand_c"],
        final_status_counts={"approved": 1, "paper": 1},
        candidate_summaries=[],
        cycle_entries=[
            PortfolioCycleLedgerEntry(
                cycle_index=1,
                cycle_name="cycle_01",
                selection_start_utc="2025-05-01 00:00:00",
                selection_end_utc="2025-05-08 00:00:00",
                forward_start_utc="2025-05-08 00:00:00",
                forward_end_utc="2025-05-15 00:00:00",
                tradeforward_plan_name="cycle_01_plan",
                tradeforward_evaluation_name="cycle_01_eval",
                ledger_balance_before=10000.0,
                ledger_balance_after_rebalance=10000.0,
                ledger_balance_after_cycle=11000.0,
                gross_cycle_pnl=1000.0,
                net_cycle_pnl=1000.0,
                portfolio_max_drawdown_pct=2.5,
                requested_risk_fraction=0.5,
                allocated_risk_fraction=0.5,
                reserve_fraction_before=1.0,
                reserve_fraction_after=0.5,
                buy_turnover_fraction=0.5,
                sell_turnover_fraction=0.0,
                gross_turnover_fraction=0.5,
                estimated_rebalance_cost=0.0,
                previous_active_candidate_ids=[],
                target_candidate_ids=["cand_a", "cand_b"],
                added_candidate_ids=["cand_a", "cand_b"],
                removed_candidate_ids=[],
                increased_candidate_ids=[],
                decreased_candidate_ids=[],
                unchanged_candidate_ids=[],
                non_tradable_selected_candidate_ids=[],
                status_counts_before={"research": 3},
                status_counts_after={"approved": 2, "research": 1},
                cluster_exposure_before={},
                cluster_exposure_after={},
                rebalance_changes=[],
            ),
            PortfolioCycleLedgerEntry(
                cycle_index=2,
                cycle_name="cycle_02",
                selection_start_utc="2025-05-08 00:00:00",
                selection_end_utc="2025-05-15 00:00:00",
                forward_start_utc="2025-05-15 00:00:00",
                forward_end_utc="2025-05-22 00:00:00",
                tradeforward_plan_name="cycle_02_plan",
                tradeforward_evaluation_name="cycle_02_eval",
                ledger_balance_before=11000.0,
                ledger_balance_after_rebalance=11000.0,
                ledger_balance_after_cycle=11550.0,
                gross_cycle_pnl=550.0,
                net_cycle_pnl=550.0,
                portfolio_max_drawdown_pct=2.5,
                requested_risk_fraction=0.5,
                allocated_risk_fraction=0.5,
                reserve_fraction_before=0.5,
                reserve_fraction_after=0.5,
                buy_turnover_fraction=0.0,
                sell_turnover_fraction=0.0,
                gross_turnover_fraction=0.0,
                estimated_rebalance_cost=0.0,
                previous_active_candidate_ids=["cand_a", "cand_b"],
                target_candidate_ids=["cand_c"],
                added_candidate_ids=["cand_c"],
                removed_candidate_ids=["cand_a", "cand_b"],
                increased_candidate_ids=[],
                decreased_candidate_ids=[],
                unchanged_candidate_ids=[],
                non_tradable_selected_candidate_ids=[],
                status_counts_before={"approved": 2, "research": 1},
                status_counts_after={"paper": 1, "approved": 1, "research": 1},
                cluster_exposure_before={},
                cluster_exposure_after={},
                rebalance_changes=[],
            ),
        ],
    )


def test_build_portfolio_baseline_report(monkeypatch, tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    registry.add_candidate(_candidate("cand_a", "A", 100.0))
    registry.add_candidate(_candidate("cand_b", "B", 200.0))
    registry.add_candidate(_candidate("cand_c", "C", 150.0))
    registry.save_rolling_conveyor(_rolling_report())
    registry.save_tradeforward_plan(_plan("cycle_01_plan", "cycle_01", ["cand_a", "cand_b"], [("cand_a", 0.25), ("cand_b", 0.25)]))
    registry.save_tradeforward_plan(_plan("cycle_02_plan", "cycle_02", ["cand_a", "cand_c"], [("cand_c", 0.5)]))

    traces = {
        ("long", 100): _episode_trace("long", 11000.0, [10000.0, 10500.0, 11000.0]),
        ("long", 200): _episode_trace("long", 10000.0, [10000.0, 10000.0, 10000.0]),
        ("/tmp/cand_a.npy", 100): _episode_trace("monolith", 12000.0, [10000.0, 11000.0, 12000.0]),
        ("/tmp/cand_b.npy", 100): _episode_trace("monolith", 10000.0, [10000.0, 10000.0, 10000.0]),
        ("/tmp/cand_b.npy", 200): _episode_trace("monolith", 11500.0, [10000.0, 10750.0, 11500.0]),
        ("/tmp/cand_c.npy", 200): _episode_trace("monolith", 10500.0, [10000.0, 10250.0, 10500.0]),
    }

    def _fake_eval(policy_name, data_path, **kwargs):
        del data_path
        key = (policy_name if policy_name == "long" else str(kwargs["weights_path"]), int(kwargs["start_step"]))
        return traces[key]

    monkeypatch.setattr("umc_nn.candidates.portfolio_baselines.evaluate_policy_trace_path", _fake_eval)

    report = build_portfolio_baseline_report(
        registry,
        "baseline_test",
        portfolio_ledger_report=_portfolio_ledger_report(),
        config=PortfolioBaselineConfig(curve_points=32),
        gate_config=PortfolioGateConfig(min_total_pnl=0.0, max_drawdown_pct=10.0, min_baselines_beaten=2),
    )

    comparisons = {item.baseline_name: item for item in report.comparisons}
    assert len(report.baselines) == 5
    assert comparisons["flat"].baseline_total_pnl == 0.0
    assert round(comparisons["long"].baseline_total_pnl, 2) == 494.05
    assert round(comparisons["equal_weight_selected_subset"].baseline_total_pnl, 2) == 743.43
    assert round(comparisons["single_best_candidate"].baseline_total_pnl, 2) == 736.92
    assert round(comparisons["naive_top_oos_rotation"].baseline_total_pnl, 2) == 248.48
    assert report.gate.overall_pass is True
    assert "flat" in report.gate.beaten_baselines
    assert "equal_weight_selected_subset" in report.gate.beaten_baselines

    path = registry.save_portfolio_baselines(report)
    assert path.exists()
    loaded = registry.load_portfolio_baselines("baseline_test")
    assert isinstance(loaded, PortfolioBaselineReport)
    assert loaded.gate.overall_pass is True
