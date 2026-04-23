from __future__ import annotations

from umc_nn.candidates import (
    CandidateRecord,
    CandidateRegistry,
    ExperimentManifest,
    LifecycleCandidateSummary,
    LifecycleDecisionRecord,
    LifecycleReport,
    PortfolioLedgerConfig,
    PortfolioLedgerReport,
    RollingConveyorReport,
    RollingCycleOutcome,
    TradeforwardAllocation,
    TradeforwardCandidateEvaluation,
    TradeforwardEvaluationReport,
    TradeforwardPlan,
    TradeforwardPortfolioEvaluation,
    build_portfolio_ledger_report,
    utc_now_text,
)


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
        econ_config={"initial_balance": 10000.0, "exchange": "binance"},
    )


def _candidate(candidate_id: str, display_name: str, status: str) -> CandidateRecord:
    return CandidateRecord(
        schema_version="1",
        candidate_id=candidate_id,
        display_name=display_name,
        engine_name="monolith",
        engine_role="candidate_engine",
        status=status,
        created_at_utc=utc_now_text(),
        tags=["test"],
        manifest=_manifest(f"/tmp/{candidate_id}.npy"),
    )


def _plan(name: str, cycle_name: str, allocations: list[tuple[str, str, str | None, float]]) -> TradeforwardPlan:
    return TradeforwardPlan(
        schema_version="1",
        name=name,
        created_at_utc=utc_now_text(),
        selection_mode="combination",
        scenario_name=f"{cycle_name}_scenario",
        data_path="/tmp/data.npz",
        forward_start_utc="2025-05-08 00:00:00",
        forward_end_utc="2025-05-15 00:00:00",
        forward_start_step=0,
        forward_max_steps=100,
        requested_risk_fraction=sum(item[3] for item in allocations),
        allocated_risk_fraction=sum(item[3] for item in allocations),
        reserve_fraction=max(0.0, 1.0 - sum(item[3] for item in allocations)),
        candidate_ids=[item[0] for item in allocations],
        allocations=[
            TradeforwardAllocation(
                candidate_id=candidate_id,
                display_name=display_name,
                checkpoint_path=f"/tmp/{candidate_id}.npy",
                engine_name="monolith",
                representation_name="fused32",
                status="research",
                cluster_id=cluster_id,
                normalized_share=1.0,
                capital_fraction=capital_fraction,
                capped=False,
                cap_reason=None,
            )
            for candidate_id, display_name, cluster_id, capital_fraction in allocations
        ],
        source_cycle_report=cycle_name,
        source_combination_report=f"{cycle_name}_combination",
    )


def _candidate_eval(candidate_id: str, display_name: str, cluster_id: str | None, capital_fraction: float, pnl: float) -> TradeforwardCandidateEvaluation:
    return TradeforwardCandidateEvaluation(
        candidate_id=candidate_id,
        display_name=display_name,
        cluster_id=cluster_id,
        checkpoint_path=f"/tmp/{candidate_id}.npy",
        capital_fraction=capital_fraction,
        normalized_share=1.0,
        capped=False,
        cap_reason=None,
        final_balance=10000.0 + pnl,
        pnl=pnl,
        max_drawdown_pct=2.0,
        trades=2,
        wins=1,
        win_rate_pct=50.0,
        curve_sample_indices=[0, 1, 2],
        normalized_balance_history=[1.0, 1.02, 1.05],
    )


def _evaluation(name: str, cycle_name: str, final_balance: float, normalized_curve: list[float], candidate_evaluations: list[TradeforwardCandidateEvaluation]) -> TradeforwardEvaluationReport:
    return TradeforwardEvaluationReport(
        schema_version="1",
        name=name,
        created_at_utc=utc_now_text(),
        source_plan=f"{cycle_name}_plan",
        selection_mode="combination",
        scenario_name=f"{cycle_name}_scenario",
        data_path="/tmp/data.npz",
        forward_start_utc="2025-05-08 00:00:00",
        forward_end_utc="2025-05-15 00:00:00",
        forward_start_step=0,
        forward_max_steps=100,
        candidate_ids=[item.candidate_id for item in candidate_evaluations],
        expectation=None,
        portfolio=TradeforwardPortfolioEvaluation(
            initial_balance=10000.0,
            final_balance=final_balance,
            pnl=final_balance - 10000.0,
            max_drawdown_pct=2.5,
            requested_risk_fraction=0.0,
            allocated_risk_fraction=0.0,
            reserve_fraction=0.0,
            component_trade_count_total=sum(item.trades for item in candidate_evaluations),
            component_win_count_total=sum(item.wins for item in candidate_evaluations),
            curve_sample_indices=list(range(len(normalized_curve))),
            normalized_balance_history=normalized_curve,
        ),
        candidate_evaluations=candidate_evaluations,
        source_cycle_report=cycle_name,
        source_combination_report=f"{cycle_name}_combination",
    )


def _rolling_report() -> RollingConveyorReport:
    return RollingConveyorReport(
        schema_version="1",
        name="rolling_test",
        created_at_utc=utc_now_text(),
        mode="reuse",
        selection_days=7,
        forward_days=7,
        step_days=7,
        initial_balance=10000.0,
        final_balance=11550.0,
        total_pnl=1550.0,
        total_return_pct=15.5,
        max_drawdown_pct=0.0,
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
                candidate_ids=["cand_a", "cand_b", "cand_c"],
                selected_candidate_ids=["cand_a", "cand_b"],
                requested_risk_fraction=0.6,
                allocated_risk_fraction=0.6,
                reserve_fraction=0.4,
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
                candidate_ids=["cand_a", "cand_b", "cand_c"],
                selected_candidate_ids=["cand_b", "cand_c"],
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


def _lifecycle_report() -> LifecycleReport:
    return LifecycleReport(
        schema_version="1",
        name="lifecycle_test",
        created_at_utc=utc_now_text(),
        source_rolling_report="rolling_test",
        source_tradeforward_evaluations=["cycle_01_eval", "cycle_02_eval"],
        config={},
        candidate_ids=["cand_a", "cand_b", "cand_c"],
        applied_status_updates=False,
        final_status_counts={"draining": 1, "paper": 1, "approved": 1},
        candidate_summaries=[
            LifecycleCandidateSummary(
                candidate_id="cand_a",
                display_name="A",
                initial_status="research",
                final_status="draining",
                total_cycles_seen=2,
                selected_cycles=1,
                successful_forward_cycles=1,
                failed_forward_cycles=0,
                no_trade_cycles=0,
                idle_cycles=1,
                transition_count=2,
                last_selected_cycle_index=1,
                last_transition_cycle_index=2,
            ),
            LifecycleCandidateSummary(
                candidate_id="cand_b",
                display_name="B",
                initial_status="research",
                final_status="paper",
                total_cycles_seen=2,
                selected_cycles=2,
                successful_forward_cycles=2,
                failed_forward_cycles=0,
                no_trade_cycles=0,
                idle_cycles=0,
                transition_count=2,
                last_selected_cycle_index=2,
                last_transition_cycle_index=2,
            ),
            LifecycleCandidateSummary(
                candidate_id="cand_c",
                display_name="C",
                initial_status="research",
                final_status="approved",
                total_cycles_seen=2,
                selected_cycles=1,
                successful_forward_cycles=1,
                failed_forward_cycles=0,
                no_trade_cycles=0,
                idle_cycles=1,
                transition_count=1,
                last_selected_cycle_index=2,
                last_transition_cycle_index=2,
            ),
        ],
        decisions=[
            LifecycleDecisionRecord(1, "cycle_01", "cand_a", "A", True, "research", "approved", "promote", ["forward_success", "promote_to_approved"]),
            LifecycleDecisionRecord(1, "cycle_01", "cand_b", "B", True, "research", "approved", "promote", ["forward_success", "promote_to_approved"]),
            LifecycleDecisionRecord(1, "cycle_01", "cand_c", "C", False, "research", "research", "hold", ["idle_hold"]),
            LifecycleDecisionRecord(2, "cycle_02", "cand_a", "A", False, "approved", "draining", "drain", ["idle_drain_threshold"]),
            LifecycleDecisionRecord(2, "cycle_02", "cand_b", "B", True, "approved", "paper", "promote", ["forward_success", "promote_to_paper"]),
            LifecycleDecisionRecord(2, "cycle_02", "cand_c", "C", True, "research", "approved", "promote", ["forward_success", "promote_to_approved"]),
        ],
    )


def test_build_portfolio_ledger_report_tracks_turnover_and_rebalance(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    registry.add_candidate(_candidate("cand_a", "A", "research"))
    registry.add_candidate(_candidate("cand_b", "B", "research"))
    registry.add_candidate(_candidate("cand_c", "C", "research"))

    registry.save_tradeforward_plan(
        _plan(
            "cycle_01_plan",
            "cycle_01",
            [("cand_a", "A", "cluster_001", 0.4), ("cand_b", "B", "cluster_001", 0.2)],
        )
    )
    registry.save_tradeforward_plan(
        _plan(
            "cycle_02_plan",
            "cycle_02",
            [("cand_b", "B", "cluster_001", 0.1), ("cand_c", "C", "cluster_002", 0.4)],
        )
    )
    registry.save_tradeforward_evaluation(
        _evaluation(
            "cycle_01_eval",
            "cycle_01",
            11000.0,
            [1.0, 1.05, 1.10],
            [
                _candidate_eval("cand_a", "A", "cluster_001", 0.4, 1200.0),
                _candidate_eval("cand_b", "B", "cluster_001", 0.2, 800.0),
            ],
        )
    )
    registry.save_tradeforward_evaluation(
        _evaluation(
            "cycle_02_eval",
            "cycle_02",
            10500.0,
            [1.0, 1.02, 1.05],
            [
                _candidate_eval("cand_b", "B", "cluster_001", 0.1, 500.0),
                _candidate_eval("cand_c", "C", "cluster_002", 0.4, 700.0),
            ],
        )
    )

    report = build_portfolio_ledger_report(
        registry,
        "portfolio_test",
        rolling_report=_rolling_report(),
        lifecycle_report=_lifecycle_report(),
        config=PortfolioLedgerConfig(turnover_cost_rate=0.01),
    )

    assert round(report.final_balance, 4) == 11377.3737
    assert round(report.total_estimated_rebalance_cost, 3) == 158.406
    assert round(report.total_gross_turnover_fraction, 3) == 1.5
    assert round(report.average_gross_turnover_fraction, 3) == 0.75
    assert report.total_churn_count == 4
    assert report.non_tradable_selection_count == 3
    assert report.final_active_candidate_ids == ["cand_b", "cand_c"]
    assert report.final_status_counts == {"draining": 1, "paper": 1, "approved": 1}
    assert round(report.peak_cluster_exposure_fraction, 3) == 0.6

    cycle_1, cycle_2 = report.cycle_entries
    assert cycle_1.added_candidate_ids == ["cand_a", "cand_b"]
    assert cycle_1.non_tradable_selected_candidate_ids == ["cand_a", "cand_b"]
    assert cycle_2.removed_candidate_ids == ["cand_a"]
    assert cycle_2.added_candidate_ids == ["cand_c"]
    assert cycle_2.decreased_candidate_ids == ["cand_b"]
    assert cycle_2.non_tradable_selected_candidate_ids == ["cand_c"]

    summaries = {item.candidate_id: item for item in report.candidate_summaries}
    assert summaries["cand_a"].removed_cycles == 1
    assert summaries["cand_b"].selected_cycles == 2
    assert round(summaries["cand_b"].average_target_capital_fraction, 3) == 0.15
    assert summaries["cand_c"].added_cycles == 1

    path = registry.save_portfolio_ledger(report)
    assert path.exists()
    loaded = registry.load_portfolio_ledger("portfolio_test")
    assert isinstance(loaded, PortfolioLedgerReport)
    assert loaded.cycle_entries[1].target_candidate_ids == ["cand_b", "cand_c"]
