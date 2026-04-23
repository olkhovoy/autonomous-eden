from __future__ import annotations

from umc_nn.candidates import (
    CandidateRecord,
    CandidateRegistry,
    ExperimentManifest,
    LifecycleConfig,
    LifecycleReport,
    RollingConveyorReport,
    RollingCycleOutcome,
    TradeforwardCandidateEvaluation,
    TradeforwardEvaluationReport,
    TradeforwardPortfolioEvaluation,
    build_lifecycle_report,
    apply_lifecycle_report,
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


def _candidate_eval(candidate_id: str, display_name: str, *, pnl: float, dd: float, trades: int) -> TradeforwardCandidateEvaluation:
    return TradeforwardCandidateEvaluation(
        candidate_id=candidate_id,
        display_name=display_name,
        cluster_id=None,
        checkpoint_path=f"/tmp/{candidate_id}.npy",
        capital_fraction=0.25,
        normalized_share=1.0,
        capped=False,
        cap_reason=None,
        final_balance=10000.0 + pnl,
        pnl=pnl,
        max_drawdown_pct=dd,
        trades=trades,
        wins=max(0, trades - 1),
        win_rate_pct=0.0 if trades <= 0 else 50.0,
        curve_sample_indices=[0, 1],
        normalized_balance_history=[1.0, 1.0 + pnl / 10000.0],
    )


def _evaluation(name: str, cycle_name: str, candidate_evaluations: list[TradeforwardCandidateEvaluation]) -> TradeforwardEvaluationReport:
    pnl = sum(item.pnl for item in candidate_evaluations)
    return TradeforwardEvaluationReport(
        schema_version="1",
        name=name,
        created_at_utc=utc_now_text(),
        source_plan=f"{cycle_name}_plan",
        selection_mode="combination",
        scenario_name="subset",
        data_path="/tmp/data.npz",
        forward_start_utc="2025-05-08 00:00:00",
        forward_end_utc="2025-05-15 00:00:00",
        forward_start_step=0,
        forward_max_steps=100,
        candidate_ids=[item.candidate_id for item in candidate_evaluations],
        expectation=None,
        portfolio=TradeforwardPortfolioEvaluation(
            initial_balance=10000.0,
            final_balance=10000.0 + pnl,
            pnl=pnl,
            max_drawdown_pct=max((item.max_drawdown_pct for item in candidate_evaluations), default=0.0),
            requested_risk_fraction=0.5,
            allocated_risk_fraction=0.5,
            reserve_fraction=0.5,
            component_trade_count_total=sum(item.trades for item in candidate_evaluations),
            component_win_count_total=sum(item.wins for item in candidate_evaluations),
            curve_sample_indices=[0, 1],
            normalized_balance_history=[1.0, 1.0 + pnl / 10000.0],
        ),
        candidate_evaluations=candidate_evaluations,
        source_cycle_report=cycle_name,
        source_allocator_report=None,
        source_combination_report=None,
        source_cluster_report=None,
    )


def _rolling_report() -> RollingConveyorReport:
    return RollingConveyorReport(
        schema_version="1",
        name="rolling_lifecycle_test",
        created_at_utc=utc_now_text(),
        mode="reuse",
        selection_days=7,
        forward_days=7,
        step_days=7,
        initial_balance=10000.0,
        final_balance=10100.0,
        total_pnl=100.0,
        total_return_pct=1.0,
        max_drawdown_pct=3.0,
        positive_cycle_count=2,
        evaluated_cycle_count=2,
        ledger_cycle_indices=[0, 1, 2],
        ledger_balance_history=[10000.0, 10050.0, 10100.0],
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
                candidate_ids=["cand_a", "cand_b", "cand_c", "cand_d"],
                selected_candidate_ids=["cand_a", "cand_b", "cand_d"],
                requested_risk_fraction=0.5,
                allocated_risk_fraction=0.5,
                reserve_fraction=0.5,
                portfolio_pnl=50.0,
                portfolio_final_balance=10050.0,
                portfolio_max_drawdown_pct=2.5,
                cycle_return_fraction=0.005,
                ledger_balance_before=10000.0,
                ledger_balance_after=10050.0,
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
                candidate_ids=["cand_a", "cand_b", "cand_c", "cand_d"],
                selected_candidate_ids=["cand_a", "cand_d"],
                requested_risk_fraction=0.5,
                allocated_risk_fraction=0.5,
                reserve_fraction=0.5,
                portfolio_pnl=50.0,
                portfolio_final_balance=10100.0,
                portfolio_max_drawdown_pct=2.0,
                cycle_return_fraction=0.004975124378109453,
                ledger_balance_before=10050.0,
                ledger_balance_after=10100.0,
            ),
        ],
        selector={"status": "research"},
    )


def test_build_lifecycle_report_transitions_and_apply(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    registry.add_candidate(_candidate("cand_a", "A", "research"))
    registry.add_candidate(_candidate("cand_b", "B", "paper"))
    registry.add_candidate(_candidate("cand_c", "C", "active"))
    registry.add_candidate(_candidate("cand_d", "D", "draining"))

    registry.save_tradeforward_evaluation(
        _evaluation(
            "cycle_01_eval",
            "cycle_01",
            [
                _candidate_eval("cand_a", "A", pnl=120.0, dd=2.0, trades=3),
                _candidate_eval("cand_b", "B", pnl=-50.0, dd=4.0, trades=2),
                _candidate_eval("cand_d", "D", pnl=90.0, dd=3.0, trades=2),
            ],
        )
    )
    registry.save_tradeforward_evaluation(
        _evaluation(
            "cycle_02_eval",
            "cycle_02",
            [
                _candidate_eval("cand_a", "A", pnl=110.0, dd=2.0, trades=2),
                _candidate_eval("cand_d", "D", pnl=80.0, dd=3.0, trades=2),
            ],
        )
    )

    report = build_lifecycle_report(
        registry,
        "lifecycle_test",
        rolling_report=_rolling_report(),
        config=LifecycleConfig(
            min_forward_pnl=0.0,
            max_forward_drawdown_pct=8.0,
            min_selected_trades=1,
            successful_selected_cycles_to_activate=2,
            successful_selected_cycles_to_recover=1,
            idle_cycles_to_drain=1,
            idle_cycles_to_retire=1,
        ),
    )

    summary = {item.candidate_id: item for item in report.candidate_summaries}
    assert summary["cand_a"].final_status == "paper"
    assert summary["cand_b"].final_status == "retired"
    assert summary["cand_c"].final_status == "retired"
    assert summary["cand_d"].final_status == "active"
    assert report.final_status_counts == {"paper": 1, "retired": 2, "active": 1}

    path = registry.save_lifecycle_report(report)
    assert path.exists()
    loaded = registry.load_lifecycle_report("lifecycle_test")
    assert isinstance(loaded, LifecycleReport)
    assert loaded.candidate_summaries[0].candidate_id

    apply_lifecycle_report(registry, report)
    assert registry.load_candidate("cand_a").status == "paper"
    assert registry.load_candidate("cand_b").status == "retired"
    assert registry.load_candidate("cand_c").status == "retired"
    updated = registry.load_candidate("cand_d")
    assert updated.status == "active"
    assert updated.metadata["lifecycle"]["source_report"] == "lifecycle_test"
