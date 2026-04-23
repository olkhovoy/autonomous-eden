from __future__ import annotations

from umc_nn.candidates import (
    CandidateRegistry,
    CombinationSearchReport,
    ContinuousSearchCycleReport,
    RollingConveyorReport,
    TradeforwardEvaluationReport,
    TradeforwardExpectation,
    TradeforwardPlan,
    TradeforwardPortfolioEvaluation,
    build_rolling_conveyor_report,
    build_rolling_window_specs,
    utc_now_text,
)


def _combo_report(name: str, start_utc: str, end_utc: str) -> CombinationSearchReport:
    return CombinationSearchReport(
        schema_version="1",
        name=name,
        created_at_utc=utc_now_text(),
        source_shortlist_report=f"{name}_shortlist",
        source_diversification_report=f"{name}_div",
        source_cluster_report=f"{name}_cluster",
        source_override_set=None,
        data_path="/tmp/data.npz",
        start_utc=start_utc,
        end_utc=end_utc,
        start_step=0,
        max_steps=10,
        pool_candidate_ids=["cand_a", "cand_b"],
        searched_subset_sizes=[1],
        evaluated_combination_count=1,
        evaluated_scenario_count=1,
        best_scenario_name=None,
        objective_config={"mode": "base"},
        scenarios=[],
    )


def _cycle_report(name: str, plan_name: str, candidate_ids: list[str]) -> ContinuousSearchCycleReport:
    return ContinuousSearchCycleReport(
        schema_version="1",
        name=name,
        created_at_utc=utc_now_text(),
        mode="reuse",
        output_dir=f"/tmp/{name}",
        cycle_tag=None,
        source_summary_path=None,
        candidate_ids=candidate_ids,
        report_names={"tradeforward": plan_name},
        steps=[],
    )


def _plan(name: str, cycle_name: str, combo_name: str, candidate_ids: list[str], forward_start: str, forward_end: str) -> TradeforwardPlan:
    return TradeforwardPlan(
        schema_version="1",
        name=name,
        created_at_utc=utc_now_text(),
        selection_mode="combination",
        scenario_name="subset_test",
        data_path="/tmp/data.npz",
        forward_start_utc=forward_start,
        forward_end_utc=forward_end,
        forward_start_step=0,
        forward_max_steps=10,
        requested_risk_fraction=0.5,
        allocated_risk_fraction=0.4,
        reserve_fraction=0.6,
        candidate_ids=candidate_ids,
        allocations=[],
        source_cycle_report=cycle_name,
        source_allocator_report=None,
        source_combination_report=combo_name,
        source_cluster_report=None,
    )


def _evaluation(
    name: str,
    plan_name: str,
    cycle_name: str,
    *,
    forward_start: str,
    forward_end: str,
    final_balance: float,
    pnl: float,
    max_drawdown_pct: float,
    normalized_curve: list[float],
    expected_original_net_profit: float,
    expected_p05_net_profit: float,
) -> TradeforwardEvaluationReport:
    return TradeforwardEvaluationReport(
        schema_version="1",
        name=name,
        created_at_utc=utc_now_text(),
        source_plan=plan_name,
        selection_mode="combination",
        scenario_name="subset_test",
        data_path="/tmp/data.npz",
        forward_start_utc=forward_start,
        forward_end_utc=forward_end,
        forward_start_step=0,
        forward_max_steps=10,
        candidate_ids=["cand_a"],
        expectation=TradeforwardExpectation(
            selection_mode="combination",
            scenario_name="subset_test",
            objective_score=0.25,
            expected_original_net_profit=expected_original_net_profit,
            expected_original_max_drawdown_pct=4.0,
            expected_p05_net_profit=expected_p05_net_profit,
            expected_p95_max_drawdown_pct=8.0,
        ),
        portfolio=TradeforwardPortfolioEvaluation(
            initial_balance=10000.0,
            final_balance=final_balance,
            pnl=pnl,
            max_drawdown_pct=max_drawdown_pct,
            requested_risk_fraction=0.5,
            allocated_risk_fraction=0.4,
            reserve_fraction=0.6,
            component_trade_count_total=5,
            component_win_count_total=3,
            curve_sample_indices=list(range(len(normalized_curve))),
            normalized_balance_history=normalized_curve,
            actual_minus_expected_original_net_profit=pnl - expected_original_net_profit,
            actual_minus_expected_original_max_drawdown_pct=max_drawdown_pct - 4.0,
            actual_minus_expected_p05_net_profit=pnl - expected_p05_net_profit,
            actual_minus_expected_p95_max_drawdown_pct=max_drawdown_pct - 8.0,
        ),
        candidate_evaluations=[],
        source_cycle_report=cycle_name,
        source_allocator_report=None,
        source_combination_report="unused",
        source_cluster_report=None,
    )


def test_build_rolling_window_specs_default_step_matches_forward_days() -> None:
    specs = build_rolling_window_specs(
        report_name="rolling_smoke",
        selection_start_utc="2025-05-01 00:00:00",
        selection_days=7,
        forward_days=7,
        cycle_count=2,
    )
    assert len(specs) == 2
    assert specs[0].selection_end_utc == "2025-05-08 00:00:00"
    assert specs[0].forward_end_utc == "2025-05-15 00:00:00"
    assert specs[1].selection_start_utc == "2025-05-08 00:00:00"


def test_build_rolling_conveyor_report_compounds_cycle_returns_and_roundtrips(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    registry.save_combination_search(_combo_report("combo_1", "2025-05-01 00:00:00", "2025-05-08 00:00:00"))
    registry.save_combination_search(_combo_report("combo_2", "2025-05-08 00:00:00", "2025-05-15 00:00:00"))

    cycle_1 = _cycle_report("cycle_1", "tf_plan_1", ["cand_a", "cand_b"])
    cycle_2 = _cycle_report("cycle_2", "tf_plan_2", ["cand_a", "cand_c"])
    plan_1 = _plan(
        "tf_plan_1",
        "cycle_1",
        "combo_1",
        ["cand_a"],
        "2025-05-08 00:00:00",
        "2025-05-15 00:00:00",
    )
    plan_2 = _plan(
        "tf_plan_2",
        "cycle_2",
        "combo_2",
        ["cand_a"],
        "2025-05-15 00:00:00",
        "2025-05-22 00:00:00",
    )
    eval_1 = _evaluation(
        "tf_eval_1",
        "tf_plan_1",
        "cycle_1",
        forward_start="2025-05-08 00:00:00",
        forward_end="2025-05-15 00:00:00",
        final_balance=11000.0,
        pnl=1000.0,
        max_drawdown_pct=2.0,
        normalized_curve=[1.0, 1.05, 1.10],
        expected_original_net_profit=600.0,
        expected_p05_net_profit=200.0,
    )
    eval_2 = _evaluation(
        "tf_eval_2",
        "tf_plan_2",
        "cycle_2",
        forward_start="2025-05-15 00:00:00",
        forward_end="2025-05-22 00:00:00",
        final_balance=10400.0,
        pnl=400.0,
        max_drawdown_pct=6.0,
        normalized_curve=[1.0, 0.95, 1.04],
        expected_original_net_profit=100.0,
        expected_p05_net_profit=-200.0,
    )

    report = build_rolling_conveyor_report(
        registry,
        "rolling_test",
        mode="reuse",
        cycle_reports=[cycle_1, cycle_2],
        tradeforward_plans=[plan_1, plan_2],
        tradeforward_evaluations=[eval_1, eval_2],
        selection_start_utc="2025-05-01 00:00:00",
        selection_days=7,
        forward_days=7,
        step_days=7,
        initial_balance=10000.0,
        selector={"candidate_ids": ["cand_a"]},
    )

    assert report.evaluated_cycle_count == 2
    assert report.final_balance == 11440.0
    assert report.total_pnl == 1440.0
    assert round(report.total_return_pct, 2) == 14.40
    assert round(report.max_drawdown_pct, 2) == 5.00
    assert report.ledger_balance_history == [10000.0, 11000.0, 11440.0]
    assert report.cycle_outcomes[1].portfolio_pnl == 440.0
    assert report.cycle_outcomes[1].actual_minus_expected_original_net_profit == 330.0
    assert report.cycle_outcomes[1].selected_candidate_ids == ["cand_a"]
    assert report.selector["candidate_ids"] == ["cand_a"]

    path = registry.save_rolling_conveyor(report)
    assert path.exists()
    loaded = registry.load_rolling_conveyor("rolling_test")
    assert isinstance(loaded, RollingConveyorReport)
    assert loaded.cycle_outcomes[0].selection_start_utc == "2025-05-01 00:00:00"
