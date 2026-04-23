from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from umc_nn.candidates import (
    AllocationWeight,
    CandidateRecord,
    CandidateRegistry,
    CombinationScenario,
    CombinationSearchReport,
    ContinuousSearchCycleReport,
    CycleStepRecord,
    ExperimentManifest,
    PortfolioResamplingStats,
    TradeforwardPlan,
    build_tradeforward_plan,
    utc_now_text,
)


def _utc_ts(text: str) -> int:
    return int(datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())


def _write_data(path) -> None:
    start = _utc_ts("2025-05-08 00:00:00")
    timestamps = np.arange(start, start + 60 * 60 * 24 * 14, 60, dtype=np.int64)
    close_prices = np.linspace(100.0, 120.0, num=timestamps.shape[0], dtype=np.float32)
    neurobars = np.zeros((timestamps.shape[0], 32), dtype=np.float32)
    np.savez(path, timestamps=timestamps, close_prices=close_prices, neurobars=neurobars)


def _manifest(data_path: str, checkpoint_path: str) -> ExperimentManifest:
    return ExperimentManifest(
        schema_version="1",
        created_at_utc=utc_now_text(),
        source_script="/tmp/source.py",
        engine_name="monolith",
        engine_role="candidate_engine",
        representation_name="fused32",
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        log_path=None,
        source_summary_path=None,
        train_window_utc={"start": "2025-05-01 00:00:00", "end": "2025-05-08 00:00:00"},
        oos_window_utc={"start": "2025-05-08 00:00:00", "end": "2025-05-15 00:00:00"},
        search_config={},
        econ_config={"initial_balance": 10000.0, "exchange": "binance"},
    )


def _candidate(candidate_id: str, data_path: str) -> CandidateRecord:
    return CandidateRecord(
        schema_version="1",
        candidate_id=candidate_id,
        display_name=candidate_id,
        engine_name="monolith",
        engine_role="candidate_engine",
        status="research",
        created_at_utc=utc_now_text(),
        tags=["cycle:test"],
        manifest=_manifest(data_path, f"/tmp/{candidate_id}.npy"),
    )


def _portfolio_stats() -> PortfolioResamplingStats:
    return PortfolioResamplingStats(
        name="combo",
        iterations=10,
        block_size=16,
        seed=42,
        steps=128,
        initial_balance=10000.0,
        requested_risk_fraction=0.5,
        allocated_risk_fraction=0.5,
        original_final_balance=10100.0,
        original_net_profit=100.0,
        original_max_drawdown_pct=4.0,
        mean_final_balance=10080.0,
        median_final_balance=10070.0,
        p05_final_balance=9900.0,
        p25_final_balance=9950.0,
        mean_net_profit=80.0,
        median_net_profit=70.0,
        p05_net_profit=-100.0,
        p25_net_profit=-20.0,
        mean_max_drawdown_pct=4.5,
        median_max_drawdown_pct=4.0,
        p75_max_drawdown_pct=5.0,
        p95_max_drawdown_pct=6.0,
        profitable_rate=0.6,
        loss_rate=0.4,
        ruin_rate=0.0,
        pessimistic_net_profit=-100.0,
        pessimistic_max_drawdown_pct=6.0,
    )


def test_build_tradeforward_plan_from_combination_and_cycle_roundtrip(tmp_path) -> None:
    data_path = tmp_path / "data.npz"
    _write_data(data_path)

    registry = CandidateRegistry(tmp_path / "registry")
    registry.add_candidate(_candidate("cand_a", str(data_path)))
    registry.add_candidate(_candidate("cand_b", str(data_path)))

    combination = CombinationSearchReport(
        schema_version="1",
        name="combo_report",
        created_at_utc=utc_now_text(),
        source_shortlist_report="shortlist_report",
        source_diversification_report="div_report",
        source_cluster_report="cluster_report",
        source_override_set=None,
        data_path=str(data_path),
        start_utc="2025-05-08 00:00:00",
        end_utc="2025-05-15 00:00:00",
        start_step=0,
        max_steps=100,
        pool_candidate_ids=["cand_a", "cand_b"],
        searched_subset_sizes=[1, 2],
        evaluated_combination_count=3,
        evaluated_scenario_count=6,
        best_scenario_name="subset_ab",
        objective_config={"mode": "base"},
        scenarios=[
            CombinationScenario(
                name="subset_ab",
                subset_candidate_ids=["cand_a", "cand_b"],
                subset_display_names=["cand_a", "cand_b"],
                subset_size=2,
                objective_score=0.4,
                requested_risk_fraction=0.5,
                allocated_risk_fraction=0.45,
                reserve_fraction=0.55,
                score_mode="base",
                curve_sample_indices=[0, 1, 2],
                normalized_balance_history=[1.0, 1.01, 1.03],
                weights=[
                    AllocationWeight(
                        candidate_id="cand_a",
                        display_name="cand_a",
                        raw_score=2.0,
                        normalized_share=0.6,
                        capital_fraction=0.27,
                        cluster_id="cluster_000",
                        capped=False,
                    ),
                    AllocationWeight(
                        candidate_id="cand_b",
                        display_name="cand_b",
                        raw_score=1.0,
                        normalized_share=0.4,
                        capital_fraction=0.18,
                        cluster_id="cluster_001",
                        capped=True,
                        cap_reason="cluster_cap",
                    ),
                ],
                resampling=_portfolio_stats(),
            )
        ],
    )
    registry.save_combination_search(combination)

    plan = build_tradeforward_plan(
        registry,
        "tf_plan",
        forward_start_utc="2025-05-15 00:00:00",
        forward_end_utc="2025-05-18 00:00:00",
        combination_report=combination,
        source_cycle_report="cycle_test",
    )
    assert plan.selection_mode == "combination"
    assert plan.scenario_name == "subset_ab"
    assert plan.candidate_ids == ["cand_a", "cand_b"]
    assert plan.forward_start_step is not None
    assert plan.forward_max_steps == 4320
    assert plan.allocations[1].cap_reason == "cluster_cap"

    path = registry.save_tradeforward_plan(plan)
    assert path.exists()
    loaded_plan = registry.load_tradeforward_plan("tf_plan")
    assert isinstance(loaded_plan, TradeforwardPlan)
    assert loaded_plan.source_cycle_report == "cycle_test"

    cycle = ContinuousSearchCycleReport(
        schema_version="1",
        name="cycle_test",
        created_at_utc=utc_now_text(),
        mode="reuse",
        output_dir=str(tmp_path / "outputs"),
        cycle_tag=None,
        source_summary_path=None,
        candidate_ids=["cand_a", "cand_b"],
        report_names={"tradeforward": "tf_plan"},
        steps=[
            CycleStepRecord(
                name="tradeforward",
                command=["python", "scripts/build_tradeforward_plan.py"],
                status="completed",
                log_path=str(tmp_path / "tradeforward.log"),
            )
        ],
    )
    cycle_path = registry.save_cycle_report(cycle)
    assert cycle_path.exists()
    loaded_cycle = registry.load_cycle_report("cycle_test")
    assert loaded_cycle.report_names["tradeforward"] == "tf_plan"
