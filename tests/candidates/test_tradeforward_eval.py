from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from umc_nn.candidates import (
    AllocationWeight,
    CandidateRecord,
    CandidateRegistry,
    CombinationScenario,
    CombinationSearchReport,
    ExperimentManifest,
    PortfolioResamplingStats,
    TradeforwardEvaluationReport,
    build_tradeforward_evaluation,
    build_tradeforward_plan,
    utc_now_text,
)
from umc_nn.trading_eval import EpisodeMetrics, EpisodeTrace


def _utc_ts(text: str) -> int:
    return int(datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())


def _write_data(path) -> None:
    start = _utc_ts("2025-05-08 00:00:00")
    timestamps = np.arange(start, start + 60 * 60 * 24 * 21, 60, dtype=np.int64)
    close_prices = np.linspace(100.0, 140.0, num=timestamps.shape[0], dtype=np.float32)
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
        allocated_risk_fraction=0.4,
        original_final_balance=10120.0,
        original_net_profit=120.0,
        original_max_drawdown_pct=4.5,
        mean_final_balance=10110.0,
        median_final_balance=10100.0,
        p05_final_balance=9940.0,
        p25_final_balance=9980.0,
        mean_net_profit=110.0,
        median_net_profit=100.0,
        p05_net_profit=-60.0,
        p25_net_profit=10.0,
        mean_max_drawdown_pct=4.4,
        median_max_drawdown_pct=4.2,
        p75_max_drawdown_pct=5.1,
        p95_max_drawdown_pct=6.7,
        profitable_rate=0.6,
        loss_rate=0.4,
        ruin_rate=0.0,
        pessimistic_net_profit=-60.0,
        pessimistic_max_drawdown_pct=6.7,
    )


def test_build_tradeforward_evaluation_from_plan(monkeypatch, tmp_path) -> None:
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
                objective_score=0.45,
                requested_risk_fraction=0.5,
                allocated_risk_fraction=0.4,
                reserve_fraction=0.6,
                score_mode="base",
                curve_sample_indices=[0, 1, 2],
                normalized_balance_history=[1.0, 1.01, 1.02],
                weights=[
                    AllocationWeight(
                        candidate_id="cand_a",
                        display_name="cand_a",
                        raw_score=2.0,
                        normalized_share=0.625,
                        capital_fraction=0.25,
                        cluster_id="cluster_000",
                        capped=False,
                    ),
                    AllocationWeight(
                        candidate_id="cand_b",
                        display_name="cand_b",
                        raw_score=1.0,
                        normalized_share=0.375,
                        capital_fraction=0.15,
                        cluster_id="cluster_001",
                        capped=True,
                        cap_reason="candidate_cap",
                    ),
                ],
                resampling=_portfolio_stats(),
            )
        ],
    )
    registry.save_combination_search(combination)
    plan = build_tradeforward_plan(
        registry,
        "tf_plan_eval",
        forward_start_utc="2025-05-15 00:00:00",
        forward_end_utc="2025-05-18 00:00:00",
        combination_report=combination,
    )

    traces = {
        "/tmp/cand_a.npy": EpisodeTrace(
            metrics=EpisodeMetrics(
                policy="monolith",
                start_step=0,
                requested_max_steps=3,
                steps_run=3,
                final_balance=10300.0,
                pnl=300.0,
                max_drawdown_pct=1.5,
                trades=3,
                wins=2,
                win_rate_pct=66.6667,
                action_counts={0: 1, 1: 2},
                position_counts={0: 1, 1: 2},
            ),
            trades=[],
            balance_history=[10000.0, 10100.0, 10200.0, 10300.0],
            action_history=[1, 1, 0],
            position_history=[0, 1, 1, 0],
        ),
        "/tmp/cand_b.npy": EpisodeTrace(
            metrics=EpisodeMetrics(
                policy="monolith",
                start_step=0,
                requested_max_steps=3,
                steps_run=3,
                final_balance=9900.0,
                pnl=-100.0,
                max_drawdown_pct=2.0,
                trades=2,
                wins=1,
                win_rate_pct=50.0,
                action_counts={0: 2, 2: 1},
                position_counts={0: 2, 2: 1},
            ),
            trades=[],
            balance_history=[10000.0, 9950.0, 9850.0, 9900.0],
            action_history=[2, 2, 0],
            position_history=[0, 2, 2, 0],
        ),
    }

    def _fake_eval(*args, **kwargs):
        del args
        return traces[str(kwargs["weights_path"])]

    monkeypatch.setattr("umc_nn.candidates.tradeforward_eval.evaluate_policy_trace_path", _fake_eval)

    report = build_tradeforward_evaluation(
        registry,
        "tf_eval",
        plan=plan,
        curve_points=8,
    )
    assert report.portfolio.pnl != 0.0
    assert report.expectation is not None
    assert report.expectation.expected_original_net_profit == 120.0
    assert len(report.candidate_evaluations) == 2
    assert report.candidate_evaluations[1].cap_reason == "candidate_cap"
    assert report.portfolio.actual_minus_expected_original_net_profit is not None

    path = registry.save_tradeforward_evaluation(report)
    assert path.exists()
    loaded = registry.load_tradeforward_evaluation("tf_eval")
    assert isinstance(loaded, TradeforwardEvaluationReport)
    assert loaded.scenario_name == "subset_ab"
