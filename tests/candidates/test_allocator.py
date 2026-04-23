from __future__ import annotations

import pytest

from umc_nn.candidates import (
    AllocationWeight,
    AllocatorConfig,
    CandidateCurveSnapshot,
    CandidateRecord,
    CandidateRegistry,
    ClusterAssignment,
    ClusterReport,
    ClusterSummary,
    DiversificationReport,
    ExperimentManifest,
    OverrideSet,
    PeriodStats,
    ShortlistCandidateScore,
    ShortlistReport,
    build_allocator_workbench_report,
    bootstrap_portfolio_step_matrix,
    empty_override_set,
    propose_allocation_weights,
    update_cluster_override,
    utc_now_text,
)
from umc_nn.candidates.allocator import PortfolioResamplingConfig


def _candidate(candidate_id: str, display_name: str) -> CandidateRecord:
    manifest = ExperimentManifest(
        schema_version="1",
        created_at_utc=utc_now_text(),
        source_script="/tmp/source.py",
        engine_name="candidate_engine_v1",
        engine_role="candidate_engine",
        representation_name="fused32",
        data_path="/tmp/data.npz",
        checkpoint_path=f"/tmp/{candidate_id}.npy",
        log_path=None,
        source_summary_path=None,
        train_window_utc={"start": "2025-01-01 00:00:00", "end": "2025-01-08 00:00:00"},
        oos_window_utc={"start": "2025-01-08 00:00:00", "end": "2025-01-15 00:00:00"},
        search_config={},
        econ_config={
            "initial_balance": 10000.0,
            "exchange": "binance",
            "execution_fee_mode": "taker",
            "slippage": 0.0,
            "position_sizing_mode": "fraction_of_equity",
            "position_notional_fraction": 1.0,
            "leverage": 1.0,
            "fixed_position_qty": 1.0,
        },
    )
    return CandidateRecord(
        schema_version="1",
        candidate_id=candidate_id,
        display_name=display_name,
        engine_name="candidate_engine_v1",
        engine_role="candidate_engine",
        status="research",
        created_at_utc=utc_now_text(),
        manifest=manifest,
        periods={
            "train": PeriodStats.from_episode(
                "train",
                {
                    "start_step": 0,
                    "requested_max_steps": 10,
                    "steps_run": 10,
                    "final_balance": 10100.0,
                    "pnl": 100.0,
                    "max_drawdown_pct": 2.0,
                    "trades": 8,
                    "wins": 4,
                    "win_rate_pct": 50.0,
                    "action_counts": {"0": 4, "1": 6},
                    "position_counts": {"0": 4, "1": 6},
                },
            ),
            "oos": PeriodStats.from_episode(
                "oos",
                {
                    "start_step": 10,
                    "requested_max_steps": 10,
                    "steps_run": 10,
                    "final_balance": 10050.0,
                    "pnl": 50.0,
                    "max_drawdown_pct": 3.0,
                    "trades": 5,
                    "wins": 3,
                    "win_rate_pct": 60.0,
                    "action_counts": {"0": 5, "1": 5},
                    "position_counts": {"0": 5, "1": 5},
                },
                beats_flat=True,
                full_window=True,
            ),
        },
        selection_flags={"oos_beats_flat": True, "oos_positive": True},
    )


def _diversification_report() -> DiversificationReport:
    return DiversificationReport(
        schema_version="1",
        name="common_window",
        created_at_utc=utc_now_text(),
        data_path="/tmp/data.npz",
        start_utc="2025-02-01 00:00:00",
        end_utc="2025-02-08 00:00:00",
        start_step=0,
        max_steps=8,
        candidate_ids=["cand_a", "cand_b"],
        candidate_curves=[
            CandidateCurveSnapshot(
                candidate_id="cand_a",
                display_name="A",
                steps_run=8,
                sample_indices=[0, 1, 2],
                normalized_balance_history=[1.0, 1.01, 1.02],
                final_balance=10200.0,
                max_drawdown_pct=2.0,
            ),
            CandidateCurveSnapshot(
                candidate_id="cand_b",
                display_name="B",
                steps_run=8,
                sample_indices=[0, 1, 2],
                normalized_balance_history=[1.0, 1.00, 1.01],
                final_balance=10100.0,
                max_drawdown_pct=2.5,
            ),
        ],
        pair_stats=[],
    )


def _shortlist_report() -> ShortlistReport:
    return ShortlistReport(
        schema_version="1",
        name="shortlist",
        created_at_utc=utc_now_text(),
        source_diversification_report="common_window",
        data_path="/tmp/data.npz",
        start_utc="2025-02-01 00:00:00",
        end_utc="2025-02-08 00:00:00",
        start_step=0,
        max_steps=8,
        resampling_name="train_bootstrap_f1.00",
        candidate_ids=["cand_a", "cand_b"],
        selected_candidate_ids=["cand_a", "cand_b"],
        selection_config={},
        candidate_scores=[
            ShortlistCandidateScore(
                candidate_id="cand_a",
                display_name="A",
                selected=True,
                selected_rank=1,
                base_score=2.0,
                marginal_score=2.5,
                brightness_hint=1.0,
                score_components={},
            ),
            ShortlistCandidateScore(
                candidate_id="cand_b",
                display_name="B",
                selected=True,
                selected_rank=2,
                base_score=1.0,
                marginal_score=1.5,
                brightness_hint=0.8,
                score_components={},
            ),
        ],
        selected_pair_scores=[],
    )


def test_propose_allocation_weights_respects_risk_dial_and_caps() -> None:
    weights, allocated = propose_allocation_weights(
        shortlisted_scores={
            "cand_a": {"display_name": "A", "base_score": 2.0, "marginal_score": 2.5},
            "cand_b": {"display_name": "B", "base_score": 1.0, "marginal_score": 1.5},
        },
        requested_risk_fraction=1.0,
        per_system_cap_fraction=0.40,
        score_mode="marginal",
        min_score_floor=0.05,
    )

    assert len(weights) == 2
    assert allocated == 0.8
    assert sum(item.capital_fraction for item in weights) == 0.8
    assert all(item.capital_fraction <= 0.4 + 1e-9 for item in weights)
    assert weights[0].capital_fraction >= weights[1].capital_fraction


def test_bootstrap_portfolio_step_matrix_returns_summary() -> None:
    matrix = [
        [0.01, -0.005],
        [-0.005, 0.01],
        [0.008, -0.002],
        [-0.004, 0.007],
    ]
    weights = [0.30, 0.20]
    stats = bootstrap_portfolio_step_matrix(
        matrix,
        capital_fractions=weights,
        config=PortfolioResamplingConfig(
            name="portfolio_risk_0.50",
            iterations=200,
            block_size=2,
            seed=11,
            initial_balance=10000.0,
        ),
        requested_risk_fraction=0.50,
        allocated_risk_fraction=0.50,
    )

    assert stats.steps == 4
    assert stats.original_net_profit != 0.0
    assert stats.p95_max_drawdown_pct >= stats.median_max_drawdown_pct
    assert 0.0 <= stats.profitable_rate <= 1.0


def test_propose_allocation_weights_respects_cluster_caps() -> None:
    weights, allocated = propose_allocation_weights(
        shortlisted_scores={
            "cand_a": {"display_name": "A", "base_score": 3.0, "marginal_score": 3.0},
            "cand_b": {"display_name": "B", "base_score": 2.0, "marginal_score": 2.0},
            "cand_c": {"display_name": "C", "base_score": 1.0, "marginal_score": 1.0},
        },
        requested_risk_fraction=1.0,
        per_system_cap_fraction=0.6,
        score_mode="base",
        min_score_floor=0.05,
        cluster_by_candidate={"cand_a": "cluster_0", "cand_b": "cluster_0", "cand_c": "cluster_1"},
        cluster_cap_overrides={"cluster_0": 0.5, "cluster_1": 0.6},
    )

    cluster_0_total = sum(item.capital_fraction for item in weights if item.cluster_id == "cluster_0")
    assert allocated == pytest.approx(1.0)
    assert cluster_0_total == pytest.approx(0.5)
    assert any(item.cap_reason == "cluster_cap" for item in weights if item.cluster_id == "cluster_0")


def test_build_allocator_workbench_report_with_synthetic_views(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    cand_a = _candidate("cand_a", "A")
    cand_b = _candidate("cand_b", "B")
    registry.add_candidate(cand_a)
    registry.add_candidate(cand_b)

    report = build_allocator_workbench_report(
        name="allocator_workbench",
        created_at_utc=utc_now_text(),
        candidates=[cand_a, cand_b],
        shortlist_report=_shortlist_report(),
        diversification_report=_diversification_report(),
        config=AllocatorConfig(
            risk_fractions=(0.40, 1.0),
            per_system_cap_fraction=0.40,
            score_mode="marginal",
            min_score_floor=0.05,
            resampling_iterations=100,
            resampling_block_size=2,
            resampling_seed=7,
            objective_max_drawdown_pct=10.0,
            curve_points=16,
        ),
        step_return_views={
            "cand_a": [0.01, -0.005, 0.008, -0.004, 0.012, -0.003, 0.004, 0.002],
            "cand_b": [-0.004, 0.009, -0.001, 0.006, -0.002, 0.008, -0.003, 0.004],
        },
    )

    assert len(report.scenarios) == 2
    assert report.chosen_scenario_name in {"risk_0.40", "risk_1.00"}
    assert report.scenarios[0].weights[0].capital_fraction > report.scenarios[0].weights[1].capital_fraction
    assert report.scenarios[1].allocated_risk_fraction == pytest.approx(0.8)
    assert report.scenarios[1].reserve_fraction == pytest.approx(0.2)

    path = registry.save_allocator_workbench(report)
    assert path.exists()
    loaded = registry.load_allocator_workbench("allocator_workbench")
    assert len(loaded.scenarios) == 2
    assert loaded.selected_candidate_ids == ["cand_a", "cand_b"]


def test_allocator_workbench_applies_override_cluster_caps(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    cand_a = _candidate("cand_a", "A")
    cand_b = _candidate("cand_b", "B")
    registry.add_candidate(cand_a)
    registry.add_candidate(cand_b)
    cluster_report = ClusterReport(
        schema_version="1",
        name="clusters",
        created_at_utc=utc_now_text(),
        source_diversification_report="common_window",
        data_path="/tmp/data.npz",
        start_utc="2025-02-01 00:00:00",
        end_utc="2025-02-08 00:00:00",
        start_step=0,
        max_steps=8,
        similarity_threshold=0.4,
        candidate_ids=["cand_a", "cand_b"],
        assignments=[
            ClusterAssignment(candidate_id="cand_a", display_name="A", cluster_id="cluster_000"),
            ClusterAssignment(candidate_id="cand_b", display_name="B", cluster_id="cluster_000"),
        ],
        clusters=[
            ClusterSummary(
                cluster_id="cluster_000",
                candidate_ids=["cand_a", "cand_b"],
                display_names=["A", "B"],
                cluster_size=2,
                mean_return_corr=0.7,
                mean_downside_corr=0.8,
                mean_simultaneous_loss_rate=0.4,
                mean_similarity_score=0.7,
            )
        ],
    )
    override_set = empty_override_set(name="ops", source_cluster_report="clusters")
    override_set = update_cluster_override(
        override_set,
        cluster_id="cluster_000",
        actor="operator",
        max_cap_fraction=0.5,
    )

    report = build_allocator_workbench_report(
        name="allocator_workbench",
        created_at_utc=utc_now_text(),
        candidates=[cand_a, cand_b],
        shortlist_report=_shortlist_report(),
        diversification_report=_diversification_report(),
        cluster_report=cluster_report,
        override_set=override_set,
        config=AllocatorConfig(
            risk_fractions=(1.0,),
            per_system_cap_fraction=0.6,
            default_cluster_cap_fraction=0.8,
            score_mode="marginal",
            min_score_floor=0.05,
            resampling_iterations=50,
            resampling_block_size=2,
            resampling_seed=7,
            objective_max_drawdown_pct=10.0,
            curve_points=16,
        ),
        step_return_views={
            "cand_a": [0.01, -0.005, 0.008, -0.004, 0.012, -0.003, 0.004, 0.002],
            "cand_b": [-0.004, 0.009, -0.001, 0.006, -0.002, 0.008, -0.003, 0.004],
        },
    )

    total_allocated = sum(item.capital_fraction for item in report.scenarios[0].weights)
    assert total_allocated == pytest.approx(0.5)
