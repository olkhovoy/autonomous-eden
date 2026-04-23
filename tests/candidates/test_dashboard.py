from __future__ import annotations

from umc_nn.candidates import (
    AllocationWeight,
    AllocatorScenario,
    AllocatorWorkbenchReport,
    CandidateOverride,
    CandidateRecord,
    CandidateRegistry,
    CandidateCurveSnapshot,
    ClusterAssignment,
    ClusterOverride,
    ClusterReport,
    ClusterSummary,
    CombinationScenario,
    CombinationSearchReport,
    DashboardFeed,
    DiversificationPairStats,
    DiversificationReport,
    ExperimentManifest,
    OverrideAuditEntry,
    OverrideSet,
    PeriodStats,
    PortfolioResamplingStats,
    ResamplingStats,
    ShortlistCandidateScore,
    ShortlistPairScore,
    ShortlistReport,
    build_dashboard_feed,
    utc_now_text,
)


def _manifest(checkpoint_path: str) -> ExperimentManifest:
    return ExperimentManifest(
        schema_version="1",
        created_at_utc=utc_now_text(),
        source_script="/tmp/run.py",
        engine_name="candidate_engine",
        engine_role="candidate_engine",
        representation_name="fused32",
        data_path="/tmp/data.npz",
        checkpoint_path=checkpoint_path,
        log_path=None,
        source_summary_path=None,
        train_window_utc={"start": "2025-05-01 00:00:00", "end": "2025-05-08 00:00:00"},
        oos_window_utc={"start": "2025-05-08 00:00:00", "end": "2025-05-15 00:00:00"},
        search_config={"generations": 8},
        econ_config={"exchange": "binance"},
    )


def _candidate(candidate_id: str, display_name: str, *, train_pnl: float, oos_pnl: float) -> CandidateRecord:
    candidate = CandidateRecord(
        schema_version="1",
        candidate_id=candidate_id,
        display_name=display_name,
        engine_name="candidate_engine",
        engine_role="candidate_engine",
        status="research",
        created_at_utc=utc_now_text(),
        tags=["walkforward"],
        manifest=_manifest(f"/tmp/{candidate_id}.npy"),
        periods={
            "train": PeriodStats.from_episode(
                "train",
                {
                    "start_step": 10,
                    "requested_max_steps": 20,
                    "steps_run": 20,
                    "final_balance": 10000.0 + train_pnl,
                    "pnl": train_pnl,
                    "max_drawdown_pct": 4.0,
                    "trades": 12,
                    "wins": 6,
                    "win_rate_pct": 50.0,
                    "action_counts": {"0": 10, "1": 10},
                    "position_counts": {"0": 10, "1": 10},
                },
                full_window=True,
            ),
            "oos": PeriodStats.from_episode(
                "oos",
                {
                    "start_step": 30,
                    "requested_max_steps": 20,
                    "steps_run": 20,
                    "final_balance": 10000.0 + oos_pnl,
                    "pnl": oos_pnl,
                    "max_drawdown_pct": 5.0,
                    "trades": 8,
                    "wins": 4,
                    "win_rate_pct": 50.0,
                    "action_counts": {"0": 8, "1": 12},
                    "position_counts": {"0": 8, "1": 12},
                },
                beats_flat=oos_pnl > 0.0,
                full_window=True,
            ),
        },
        selection_flags={"oos_positive": oos_pnl > 0.0, "oos_beats_flat": oos_pnl > 0.0},
        resampling_results={
            "train_bootstrap_f1.00": ResamplingStats(
                name="train_bootstrap_f1.00",
                period_name="train",
                iterations=100,
                sample_size=12,
                seed=42,
                sizing_mode="fractional_returns",
                fraction=1.0,
                initial_balance=10000.0,
                original_trade_count=12,
                original_final_balance=10000.0 + train_pnl,
                original_net_profit=train_pnl,
                original_max_drawdown_pct=4.0,
                mean_final_balance=10000.0 + train_pnl,
                median_final_balance=10000.0 + train_pnl,
                p05_final_balance=9950.0,
                p25_final_balance=9980.0,
                mean_net_profit=train_pnl,
                median_net_profit=train_pnl,
                p05_net_profit=train_pnl - 120.0,
                p25_net_profit=train_pnl - 60.0,
                mean_max_drawdown_pct=4.0,
                median_max_drawdown_pct=4.0,
                p75_max_drawdown_pct=6.0,
                p95_max_drawdown_pct=8.0,
                profitable_rate=0.55,
                loss_rate=0.45,
                ruin_rate=0.0,
                pessimistic_net_profit=train_pnl - 120.0,
                pessimistic_max_drawdown_pct=8.0,
            )
        },
    )
    return candidate


def _portfolio_stats(name: str, requested: float, allocated: float) -> PortfolioResamplingStats:
    return PortfolioResamplingStats(
        name=name,
        iterations=64,
        block_size=16,
        seed=42,
        steps=128,
        initial_balance=10000.0,
        requested_risk_fraction=requested,
        allocated_risk_fraction=allocated,
        original_final_balance=10120.0,
        original_net_profit=120.0,
        original_max_drawdown_pct=4.0,
        mean_final_balance=10110.0,
        median_final_balance=10100.0,
        p05_final_balance=9920.0,
        p25_final_balance=9980.0,
        mean_net_profit=110.0,
        median_net_profit=100.0,
        p05_net_profit=-80.0,
        p25_net_profit=-20.0,
        mean_max_drawdown_pct=4.2,
        median_max_drawdown_pct=4.0,
        p75_max_drawdown_pct=5.6,
        p95_max_drawdown_pct=7.8,
        profitable_rate=0.60,
        loss_rate=0.40,
        ruin_rate=0.0,
        pessimistic_net_profit=-80.0,
        pessimistic_max_drawdown_pct=7.8,
    )


def test_build_dashboard_feed_and_registry_roundtrip(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    candidate_a = _candidate("cand_a", "wf_a", train_pnl=200.0, oos_pnl=80.0)
    candidate_b = _candidate("cand_b", "wf_b", train_pnl=120.0, oos_pnl=-20.0)
    registry.add_candidate(candidate_a)
    registry.add_candidate(candidate_b)

    diversification = DiversificationReport(
        schema_version="1",
        name="common_window",
        created_at_utc=utc_now_text(),
        data_path="/tmp/data.npz",
        start_utc="2025-05-08 00:00:00",
        end_utc="2025-05-15 00:00:00",
        start_step=100,
        max_steps=128,
        candidate_ids=["cand_a", "cand_b"],
        candidate_curves=[
            CandidateCurveSnapshot(
                candidate_id="cand_a",
                display_name="wf_a",
                steps_run=128,
                sample_indices=[0, 64, 127],
                normalized_balance_history=[1.0, 1.03, 1.08],
                final_balance=10080.0,
                max_drawdown_pct=4.0,
            ),
            CandidateCurveSnapshot(
                candidate_id="cand_b",
                display_name="wf_b",
                steps_run=128,
                sample_indices=[0, 64, 127],
                normalized_balance_history=[1.0, 0.99, 0.998],
                final_balance=9980.0,
                max_drawdown_pct=6.0,
            ),
        ],
        pair_stats=[
            DiversificationPairStats(
                left_candidate_id="cand_a",
                right_candidate_id="cand_b",
                left_display_name="wf_a",
                right_display_name="wf_b",
                steps=128,
                return_corr=-0.2,
                downside_corr=0.1,
                simultaneous_loss_rate=0.25,
                simultaneous_drawdown_rate=0.20,
                action_agreement_rate=0.35,
                same_nonflat_rate=0.20,
                opposite_nonflat_rate=0.10,
                equal_weight_final_balance=10040.0,
                equal_weight_net_profit=40.0,
                equal_weight_max_drawdown_pct=3.5,
                avg_individual_max_drawdown_pct=5.0,
                drawdown_improvement_pct_points=1.5,
            )
        ],
    )
    registry.save_diversification_report(diversification)

    cluster_report = ClusterReport(
        schema_version="1",
        name="clusters",
        created_at_utc=utc_now_text(),
        source_diversification_report="common_window",
        data_path="/tmp/data.npz",
        start_utc="2025-05-08 00:00:00",
        end_utc="2025-05-15 00:00:00",
        start_step=100,
        max_steps=128,
        similarity_threshold=0.40,
        candidate_ids=["cand_a", "cand_b"],
        assignments=[
            ClusterAssignment(candidate_id="cand_a", display_name="wf_a", cluster_id="cluster_000"),
            ClusterAssignment(candidate_id="cand_b", display_name="wf_b", cluster_id="cluster_001"),
        ],
        clusters=[
            ClusterSummary(
                cluster_id="cluster_000",
                candidate_ids=["cand_a"],
                display_names=["wf_a"],
                cluster_size=1,
                mean_return_corr=0.0,
                mean_downside_corr=0.0,
                mean_simultaneous_loss_rate=0.0,
                mean_similarity_score=0.0,
            ),
            ClusterSummary(
                cluster_id="cluster_001",
                candidate_ids=["cand_b"],
                display_names=["wf_b"],
                cluster_size=1,
                mean_return_corr=0.0,
                mean_downside_corr=0.0,
                mean_simultaneous_loss_rate=0.0,
                mean_similarity_score=0.0,
            ),
        ],
    )
    registry.save_cluster_report(cluster_report)

    override_set = OverrideSet(
        schema_version="1",
        name="ops_test",
        created_at_utc=utc_now_text(),
        updated_at_utc=utc_now_text(),
        source_cluster_report="clusters",
        description="test overrides",
        candidate_overrides=[
            CandidateOverride(candidate_id="cand_a", pin=True, note="keep visible"),
        ],
        cluster_overrides=[
            ClusterOverride(cluster_id="cluster_000", max_cap_fraction=0.5, note="cap cluster"),
        ],
        audit_entries=[
            OverrideAuditEntry(
                created_at_utc=utc_now_text(),
                actor="operator",
                target_type="candidate",
                target_id="cand_a",
                action="update_candidate_override",
                changes={"pin": True},
                note="keep visible",
            )
        ],
    )
    registry.save_override_set(override_set)

    shortlist = ShortlistReport(
        schema_version="1",
        name="shortlist",
        created_at_utc=utc_now_text(),
        source_diversification_report="common_window",
        data_path="/tmp/data.npz",
        start_utc="2025-05-08 00:00:00",
        end_utc="2025-05-15 00:00:00",
        start_step=100,
        max_steps=128,
        resampling_name="train_bootstrap_f1.00",
        candidate_ids=["cand_a", "cand_b"],
        selected_candidate_ids=["cand_a"],
        selection_config={"max_candidates": 1},
        candidate_scores=[
            ShortlistCandidateScore(
                candidate_id="cand_a",
                display_name="wf_a",
                selected=True,
                selected_rank=1,
                base_score=2.0,
                marginal_score=2.0,
                brightness_hint=1.0,
                score_components={"oos": 1.0},
                exception_flags=[],
            ),
            ShortlistCandidateScore(
                candidate_id="cand_b",
                display_name="wf_b",
                selected=False,
                selected_rank=None,
                base_score=1.1,
                marginal_score=1.1,
                brightness_hint=0.55,
                score_components={"oos": -0.2},
                exception_flags=["near_miss"],
            ),
        ],
        selected_pair_scores=[
            ShortlistPairScore(
                left_candidate_id="cand_a",
                right_candidate_id="cand_b",
                compatibility_score=0.8,
                downside_corr=0.1,
                simultaneous_loss_rate=0.25,
                action_agreement_rate=0.35,
                drawdown_bonus=1.0,
                downside_penalty=0.2,
                simultaneous_loss_penalty=0.1,
                action_agreement_penalty=0.05,
            )
        ],
    )
    registry.save_shortlist_report(shortlist)

    allocator = AllocatorWorkbenchReport(
        schema_version="1",
        name="allocator",
        created_at_utc=utc_now_text(),
        source_shortlist_report="shortlist",
        source_diversification_report="common_window",
        source_cluster_report="clusters",
        source_override_set="ops_test",
        data_path="/tmp/data.npz",
        start_utc="2025-05-08 00:00:00",
        end_utc="2025-05-15 00:00:00",
        start_step=100,
        max_steps=128,
        selected_candidate_ids=["cand_a"],
        requested_risk_fractions=[0.25],
        chosen_scenario_name="risk_0.25",
        objective_config={"mode": "base"},
        scenarios=[
            AllocatorScenario(
                name="risk_0.25",
                objective_score=0.4,
                requested_risk_fraction=0.25,
                allocated_risk_fraction=0.25,
                reserve_fraction=0.75,
                per_system_cap_fraction=0.5,
                score_mode="base",
                curve_sample_indices=[0, 64, 127],
                normalized_balance_history=[1.0, 1.01, 1.02],
                weights=[
                    AllocationWeight(
                        candidate_id="cand_a",
                        display_name="wf_a",
                        raw_score=2.0,
                        normalized_share=1.0,
                        capital_fraction=0.25,
                        cluster_id="cluster_000",
                        capped=False,
                    )
                ],
                resampling=_portfolio_stats("risk_0.25", 0.25, 0.25),
            )
        ],
    )
    registry.save_allocator_workbench(allocator)

    combination = CombinationSearchReport(
        schema_version="1",
        name="combos",
        created_at_utc=utc_now_text(),
        source_shortlist_report="shortlist",
        source_diversification_report="common_window",
        source_cluster_report="clusters",
        source_override_set="ops_test",
        data_path="/tmp/data.npz",
        start_utc="2025-05-08 00:00:00",
        end_utc="2025-05-15 00:00:00",
        start_step=100,
        max_steps=128,
        pool_candidate_ids=["cand_a", "cand_b"],
        searched_subset_sizes=[1, 2],
        evaluated_combination_count=3,
        evaluated_scenario_count=3,
        best_scenario_name="subset_ab",
        objective_config={"mode": "base"},
        scenarios=[
            CombinationScenario(
                name="subset_ab",
                subset_candidate_ids=["cand_a", "cand_b"],
                subset_display_names=["wf_a", "wf_b"],
                subset_size=2,
                objective_score=0.5,
                requested_risk_fraction=0.25,
                allocated_risk_fraction=0.25,
                reserve_fraction=0.75,
                score_mode="base",
                curve_sample_indices=[0, 64, 127],
                normalized_balance_history=[1.0, 1.005, 1.015],
                weights=[
                    AllocationWeight(
                        candidate_id="cand_a",
                        display_name="wf_a",
                        raw_score=2.0,
                        normalized_share=0.7,
                        capital_fraction=0.175,
                        cluster_id="cluster_000",
                        capped=False,
                    ),
                    AllocationWeight(
                        candidate_id="cand_b",
                        display_name="wf_b",
                        raw_score=1.1,
                        normalized_share=0.3,
                        capital_fraction=0.075,
                        cluster_id="cluster_001",
                        capped=False,
                    ),
                ],
                resampling=_portfolio_stats("subset_ab", 0.25, 0.25),
            )
        ],
    )
    registry.save_combination_search(combination)

    feed = build_dashboard_feed(
        registry,
        "dashboard_test",
        shortlist_report=shortlist,
        diversification_report=diversification,
        cluster_report=cluster_report,
        override_set=override_set,
        allocator_report=allocator,
        combination_report=combination,
    )

    assert feed.summary["total_candidates"] == 2
    assert feed.summary["selected_candidate_count"] == 1
    assert feed.summary["pinned_candidate_count"] == 1
    assert feed.broom is not None
    assert feed.broom["line_count"] == 2
    assert feed.allocator is not None
    assert feed.allocator["chosen_scenario_name"] == "risk_0.25"
    assert feed.combinations is not None
    assert feed.combinations["best_scenario_name"] == "subset_ab"

    row_by_id = {item["candidate_id"]: item for item in feed.candidates}
    assert row_by_id["cand_a"]["cluster_id"] == "cluster_000"
    assert row_by_id["cand_a"]["shortlist"]["selected"] is True
    assert row_by_id["cand_a"]["overrides"]["pin"] is True
    assert row_by_id["cand_b"]["shortlist"]["exception_flags"] == ["near_miss"]

    path = registry.save_dashboard_feed(feed)
    assert path.exists()
    loaded = registry.load_dashboard_feed("dashboard_test")
    assert isinstance(loaded, DashboardFeed)
    assert loaded.summary["cluster_count"] == 2
