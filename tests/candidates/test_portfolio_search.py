from __future__ import annotations

from typing import Iterable

from umc_nn.candidates import (
    AllocatorConfig,
    CandidateCurveSnapshot,
    CandidateRecord,
    CandidateRegistry,
    CombinationSearchConfig,
    DiversificationReport,
    ExperimentManifest,
    PeriodStats,
    ShortlistCandidateScore,
    ShortlistReport,
    build_combination_search_report,
    empty_override_set,
    shortlist_pool_candidate_ids,
    update_candidate_override,
    utc_now_text,
)


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
                    "requested_max_steps": 8,
                    "steps_run": 8,
                    "final_balance": 10100.0,
                    "pnl": 100.0,
                    "max_drawdown_pct": 2.0,
                    "trades": 8,
                    "wins": 4,
                    "win_rate_pct": 50.0,
                    "action_counts": {"0": 4, "1": 4},
                    "position_counts": {"0": 4, "1": 4},
                },
            ),
            "oos": PeriodStats.from_episode(
                "oos",
                {
                    "start_step": 8,
                    "requested_max_steps": 8,
                    "steps_run": 8,
                    "final_balance": 10050.0,
                    "pnl": 50.0,
                    "max_drawdown_pct": 2.5,
                    "trades": 6,
                    "wins": 3,
                    "win_rate_pct": 50.0,
                    "action_counts": {"0": 4, "1": 4},
                    "position_counts": {"0": 4, "1": 4},
                },
                beats_flat=True,
                full_window=True,
            ),
        },
    )


def _diversification_report(candidate_ids: Iterable[str]) -> DiversificationReport:
    ids = list(candidate_ids)
    return DiversificationReport(
        schema_version="1",
        name="common_window",
        created_at_utc=utc_now_text(),
        data_path="/tmp/data.npz",
        start_utc="2025-02-01 00:00:00",
        end_utc="2025-02-08 00:00:00",
        start_step=0,
        max_steps=8,
        candidate_ids=ids,
        candidate_curves=[
            CandidateCurveSnapshot(
                candidate_id=candidate_id,
                display_name=candidate_id.upper(),
                steps_run=8,
                sample_indices=[0, 1, 2],
                normalized_balance_history=[1.0, 1.01, 1.02],
                final_balance=10200.0,
                max_drawdown_pct=2.0,
            )
            for candidate_id in ids
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
        candidate_ids=["cand_a", "cand_b", "cand_c", "cand_d"],
        selected_candidate_ids=["cand_a", "cand_c"],
        selection_config={},
        candidate_scores=[
            ShortlistCandidateScore(
                candidate_id="cand_a",
                display_name="A",
                selected=True,
                selected_rank=1,
                base_score=2.0,
                marginal_score=2.0,
                brightness_hint=1.0,
                score_components={},
            ),
            ShortlistCandidateScore(
                candidate_id="cand_c",
                display_name="C",
                selected=True,
                selected_rank=2,
                base_score=1.8,
                marginal_score=1.7,
                brightness_hint=0.9,
                score_components={},
            ),
            ShortlistCandidateScore(
                candidate_id="cand_b",
                display_name="B",
                selected=False,
                selected_rank=None,
                base_score=1.2,
                marginal_score=None,
                brightness_hint=0.6,
                score_components={},
                exception_flags=["diversifier_outlier"],
            ),
            ShortlistCandidateScore(
                candidate_id="cand_d",
                display_name="D",
                selected=False,
                selected_rank=None,
                base_score=1.5,
                marginal_score=None,
                brightness_hint=0.7,
                score_components={},
            ),
        ],
        selected_pair_scores=[],
    )


def test_shortlist_pool_candidate_ids_include_selected_and_exceptions() -> None:
    pool = shortlist_pool_candidate_ids(
        _shortlist_report(),
        max_pool_size=3,
        include_selected=True,
        include_exception_flags=True,
    )
    assert pool == ["cand_a", "cand_c", "cand_b"]


def test_combination_search_prefers_diversified_pair() -> None:
    shortlist = _shortlist_report()
    candidates = [_candidate("cand_a", "A"), _candidate("cand_b", "B"), _candidate("cand_c", "C")]
    report = build_combination_search_report(
        name="combo_search",
        created_at_utc=utc_now_text(),
        candidates=candidates,
        shortlist_report=shortlist,
        diversification_report=_diversification_report(["cand_a", "cand_b", "cand_c"]),
        config=CombinationSearchConfig(
            max_pool_size=3,
            min_subset_size=1,
            max_subset_size=2,
            include_selected=True,
            include_exception_flags=True,
            max_stored_scenarios=16,
            allocator_config=AllocatorConfig(
                risk_fractions=(1.0,),
                per_system_cap_fraction=0.6,
                score_mode="base",
                min_score_floor=0.05,
                resampling_iterations=100,
                resampling_block_size=2,
                resampling_seed=11,
                objective_max_drawdown_pct=10.0,
                curve_points=32,
            ),
        ),
        step_return_views={
            "cand_a": [0.06, -0.055, 0.06, -0.055, 0.06, -0.055, 0.06, -0.055],
            "cand_b": [-0.085, 0.085, -0.085, 0.085, -0.085, 0.085, -0.085, 0.085],
            "cand_c": [0.06, -0.080, 0.06, -0.080, 0.06, -0.080, 0.06, -0.080],
        },
    )

    assert report.evaluated_combination_count == 6
    assert report.evaluated_scenario_count == 6
    best = report.scenarios[0]
    assert set(best.subset_candidate_ids) == {"cand_a", "cand_b"}
    assert best.subset_size == 2


def test_registry_saves_and_loads_combination_search(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    cand_a = _candidate("cand_a", "A")
    cand_b = _candidate("cand_b", "B")
    registry.add_candidate(cand_a)
    registry.add_candidate(cand_b)
    shortlist = _shortlist_report()
    shortlist.candidate_ids = ["cand_a", "cand_b"]
    shortlist.selected_candidate_ids = ["cand_a", "cand_b"]
    shortlist.candidate_scores = [
        ShortlistCandidateScore(
            candidate_id="cand_a",
            display_name="A",
            selected=True,
            selected_rank=1,
            base_score=2.0,
            marginal_score=2.0,
            brightness_hint=1.0,
            score_components={},
        ),
        ShortlistCandidateScore(
            candidate_id="cand_b",
            display_name="B",
            selected=True,
            selected_rank=2,
            base_score=1.2,
            marginal_score=1.1,
            brightness_hint=0.7,
            score_components={},
        ),
    ]
    report = build_combination_search_report(
        name="combo_search",
        created_at_utc=utc_now_text(),
        candidates=[cand_a, cand_b],
        shortlist_report=shortlist,
        diversification_report=_diversification_report(["cand_a", "cand_b"]),
        config=CombinationSearchConfig(
            max_pool_size=2,
            min_subset_size=1,
            max_subset_size=2,
            include_selected=True,
            include_exception_flags=False,
            max_stored_scenarios=8,
            allocator_config=AllocatorConfig(
                risk_fractions=(0.5,),
                per_system_cap_fraction=0.5,
                score_mode="base",
                min_score_floor=0.05,
                resampling_iterations=50,
                resampling_block_size=2,
                resampling_seed=7,
                objective_max_drawdown_pct=10.0,
                curve_points=16,
            ),
        ),
        step_return_views={
            "cand_a": [0.01, -0.005, 0.012, -0.004, 0.008, -0.003, 0.005, 0.002],
            "cand_b": [-0.004, 0.009, -0.001, 0.006, -0.002, 0.008, -0.003, 0.004],
        },
    )

    path = registry.save_combination_search(report)
    assert path.exists()
    loaded = registry.load_combination_search("combo_search")
    assert loaded.evaluated_combination_count == 3
    assert loaded.best_scenario_name is not None


def test_combination_search_respects_pin_and_exclude_overrides() -> None:
    shortlist = _shortlist_report()
    shortlist.candidate_ids = ["cand_a", "cand_b", "cand_c"]
    shortlist.candidate_scores = shortlist.candidate_scores[:3]
    override_set = empty_override_set(name="ops")
    override_set = update_candidate_override(
        override_set,
        candidate_id="cand_a",
        actor="operator",
        exclude=True,
    )
    override_set = update_candidate_override(
        override_set,
        candidate_id="cand_b",
        actor="operator",
        pin=True,
    )
    candidates = [_candidate("cand_a", "A"), _candidate("cand_b", "B"), _candidate("cand_c", "C")]
    report = build_combination_search_report(
        name="combo_search",
        created_at_utc=utc_now_text(),
        candidates=candidates,
        shortlist_report=shortlist,
        diversification_report=_diversification_report(["cand_a", "cand_b", "cand_c"]),
        override_set=override_set,
        config=CombinationSearchConfig(
            max_pool_size=3,
            min_subset_size=1,
            max_subset_size=2,
            include_selected=True,
            include_exception_flags=True,
            max_stored_scenarios=16,
            allocator_config=AllocatorConfig(
                risk_fractions=(1.0,),
                per_system_cap_fraction=0.6,
                score_mode="base",
                min_score_floor=0.05,
                resampling_iterations=50,
                resampling_block_size=2,
                resampling_seed=11,
                objective_max_drawdown_pct=10.0,
                curve_points=16,
            ),
        ),
        step_return_views={
            "cand_a": [0.06, -0.055, 0.06, -0.055, 0.06, -0.055, 0.06, -0.055],
            "cand_b": [-0.085, 0.085, -0.085, 0.085, -0.085, 0.085, -0.085, 0.085],
            "cand_c": [0.06, -0.080, 0.06, -0.080, 0.06, -0.080, 0.06, -0.080],
        },
    )

    assert "cand_a" not in report.pool_candidate_ids
    assert "cand_b" in report.pool_candidate_ids
    assert all("cand_b" in scenario.subset_candidate_ids for scenario in report.scenarios)
