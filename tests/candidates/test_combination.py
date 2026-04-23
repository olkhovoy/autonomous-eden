from __future__ import annotations

from umc_nn.candidates import (
    CandidateCurveSnapshot,
    CandidateRecord,
    CandidateRegistry,
    DiversificationPairStats,
    DiversificationReport,
    ExperimentManifest,
    PeriodStats,
    ResamplingStats,
    ShortlistConfig,
    build_shortlist_report,
    utc_now_text,
)


def _candidate(
    candidate_id: str,
    display_name: str,
    *,
    train_pnl: float,
    oos_pnl: float,
    oos_trades: int,
    oos_beats_flat: bool,
    resampling_pess_net: float,
    profitable_rate: float,
    pessimistic_dd: float,
) -> CandidateRecord:
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
        search_config={"population": 8},
        econ_config={"initial_balance": 10000.0, "exchange": "binance"},
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
                    "requested_max_steps": 100,
                    "steps_run": 100,
                    "final_balance": 10000.0 + train_pnl,
                    "pnl": train_pnl,
                    "max_drawdown_pct": 3.0,
                    "trades": 24,
                    "wins": 13,
                    "win_rate_pct": 54.0,
                    "action_counts": {"0": 50, "1": 30, "2": 20},
                    "position_counts": {"0": 50, "1": 30, "2": 20},
                },
            ),
            "oos": PeriodStats.from_episode(
                "oos",
                {
                    "start_step": 100,
                    "requested_max_steps": 100,
                    "steps_run": 100,
                    "final_balance": 10000.0 + oos_pnl,
                    "pnl": oos_pnl,
                    "max_drawdown_pct": 4.0,
                    "trades": oos_trades,
                    "wins": max(0, int(round(oos_trades * 0.5))),
                    "win_rate_pct": 50.0,
                    "action_counts": {"0": 40, "1": 35, "2": 25},
                    "position_counts": {"0": 40, "1": 35, "2": 25},
                },
                beats_flat=oos_beats_flat,
                full_window=True,
            ),
        },
        selection_flags={"oos_beats_flat": oos_beats_flat, "oos_positive": oos_pnl > 0.0},
        resampling_results={
            "train_bootstrap_f1.00": ResamplingStats(
                name="train_bootstrap_f1.00",
                period_name="train",
                iterations=100,
                sample_size=24,
                seed=42,
                sizing_mode="fractional_returns",
                fraction=1.0,
                initial_balance=10000.0,
                original_trade_count=24,
                original_final_balance=10000.0 + train_pnl,
                original_net_profit=train_pnl,
                original_max_drawdown_pct=3.0,
                mean_final_balance=10000.0 + train_pnl,
                median_final_balance=10000.0 + train_pnl,
                p05_final_balance=10000.0 + resampling_pess_net,
                p25_final_balance=10000.0 + max(resampling_pess_net, train_pnl * 0.25),
                mean_net_profit=train_pnl,
                median_net_profit=train_pnl,
                p05_net_profit=resampling_pess_net,
                p25_net_profit=max(resampling_pess_net, train_pnl * 0.25),
                mean_max_drawdown_pct=3.0,
                median_max_drawdown_pct=3.0,
                p75_max_drawdown_pct=4.0,
                p95_max_drawdown_pct=max(4.0, pessimistic_dd),
                profitable_rate=profitable_rate,
                loss_rate=1.0 - profitable_rate,
                ruin_rate=0.0,
                pessimistic_net_profit=resampling_pess_net,
                pessimistic_max_drawdown_pct=pessimistic_dd,
            )
        },
    )


def _curve(candidate_id: str, display_name: str, values: list[float]) -> CandidateCurveSnapshot:
    return CandidateCurveSnapshot(
        candidate_id=candidate_id,
        display_name=display_name,
        steps_run=len(values) - 1,
        sample_indices=list(range(len(values))),
        normalized_balance_history=values,
        final_balance=10000.0 * values[-1],
        max_drawdown_pct=5.0,
    )


def test_shortlist_prefers_diversified_pair() -> None:
    cand_a = _candidate(
        "cand_a",
        "A",
        train_pnl=500.0,
        oos_pnl=300.0,
        oos_trades=32,
        oos_beats_flat=True,
        resampling_pess_net=120.0,
        profitable_rate=0.92,
        pessimistic_dd=1.0,
    )
    cand_b = _candidate(
        "cand_b",
        "B",
        train_pnl=450.0,
        oos_pnl=180.0,
        oos_trades=30,
        oos_beats_flat=True,
        resampling_pess_net=90.0,
        profitable_rate=0.88,
        pessimistic_dd=1.1,
    )
    cand_c = _candidate(
        "cand_c",
        "C",
        train_pnl=200.0,
        oos_pnl=-40.0,
        oos_trades=26,
        oos_beats_flat=False,
        resampling_pess_net=25.0,
        profitable_rate=0.78,
        pessimistic_dd=1.5,
    )

    report = DiversificationReport(
        schema_version="1",
        name="common_window",
        created_at_utc=utc_now_text(),
        data_path="/tmp/data.npz",
        start_utc="2025-02-01 00:00:00",
        end_utc="2025-02-08 00:00:00",
        start_step=0,
        max_steps=100,
        candidate_ids=["cand_a", "cand_b", "cand_c"],
        candidate_curves=[
            _curve("cand_a", "A", [1.0, 1.02, 1.03]),
            _curve("cand_b", "B", [1.0, 1.015, 1.018]),
            _curve("cand_c", "C", [1.0, 0.99, 1.01]),
        ],
        pair_stats=[
            DiversificationPairStats(
                left_candidate_id="cand_a",
                right_candidate_id="cand_b",
                left_display_name="A",
                right_display_name="B",
                steps=100,
                return_corr=0.80,
                downside_corr=0.90,
                simultaneous_loss_rate=0.80,
                simultaneous_drawdown_rate=0.85,
                action_agreement_rate=0.95,
                same_nonflat_rate=0.90,
                opposite_nonflat_rate=0.00,
                equal_weight_final_balance=10100.0,
                equal_weight_net_profit=100.0,
                equal_weight_max_drawdown_pct=6.0,
                avg_individual_max_drawdown_pct=6.5,
                drawdown_improvement_pct_points=0.5,
            ),
            DiversificationPairStats(
                left_candidate_id="cand_a",
                right_candidate_id="cand_c",
                left_display_name="A",
                right_display_name="C",
                steps=100,
                return_corr=-0.50,
                downside_corr=0.10,
                simultaneous_loss_rate=0.15,
                simultaneous_drawdown_rate=0.25,
                action_agreement_rate=0.20,
                same_nonflat_rate=0.10,
                opposite_nonflat_rate=0.40,
                equal_weight_final_balance=10250.0,
                equal_weight_net_profit=250.0,
                equal_weight_max_drawdown_pct=3.0,
                avg_individual_max_drawdown_pct=6.0,
                drawdown_improvement_pct_points=3.0,
            ),
            DiversificationPairStats(
                left_candidate_id="cand_b",
                right_candidate_id="cand_c",
                left_display_name="B",
                right_display_name="C",
                steps=100,
                return_corr=-0.20,
                downside_corr=0.12,
                simultaneous_loss_rate=0.20,
                simultaneous_drawdown_rate=0.28,
                action_agreement_rate=0.30,
                same_nonflat_rate=0.12,
                opposite_nonflat_rate=0.25,
                equal_weight_final_balance=10180.0,
                equal_weight_net_profit=180.0,
                equal_weight_max_drawdown_pct=3.5,
                avg_individual_max_drawdown_pct=5.5,
                drawdown_improvement_pct_points=2.0,
            ),
        ],
    )

    shortlist = build_shortlist_report(
        name="shortlist_week1",
        created_at_utc=utc_now_text(),
        candidates=[cand_a, cand_b, cand_c],
        diversification_report=report,
        config=ShortlistConfig(resampling_name="train_bootstrap_f1.00", max_candidates=2, min_marginal_score=0.0),
    )

    assert shortlist.selected_candidate_ids == ["cand_a", "cand_c"]
    selected_scores = [item for item in shortlist.candidate_scores if item.selected]
    assert [item.candidate_id for item in selected_scores] == ["cand_a", "cand_c"]
    assert all(0.0 < item.brightness_hint <= 1.0 for item in shortlist.candidate_scores)


def test_registry_saves_and_loads_shortlist_report(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    candidate = _candidate(
        "cand_a",
        "A",
        train_pnl=500.0,
        oos_pnl=300.0,
        oos_trades=20,
        oos_beats_flat=True,
        resampling_pess_net=120.0,
        profitable_rate=0.92,
        pessimistic_dd=1.0,
    )
    registry.add_candidate(candidate)
    report = DiversificationReport(
        schema_version="1",
        name="diversification",
        created_at_utc=utc_now_text(),
        data_path="/tmp/data.npz",
        start_utc="2025-02-01 00:00:00",
        end_utc="2025-02-08 00:00:00",
        start_step=0,
        max_steps=100,
        candidate_ids=["cand_a"],
        candidate_curves=[_curve("cand_a", "A", [1.0, 1.01, 1.02])],
        pair_stats=[],
    )
    shortlist = build_shortlist_report(
        name="shortlist",
        created_at_utc=utc_now_text(),
        candidates=[candidate],
        diversification_report=report,
        config=ShortlistConfig(resampling_name="train_bootstrap_f1.00", max_candidates=1, min_marginal_score=0.0),
    )

    path = registry.save_shortlist_report(shortlist)
    assert path.exists()

    loaded = registry.load_shortlist_report("shortlist")
    assert loaded.selected_candidate_ids == ["cand_a"]
    assert loaded.candidate_scores[0].candidate_id == "cand_a"
