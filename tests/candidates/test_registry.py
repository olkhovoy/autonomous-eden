from __future__ import annotations

from umc_nn.candidates import (
    CandidateRecord,
    CandidateRegistry,
    ExperimentManifest,
    PeriodSpan,
    PeriodStats,
    ResamplingStats,
    TracePeriodRecord,
    TradeRecord,
    utc_now_text,
)


def _candidate() -> CandidateRecord:
    manifest = ExperimentManifest(
        schema_version="1",
        created_at_utc=utc_now_text(),
        source_script="/tmp/source.py",
        engine_name="monolith",
        engine_role="candidate_engine",
        representation_name="neurobars",
        data_path="/tmp/data.npz",
        checkpoint_path="/tmp/checkpoint.npy",
        log_path="/tmp/run.log",
        source_summary_path="/tmp/summary.json",
        train_window_utc={"start": "2024-01-01 00:00:00", "end": "2024-01-08 00:00:00"},
        oos_window_utc={"start": "2024-01-08 00:00:00", "end": "2024-01-15 00:00:00"},
        search_config={"generations": 8},
        econ_config={"exchange": "binance"},
    )
    return CandidateRecord(
        schema_version="1",
        candidate_id="cand_test123456",
        display_name="wf_01_run01",
        engine_name="monolith",
        engine_role="candidate_engine",
        status="research",
        created_at_utc=utc_now_text(),
        engine_family="umc",
        engine_config_id="eng_test",
        representation_branch="fused32",
        fitness_profile="hunter",
        train_span=PeriodSpan.from_window(
            "train",
            start_utc="2024-01-01 00:00:00",
            end_utc="2024-01-08 00:00:00",
            start_step=100,
            max_steps=10,
        ),
        valid_span=PeriodSpan.from_window(
            "valid",
            start_utc="2024-01-08 00:00:00",
            end_utc="2024-01-10 00:00:00",
            start_step=110,
            max_steps=2,
        ),
        oos_adjacent_span=PeriodSpan.from_window(
            "oos_adjacent",
            start_utc="2024-01-10 00:00:00",
            end_utc="2024-01-15 00:00:00",
            start_step=112,
            max_steps=5,
        ),
        total_span=PeriodSpan.from_window(
            "total",
            start_utc="2024-01-01 00:00:00",
            end_utc="2024-01-15 00:00:00",
            start_step=100,
            max_steps=15,
        ),
        tags=["walkforward", "probe"],
        manifest=manifest,
        periods={
            "train": PeriodStats.from_episode(
                "train",
                {
                    "start_step": 100,
                    "requested_max_steps": 10,
                    "steps_run": 10,
                    "final_balance": 10100.0,
                    "pnl": 100.0,
                    "max_drawdown_pct": 2.0,
                    "trades": 8,
                    "wins": 4,
                    "win_rate_pct": 50.0,
                    "action_counts": {"0": 4, "1": 6},
                    "position_counts": {"0": 5, "1": 6},
                },
                beats_flat=True,
                full_window=True,
            ),
            "oos": PeriodStats.from_episode(
                "oos",
                {
                    "start_step": 110,
                    "requested_max_steps": 10,
                    "steps_run": 10,
                    "final_balance": 10050.0,
                    "pnl": 50.0,
                    "max_drawdown_pct": 3.0,
                    "trades": 5,
                    "wins": 3,
                    "win_rate_pct": 60.0,
                    "action_counts": {"0": 2, "1": 8},
                    "position_counts": {"0": 3, "1": 8},
                },
                beats_flat=True,
                beats_best_baseline=False,
                full_window=True,
            ),
        },
        selection_flags={"oos_positive": True, "oos_beats_flat": True},
    )


def test_registry_add_load_and_index(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    candidate = _candidate()

    path = registry.add_candidate(candidate)
    assert path.exists()

    loaded = registry.load_candidate(candidate.candidate_id)
    assert loaded.display_name == candidate.display_name
    assert loaded.periods["oos"].pnl == 50.0
    assert loaded.oos_adjacent_span is not None
    assert loaded.train_span is not None
    assert loaded.engine_family == "umc"

    index_payload = registry.index_path.read_text()
    assert candidate.candidate_id in index_payload


def test_registry_update_status_and_tags(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    candidate = _candidate()
    registry.add_candidate(candidate)

    updated = registry.update_status(
        candidate.candidate_id,
        "approved",
        note="manual review passed",
        add_tags=["selected"],
    )
    assert updated.status == "approved"
    assert "selected" in updated.tags
    assert "manual review passed" in (updated.notes or "")

    approved = registry.list_candidates(status="approved")
    assert len(approved) == 1
    assert approved[0].candidate_id == candidate.candidate_id


def test_registry_attaches_trade_and_resampling_artifacts(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    candidate = _candidate()
    registry.add_candidate(candidate)

    trade = TradeRecord(
        trade_id="train_t1",
        period_name="train",
        direction="long",
        entry_step=0,
        exit_step=1,
        entry_timestamp_utc=None,
        exit_timestamp_utc=None,
        entry_price=100.0,
        exit_price=101.0,
        quantity=1.0,
        gross_pnl=105.0,
        net_pnl=100.0,
        fees_paid=5.0,
        duration_steps=1,
        entry_balance=10000.0,
        exit_balance=10100.0,
        return_on_equity=0.01,
    )
    trades_path = registry.attach_trade_records(candidate.candidate_id, [trade])
    assert trades_path.exists()
    loaded_trades = registry.load_trade_records(candidate.candidate_id)
    assert len(loaded_trades) == 1
    assert loaded_trades[0].trade_id == "train_t1"

    trace = TracePeriodRecord(
        period_name="train",
        source="test",
        start_step=0,
        requested_max_steps=1,
        steps_run=1,
        start_utc=None,
        end_utc=None,
        balance_history=[10000.0, 10100.0],
        action_history=[1],
        position_history=[0, 1],
    )
    traces_path = registry.attach_trace_records(candidate.candidate_id, [trace])
    assert traces_path.exists()
    loaded_traces = registry.load_trace_records(candidate.candidate_id)
    assert len(loaded_traces) == 1
    assert loaded_traces[0].period_name == "train"

    stats = ResamplingStats(
        name="bootstrap_f1.00",
        period_name="train",
        iterations=100,
        sample_size=1,
        seed=42,
        sizing_mode="fractional_returns",
        fraction=1.0,
        initial_balance=10000.0,
        original_trade_count=1,
        original_final_balance=10100.0,
        original_net_profit=100.0,
        original_max_drawdown_pct=0.0,
        mean_final_balance=10100.0,
        median_final_balance=10100.0,
        p05_final_balance=10100.0,
        p25_final_balance=10100.0,
        mean_net_profit=100.0,
        median_net_profit=100.0,
        p05_net_profit=100.0,
        p25_net_profit=100.0,
        mean_max_drawdown_pct=0.0,
        median_max_drawdown_pct=0.0,
        p75_max_drawdown_pct=0.0,
        p95_max_drawdown_pct=0.0,
        profitable_rate=1.0,
        loss_rate=0.0,
        ruin_rate=0.0,
        pessimistic_net_profit=100.0,
        pessimistic_max_drawdown_pct=0.0,
    )
    resampling_path = registry.attach_resampling_results(candidate.candidate_id, {"bootstrap_f1.00": stats})
    assert resampling_path.exists()

    loaded = registry.load_candidate(candidate.candidate_id)
    assert loaded.trade_records_path is not None
    assert loaded.trace_records_path is not None
    assert "bootstrap_f1.00" in loaded.resampling_results
