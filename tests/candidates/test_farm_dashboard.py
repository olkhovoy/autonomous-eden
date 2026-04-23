from __future__ import annotations

from umc_nn.candidates import (
    CandidateRegistry,
    FarmProgressEvent,
    FarmProgressLog,
    RollingConveyorReport,
    RollingCycleOutcome,
    build_candidate_farm_report,
    build_candidate_farm_scenario_report,
    build_farm_dashboard_feed,
)


def _rolling_report(name: str, start: str, end: str, balances: list[float], pnl: float, dd: float) -> RollingConveyorReport:
    return RollingConveyorReport(
        schema_version="1",
        name=name,
        created_at_utc="2026-03-10T00:00:00Z",
        mode="reuse",
        selection_days=7,
        forward_days=7,
        step_days=7,
        initial_balance=float(balances[0]),
        final_balance=float(balances[-1]),
        total_pnl=float(pnl),
        total_return_pct=((float(balances[-1]) / float(balances[0])) - 1.0) * 100.0,
        max_drawdown_pct=float(dd),
        positive_cycle_count=1 if pnl > 0 else 0,
        evaluated_cycle_count=len(balances) - 1,
        ledger_cycle_indices=list(range(len(balances))),
        ledger_balance_history=[float(item) for item in balances],
        cycle_outcomes=[
            RollingCycleOutcome(
                cycle_index=1,
                cycle_name=f"{name}_cycle_01",
                cycle_report_name=f"{name}_cycle_01",
                tradeforward_plan_name=f"{name}_plan_01",
                tradeforward_evaluation_name=f"{name}_eval_01",
                selection_start_utc=start,
                selection_end_utc=end,
                forward_start_utc=end,
                forward_end_utc="2025-05-22 00:00:00",
                candidate_ids=["cand_a", "cand_b"],
                selected_candidate_ids=["cand_a"],
                requested_risk_fraction=0.5,
                allocated_risk_fraction=0.35,
                reserve_fraction=0.65,
                portfolio_pnl=float(pnl),
                portfolio_final_balance=float(balances[-1]),
                portfolio_max_drawdown_pct=float(dd),
                cycle_return_fraction=(float(balances[-1]) / float(balances[0])) - 1.0,
                ledger_balance_before=float(balances[0]),
                ledger_balance_after=float(balances[-1]),
            )
        ],
        selector={"tags": ["walkforward", "probe", "fused32"]},
        notes=None,
    )


def test_build_farm_dashboard_feed_includes_scenario_rows_and_broom(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    rolling_a = _rolling_report("rolling_a", "2025-05-01 00:00:00", "2025-05-08 00:00:00", [10000.0, 10120.0, 10240.0], 240.0, 1.8)
    rolling_b = _rolling_report("rolling_b", "2025-05-08 00:00:00", "2025-05-15 00:00:00", [10000.0, 9950.0, 10110.0], 110.0, 2.6)
    registry.save_rolling_conveyor(rolling_a)
    registry.save_rolling_conveyor(rolling_b)

    scenario_a = build_candidate_farm_scenario_report(
        scenario_name="scenario_a",
        status="completed",
        mode="reuse",
        output_dir="/tmp/scenario_a",
        config={"rolling_args": {"selection_start_utc": "2025-05-01 00:00:00", "selection_days": 7, "num_cycles": 2}},
        log_paths={},
        progress_stage="completed",
        rolling_report=rolling_a,
        notes="a",
    )
    scenario_b = build_candidate_farm_scenario_report(
        scenario_name="scenario_b",
        status="running",
        mode="generate",
        output_dir="/tmp/scenario_b",
        config={"rolling_args": {"selection_start_utc": "2025-05-08 00:00:00", "selection_days": 7, "num_cycles": 2}},
        log_paths={},
        progress_stage="portfolio_baselines",
        rolling_report=rolling_b,
        notes="b",
    )
    farm = build_candidate_farm_report(
        name="farm_feed_test",
        registry_root=str(registry.root),
        scenarios=[scenario_a, scenario_b],
        source_manifest_path=str(tmp_path / "manifest.json"),
    )
    registry.save_farm_report(farm)
    registry.save_farm_progress_log(
        FarmProgressLog(
            schema_version="1",
            farm_name=farm.name,
            created_at_utc="2026-03-10T00:00:00Z",
            updated_at_utc="2026-03-10T00:08:00Z",
            events=[
                FarmProgressEvent(
                    sequence=1,
                    created_at_utc="2026-03-10T00:00:00Z",
                    scenario_name="scenario_a",
                    status="planned",
                    progress_stage="queued",
                    event_kind="queued",
                ),
                FarmProgressEvent(
                    sequence=2,
                    created_at_utc="2026-03-10T00:04:00Z",
                    scenario_name="scenario_a",
                    status="completed",
                    progress_stage="completed",
                    event_kind="completed",
                    gate_pass=True,
                    total_pnl=240.0,
                ),
                FarmProgressEvent(
                    sequence=3,
                    created_at_utc="2026-03-10T00:08:00Z",
                    scenario_name="scenario_b",
                    status="running",
                    progress_stage="portfolio_baselines",
                    event_kind="stage",
                ),
            ],
        )
    )

    feed = build_farm_dashboard_feed(
        registry,
        "farm_feed",
        farm_report=farm,
        max_scenarios=50,
        max_broom_lines=20,
    )

    assert feed.summary["scenario_count"] == 2
    assert feed.summary["running_scenarios"] == 1
    assert feed.summary["completed_or_reused_scenarios"] == 1
    assert len(feed.scenarios) == 2
    assert feed.scenarios[0]["scenario_name"] == "scenario_b"
    assert feed.scenarios[1]["scenario_name"] == "scenario_a"
    assert feed.broom is not None
    assert feed.broom["line_count"] == 1
    assert feed.broom["lines"][0]["scenario_name"] == "scenario_a"
    assert feed.broom["lines"][0]["normalized_balance_history"][0] == 1.0
    assert feed.monitoring["event_count"] == 3
    assert feed.monitoring["completion_event_count"] == 1
    assert feed.monitoring["gate_pass_completion_count"] == 1
    assert feed.monitoring["heartbeat_state"] in {"fresh", "watch", "stale"}
    assert feed.monitoring["stagnation_state"] in {"healthy", "stagnating", "searching", "pre-first-completion"}
    assert len(feed.monitoring["recent_events"]) == 3

    path = registry.save_farm_dashboard_feed(feed)
    assert path.exists()
    loaded = registry.load_farm_dashboard_feed("farm_feed")
    assert loaded.summary["scenario_count"] == 2
