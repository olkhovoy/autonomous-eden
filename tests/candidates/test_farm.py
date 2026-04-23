from __future__ import annotations

from umc_nn.candidates import (
    CandidateFarmScenarioReport,
    CandidateRegistry,
    FarmProgressEvent,
    FarmProgressLog,
    LifecycleReport,
    PortfolioBaselineReport,
    PortfolioGateResult,
    RollingConveyorReport,
    RollingCycleOutcome,
    build_candidate_farm_report,
    build_candidate_farm_scenario_report,
    expand_farm_manifest_scenarios,
)


def _rolling_report() -> RollingConveyorReport:
    return RollingConveyorReport(
        schema_version="1",
        name="rolling_farm_test",
        created_at_utc="2026-03-10T00:00:00Z",
        mode="reuse",
        selection_days=7,
        forward_days=7,
        step_days=7,
        initial_balance=10000.0,
        final_balance=10325.0,
        total_pnl=325.0,
        total_return_pct=3.25,
        max_drawdown_pct=2.4,
        positive_cycle_count=1,
        evaluated_cycle_count=2,
        ledger_cycle_indices=[1, 2],
        ledger_balance_history=[10000.0, 10150.0, 10325.0],
        cycle_outcomes=[
            RollingCycleOutcome(
                cycle_index=1,
                cycle_name="cycle_01",
                cycle_report_name="cycle_01",
                tradeforward_plan_name="plan_01",
                tradeforward_evaluation_name="eval_01",
                selection_start_utc="2025-05-01 00:00:00",
                selection_end_utc="2025-05-08 00:00:00",
                forward_start_utc="2025-05-08 00:00:00",
                forward_end_utc="2025-05-15 00:00:00",
                candidate_ids=["cand_a", "cand_b"],
                selected_candidate_ids=["cand_a"],
                requested_risk_fraction=0.5,
                allocated_risk_fraction=0.35,
                reserve_fraction=0.65,
                portfolio_pnl=150.0,
                portfolio_final_balance=10150.0,
                portfolio_max_drawdown_pct=1.8,
                cycle_return_fraction=0.015,
                ledger_balance_before=10000.0,
                ledger_balance_after=10150.0,
            ),
            RollingCycleOutcome(
                cycle_index=2,
                cycle_name="cycle_02",
                cycle_report_name="cycle_02",
                tradeforward_plan_name="plan_02",
                tradeforward_evaluation_name="eval_02",
                selection_start_utc="2025-05-08 00:00:00",
                selection_end_utc="2025-05-15 00:00:00",
                forward_start_utc="2025-05-15 00:00:00",
                forward_end_utc="2025-05-22 00:00:00",
                candidate_ids=["cand_b", "cand_c"],
                selected_candidate_ids=["cand_c"],
                requested_risk_fraction=0.5,
                allocated_risk_fraction=0.35,
                reserve_fraction=0.65,
                portfolio_pnl=175.0,
                portfolio_final_balance=10325.0,
                portfolio_max_drawdown_pct=2.4,
                cycle_return_fraction=0.017241,
                ledger_balance_before=10150.0,
                ledger_balance_after=10325.0,
            ),
        ],
        selector={"tags": ["walkforward", "probe", "fused32"]},
        notes=None,
    )


def _lifecycle_report() -> LifecycleReport:
    return LifecycleReport(
        schema_version="1",
        name="lifecycle_farm_test",
        created_at_utc="2026-03-10T00:00:00Z",
        source_rolling_report="rolling_farm_test",
        source_tradeforward_evaluations=["eval_01", "eval_02"],
        config={},
        candidate_ids=["cand_a", "cand_b", "cand_c"],
        applied_status_updates=False,
        final_status_counts={"research": 1, "approved": 1, "paper": 1},
        candidate_summaries=[],
        decisions=[],
        notes=None,
    )


def _baseline_report() -> PortfolioBaselineReport:
    return PortfolioBaselineReport(
        schema_version="1",
        name="baseline_farm_test",
        created_at_utc="2026-03-10T00:00:00Z",
        source_portfolio_ledger_report="portfolio_farm_test",
        source_rolling_report="rolling_farm_test",
        config={},
        conveyor_total_pnl=325.0,
        conveyor_total_return_pct=3.25,
        conveyor_max_drawdown_pct=2.4,
        baselines=[],
        comparisons=[],
        gate=PortfolioGateResult(
            overall_pass=True,
            config={},
            checks={"minimum_baselines_beaten": True},
            beaten_baselines=["flat", "equal_weight_selected_subset", "single_best_candidate"],
            failed_required_baselines=[],
        ),
        notes=None,
    )


def test_build_candidate_farm_scenario_report_from_artifacts() -> None:
    report = build_candidate_farm_scenario_report(
        scenario_name="reuse_may",
        status="completed",
        mode="reuse",
        output_dir="/tmp/farm/reuse_may",
        config={"rolling_args": {"mode": "reuse"}},
        log_paths={"rolling": "/tmp/farm/reuse_may/01_rolling.log"},
        rolling_report=_rolling_report(),
        lifecycle_report=_lifecycle_report(),
        portfolio_baselines_report=_baseline_report(),
        notes="smoke",
    )

    assert report.candidate_pool_ids == ["cand_a", "cand_b", "cand_c"]
    assert report.selected_candidate_ids == ["cand_a", "cand_c"]
    assert report.selector["tags"] == ["walkforward", "probe", "fused32"]
    assert report.final_status_counts["paper"] == 1
    assert report.gate_pass is True
    assert report.total_pnl == 325.0
    assert report.evaluated_cycle_count == 2
    assert report.positive_cycle_count == 1


def test_candidate_farm_report_summary_and_registry_roundtrip(tmp_path) -> None:
    completed = build_candidate_farm_scenario_report(
        scenario_name="reuse_may",
        status="completed",
        mode="reuse",
        output_dir="/tmp/farm/reuse_may",
        config={},
        log_paths={},
        rolling_report=_rolling_report(),
        lifecycle_report=_lifecycle_report(),
        portfolio_baselines_report=_baseline_report(),
    )
    failed = CandidateFarmScenarioReport(
        scenario_name="generate_fail",
        status="failed",
        mode="generate",
        output_dir="/tmp/farm/generate_fail",
        config={"rolling_args": {"mode": "generate"}},
        log_paths={"rolling": "/tmp/farm/generate_fail/01_rolling.log"},
        error_message="boom",
    )

    farm = build_candidate_farm_report(
        name="farm_smoke",
        registry_root=str(tmp_path),
        scenarios=[completed, failed],
        source_manifest_path=str(tmp_path / "manifest.json"),
        notes="smoke",
    )

    assert farm.summary["scenario_count"] == 2
    assert farm.summary["completed_scenarios"] == 1
    assert farm.summary["failed_scenarios"] == 1
    assert farm.summary["gate_pass_count"] == 1
    assert farm.summary["best_scenario_by_pnl"] == "reuse_may"
    assert farm.summary["ranked_scenarios"] == ["reuse_may"]

    registry = CandidateRegistry(tmp_path / "registry")
    path = registry.save_farm_report(farm)
    assert path.exists()
    loaded = registry.load_farm_report("farm_smoke")
    assert loaded.summary["gate_pass_count"] == 1
    assert loaded.scenarios[0].candidate_pool_ids == ["cand_a", "cand_b", "cand_c"]

    progress = FarmProgressLog(
        schema_version="1",
        farm_name="farm_smoke",
        created_at_utc="2026-03-10T00:00:00Z",
        updated_at_utc="2026-03-10T00:05:00Z",
        events=[
            FarmProgressEvent(
                sequence=1,
                created_at_utc="2026-03-10T00:00:00Z",
                scenario_name="reuse_may",
                status="planned",
                progress_stage="queued",
                event_kind="queued",
            ),
            FarmProgressEvent(
                sequence=2,
                created_at_utc="2026-03-10T00:05:00Z",
                scenario_name="reuse_may",
                status="completed",
                progress_stage="completed",
                event_kind="completed",
                gate_pass=True,
                total_pnl=325.0,
            ),
        ],
    )
    progress_path = registry.save_farm_progress_log(progress)
    assert progress_path.exists()
    loaded_progress = registry.load_farm_progress_log("farm_smoke")
    assert loaded_progress.events[-1].gate_pass is True
    assert loaded_progress.events[-1].total_pnl == 325.0


def test_expand_farm_manifest_scenarios_supports_defaults_and_matrix() -> None:
    manifest = {
        "defaults": {
            "rolling_args": {
                "selection_days": 7,
                "forward_days": 7,
            },
            "forwarded_cycle_args": ["--tag", "walkforward"],
        },
        "context": {
            "registry_tag": "fused32",
        },
        "scenario_templates": [
            {
                "name": "gen_{selection_label}_g{generations}",
                "rolling_args": {
                    "mode": "generate",
                    "selection_start_utc": "{selection_start_utc}",
                },
                "forwarded_cycle_args": [
                    "--tag",
                    "{registry_tag}",
                    "--generations",
                    "{generations}",
                ],
                "cases": [
                    {"selection_label": "may01", "selection_start_utc": "2025-05-01 00:00:00"},
                    {"selection_label": "may08", "selection_start_utc": "2025-05-08 00:00:00"},
                ],
                "matrix": {"generations": [4, 8]},
            }
        ],
    }

    scenarios = expand_farm_manifest_scenarios(manifest)
    scenario_names = [item["name"] for item in scenarios]

    assert len(scenarios) == 4
    assert "gen_may01_g4" in scenario_names
    assert "gen_may08_g8" in scenario_names
    assert scenarios[0]["rolling_args"]["forward_days"] == 7
    assert "--tag" in scenarios[0]["forwarded_cycle_args"]
    assert "walkforward" in scenarios[0]["forwarded_cycle_args"]
    assert "fused32" in scenarios[0]["forwarded_cycle_args"]
