from __future__ import annotations

from umc_nn.candidates import CandidateRecord, CandidateRegistry, ExperimentManifest, PeriodStats, ResamplingStats, RuleSetRecord, parse_rule, utc_now_text


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
        candidate_id="cand_test654321",
        display_name="wf_01_run02",
        engine_name="monolith",
        engine_role="candidate_engine",
        status="research",
        created_at_utc=utc_now_text(),
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
            ),
        },
        selection_flags={"oos_positive": True, "oos_beats_flat": True},
        resampling_results={
            "train_bootstrap_f1.00": ResamplingStats(
                name="train_bootstrap_f1.00",
                period_name="train",
                iterations=100,
                sample_size=2,
                seed=42,
                sizing_mode="fractional_returns",
                fraction=1.0,
                initial_balance=10000.0,
                original_trade_count=2,
                original_final_balance=10100.0,
                original_net_profit=100.0,
                original_max_drawdown_pct=1.0,
                mean_final_balance=10100.0,
                median_final_balance=10100.0,
                p05_final_balance=10010.0,
                p25_final_balance=10050.0,
                mean_net_profit=100.0,
                median_net_profit=100.0,
                p05_net_profit=10.0,
                p25_net_profit=50.0,
                mean_max_drawdown_pct=1.0,
                median_max_drawdown_pct=1.0,
                p75_max_drawdown_pct=2.0,
                p95_max_drawdown_pct=3.0,
                profitable_rate=0.9,
                loss_rate=0.1,
                ruin_rate=0.0,
                pessimistic_net_profit=10.0,
                pessimistic_max_drawdown_pct=3.0,
            )
        },
    )


def test_parse_and_evaluate_rules(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    candidate = _candidate()
    registry.add_candidate(candidate)

    rules = [
        parse_rule("periods.oos.pnl:gt:0"),
        parse_rule("selection_flags.oos_positive:eq:true"),
        parse_rule("tags:contains:\"walkforward\""),
    ]
    matches = registry.filter_candidates(rules)
    assert len(matches) == 1
    assert matches[0][0].candidate_id == candidate.candidate_id
    assert all(item.matched for item in matches[0][1])


def test_rule_set_round_trip(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    rule_set = RuleSetRecord(
        schema_version="1",
        name="positive_oos",
        created_at_utc=utc_now_text(),
        description="Candidates with profitable OOS",
        require_all=True,
        rules=[parse_rule("periods.oos.pnl:gt:0").to_dict()],
    )

    path = registry.save_rule_set(rule_set)
    assert path.exists()

    loaded = registry.load_rule_set("positive_oos")
    assert loaded.name == "positive_oos"
    assert loaded.rules[0]["field"] == "periods.oos.pnl"


def test_rules_can_resolve_dict_keys_with_dots(tmp_path) -> None:
    registry = CandidateRegistry(tmp_path / "registry")
    candidate = _candidate()
    registry.add_candidate(candidate)

    rules = [
        parse_rule("resampling_results.train_bootstrap_f1.00.profitable_rate:gte:0.9"),
        parse_rule("resampling_results.train_bootstrap_f1.00.p05_net_profit:gt:0"),
    ]
    matches = registry.filter_candidates(rules)
    assert len(matches) == 1
