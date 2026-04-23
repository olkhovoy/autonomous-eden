from __future__ import annotations

import pytest

from umc_nn.candidates import (
    TracePeriodRecord,
    TradeRecord,
    bootstrap_trade_resampling,
    replay_trace_resampling,
    scan_fraction_grid,
)
from umc_nn.candidates.resampling import TradeResamplingConfig


def _trades() -> list[TradeRecord]:
    return [
        TradeRecord(
            trade_id="t1",
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
        ),
        TradeRecord(
            trade_id="t2",
            period_name="train",
            direction="long",
            entry_step=2,
            exit_step=3,
            entry_timestamp_utc=None,
            exit_timestamp_utc=None,
            entry_price=100.0,
            exit_price=99.0,
            quantity=1.0,
            gross_pnl=-45.0,
            net_pnl=-50.0,
            fees_paid=5.0,
            duration_steps=1,
            entry_balance=10100.0,
            exit_balance=10050.0,
            return_on_equity=-0.0049504950495049506,
        ),
    ]


def test_bootstrap_trade_resampling_returns_summary() -> None:
    stats = bootstrap_trade_resampling(
        _trades(),
        TradeResamplingConfig(
            name="bootstrap_f1.00",
            period_name="train",
            iterations=200,
            seed=7,
            fraction=1.0,
            initial_balance=10000.0,
        ),
    )

    assert stats.original_trade_count == 2
    assert stats.original_net_profit == pytest.approx(50.0, abs=1e-6)
    assert stats.mean_max_drawdown_pct >= 0.0
    assert stats.p95_max_drawdown_pct >= stats.median_max_drawdown_pct
    assert 0.0 <= stats.profitable_rate <= 1.0


def test_scan_fraction_grid_keeps_names_distinct() -> None:
    results = scan_fraction_grid(
        _trades(),
        "train",
        fractions=[0.5, 1.0],
        iterations=50,
        seed=3,
        initial_balance=10000.0,
        name_prefix="train_bootstrap",
    )

    assert len(results) == 2
    assert results[0].name != results[1].name


def test_identity_replay_trace_resampling_reproduces_original_path() -> None:
    trace = TracePeriodRecord(
        period_name="train",
        source="test",
        start_step=0,
        requested_max_steps=3,
        steps_run=3,
        start_utc=None,
        end_utc=None,
        balance_history=[10000.0, 10100.0, 10050.0, 10120.0],
        action_history=[1, 1, 0],
        position_history=[0, 1, 1, 0],
    )
    stats = replay_trace_resampling(
        trace,
        TradeResamplingConfig(
            name="identity_f1.00",
            period_name="train",
            iterations=10,
            seed=7,
            fraction=1.0,
            initial_balance=10000.0,
            replay_mode="identity_replay",
        ),
        original_trade_count=2,
    )

    assert stats.replay_mode == "identity_replay"
    assert stats.original_final_balance == pytest.approx(10120.0)
    assert stats.original_net_profit == pytest.approx(120.0)
    assert stats.mean_final_balance == pytest.approx(10120.0)
    assert stats.p05_net_profit == pytest.approx(120.0)


def test_scan_fraction_grid_uses_trace_block_bootstrap_when_trace_present() -> None:
    trace = TracePeriodRecord(
        period_name="train",
        source="test",
        start_step=0,
        requested_max_steps=3,
        steps_run=3,
        start_utc=None,
        end_utc=None,
        balance_history=[10000.0, 10100.0, 10050.0, 10120.0],
        action_history=[1, 1, 0],
        position_history=[0, 1, 1, 0],
    )
    stats = scan_fraction_grid(
        _trades(),
        "train",
        fractions=[1.0],
        iterations=20,
        seed=5,
        initial_balance=10000.0,
        trace=trace,
        replay_mode="trace_block_bootstrap",
    )[0]

    assert stats.replay_mode == "trace_block_bootstrap"
    assert stats.sampler_id == "moving_block_step_sampler"
    assert stats.steps == 3
