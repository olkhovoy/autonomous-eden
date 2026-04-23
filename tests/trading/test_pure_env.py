import json

import pytest

from umc_nn.pure_env import EXCHANGE_FEE_PRESETS, PureTradingEnv


def _write_history(tmp_path, closes):
    rows = []
    for idx, close in enumerate(closes):
        rows.append([idx, close, close, close, close, 1.0])
    path = tmp_path / "history.json"
    path.write_text(json.dumps(rows))
    return path


def test_flat_policy_preserves_balance(tmp_path):
    data_path = _write_history(tmp_path, [100.0, 101.0, 99.0, 102.0])
    env = PureTradingEnv(max_steps=3, initial_balance=10000.0, data_path=str(data_path))

    env.reset()
    for _ in range(3):
        env.step(0)

    assert env.balance == pytest.approx(10000.0)


@pytest.mark.parametrize(
    ("exchange", "maker_fee", "taker_fee"),
    [
        ("binance", 0.0002, 0.0005),
        ("bybit", 0.0002, 0.00055),
        ("okx", 0.0002, 0.0005),
    ],
)
def test_exchange_fee_presets_are_loaded(exchange, maker_fee, taker_fee):
    preset = EXCHANGE_FEE_PRESETS[exchange]
    assert preset.maker_fee_rate == pytest.approx(maker_fee)
    assert preset.taker_fee_rate == pytest.approx(taker_fee)


def test_fraction_of_equity_long_trade_uses_10k_account_notional(tmp_path):
    data_path = _write_history(tmp_path, [100.0, 101.0, 102.0, 103.0])
    env = PureTradingEnv(max_steps=3, initial_balance=10000.0, data_path=str(data_path))

    env.reset()
    env.step(1)
    env.step(1)
    env.step(1)

    expected_entry_fee = 10000.0 * 0.0005
    expected_balance = 10000.0 - expected_entry_fee + 100.0 + 100.0
    assert env.balance == pytest.approx(expected_balance)
    assert env.position_qty == pytest.approx(100.0)
    assert env.position_notional == pytest.approx(10000.0)


def test_reversal_charges_close_and_reentry_fee_with_updated_equity(tmp_path):
    data_path = _write_history(tmp_path, [100.0, 100.0, 100.0])
    env = PureTradingEnv(max_steps=2, initial_balance=10000.0, data_path=str(data_path))

    env.reset()
    env.step(1)
    _, _, done, info = env.step(2)

    expected_entry_fee = 10000.0 * 0.0005
    expected_close_fee = 10000.0 * 0.0005
    expected_reentry_notional = 10000.0 - expected_entry_fee - expected_close_fee
    expected_reentry_fee = expected_reentry_notional * 0.0005
    expected_reversal_friction = expected_close_fee + expected_reentry_fee
    expected_balance = 10000.0 - expected_entry_fee - expected_reversal_friction

    assert info["friction_cost"] == pytest.approx(expected_reversal_friction)
    assert env.balance == pytest.approx(expected_balance)
    assert env.position_qty == pytest.approx(expected_reentry_notional / 100.0)
    assert done is True


def test_fixed_quantity_mode_keeps_legacy_one_unit_behavior(tmp_path):
    data_path = _write_history(tmp_path, [100.0, 101.0, 102.0, 103.0])
    env = PureTradingEnv(
        max_steps=3,
        initial_balance=1000.0,
        data_path=str(data_path),
        position_sizing_mode="fixed_quantity",
        fixed_position_qty=1.0,
    )

    env.reset()
    env.step(1)
    env.step(1)
    env.step(1)

    expected_balance = 1000.0 - (100.0 * 0.0005) + 1.0 + 1.0
    assert env.balance == pytest.approx(expected_balance)
    assert env.position_qty == pytest.approx(1.0)


def test_max_steps_are_relative_to_start_step(tmp_path):
    data_path = _write_history(tmp_path, [100.0, 101.0, 102.0, 103.0, 104.0])
    env = PureTradingEnv(
        max_steps=2,
        initial_balance=10000.0,
        data_path=str(data_path),
        start_step=1,
    )

    env.reset()
    _, _, done_first, _ = env.step(0)
    _, _, done_second, _ = env.step(0)

    assert done_first is False
    assert done_second is True
    assert env.current_step == 3


def test_missing_data_path_is_hard_failure_by_default(tmp_path):
    missing_path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        PureTradingEnv(max_steps=2, initial_balance=10000.0, data_path=str(missing_path))
