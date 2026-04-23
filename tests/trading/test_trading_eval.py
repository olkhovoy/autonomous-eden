import json
import numpy as np

import pytest

from umc_nn.candidate_engines import EngineConfig
from umc_nn.pure_env import PureTradingEnv
from umc_nn.trading_eval import evaluate_engine_vector, evaluate_policy, evaluate_policy_path, evaluate_policy_trace, evaluate_policy_trace_path


class SequencePolicy:
    def __init__(self, actions):
        self.actions = list(actions)
        self.index = 0

    def reset(self):
        self.index = 0

    def act(self, obs):
        del obs
        action = self.actions[self.index]
        self.index += 1
        return action


def _write_history(tmp_path, closes):
    rows = []
    for idx, close in enumerate(closes):
        rows.append([idx, close, close, close, close, 1.0])
    path = tmp_path / "history.json"
    path.write_text(json.dumps(rows))
    return path


def test_flat_metrics_are_exact(tmp_path):
    data_path = _write_history(tmp_path, [100.0, 101.0, 99.0, 102.0, 103.0])
    metrics = evaluate_policy_path(
        "flat",
        data_path,
        use_neurobars=False,
        start_step=0,
        max_steps=3,
        initial_balance=10000.0,
    )

    assert metrics.final_balance == pytest.approx(10000.0)
    assert metrics.pnl == pytest.approx(0.0)
    assert metrics.max_drawdown_pct == pytest.approx(0.0)
    assert metrics.steps_run == 3
    assert metrics.trades == 0
    assert metrics.action_counts == {0: 3}
    assert metrics.position_counts == {0: 4}


def test_long_policy_reports_expected_balance_with_equity_sizing(tmp_path):
    data_path = _write_history(tmp_path, [100.0, 101.0, 102.0, 103.0, 104.0])
    metrics = evaluate_policy_path(
        "long",
        data_path,
        use_neurobars=False,
        start_step=0,
        max_steps=3,
        initial_balance=10000.0,
    )

    assert metrics.final_balance == pytest.approx(10195.0)
    assert metrics.pnl == pytest.approx(195.0, abs=1e-4)
    assert metrics.action_counts == {1: 3}


def test_closed_trade_count_and_win_rate_follow_realized_balance(tmp_path):
    data_path = _write_history(tmp_path, [100.0, 101.0, 103.0, 103.0, 103.0])
    env = PureTradingEnv(max_steps=3, initial_balance=10000.0, data_path=str(data_path))
    policy = SequencePolicy([1, 1, 0])

    metrics = evaluate_policy(env, policy_name="sequence", policy=policy)

    assert metrics.trades == 1
    assert metrics.wins == 1
    assert metrics.win_rate_pct == pytest.approx(100.0)
    assert metrics.final_balance == pytest.approx(10189.85)
    assert metrics.action_counts == {0: 1, 1: 2}


def test_evaluate_policy_trace_exports_closed_trade_details(tmp_path):
    data_path = _write_history(tmp_path, [100.0, 101.0, 103.0, 103.0, 103.0])
    env = PureTradingEnv(max_steps=3, initial_balance=10000.0, data_path=str(data_path))
    policy = SequencePolicy([1, 1, 0])

    trace = evaluate_policy_trace(env, policy_name="sequence", policy=policy)

    assert trace.metrics.trades == 1
    assert len(trace.trades) == 1
    trade = trace.trades[0]
    assert trade.direction == "long"
    assert trade.entry_step == 0
    assert trade.exit_step == 2
    assert trade.entry_price == pytest.approx(100.0)
    assert trade.exit_price == pytest.approx(103.0)
    assert trade.quantity == pytest.approx(100.0)
    assert trade.fees_paid == pytest.approx(10.15)
    assert trade.net_pnl == pytest.approx(194.85)
    assert trade.gross_pnl == pytest.approx(205.0)
    assert trade.return_on_equity == pytest.approx(194.85 / 9995.0)


def test_linear_score_family_matches_long_policy_when_biases_force_long(tmp_path):
    data_path = _write_history(tmp_path, [100.0, 101.0, 102.0, 103.0, 104.0])
    weights_path = tmp_path / "linear_long.npy"
    vector = np.zeros(18, dtype=np.float32)
    vector[-3:] = np.array([0.0, 10.0, -10.0], dtype=np.float32)
    np.save(weights_path, vector)

    long_metrics = evaluate_policy_path(
        "long",
        data_path,
        use_neurobars=False,
        start_step=0,
        max_steps=3,
        initial_balance=10000.0,
    )
    family_metrics = evaluate_policy_path(
        "monolith",
        data_path,
        use_neurobars=False,
        start_step=0,
        max_steps=3,
        initial_balance=10000.0,
        weights_path=weights_path,
        engine_config=EngineConfig(family="linear_score"),
    )

    assert family_metrics.final_balance == pytest.approx(long_metrics.final_balance)
    assert family_metrics.pnl == pytest.approx(long_metrics.pnl)


def test_evaluate_engine_vector_matches_weights_path_trace(tmp_path):
    data_path = _write_history(tmp_path, [100.0, 100.5, 101.0, 100.0, 99.0])
    weights_path = tmp_path / "linear_short.npy"
    vector = np.zeros(18, dtype=np.float32)
    vector[-3:] = np.array([0.0, -10.0, 10.0], dtype=np.float32)
    np.save(weights_path, vector)
    config = EngineConfig(family="linear_score")

    env = PureTradingEnv(max_steps=3, initial_balance=10000.0, data_path=str(data_path))
    trace_from_vector = evaluate_engine_vector(env, vector, engine_config=config)
    trace_from_path = evaluate_policy_trace_path(
        "monolith",
        data_path,
        use_neurobars=False,
        start_step=0,
        max_steps=3,
        initial_balance=10000.0,
        weights_path=weights_path,
        engine_config=config,
    )

    assert trace_from_vector.metrics.final_balance == pytest.approx(trace_from_path.metrics.final_balance)
    assert trace_from_vector.metrics.pnl == pytest.approx(trace_from_path.metrics.pnl)
    assert trace_from_vector.action_history == trace_from_path.action_history
