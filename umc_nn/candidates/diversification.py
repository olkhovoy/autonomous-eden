from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from umc_nn.trading_eval import EpisodeTrace

from .schema import CandidateCurveSnapshot, DiversificationPairStats


@dataclass(frozen=True)
class CandidateTraceView:
    candidate_id: str
    display_name: str
    requested_max_steps: int
    initial_balance: float
    balance_history: list[float]
    action_history: list[int]
    max_drawdown_pct: float


def build_trace_view(candidate_id: str, display_name: str, trace: EpisodeTrace) -> CandidateTraceView:
    metrics = trace.metrics
    initial_balance = trace.balance_history[0]
    return CandidateTraceView(
        candidate_id=candidate_id,
        display_name=display_name,
        requested_max_steps=metrics.requested_max_steps,
        initial_balance=float(initial_balance),
        balance_history=list(trace.balance_history),
        action_history=list(trace.action_history),
        max_drawdown_pct=float(metrics.max_drawdown_pct),
    )


def pairwise_diversification_stats(
    left: CandidateTraceView,
    right: CandidateTraceView,
) -> DiversificationPairStats:
    left_balances = _pad_balance_history(left.balance_history, left.requested_max_steps + 1)
    right_balances = _pad_balance_history(right.balance_history, right.requested_max_steps + 1)
    left_actions = _pad_action_history(left.action_history, left.requested_max_steps)
    right_actions = _pad_action_history(right.action_history, right.requested_max_steps)

    steps = min(len(left_balances), len(right_balances)) - 1
    left_balances = left_balances[: steps + 1]
    right_balances = right_balances[: steps + 1]
    left_actions = left_actions[:steps]
    right_actions = right_actions[:steps]

    left_returns = _step_returns(left_balances)
    right_returns = _step_returns(right_balances)
    left_drawdowns = _drawdowns(left_balances)[1:]
    right_drawdowns = _drawdowns(right_balances)[1:]

    return_corr = _safe_corr(left_returns, right_returns)
    downside_corr = _safe_corr(np.minimum(left_returns, 0.0), np.minimum(right_returns, 0.0))
    simultaneous_loss_rate = float(np.mean((left_returns < 0.0) & (right_returns < 0.0)))
    simultaneous_drawdown_rate = float(np.mean((left_drawdowns > 0.0) & (right_drawdowns > 0.0)))
    action_agreement_rate = float(np.mean(left_actions == right_actions))
    same_nonflat_rate = float(np.mean((left_actions != 0) & (right_actions != 0) & (left_actions == right_actions)))
    opposite_nonflat_rate = float(np.mean((left_actions != 0) & (right_actions != 0) & (left_actions != right_actions)))

    portfolio_returns = 0.5 * left_returns + 0.5 * right_returns
    initial_balance = 0.5 * (left.initial_balance + right.initial_balance)
    portfolio_equity = _equity_from_returns(portfolio_returns, initial_balance)
    portfolio_drawdowns = _drawdowns(portfolio_equity)
    portfolio_max_dd = float(np.max(portfolio_drawdowns) * 100.0)
    equal_weight_final_balance = float(portfolio_equity[-1])
    equal_weight_net_profit = float(equal_weight_final_balance - initial_balance)
    avg_individual_max_dd = float((left.max_drawdown_pct + right.max_drawdown_pct) / 2.0)

    return DiversificationPairStats(
        left_candidate_id=left.candidate_id,
        right_candidate_id=right.candidate_id,
        left_display_name=left.display_name,
        right_display_name=right.display_name,
        steps=steps,
        return_corr=return_corr,
        downside_corr=downside_corr,
        simultaneous_loss_rate=simultaneous_loss_rate,
        simultaneous_drawdown_rate=simultaneous_drawdown_rate,
        action_agreement_rate=action_agreement_rate,
        same_nonflat_rate=same_nonflat_rate,
        opposite_nonflat_rate=opposite_nonflat_rate,
        equal_weight_final_balance=equal_weight_final_balance,
        equal_weight_net_profit=equal_weight_net_profit,
        equal_weight_max_drawdown_pct=portfolio_max_dd,
        avg_individual_max_drawdown_pct=avg_individual_max_dd,
        drawdown_improvement_pct_points=avg_individual_max_dd - portfolio_max_dd,
    )


def all_pairwise_diversification(traces: Sequence[CandidateTraceView]) -> list[DiversificationPairStats]:
    results: list[DiversificationPairStats] = []
    for left, right in combinations(traces, 2):
        results.append(pairwise_diversification_stats(left, right))
    return results


def curve_snapshot(trace: CandidateTraceView, *, max_points: int = 512) -> CandidateCurveSnapshot:
    balances = _pad_balance_history(trace.balance_history, trace.requested_max_steps + 1)
    normalized = balances / max(trace.initial_balance, 1e-12)
    sample_indices = _downsample_indices(len(normalized), max_points=max_points)
    return CandidateCurveSnapshot(
        candidate_id=trace.candidate_id,
        display_name=trace.display_name,
        steps_run=len(trace.balance_history) - 1,
        sample_indices=[int(idx) for idx in sample_indices],
        normalized_balance_history=[float(normalized[idx]) for idx in sample_indices],
        final_balance=float(balances[-1]),
        max_drawdown_pct=float(trace.max_drawdown_pct),
    )


def _pad_balance_history(values: Sequence[float], target_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) >= target_len:
        return arr[:target_len]
    if len(arr) == 0:
        raise ValueError("balance history cannot be empty")
    return np.pad(arr, (0, target_len - len(arr)), constant_values=arr[-1])


def _pad_action_history(values: Sequence[int], target_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if len(arr) >= target_len:
        return arr[:target_len]
    if len(arr) == 0:
        return np.zeros(target_len, dtype=np.int64)
    return np.pad(arr, (0, target_len - len(arr)), constant_values=arr[-1])


def _step_returns(balance_history: np.ndarray) -> np.ndarray:
    prev = np.maximum(balance_history[:-1], 1e-12)
    return (balance_history[1:] - balance_history[:-1]) / prev


def _drawdowns(balance_history: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(balance_history)
    return np.divide(
        running_max - balance_history,
        running_max,
        out=np.zeros_like(balance_history),
        where=running_max > 0,
    )


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _equity_from_returns(step_returns: np.ndarray, initial_balance: float) -> np.ndarray:
    equity = np.empty(step_returns.size + 1, dtype=np.float64)
    equity[0] = initial_balance
    for idx, step_return in enumerate(step_returns, start=1):
        equity[idx] = equity[idx - 1] * (1.0 + float(step_return))
    return equity


def _downsample_indices(length: int, *, max_points: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.int64)
    if length <= max_points:
        return np.arange(length, dtype=np.int64)
    raw = np.linspace(0, length - 1, num=max_points, dtype=np.int64)
    return np.unique(raw)
