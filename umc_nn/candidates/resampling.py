from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .schema import ResamplingStats, TracePeriodRecord, TradeRecord


RESAMPLING_SIZING_MODES = {"fractional_returns"}
REPLAY_MODES = {"trade_bootstrap_legacy", "trace_block_bootstrap", "identity_replay"}


@dataclass(frozen=True)
class TradeResamplingConfig:
    name: str
    period_name: str
    iterations: int = 1000
    sample_size: int | None = None
    block_size: int = 64
    seed: int = 42
    sizing_mode: str = "fractional_returns"
    fraction: float = 1.0
    initial_balance: float = 10000.0
    replay_mode: str = "trace_block_bootstrap"
    sampler_id: str | None = None
    overlap_policy: str | None = None

    def validate(self) -> None:
        if self.iterations <= 0:
            raise ValueError("iterations must be > 0")
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError("sample_size must be > 0 when provided")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.sizing_mode not in RESAMPLING_SIZING_MODES:
            raise ValueError(f"Unsupported sizing mode: {self.sizing_mode}")
        if self.fraction < 0:
            raise ValueError("fraction must be >= 0")
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be > 0")
        if self.replay_mode not in REPLAY_MODES:
            raise ValueError(f"Unsupported replay_mode: {self.replay_mode}")

    @property
    def resolved_sampler_id(self) -> str:
        if self.sampler_id:
            return self.sampler_id
        if self.replay_mode == "trace_block_bootstrap":
            return "moving_block_step_sampler"
        if self.replay_mode == "identity_replay":
            return "identity_path_sampler"
        return "iid_trade_bootstrap"

    @property
    def resolved_overlap_policy(self) -> str:
        if self.overlap_policy:
            return self.overlap_policy
        if self.replay_mode == "trace_block_bootstrap":
            return "time_ordered_step_blocks"
        if self.replay_mode == "identity_replay":
            return "original_path"
        return "trade_independent"


def closed_trades_for_period(trades: Iterable[TradeRecord], period_name: str) -> list[TradeRecord]:
    return [trade for trade in trades if trade.period_name == period_name]


def closed_trace_for_period(traces: Iterable[TracePeriodRecord], period_name: str) -> TracePeriodRecord | None:
    for trace in traces:
        if trace.period_name == period_name:
            return trace
    return None


def bootstrap_trade_resampling(
    trades: Sequence[TradeRecord],
    config: TradeResamplingConfig,
) -> ResamplingStats:
    config.validate()
    if not trades:
        raise ValueError("bootstrap_trade_resampling requires at least one trade")

    returns = np.array([_trade_return(trade) for trade in trades], dtype=np.float64)
    sample_size = config.sample_size or len(trades)
    rng = np.random.RandomState(config.seed)

    original_final_balance, original_net_profit, original_max_drawdown = _simulate_fractional_path(
        returns,
        initial_balance=config.initial_balance,
        fraction=config.fraction,
    )

    final_balances = np.zeros(config.iterations, dtype=np.float64)
    net_profits = np.zeros(config.iterations, dtype=np.float64)
    max_drawdowns = np.zeros(config.iterations, dtype=np.float64)

    for idx in range(config.iterations):
        sample_indices = rng.randint(0, len(trades), size=sample_size)
        sampled_returns = returns[sample_indices]
        final_balance, net_profit, max_drawdown = _simulate_fractional_path(
            sampled_returns,
            initial_balance=config.initial_balance,
            fraction=config.fraction,
        )
        final_balances[idx] = final_balance
        net_profits[idx] = net_profit
        max_drawdowns[idx] = max_drawdown

    return _stats_from_samples(
        config=config,
        original_trade_count=len(trades),
        sample_size=sample_size,
        original_final_balance=original_final_balance,
        original_net_profit=original_net_profit,
        original_max_drawdown=original_max_drawdown,
        final_balances=final_balances,
        net_profits=net_profits,
        max_drawdowns=max_drawdowns,
        steps=None,
    )


def replay_trace_resampling(
    trace: TracePeriodRecord,
    config: TradeResamplingConfig,
    *,
    original_trade_count: int,
) -> ResamplingStats:
    config.validate()
    if not trace.balance_history:
        raise ValueError("replay_trace_resampling requires non-empty balance_history")

    step_returns = _step_returns(np.asarray(trace.balance_history, dtype=np.float64))
    if step_returns.size == 0:
        raise ValueError("trace balance history must contain at least two values")

    original_equity = _equity_from_step_returns(
        step_returns,
        initial_balance=config.initial_balance,
        fraction=config.fraction,
    )
    original_final_balance = float(original_equity[-1])
    original_net_profit = float(original_final_balance - config.initial_balance)
    original_max_drawdown = _max_drawdown_pct(original_equity)

    if config.replay_mode == "identity_replay":
        final_balances = np.array([original_final_balance], dtype=np.float64)
        net_profits = np.array([original_net_profit], dtype=np.float64)
        max_drawdowns = np.array([original_max_drawdown], dtype=np.float64)
        iterations = 1
    else:
        rng = np.random.RandomState(config.seed)
        final_balances = np.zeros(config.iterations, dtype=np.float64)
        net_profits = np.zeros(config.iterations, dtype=np.float64)
        max_drawdowns = np.zeros(config.iterations, dtype=np.float64)
        for idx in range(config.iterations):
            sampled_returns = _sample_step_blocks(
                step_returns,
                target_steps=step_returns.size,
                block_size=min(config.block_size, step_returns.size),
                rng=rng,
            )
            sampled_equity = _equity_from_step_returns(
                sampled_returns,
                initial_balance=config.initial_balance,
                fraction=config.fraction,
            )
            final_balances[idx] = float(sampled_equity[-1])
            net_profits[idx] = float(sampled_equity[-1] - config.initial_balance)
            max_drawdowns[idx] = _max_drawdown_pct(sampled_equity)
        iterations = config.iterations

    replay_config = TradeResamplingConfig(
        name=config.name,
        period_name=config.period_name,
        iterations=iterations,
        sample_size=step_returns.size,
        block_size=config.block_size,
        seed=config.seed,
        sizing_mode=config.sizing_mode,
        fraction=config.fraction,
        initial_balance=config.initial_balance,
        replay_mode=config.replay_mode,
        sampler_id=config.sampler_id,
        overlap_policy=config.overlap_policy,
    )
    return _stats_from_samples(
        config=replay_config,
        original_trade_count=original_trade_count,
        sample_size=step_returns.size,
        original_final_balance=original_final_balance,
        original_net_profit=original_net_profit,
        original_max_drawdown=original_max_drawdown,
        final_balances=final_balances,
        net_profits=net_profits,
        max_drawdowns=max_drawdowns,
        steps=step_returns.size,
    )


def scan_fraction_grid(
    trades: Sequence[TradeRecord],
    period_name: str,
    *,
    fractions: Iterable[float],
    iterations: int = 1000,
    sample_size: int | None = None,
    block_size: int = 64,
    seed: int = 42,
    initial_balance: float = 10000.0,
    name_prefix: str = "bootstrap",
    trace: TracePeriodRecord | None = None,
    replay_mode: str = "trace_block_bootstrap",
) -> list[ResamplingStats]:
    results: list[ResamplingStats] = []
    for fraction in fractions:
        config = TradeResamplingConfig(
            name=f"{name_prefix}_f{fraction:.2f}",
            period_name=period_name,
            iterations=iterations,
            sample_size=sample_size,
            block_size=block_size,
            seed=seed,
            sizing_mode="fractional_returns",
            fraction=float(fraction),
            initial_balance=initial_balance,
            replay_mode=replay_mode,
        )
        if replay_mode in {"trace_block_bootstrap", "identity_replay"}:
            if trace is None:
                if replay_mode == "trace_block_bootstrap":
                    results.append(
                        bootstrap_trade_resampling(
                            trades,
                            TradeResamplingConfig(
                                name=config.name,
                                period_name=config.period_name,
                                iterations=config.iterations,
                                sample_size=config.sample_size,
                                block_size=config.block_size,
                                seed=config.seed,
                                sizing_mode=config.sizing_mode,
                                fraction=config.fraction,
                                initial_balance=config.initial_balance,
                                replay_mode="trade_bootstrap_legacy",
                            ),
                        )
                    )
                    continue
                raise ValueError(f"{replay_mode} requires a trace artifact for period {period_name}")
            results.append(replay_trace_resampling(trace, config, original_trade_count=len(trades)))
        else:
            results.append(bootstrap_trade_resampling(trades, config))
    return results


def _trade_return(trade: TradeRecord) -> float:
    if trade.return_on_equity is not None:
        return float(trade.return_on_equity)
    if trade.entry_balance is not None and trade.entry_balance > 0 and trade.net_pnl is not None:
        return float(trade.net_pnl) / float(trade.entry_balance)
    raise ValueError(f"Trade {trade.trade_id} is missing return_on_equity and entry_balance")


def _simulate_fractional_path(
    returns: np.ndarray,
    *,
    initial_balance: float,
    fraction: float,
) -> tuple[float, float, float]:
    equity = _equity_from_trade_returns(returns, initial_balance=initial_balance, fraction=fraction)
    final_balance = float(equity[-1])
    net_profit = float(final_balance - initial_balance)
    return final_balance, net_profit, _max_drawdown_pct(equity)


def _equity_from_trade_returns(
    returns: np.ndarray,
    *,
    initial_balance: float,
    fraction: float,
) -> np.ndarray:
    equity = np.empty(returns.size + 1, dtype=np.float64)
    equity[0] = float(initial_balance)
    for idx, trade_return in enumerate(returns, start=1):
        equity[idx] = equity[idx - 1] * (1.0 + float(trade_return) * fraction)
    return equity


def _equity_from_step_returns(
    step_returns: np.ndarray,
    *,
    initial_balance: float,
    fraction: float,
) -> np.ndarray:
    equity = np.empty(step_returns.size + 1, dtype=np.float64)
    equity[0] = float(initial_balance)
    for idx, step_return in enumerate(step_returns, start=1):
        equity[idx] = equity[idx - 1] * (1.0 + float(step_return) * fraction)
    return equity


def _step_returns(balance_history: np.ndarray) -> np.ndarray:
    prev = np.maximum(balance_history[:-1], 1e-12)
    return (balance_history[1:] - balance_history[:-1]) / prev


def _max_drawdown_pct(equity: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity)
    drawdowns = np.divide(
        running_max - equity,
        running_max,
        out=np.zeros_like(equity),
        where=running_max > 0,
    )
    return float(np.max(drawdowns) * 100.0)


def _sample_step_blocks(
    step_returns: np.ndarray,
    *,
    target_steps: int,
    block_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    blocks: list[np.ndarray] = []
    while sum(block.shape[0] for block in blocks) < target_steps:
        start = int(rng.randint(0, step_returns.size))
        end = min(start + block_size, step_returns.size)
        block = step_returns[start:end]
        if block.size == 0:
            continue
        blocks.append(block)
    return np.concatenate(blocks, axis=0)[:target_steps]


def _stats_from_samples(
    *,
    config: TradeResamplingConfig,
    original_trade_count: int,
    sample_size: int,
    original_final_balance: float,
    original_net_profit: float,
    original_max_drawdown: float,
    final_balances: np.ndarray,
    net_profits: np.ndarray,
    max_drawdowns: np.ndarray,
    steps: int | None,
) -> ResamplingStats:
    return ResamplingStats(
        name=config.name,
        period_name=config.period_name,
        iterations=int(final_balances.size),
        sample_size=sample_size,
        seed=config.seed,
        sizing_mode=config.sizing_mode,
        fraction=config.fraction,
        initial_balance=config.initial_balance,
        original_trade_count=original_trade_count,
        original_final_balance=float(original_final_balance),
        original_net_profit=float(original_net_profit),
        original_max_drawdown_pct=float(original_max_drawdown),
        mean_final_balance=float(np.mean(final_balances)),
        median_final_balance=float(np.median(final_balances)),
        p05_final_balance=float(np.percentile(final_balances, 5)),
        p25_final_balance=float(np.percentile(final_balances, 25)),
        mean_net_profit=float(np.mean(net_profits)),
        median_net_profit=float(np.median(net_profits)),
        p05_net_profit=float(np.percentile(net_profits, 5)),
        p25_net_profit=float(np.percentile(net_profits, 25)),
        mean_max_drawdown_pct=float(np.mean(max_drawdowns)),
        median_max_drawdown_pct=float(np.median(max_drawdowns)),
        p75_max_drawdown_pct=float(np.percentile(max_drawdowns, 75)),
        p95_max_drawdown_pct=float(np.percentile(max_drawdowns, 95)),
        profitable_rate=float(np.mean(net_profits > 0.0)),
        loss_rate=float(np.mean(net_profits <= 0.0)),
        ruin_rate=float(np.mean(final_balances <= 0.0)),
        pessimistic_net_profit=float(np.percentile(net_profits, 5)),
        pessimistic_max_drawdown_pct=float(np.percentile(max_drawdowns, 95)),
        replay_mode=config.replay_mode,
        sampler_id=config.resolved_sampler_id,
        overlap_policy=config.resolved_overlap_policy,
        steps=steps,
        capital_path_distribution={
            "p05_final_balance": float(np.percentile(final_balances, 5)),
            "p50_final_balance": float(np.percentile(final_balances, 50)),
            "p95_final_balance": float(np.percentile(final_balances, 95)),
            "p05_net_profit": float(np.percentile(net_profits, 5)),
            "p95_max_drawdown_pct": float(np.percentile(max_drawdowns, 95)),
        },
    )
