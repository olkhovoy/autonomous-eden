from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Protocol

import numpy as np
import torch

from umc_nn.candidate_engines import EngineConfig, VectorEnginePolicy
from umc_nn.pure_env import (
    EXECUTION_FEE_MODES,
    EXCHANGE_FEE_PRESETS,
    POSITION_SIZING_MODES,
    PureTradingEnv,
)


DEFAULT_DATE_WINDOWS = {
    "2021_2023": ("2021-07-05 15:00:00", "2023-05-31 01:40:00"),
    "2023_2025": ("2023-05-31 01:40:00", "2025-04-24 12:20:00"),
    "2025_2026": ("2025-04-24 12:20:00", "2026-01-19 17:45:00"),
    "train_2024_2025": ("2024-01-01 00:00:00", "2026-01-01 00:00:00"),
}


class Policy(Protocol):
    def reset(self) -> None:
        ...

    def act(self, obs: np.ndarray) -> int:
        ...


@dataclass
class EpisodeMetrics:
    policy: str
    start_step: int
    requested_max_steps: int
    steps_run: int
    final_balance: float
    pnl: float
    max_drawdown_pct: float
    trades: int
    wins: int
    win_rate_pct: float
    action_counts: Dict[int, int]
    position_counts: Dict[int, int]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class EpisodeTrade:
    trade_id: str
    direction: str
    entry_step: int
    exit_step: int
    entry_timestamp_utc: str | None
    exit_timestamp_utc: str | None
    entry_price: float | None
    exit_price: float | None
    quantity: float | None
    gross_pnl: float
    net_pnl: float
    fees_paid: float
    duration_steps: int
    entry_balance: float | None = None
    exit_balance: float | None = None
    return_on_equity: float | None = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class EpisodeTrace:
    metrics: EpisodeMetrics
    trades: list[EpisodeTrade]
    balance_history: list[float]
    action_history: list[int]
    position_history: list[int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "metrics": self.metrics.to_dict(),
            "trades": [trade.to_dict() for trade in self.trades],
            "balance_history": list(self.balance_history),
            "action_history": list(self.action_history),
            "position_history": list(self.position_history),
        }


class ConstantPolicy:
    def __init__(self, action: int):
        self.action = action

    def reset(self) -> None:
        return None

    def act(self, obs: np.ndarray) -> int:
        del obs
        return self.action


def build_policy(
    policy_name: str,
    input_dim: int,
    hidden_dim: int = 64,
    weights_path: Optional[str | Path] = None,
    device: Optional[str | torch.device] = None,
    engine_config: EngineConfig | None = None,
) -> Policy:
    if policy_name == "flat":
        return ConstantPolicy(0)
    if policy_name == "long":
        return ConstantPolicy(1)
    if policy_name == "short":
        return ConstantPolicy(2)
    if policy_name == "monolith":
        if weights_path is None:
            raise ValueError("weights_path is required for monolith policy")
        torch_device = _resolve_device(device)
        config = engine_config or EngineConfig(hidden_dim=hidden_dim)
        return VectorEnginePolicy(
            input_dim=input_dim,
            engine_config=config,
            device=torch_device,
            weights_path=weights_path,
        )
    raise ValueError(f"Unknown policy: {policy_name}")


def evaluate_policy(env: PureTradingEnv, policy_name: str, policy: Policy) -> EpisodeMetrics:
    return evaluate_policy_trace(env, policy_name=policy_name, policy=policy).metrics


def evaluate_policy_trace(env: PureTradingEnv, policy_name: str, policy: Policy) -> EpisodeTrace:
    obs = env.reset()
    policy.reset()

    action_history: list[int] = []
    position_history: list[int] = [env.current_position]
    balance_history: list[float] = [float(env.balance)]
    trade_history: list[EpisodeTrade] = []
    open_trade: dict[str, object] | None = None
    prev_action = env.current_position

    done = False
    while not done:
        action = int(policy.act(obs))
        obs, reward, done, info = env.step(action)
        del reward

        current_balance = float(env.balance)
        executed_action = int(info["executed_action"])
        step_index = int(info["step_index"])
        current_price = float(info["current_price"])
        open_fee = float(info.get("open_fee", 0.0))
        close_fee = float(info.get("close_fee", 0.0))
        close_balance = current_balance + open_fee if executed_action != 0 else current_balance

        action_history.append(executed_action)
        position_history.append(env.current_position)
        balance_history.append(current_balance)

        if executed_action != prev_action:
            if prev_action != 0 and open_trade is not None:
                entry_balance = float(open_trade["entry_balance"])
                net_pnl = close_balance - entry_balance
                fees_paid = float(open_trade["entry_fee"]) + close_fee
                gross_pnl = net_pnl + fees_paid
                trade_history.append(
                    EpisodeTrade(
                        trade_id=f"{policy_name}_{len(trade_history) + 1:04d}",
                        direction=str(open_trade["direction"]),
                        entry_step=int(open_trade["entry_step"]),
                        exit_step=step_index,
                        entry_timestamp_utc=open_trade["entry_timestamp_utc"],  # type: ignore[arg-type]
                        exit_timestamp_utc=_step_timestamp_text(env, step_index),
                        entry_price=float(open_trade["entry_price"]),
                        exit_price=current_price,
                        quantity=float(open_trade["quantity"]),
                        gross_pnl=float(gross_pnl),
                        net_pnl=float(net_pnl),
                        fees_paid=float(fees_paid),
                        duration_steps=max(step_index - int(open_trade["entry_step"]), 0),
                        entry_balance=entry_balance,
                        exit_balance=float(close_balance),
                        return_on_equity=(net_pnl / entry_balance) if entry_balance > 0 else None,
                    )
                )
                open_trade = None

            if executed_action != 0:
                open_trade = {
                    "direction": "long" if executed_action == 1 else "short",
                    "entry_step": step_index,
                    "entry_timestamp_utc": _step_timestamp_text(env, step_index),
                    "entry_price": current_price,
                    "quantity": float(info["position_qty_after"]),
                    "entry_balance": current_balance,
                    "entry_fee": open_fee,
                }

            prev_action = executed_action

    balance_arr = np.array(balance_history, dtype=np.float64)
    running_max = np.maximum.accumulate(balance_arr)
    drawdowns = np.divide(
        running_max - balance_arr,
        running_max,
        out=np.zeros_like(balance_arr),
        where=running_max > 0,
    )

    wins = sum(1 for trade in trade_history if trade.net_pnl > 0.0)
    metrics = EpisodeMetrics(
        policy=policy_name,
        start_step=env.start_step,
        requested_max_steps=env.max_steps,
        steps_run=len(action_history),
        final_balance=float(balance_history[-1]),
        pnl=float(balance_history[-1] - env.initial_balance),
        max_drawdown_pct=float(np.max(drawdowns) * 100.0),
        trades=len(trade_history),
        wins=wins,
        win_rate_pct=float((wins / len(trade_history)) * 100.0) if trade_history else 0.0,
        action_counts=_counter_dict(action_history),
        position_counts=_counter_dict(position_history),
    )
    return EpisodeTrace(
        metrics=metrics,
        trades=trade_history,
        balance_history=balance_history,
        action_history=action_history,
        position_history=position_history,
    )


def evaluate_policy_path(
    policy_name: str,
    data_path: str | Path,
    *,
    use_neurobars: bool,
    start_step: int,
    max_steps: int,
    initial_balance: float = 10000.0,
    exchange: str = "binance",
    maker_fee_rate: float | None = None,
    taker_fee_rate: float | None = None,
    execution_fee_mode: str = "taker",
    slippage: float = 0.0,
    position_sizing_mode: str = "fraction_of_equity",
    position_notional_fraction: float = 1.0,
    leverage: float = 1.0,
    fixed_position_qty: float = 1.0,
    hidden_dim: int = 64,
    weights_path: Optional[str | Path] = None,
    device: Optional[str | torch.device] = None,
    engine_config: EngineConfig | None = None,
) -> EpisodeMetrics:
    return evaluate_policy_trace_path(
        policy_name,
        data_path,
        use_neurobars=use_neurobars,
        start_step=start_step,
        max_steps=max_steps,
        initial_balance=initial_balance,
        exchange=exchange,
        maker_fee_rate=maker_fee_rate,
        taker_fee_rate=taker_fee_rate,
        execution_fee_mode=execution_fee_mode,
        slippage=slippage,
        position_sizing_mode=position_sizing_mode,
        position_notional_fraction=position_notional_fraction,
        leverage=leverage,
        fixed_position_qty=fixed_position_qty,
        hidden_dim=hidden_dim,
        weights_path=weights_path,
        device=device,
        engine_config=engine_config,
    ).metrics


def evaluate_policy_trace_path(
    policy_name: str,
    data_path: str | Path,
    *,
    use_neurobars: bool,
    start_step: int,
    max_steps: int,
    initial_balance: float = 10000.0,
    exchange: str = "binance",
    maker_fee_rate: float | None = None,
    taker_fee_rate: float | None = None,
    execution_fee_mode: str = "taker",
    slippage: float = 0.0,
    position_sizing_mode: str = "fraction_of_equity",
    position_notional_fraction: float = 1.0,
    leverage: float = 1.0,
    fixed_position_qty: float = 1.0,
    hidden_dim: int = 64,
    weights_path: Optional[str | Path] = None,
    device: Optional[str | torch.device] = None,
    engine_config: EngineConfig | None = None,
) -> EpisodeTrace:
    if exchange not in EXCHANGE_FEE_PRESETS:
        raise ValueError(f"Unknown exchange preset: {exchange}")
    if execution_fee_mode not in EXECUTION_FEE_MODES:
        raise ValueError(f"Unknown execution fee mode: {execution_fee_mode}")
    if position_sizing_mode not in POSITION_SIZING_MODES:
        raise ValueError(f"Unknown position sizing mode: {position_sizing_mode}")

    env = PureTradingEnv(
        max_steps=max_steps,
        initial_balance=initial_balance,
        exchange=exchange,
        maker_fee_rate=maker_fee_rate,
        taker_fee_rate=taker_fee_rate,
        execution_fee_mode=execution_fee_mode,
        slippage=slippage,
        position_sizing_mode=position_sizing_mode,
        position_notional_fraction=position_notional_fraction,
        leverage=leverage,
        fixed_position_qty=fixed_position_qty,
        data_path=str(data_path),
        use_neurobars=use_neurobars,
        start_step=start_step,
    )
    env.start_step = start_step

    policy = build_policy(
        policy_name=policy_name,
        input_dim=env.input_dim,
        hidden_dim=hidden_dim,
        weights_path=weights_path,
        device=device,
        engine_config=engine_config,
    )
    return evaluate_policy_trace(env, policy_name=policy_name, policy=policy)


def evaluate_engine_vector(
    env: PureTradingEnv,
    vector: np.ndarray,
    *,
    engine_config: EngineConfig,
    device: Optional[str | torch.device] = None,
    policy_name: str = "monolith",
) -> EpisodeTrace:
    policy = VectorEnginePolicy(
        input_dim=env.input_dim,
        engine_config=engine_config,
        device=_resolve_device(device),
        vector=np.asarray(vector, dtype=np.float32),
    )
    return evaluate_policy_trace(env, policy_name=policy_name, policy=policy)


def resolve_date_window(
    data_path: str | Path,
    start_utc: str,
    end_utc: str,
) -> tuple[int, int]:
    timestamps = load_timestamps(data_path)
    start_ts = _parse_utc(start_utc)
    end_ts = _parse_utc(end_utc)

    start_step = int(np.searchsorted(timestamps, start_ts, side="left"))
    end_step = int(np.searchsorted(timestamps, end_ts, side="left"))
    if end_step < start_step:
        raise ValueError("end_utc must be >= start_utc")
    return start_step, end_step - start_step


def load_timestamps(data_path: str | Path) -> np.ndarray:
    data_path = Path(data_path)
    if data_path.suffix != ".npz":
        raise ValueError("Timestamp-based windows currently require an .npz file with timestamps")

    data = np.load(data_path)
    if "timestamps" not in data.files:
        raise ValueError(f"{data_path} does not contain a 'timestamps' array")
    return data["timestamps"]


def format_metrics_table(metrics: Iterable[EpisodeMetrics]) -> str:
    rows = [
        (
            metric.policy,
            f"{metric.pnl:9.2f}",
            f"{metric.final_balance:11.2f}",
            f"{metric.max_drawdown_pct:7.2f}",
            f"{metric.steps_run:9d}",
            f"{metric.trades:6d}",
            f"{metric.win_rate_pct:7.2f}",
        )
        for metric in metrics
    ]
    header = ("policy", "pnl", "final_balance", "max_dd", "steps", "trades", "win_pct")
    widths = [max(len(header[idx]), max((len(row[idx]) for row in rows), default=0)) for idx in range(len(header))]
    lines = [
        "  ".join(header[idx].ljust(widths[idx]) for idx in range(len(header))),
        "  ".join("-" * widths[idx] for idx in range(len(header))),
    ]
    for row in rows:
        lines.append("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(row))))
    return "\n".join(lines)


def _counter_dict(values: Iterable[int]) -> Dict[int, int]:
    counter = Counter(int(value) for value in values)
    return dict(sorted(counter.items()))


def _parse_utc(value: str) -> int:
    return int(datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())


def _step_timestamp_text(env: PureTradingEnv, step_index: int) -> str | None:
    if env.timestamps is None:
        return None
    ts = int(env.timestamps[step_index])
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _resolve_device(device: Optional[str | torch.device]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)
