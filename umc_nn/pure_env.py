from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ExchangeFeePreset:
    exchange: str
    maker_fee_rate: float
    taker_fee_rate: float
    source_note: str


EXCHANGE_FEE_PRESETS: Dict[str, ExchangeFeePreset] = {
    "binance": ExchangeFeePreset(
        exchange="binance",
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
        source_note="Binance futures non-VIP base rates from official Binance blog/fee schedule",
    ),
    "bybit": ExchangeFeePreset(
        exchange="bybit",
        maker_fee_rate=0.0002,
        taker_fee_rate=0.00055,
        source_note="Bybit Perpetual & Futures Contract non-VIP rates from official help center",
    ),
    "okx": ExchangeFeePreset(
        exchange="okx",
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0005,
        source_note="OKX lv1 futures rates from official help center",
    ),
}

POSITION_SIZING_MODES = {"fraction_of_equity", "fixed_quantity"}
EXECUTION_FEE_MODES = {"taker", "maker"}


class PureTradingEnv:
    """
    Minimal futures-style environment with explicit position sizing.

    The default mode uses a fraction of current equity to size each new position,
    so a 10k USDT account no longer behaves like it is trading 1 BTC by default.
    """

    def __init__(
        self,
        max_steps: int = 10000,
        initial_balance: float = 10000.0,
        taker_fee_rate: float | None = None,
        slippage: float = 0.0,
        data_path: str | None = None,
        use_neurobars: bool = False,
        start_step: int = 0,
        exchange: str = "binance",
        maker_fee_rate: float | None = None,
        execution_fee_mode: str = "taker",
        position_sizing_mode: str = "fraction_of_equity",
        position_notional_fraction: float = 1.0,
        leverage: float = 1.0,
        fixed_position_qty: float = 1.0,
        allow_synthetic_data: bool = False,
    ):
        if exchange not in EXCHANGE_FEE_PRESETS:
            raise ValueError(f"Unsupported exchange preset: {exchange}")
        if execution_fee_mode not in EXECUTION_FEE_MODES:
            raise ValueError(f"Unsupported execution fee mode: {execution_fee_mode}")
        if position_sizing_mode not in POSITION_SIZING_MODES:
            raise ValueError(f"Unsupported position sizing mode: {position_sizing_mode}")
        if position_notional_fraction < 0:
            raise ValueError("position_notional_fraction must be >= 0")
        if leverage <= 0:
            raise ValueError("leverage must be > 0")
        if fixed_position_qty < 0:
            raise ValueError("fixed_position_qty must be >= 0")

        preset = EXCHANGE_FEE_PRESETS[exchange]

        self.max_steps = max_steps
        self.initial_balance = initial_balance
        self.exchange = exchange
        self.execution_fee_mode = execution_fee_mode
        self.position_sizing_mode = position_sizing_mode
        self.position_notional_fraction = position_notional_fraction
        self.leverage = leverage
        self.fixed_position_qty = fixed_position_qty
        self.slippage = slippage
        self.use_neurobars = use_neurobars
        self.start_step = start_step
        self.allow_synthetic_data = allow_synthetic_data

        self.maker_fee_rate = preset.maker_fee_rate if maker_fee_rate is None else maker_fee_rate
        self.taker_fee_rate = preset.taker_fee_rate if taker_fee_rate is None else taker_fee_rate

        self.current_step = 0
        self.balance = self.initial_balance
        self.cumulative_pnl = 0.0

        # 0: Flat, 1: Long, 2: Short
        self.current_position = 0
        self.position_qty = 0.0
        self.position_notional = 0.0

        self.random_state = np.random.RandomState(42)
        self.input_dim = 32 if use_neurobars else 5
        self.timestamps: np.ndarray | None = None

        if data_path and os.path.exists(data_path):
            try:
                if self.use_neurobars and data_path.endswith(".npz"):
                    data = np.load(data_path)
                    self.features = data["neurobars"]
                    self.prices = data["close_prices"]
                    self.timestamps = data["timestamps"] if "timestamps" in data.files else None
                    self.input_dim = self.features.shape[1]
                else:
                    with open(data_path, "r") as f:
                        raw_data = json.load(f)

                    self.features = np.array(raw_data, dtype=np.float32)
                    self.features = self.features[:, 1:6]
                    self.prices = self.features[:, 3].copy()

                    means = self.features.mean(axis=0)
                    stds = self.features.std(axis=0)
                    stds[stds == 0] = 1e-8
                    self.features = (self.features - means) / stds
                    self.input_dim = 5

                self.max_steps = min(self.max_steps, len(self.prices) - 2)
            except Exception as e:
                if self.allow_synthetic_data:
                    print(f"Error loading {data_path}, falling back to synthetic data: {e}")
                    self._generate_synthetic_data()
                else:
                    raise RuntimeError(f"Failed to load trading data from {data_path}") from e
        else:
            if self.allow_synthetic_data:
                self._generate_synthetic_data()
            else:
                raise FileNotFoundError(f"Trading data path is missing or does not exist: {data_path}")

    def _generate_synthetic_data(self):
        self.input_dim = 5
        self.prices = np.ones(self.max_steps + 1) * 100.0
        price_changes = self.random_state.randn(self.max_steps) * 0.5
        self.prices[1:] += np.cumsum(price_changes)

        self.features = self.random_state.randn(self.max_steps + 1, self.input_dim)
        means = self.features.mean(axis=0)
        stds = self.features.std(axis=0)
        stds[stds == 0] = 1e-8
        self.features = (self.features - means) / stds

    def reset(self) -> np.ndarray:
        self.current_step = self.start_step
        self.balance = self.initial_balance
        self.cumulative_pnl = 0.0
        self.current_position = 0
        self.position_qty = 0.0
        self.position_notional = 0.0
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        return self.features[self.current_step]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        assert 0 <= action <= 2, f"Invalid action {action}"

        requested_action = action
        current_price = float(self.prices[self.current_step])
        next_price = float(self.prices[self.current_step + 1])
        price_change = next_price - current_price
        step_index = self.current_step
        previous_position = self.current_position
        position_qty_before = self.position_qty
        position_notional_before = self.position_notional
        balance_before = self.balance

        step_pnl = self._mark_to_market(price_change)

        balance_after_pnl = self.balance + step_pnl
        friction_cost = 0.0
        close_fee = 0.0
        open_fee = 0.0

        if action != self.current_position:
            if self.current_position != 0 and self.position_qty > 0:
                close_notional = self.position_qty * current_price
                close_fee = close_notional * self._effective_fee_rate()
                friction_cost += close_fee
                self.position_qty = 0.0
                self.position_notional = 0.0

            if action != 0:
                available_equity = max(balance_after_pnl - friction_cost, 0.0)
                qty, notional = self._size_new_position(available_equity, current_price)
                if qty > 0.0:
                    open_fee = notional * self._effective_fee_rate()
                    friction_cost += open_fee
                    self.position_qty = qty
                    self.position_notional = notional
                else:
                    action = 0

        self.balance = balance_after_pnl - friction_cost
        self.cumulative_pnl += step_pnl - friction_cost
        self.current_position = action
        self.current_step += 1

        done = self.current_step >= self._terminal_step() or self.balance <= 0
        info = {
            "balance": self.balance,
            "cumulative_pnl": self.cumulative_pnl,
            "current_position": self.current_position,
            "friction_cost": friction_cost,
            "close_fee": close_fee,
            "open_fee": open_fee,
            "step_index": step_index,
            "current_price": current_price,
            "next_price": next_price,
            "step_pnl": step_pnl,
            "balance_before": balance_before,
            "balance_after_pnl": balance_after_pnl,
            "previous_position": previous_position,
            "requested_action": requested_action,
            "executed_action": self.current_position,
            "position_qty_before": position_qty_before,
            "position_qty": self.position_qty,
            "position_qty_after": self.position_qty,
            "position_notional_before": position_notional_before,
            "position_notional": self.position_notional,
            "position_notional_after": self.position_notional,
            "exchange": self.exchange,
            "execution_fee_mode": self.execution_fee_mode,
        }
        reward = step_pnl - friction_cost
        return self.get_observation(), reward, done, info

    def _mark_to_market(self, price_change: float) -> float:
        if self.current_position == 1:
            return self.position_qty * price_change
        if self.current_position == 2:
            return -self.position_qty * price_change
        return 0.0

    def _size_new_position(self, available_equity: float, current_price: float) -> tuple[float, float]:
        if available_equity <= 0 or current_price <= 0:
            return 0.0, 0.0

        if self.position_sizing_mode == "fixed_quantity":
            qty = self.fixed_position_qty
            notional = qty * current_price
            return qty, notional

        notional = available_equity * self.position_notional_fraction * self.leverage
        if notional <= 0:
            return 0.0, 0.0
        qty = notional / current_price
        return qty, notional

    def _effective_fee_rate(self) -> float:
        base_fee = self.taker_fee_rate if self.execution_fee_mode == "taker" else self.maker_fee_rate
        if self.execution_fee_mode == "taker":
            return base_fee + self.slippage
        return base_fee

    def _terminal_step(self) -> int:
        last_price_step = len(self.prices) - 1
        return min(self.start_step + self.max_steps, last_price_step)
