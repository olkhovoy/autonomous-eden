from __future__ import annotations

from dataclasses import dataclass

import torch

from src.envs.execution import ExecutionModel
from src.self_state import PositionInfo, PsychoModule


@dataclass(frozen=True)
class VecEnvConfig:
    seq_len: int
    close_idx: int
    leverage: float = 3.0
    initial_balance: float = 1.0
    action_sizes: tuple[float, ...] | None = None


class VecMarketEnv:
    """GPU-first vectorized market environment."""

    def __init__(
        self,
        market: torch.Tensor,
        raw: torch.Tensor,
        *,
        config: VecEnvConfig,
        psycho: PsychoModule,
        execution: ExecutionModel | None = None,
        device: torch.device | None = None,
    ) -> None:
        if market.ndim != 2 or raw.ndim != 2:
            raise ValueError("market/raw must be (steps, features).")
        if market.shape != raw.shape:
            raise ValueError("market and raw must have matching shapes.")
        self.device = device or market.device
        self.market = market.to(self.device) if market.device != self.device else market
        self.raw = raw.to(self.device) if raw.device != self.device else raw
        self.config = config
        self.psycho = psycho.to(self.device)
        self.execution = execution or ExecutionModel()
        self.action_sizes = (
            torch.tensor(config.action_sizes, device=self.device, dtype=self.market.dtype)
            if config.action_sizes is not None
            else None
        )
        self._offsets = torch.arange(
            -self.config.seq_len, 0, device=self.device
        ).view(1, -1)
        self.cursor: torch.Tensor | None = None
        self.balance: torch.Tensor | None = None
        self.position_size: torch.Tensor | None = None
        self.entry_price: torch.Tensor | None = None
        self.prev_cortisol: torch.Tensor | None = None
        self.flat_steps: torch.Tensor | None = None

    def reset(
        self,
        *,
        batch_size: int,
        start_indices: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if start_indices is None:
            min_start = max(self.config.seq_len, 1)
            max_start = self.market.shape[0] - 1
            start_indices = torch.randint(
                min_start, max_start, (batch_size,), device=self.device
            )
        self.cursor = start_indices.to(self.device)
        self.balance = torch.full(
            (batch_size,), self.config.initial_balance, device=self.device
        )
        self.position_size = torch.zeros(batch_size, device=self.device)
        self.entry_price = self._price_at(self.cursor)
        self.prev_cortisol = torch.zeros(batch_size, device=self.device)
        self.flat_steps = torch.zeros(batch_size, device=self.device)
        obs, self_state = self._observe()
        return obs, self_state

    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.cursor is None:
            raise RuntimeError("Environment must be reset before stepping.")
        action = action.to(self.device).view(-1)
        if self.action_sizes is not None:
            if action.min() < 0 or action.max() >= self.action_sizes.numel():
                raise ValueError("Action index out of range for action_sizes.")
            action_values = self.action_sizes[action.long()]
        else:
            action_values = action.to(self.market.dtype)
        price_prev = self._price_at(self.cursor - 1)
        price_now = self._price_at(self.cursor)
        pos_prev = self.position_size
        target_pos = action_values * self.balance
        pos_delta = target_pos - pos_prev
        pnl = pos_prev * (price_now - price_prev) / price_prev * self.config.leverage
        cost = self.execution.cost(price_now, pos_delta)
        self.balance = torch.clamp(self.balance + pnl - cost, min=1e-8)
        self.position_size = target_pos
        entry_update = (pos_prev == 0) | (torch.sign(pos_prev) != torch.sign(target_pos))
        self.entry_price = torch.where(
            (target_pos != 0) & entry_update, price_now, self.entry_price
        )
        self.flat_steps = torch.where(
            target_pos == 0, self.flat_steps + 1.0, torch.zeros_like(self.flat_steps)
        )
        reward = pnl - cost
        self.cursor = self.cursor + 1
        done = self.cursor >= (self.market.shape[0] - 1)
        obs, self_state = self._observe()
        info = {
            "pnl": pnl,
            "cost": cost,
            "balance": self.balance,
            "position_size": self.position_size,
        }
        return obs, reward, done, info

    def _observe(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        market_window = self._window(self.market)
        raw_window = self._window(self.raw)
        position = PositionInfo(
            initial_balance=self.config.initial_balance,
            balance=self.balance,
            position_size=self.position_size,
            entry_price=self.entry_price,
            leverage=self.config.leverage,
        )
        state = self.psycho(
            raw_window,
            position,
            prev_cortisol=self.prev_cortisol,
            flat_steps=self.flat_steps,
        )
        self.prev_cortisol = state.cortisol.detach()
        obs = {"market": market_window, "self_state": state.as_tensor()}
        return obs, state.as_tensor()

    def _window(self, data: torch.Tensor) -> torch.Tensor:
        idx = self.cursor.view(-1, 1) + self._offsets
        return data[idx]

    def _price_at(self, idx: torch.Tensor) -> torch.Tensor:
        return self.raw[idx, self.config.close_idx]
