from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SelfState:
    balance: torch.Tensor
    exposure: torch.Tensor
    pnl_unrealized: torch.Tensor
    pain_distance: torch.Tensor
    dopamine: torch.Tensor
    cortisol: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.stack(
            [
                self.balance,
                self.exposure,
                self.pnl_unrealized,
                self.pain_distance,
                self.dopamine,
                self.cortisol,
            ],
            dim=1,
        )


@dataclass(frozen=True)
class PositionInfo:
    initial_balance: float | torch.Tensor
    position_size: float | torch.Tensor
    entry_price: float | torch.Tensor
    leverage: float | torch.Tensor
    balance: float | torch.Tensor | None = None
    liq_price: float | torch.Tensor | None = None


@dataclass(frozen=True)
class PsychoConfig:
    cortisol_decay: float = 0.97
    flat_cortisol_decay: float = 0.7
    pain_scale: float = 80.0
    stress_scale: float = 20.0
    eps: float = 1e-8
    inputs_are_log_returns: bool = True
    update_balance_from_pnl: bool = True
    debug: bool = False


class PsychoModule(nn.Module):
    def __init__(self, close_idx: int, config: PsychoConfig) -> None:
        super().__init__()
        self.close_idx = int(close_idx)
        self.config = config

    @staticmethod
    def _expand(
        value: float | torch.Tensor | None,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        return torch.full((batch,), float(value), device=device, dtype=dtype)

    def forward(
        self,
        window: torch.Tensor,
        position: PositionInfo,
        *,
        expected_pnl: float | torch.Tensor | None = None,
        prev_cortisol: float | torch.Tensor | None = None,
        flat_steps: float | torch.Tensor | None = None,
    ) -> SelfState:
        if window.ndim != 3:
            raise ValueError("window must be (batch, seq_len, features).")
        batch = window.shape[0]
        device = window.device
        dtype = window.dtype

        initial_balance = self._expand(
            position.initial_balance, batch, device, dtype
        )
        position_size = self._expand(
            position.position_size, batch, device, dtype
        )
        entry_price = self._expand(position.entry_price, batch, device, dtype)
        leverage = self._expand(position.leverage, batch, device, dtype)
        balance = self._expand(position.balance, batch, device, dtype)

        if balance is None:
            balance = initial_balance.clone()
        leverage = torch.clamp(leverage, min=self.config.eps)

        close_series = window[:, :, self.close_idx]
        if self.config.inputs_are_log_returns:
            log_price = close_series.cumsum(dim=1)
            current_price = entry_price * log_price[:, -1].exp()
            volatility = close_series.std(dim=1, unbiased=False) * 100.0
        else:
            current_price = close_series[:, -1]
            if close_series.shape[1] < 2:
                volatility = torch.zeros_like(current_price)
            else:
                returns = (
                    close_series[:, 1:]
                    / (close_series[:, :-1] + self.config.eps)
                ).log()
                volatility = returns.std(dim=1, unbiased=False) * 100.0

        exposure = position_size / (balance + self.config.eps)
        direction = torch.sign(position_size)
        no_position = exposure.abs() < self.config.eps
        pnl_unrealized = (
            (current_price - entry_price)
            / (entry_price + self.config.eps)
            * leverage
            * position_size
        )
        base_balance = balance
        if self.config.update_balance_from_pnl:
            balance = base_balance + pnl_unrealized
            balance = torch.clamp(balance, min=self.config.eps)
            exposure = position_size / (balance + self.config.eps)

        liq_price = self._expand(position.liq_price, batch, device, dtype)
        if liq_price is None:
            long_liq = entry_price * (1.0 - 1.0 / leverage)
            short_liq = entry_price * (1.0 + 1.0 / leverage)
            liq_price = torch.where(direction >= 0, long_liq, short_liq)

        distance = (current_price - liq_price).abs() + self.config.eps
        if self.config.debug:
            pnl_mean = float(pnl_unrealized.mean().item())
            pnl_min = float(pnl_unrealized.min().item())
            pnl_max = float(pnl_unrealized.max().item())
            dist_mean = float(distance.mean().item())
            dist_min = float(distance.min().item())
            dist_max = float(distance.max().item())
            print(
                "psycho_debug",
                f"pnl_mean={pnl_mean:.6f}",
                f"pnl_min={pnl_min:.6f}",
                f"pnl_max={pnl_max:.6f}",
                f"dist_mean={dist_mean:.6f}",
                f"dist_min={dist_min:.6f}",
                f"dist_max={dist_max:.6f}",
            )
        distance_norm = 1.0 - 1.0 / distance
        pain_distance = torch.sigmoid(-distance_norm * self.config.pain_scale)
        pain_distance = torch.where(
            no_position, torch.zeros_like(pain_distance), pain_distance
        )
        if flat_steps is not None:
            flat_steps_t = self._expand(flat_steps, batch, device, dtype)
            boredom = torch.clamp(flat_steps_t - 10.0, min=0.0) * 0.01
            pain_distance = torch.where(
                no_position,
                torch.clamp(pain_distance + boredom, max=1.0),
                pain_distance,
            )

        expected = self._expand(expected_pnl, batch, device, dtype)
        if expected is None:
            expected = torch.zeros_like(pnl_unrealized)
        dopamine = pnl_unrealized - expected

        prev_cort = self._expand(prev_cortisol, batch, device, dtype)
        if prev_cort is None:
            prev_cort = torch.zeros_like(pnl_unrealized)
        decay = torch.where(
            no_position,
            torch.full_like(prev_cort, self.config.flat_cortisol_decay),
            torch.full_like(prev_cort, self.config.cortisol_decay),
        )
        cortisol = prev_cort * decay + volatility * exposure.abs() * self.config.stress_scale
        cortisol = torch.clamp(cortisol, max=1.0)

        balance_norm = balance / (initial_balance + self.config.eps)

        return SelfState(
            balance=balance_norm,
            exposure=exposure,
            pnl_unrealized=pnl_unrealized,
            pain_distance=pain_distance,
            dopamine=dopamine,
            cortisol=cortisol,
        )
