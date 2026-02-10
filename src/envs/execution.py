from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExecutionConfig:
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004
    slippage: float = 0.0001
    use_taker: bool = True


class ExecutionModel:
    """Lightweight execution cost model for vectorized sims."""

    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()

    def cost(self, price: torch.Tensor, position_delta: torch.Tensor) -> torch.Tensor:
        notional = position_delta.abs() * price
        fee_rate = self.config.taker_fee if self.config.use_taker else self.config.maker_fee
        fees = notional * fee_rate
        slippage = notional * self.config.slippage
        return fees + slippage
