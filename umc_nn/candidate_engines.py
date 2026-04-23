from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from umc_nn.umc_cell import UMCTradingCell


ENGINE_FAMILIES = {"umc", "linear_score", "small_recurrent"}
ACTION_HEAD_MODES = {"argmax", "confidence_gate"}


def _config_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"eng_{hashlib.sha1(encoded).hexdigest()[:12]}"


@dataclass(frozen=True, slots=True)
class EngineConfig:
    family: str = "umc"
    hidden_dim: int = 64
    alpha: float = 0.5
    action_head_mode: str = "argmax"
    action_threshold: float = 0.55

    def validate(self) -> None:
        if self.family not in ENGINE_FAMILIES:
            raise ValueError(f"Unsupported engine family: {self.family}")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        if self.action_head_mode not in ACTION_HEAD_MODES:
            raise ValueError(f"Unsupported action_head_mode: {self.action_head_mode}")
        if not (0.0 < self.action_threshold <= 1.0):
            raise ValueError("action_threshold must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_id(self) -> str:
        return _config_id(self.to_dict())


class _EngineModule(nn.Module):
    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        raise NotImplementedError

    def step(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError

    def get_num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def load_vector(self, vector: np.ndarray) -> None:
        _load_flat_parameters(self, vector)


class LinearScoreCell(_EngineModule):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.action_head = nn.Linear(input_dim, 3, bias=True)
        for param in self.parameters():
            param.requires_grad = False

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        del batch_size, device
        return None

    def step(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del state
        return self.action_head(x), None


class SmallRecurrentTradingCell(_EngineModule):
    def __init__(self, input_dim: int, hidden_dim: int = 32, alpha: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.W_x = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.activation = nn.Tanh()
        self.action_head = nn.Linear(hidden_dim, 3, bias=True)
        for param in self.parameters():
            param.requires_grad = False

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        target = self.activation(self.W_x(x) + self.W_h(state))
        next_state = (1.0 - self.alpha) * state + self.alpha * target
        return self.action_head(next_state), next_state


class UMCTradingEngine(_EngineModule):
    def __init__(self, input_dim: int, hidden_dim: int = 64, alpha: float = 0.5):
        super().__init__()
        self.cell = UMCTradingCell(input_dim=input_dim, hidden_dim=hidden_dim, alpha=alpha)
        self.hidden_dim = hidden_dim

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = torch.zeros(x.shape[0], self.hidden_dim, device=x.device)
        return self.cell(x, state)

    def get_num_parameters(self) -> int:
        return self.cell.get_num_parameters()

    def load_vector(self, vector: np.ndarray) -> None:
        self.cell.set_weights_from_vector(vector)


def build_engine_module(input_dim: int, config: EngineConfig) -> _EngineModule:
    config.validate()
    if config.family == "umc":
        return UMCTradingEngine(input_dim=input_dim, hidden_dim=config.hidden_dim, alpha=config.alpha)
    if config.family == "linear_score":
        return LinearScoreCell(input_dim=input_dim)
    if config.family == "small_recurrent":
        return SmallRecurrentTradingCell(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            alpha=config.alpha,
        )
    raise ValueError(f"Unsupported engine family: {config.family}")


def engine_num_parameters(input_dim: int, config: EngineConfig) -> int:
    return build_engine_module(input_dim, config).get_num_parameters()


def select_action(
    logits: torch.Tensor,
    *,
    action_head_mode: str,
    action_threshold: float,
) -> int:
    if action_head_mode == "argmax":
        return int(torch.argmax(logits, dim=1).item())
    if action_head_mode == "confidence_gate":
        probs = torch.softmax(logits, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)
        action = int(top_idx.item())
        if action != 0 and float(top_prob.item()) < action_threshold:
            return 0
        return action
    raise ValueError(f"Unsupported action_head_mode: {action_head_mode}")


class VectorEnginePolicy:
    def __init__(
        self,
        *,
        input_dim: int,
        engine_config: EngineConfig,
        device: torch.device,
        vector: np.ndarray | None = None,
        weights_path: str | Path | None = None,
    ):
        if vector is None and weights_path is None:
            raise ValueError("Either vector or weights_path must be provided")
        self.engine_config = engine_config
        self.device = device
        self.model = build_engine_module(input_dim=input_dim, config=engine_config).to(device)
        payload = vector if vector is not None else np.load(Path(weights_path))  # type: ignore[arg-type]
        self.model.load_vector(np.asarray(payload, dtype=np.float32))
        self.model.eval()
        self.state = self.model.initial_state(batch_size=1, device=device)

    def reset(self) -> None:
        self.state = self.model.initial_state(batch_size=1, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        x_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        logits, next_state = self.model.step(x_tensor, self.state)
        self.state = next_state
        return select_action(
            logits,
            action_head_mode=self.engine_config.action_head_mode,
            action_threshold=self.engine_config.action_threshold,
        )


def _load_flat_parameters(module: nn.Module, vector: np.ndarray) -> None:
    flat = np.asarray(vector, dtype=np.float32)
    expected = sum(param.numel() for param in module.parameters())
    if flat.size != expected:
        raise ValueError(f"Vector length {flat.size} does not match parameter count {expected}")

    index = 0
    with torch.no_grad():
        for param in module.parameters():
            size = param.numel()
            values = flat[index : index + size]
            index += size
            param.copy_(torch.from_numpy(values).view_as(param))
