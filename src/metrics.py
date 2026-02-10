from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


def _reset_env(env: Any, seed: int | None) -> Any:
    if seed is None:
        out = env.reset()
    else:
        try:
            out = env.reset(seed=seed)
        except TypeError:
            out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0]
    return out


def _step_env(env: Any, action: Any) -> tuple[Any, float, bool]:
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, _ = out
        done = terminated or truncated
    elif isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, _ = out
    else:
        raise ValueError("env.step must return 4 or 5 values.")
    return obs, float(reward), bool(done)


def _extract_market_self(obs: Any) -> tuple[Any, Any | None]:
    if isinstance(obs, dict):
        market = None
        self_state = None
        for key in ("market", "market_input", "observation", "obs", "state"):
            if key in obs:
                market = obs[key]
                break
        for key in ("self", "self_state", "self_vector", "self_vec"):
            if key in obs:
                self_state = obs[key]
                break
        if market is None:
            market = obs
        return market, self_state
    if isinstance(obs, (list, tuple)) and len(obs) == 2:
        return obs[0], obs[1]
    return obs, None


def _infer_self_dim(agent: Any) -> int | None:
    for attr in ("config", "model_config"):
        cfg = getattr(agent, attr, None)
        if cfg is not None and hasattr(cfg, "self_dim"):
            return int(cfg.self_dim)
    return None


def _to_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if value is None:
        raise ValueError("value is required for tensor conversion.")
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device=device)
    if isinstance(value, (float, int)):
        return torch.tensor([value], device=device, dtype=torch.float32)
    return torch.as_tensor(value, device=device)


def _ensure_market_shape(market: torch.Tensor) -> torch.Tensor:
    if market.ndim == 1:
        return market.unsqueeze(0).unsqueeze(0)
    if market.ndim == 2:
        return market.unsqueeze(0)
    return market


def _ensure_self_shape(self_state: torch.Tensor, batch: int) -> torch.Tensor:
    if self_state.ndim == 1:
        return self_state.unsqueeze(0).expand(batch, -1)
    return self_state


def _agent_forward(agent: Any, market: torch.Tensor, self_state: torch.Tensor | None) -> Any:
    if hasattr(agent, "act"):
        return agent.act(market, self_state)
    if self_state is None:
        return agent(market, None)
    return agent(market, self_state)


def _extract_action_logits(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ("action_logits", "logits"):
            if key in output:
                output = output[key]
                break
    if isinstance(output, (list, tuple)):
        output = output[0]
    if torch.is_tensor(output):
        return output
    if isinstance(output, np.ndarray):
        return torch.from_numpy(output)
    raise ValueError("Agent output does not contain logits.")


def _parse_agent_output(output: Any) -> tuple[int | None, torch.Tensor | None, torch.Tensor | None]:
    action = None
    logits = None
    self_pred = None
    if isinstance(output, dict):
        if "action" in output:
            action = int(output["action"])
        for key in ("action_logits", "logits"):
            if key in output:
                logits = _extract_action_logits(output[key])
                break
        for key in ("self_state", "self", "new_self"):
            if key in output:
                self_pred = output[key]
                break
        return action, logits, self_pred
    if isinstance(output, (list, tuple)):
        if output:
            if torch.is_tensor(output[0]) or isinstance(output[0], np.ndarray):
                logits = _extract_action_logits(output[0])
        if len(output) > 1:
            self_pred = output[1]
        return action, logits, self_pred
    if torch.is_tensor(output) or isinstance(output, np.ndarray):
        return None, _extract_action_logits(output), None
    if isinstance(output, (float, int)):
        return int(output), None, None
    raise ValueError("Unsupported agent output type.")


def _select_action(logits: torch.Tensor) -> int:
    if logits.ndim > 1:
        logits = logits[0]
    return int(torch.argmax(logits, dim=-1).item())


def _rollout(
    agent: Any,
    env: Any,
    *,
    seed: int | None,
    max_steps: int | None,
    mask_self: bool,
    device: torch.device,
) -> float:
    obs = _reset_env(env, seed)
    total_reward = 0.0
    steps = 0
    if hasattr(agent, "eval"):
        agent.eval()
    while True:
        market, self_state = _extract_market_self(obs)
        market_t = _ensure_market_shape(_to_tensor(market, device))
        if self_state is None:
            self_dim = _infer_self_dim(agent)
            if self_dim is not None:
                self_state_t = torch.zeros(
                    (market_t.shape[0], self_dim), device=device, dtype=market_t.dtype
                )
            else:
                self_state_t = None
        else:
            self_state_t = _to_tensor(self_state, device)
            self_state_t = _ensure_self_shape(self_state_t, market_t.shape[0])
        if mask_self and self_state_t is not None:
            self_state_t = torch.zeros_like(self_state_t)
        with torch.no_grad():
            output = _agent_forward(agent, market_t, self_state_t)
            action, logits, _ = _parse_agent_output(output)
            if action is None:
                if logits is None:
                    raise ValueError("Agent output did not provide action or logits.")
                action = _select_action(logits)
        obs, reward, done = _step_env(env, action)
        total_reward += reward
        steps += 1
        if done or (max_steps is not None and steps >= max_steps):
            break
    return total_reward


def calc_partition_loss(
    agent: Any,
    env: Any,
    *,
    seed: int | None = 0,
    max_steps: int | None = None,
    device: str | torch.device | None = None,
) -> float:
    """Reward drop when Self_Vector is masked."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reward_full = _rollout(
        agent, env, seed=seed, max_steps=max_steps, mask_self=False, device=dev
    )
    reward_masked = _rollout(
        agent, env, seed=seed, max_steps=max_steps, mask_self=True, device=dev
    )
    return reward_full - reward_masked


def calc_downward_causality(
    agent: Any,
    env: Any,
    *,
    cortisol_index: int = 5,
    cortisol_value: float = 5.0,
    exit_action_idx: int = 0,
    seed: int | None = 0,
    device: str | torch.device | None = None,
    return_details: bool = False,
) -> float | dict[str, float]:
    """Measure action sensitivity to a cortisol spike."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    obs = _reset_env(env, seed)
    market, self_state = _extract_market_self(obs)
    market_t = _ensure_market_shape(_to_tensor(market, dev))
    if self_state is None:
        self_dim = _infer_self_dim(agent)
        if self_dim is None:
            raise ValueError("Self_state missing and self_dim could not be inferred.")
        self_state_t = torch.zeros(
            (market_t.shape[0], self_dim), device=dev, dtype=market_t.dtype
        )
    else:
        self_state_t = _to_tensor(self_state, dev)
        self_state_t = _ensure_self_shape(self_state_t, market_t.shape[0])
    with torch.no_grad():
        base_logits = _extract_action_logits(
            _agent_forward(agent, market_t, self_state_t)
        )
        base_probs = F.softmax(base_logits, dim=-1)
        base_exit = base_probs[..., exit_action_idx].mean().item()

    injected = self_state_t.clone().detach()
    injected.requires_grad_(True)
    injected[:, cortisol_index] = cortisol_value
    logits = _extract_action_logits(_agent_forward(agent, market_t, injected))
    probs = F.softmax(logits, dim=-1)
    exit_prob = probs[..., exit_action_idx].mean()
    exit_prob.backward()
    grad = injected.grad
    grad_norm = float(grad.norm(p=2).item()) if grad is not None else 0.0
    prob_shift = float(exit_prob.detach().item() - base_exit)
    if return_details:
        return {"grad_norm": grad_norm, "prob_shift": prob_shift}
    return grad_norm


def calc_fixed_point_stability(
    agent: Any,
    inputs: Any,
    *,
    self_state: Any | None = None,
    steps: int = 32,
    action_index: int | None = None,
    threshold: float = 1e-3,
    device: str | torch.device | None = None,
) -> dict[str, float | bool | list[float]]:
    """Run recursive prediction without new market data and track convergence."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    market_t = _ensure_market_shape(_to_tensor(inputs, dev))
    if self_state is None:
        self_dim = _infer_self_dim(agent)
        if self_dim is None:
            raise ValueError("self_state missing and self_dim could not be inferred.")
        self_state_t = torch.zeros(
            (market_t.shape[0], self_dim), device=dev, dtype=market_t.dtype
        )
    else:
        self_state_t = _ensure_self_shape(_to_tensor(self_state, dev), market_t.shape[0])

    deltas: list[float] = []
    current = self_state_t
    with torch.no_grad():
        for _ in range(steps):
            output = _agent_forward(agent, market_t, current)
            action, logits, new_self = _parse_agent_output(output)
            if new_self is None:
                raise ValueError("Agent output missing self_state prediction.")
            if torch.is_tensor(new_self):
                new_self_t = new_self
            else:
                new_self_t = _to_tensor(new_self, dev)
            if new_self_t.ndim == 3:
                if action_index is None:
                    if action is None:
                        if logits is None:
                            raise ValueError("Action index required for multi-action self_state.")
                        action = _select_action(logits)
                    action_index = action
                next_state = new_self_t[:, action_index, :]
            else:
                next_state = new_self_t
            delta = float(torch.norm(next_state - current, p=2).item())
            deltas.append(delta)
            current = next_state
    final_delta = deltas[-1] if deltas else 0.0
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    converged = final_delta < threshold
    return {
        "final_delta": final_delta,
        "mean_delta": mean_delta,
        "converged": converged,
        "deltas": deltas,
    }
