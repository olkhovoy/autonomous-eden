from __future__ import annotations

import torch

from src.envs import ExecutionModel, VecEnvConfig, VecMarketEnv
from src.self_state import PsychoConfig, PsychoModule


def build_synthetic_series(steps: int, close_idx: int) -> torch.Tensor:
    base = torch.linspace(20000.0, 21000.0, steps)
    noise = torch.randn(steps) * 50.0
    close = base + noise
    series = torch.zeros(steps, 5)
    series[:, close_idx] = close
    series[:, 0] = close * 0.999
    series[:, 1] = close * 1.001
    series[:, 2] = close * 0.998
    series[:, 4] = torch.rand(steps) * 1000.0
    return series


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = 512
    close_idx = 3
    raw = build_synthetic_series(steps, close_idx)
    market = raw.clone()
    config = VecEnvConfig(
        seq_len=32,
        close_idx=close_idx,
        action_sizes=(0.0, 0.5, 1.0, 2.0, -1.0),
    )
    psycho = PsychoModule(
        close_idx=close_idx,
        config=PsychoConfig(inputs_are_log_returns=False),
    )
    env = VecMarketEnv(
        market,
        raw,
        config=config,
        psycho=psycho,
        execution=ExecutionModel(),
        device=device,
    )
    obs, self_state = env.reset(batch_size=4)
    print("obs", obs["market"].shape, "self_state", self_state.shape)
    for step in range(5):
        action = torch.randint(0, 5, (4,), device=device)
        obs, reward, done, info = env.step(action)
        print(
            "step",
            step,
            "reward",
            float(reward.mean().item()),
            "done",
            bool(done.any().item()),
        )


if __name__ == "__main__":
    main()
