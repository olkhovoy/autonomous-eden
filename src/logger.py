from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from src import metrics


@dataclass(frozen=True)
class ThoughtThresholds:
    panic_cortisol: float = 0.8
    flow_dopamine: float = 0.8
    agony_pain: float = 0.8
    boredom_cortisol: float = 0.2
    pnl_near_zero: float = 0.02


def decode_thoughts(
    self_state: Any,
    *,
    thresholds: ThoughtThresholds | None = None,
) -> str:
    """Translate a self-state vector into a human-readable thought."""
    if thresholds is None:
        thresholds = ThoughtThresholds()
    vec = _to_numpy(self_state)
    if vec.ndim == 2:
        vec = vec[0]
    if vec.size < 6:
        raise ValueError("self_state must have at least 6 elements.")
    balance, exposure, pnl, pain, dopamine, cortisol = vec[:6]
    if pain > 0.5:
        base = "FEAR: Liquidation approaching."
    elif cortisol < 0.1 and abs(exposure) < 1e-6:
        base = "BOREDOM: Seeking stimuli."
    elif cortisol > 0.8:
        base = "PANIC: Heart racing. Too much risk!"
    elif cortisol >= 0.4:
        base = "ANXIETY: The market is shaking. Be careful."
    elif cortisol >= 0.1:
        base = "ALERT: Watching closely."
    else:
        base = "CALM: All is quiet."

    overlay = ""
    if dopamine > 0.01:
        overlay = " EUPHORIA: I am a genius!"
    elif dopamine < -0.01:
        overlay = " REGRET: I made a mistake."
    return base + overlay


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("self_state is required.")
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value, dtype=np.float32)


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _colorize(text: str, color: str | None, enabled: bool) -> str:
    if not enabled or color is None:
        return text
    codes = {
        "red": "\x1b[31m",
        "green": "\x1b[32m",
        "yellow": "\x1b[33m",
        "reset": "\x1b[0m",
    }
    return f"{codes[color]}{text}{codes['reset']}"


def _stress_color(value: float) -> str:
    if value >= 0.8:
        return "red"
    if value >= 0.5:
        return "yellow"
    return "green"


def _dopamine_color(value: float) -> str:
    if value >= 0.8:
        return "green"
    if value >= 0.5:
        return "yellow"
    return "red"


class UnitaryLogger:
    def __init__(
        self,
        *,
        log_dir: str | Path = "logs",
        log_every: int = 1,
        enable_color: bool = True,
        write_jsonl: bool = False,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"thought_stream_{stamp}.csv"
        self.csv_file = self.csv_path.open("w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "step",
                "timestamp",
                "price",
                "pnl",
                "position",
                "balance",
                "exposure",
                "pnl_unrealized",
                "pain",
                "dopamine",
                "cortisol",
                "thought",
            ]
        )
        self.jsonl_file = None
        if write_jsonl:
            self.jsonl_file = (self.log_dir / f"thought_stream_{stamp}.jsonl").open("w")
        self.log_every = max(1, int(log_every))
        self.enable_color = bool(enable_color)

    def log_step(
        self,
        *,
        step: int,
        price: float,
        pnl: float,
        self_state: Any,
        position_size: float,
        thought: str | None = None,
    ) -> None:
        vec = _to_numpy(self_state)
        if vec.ndim == 2:
            vec = vec[0]
        if vec.size < 6:
            raise ValueError("self_state must have at least 6 elements.")
        balance, exposure, pnl_unrealized, pain, dopamine, cortisol = vec[:6]
        if thought is None:
            thought = decode_thoughts(vec)
        timestamp = time.time()

        self.csv_writer.writerow(
            [
                step,
                timestamp,
                price,
                pnl,
                position_size,
                balance,
                exposure,
                pnl_unrealized,
                pain,
                dopamine,
                cortisol,
                thought,
            ]
        )
        self.csv_file.flush()
        if self.jsonl_file is not None:
            payload = {
                "step": step,
                "timestamp": timestamp,
                "price": price,
                "pnl": pnl,
                "position": position_size,
                "balance": float(balance),
                "exposure": float(exposure),
                "pnl_unrealized": float(pnl_unrealized),
                "pain": float(pain),
                "dopamine": float(dopamine),
                "cortisol": float(cortisol),
                "thought": thought,
            }
            self.jsonl_file.write(json.dumps(payload) + "\n")
            self.jsonl_file.flush()

        if step % self.log_every == 0:
            stress_text = _colorize(
                f"{_format_pct(cortisol)}", _stress_color(cortisol), self.enable_color
            )
            pain_text = _colorize(
                f"{_format_pct(pain)}", _stress_color(pain), self.enable_color
            )
            dopamine_text = _colorize(
                f"{_format_pct(dopamine)}",
                _dopamine_color(dopamine),
                self.enable_color,
            )
            print(
                f"[Step {step}] Price: {price:.2f} | PnL: {pnl:+.2f} | "
                f"Stress: {stress_text} | Dopamine: {dopamine_text} | "
                f"Pain: {pain_text} | Position: {position_size:.2f}"
            )
            print(f'>> THOUGHT: "{thought}"')

    def log_autopsy(
        self,
        *,
        agent: Any | None = None,
        env: Any | None = None,
        inputs: Any | None = None,
        partition_loss: float | None = None,
        downward_causality: float | dict[str, float] | None = None,
        fixed_point: dict[str, Any] | None = None,
        baseline_reward: float | None = None,
    ) -> None:
        if partition_loss is None and agent is not None and env is not None:
            partition_loss = metrics.calc_partition_loss(agent, env)
        if downward_causality is None and agent is not None and env is not None:
            downward_causality = metrics.calc_downward_causality(
                agent, env, return_details=True
            )
        if fixed_point is None and agent is not None and inputs is not None:
            fixed_point = metrics.calc_fixed_point_stability(agent, inputs)

        print("[END REPORT]")
        if partition_loss is not None:
            if baseline_reward is not None:
                pct = partition_loss / (abs(baseline_reward) + 1e-8) * 100.0
                print(f"> Consciousness Bonus: {pct:+.1f}% (PL={partition_loss:+.3f})")
            else:
                print(f"> Consciousness Bonus: {partition_loss:+.3f}")
        if downward_causality is not None:
            will_text = _interpret_willpower(downward_causality)
            print(f"> Willpower Index: {will_text}")
        if fixed_point is not None:
            diagnosis = _interpret_stability(fixed_point)
            print(f"> Diagnosis: {diagnosis}")

    def close(self) -> None:
        if self.csv_file:
            self.csv_file.close()
        if self.jsonl_file:
            self.jsonl_file.close()


def _interpret_willpower(value: float | dict[str, float]) -> str:
    if isinstance(value, dict):
        grad = value.get("grad_norm", 0.0)
        shift = value.get("prob_shift", 0.0)
        score = grad + abs(shift)
        shift_pct = shift * 100.0
    else:
        score = float(value)
        shift_pct = None
    if score >= 1.0:
        detail = "High (Agent actively manages its own chemistry)"
    if score >= 0.2:
        detail = "Medium (Internal state nudges actions)"
    else:
        detail = "Low (Actions ignore internal stress)"
    if shift_pct is None:
        return detail
    return f"{detail} | Exit shift: {shift_pct:+.1f}%"


def _interpret_stability(result: dict[str, Any]) -> str:
    final_delta = float(result.get("final_delta", 0.0))
    mean_delta = float(result.get("mean_delta", 0.0))
    converged = bool(result.get("converged", False))
    if converged and mean_delta < 1e-2:
        return "Stable (Sanity Score: High)"
    if final_delta < 0.1:
        return "Neurotic (Oscillatory but bounded)"
    return "Chaotic (Divergent or unstable)"
