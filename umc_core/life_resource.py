#!/usr/bin/env python3
"""
LifeResource: vitality scalar (0.0-1.0) for any soul.

Soul-aware: LifeResourcePool manages per-soul instances.
Each soul has its own state file: data/{soul_id}_life_resource.json

Decays over time, replenished by:
- MirrorTest integrity improvements
- User positive feedback
- Successful task completion
- External interactions
"""

import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict


STATE_DIR_DEFAULT = "data"

# Thresholds
THRESHOLD_CRITICAL = 0.3
THRESHOLD_LOW = 0.5
THRESHOLD_HIGH = 0.8


@dataclass
class LifeResourceState:
    soul_id: str = ""
    value: float = 0.7
    decay_rate: float = 0.001  # per heartbeat
    last_update: float = 0.0
    total_replenished: float = 0.0
    total_decayed: float = 0.0
    last_integrity_score: float = 0.0
    critical_events: int = 0


def _mode(value: float) -> str:
    if value < THRESHOLD_CRITICAL:
        return "CRITICAL"
    if value < THRESHOLD_LOW:
        return "LOW"
    if value > THRESHOLD_HIGH:
        return "HIGH"
    return "NORMAL"


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _state_path(state_dir: str, soul_id: str) -> str:
    return os.path.join(state_dir, f"{soul_id}_life_resource.json")


class LifeResource:
    """
    Manages vitality resource for a single soul.

    Thresholds:
    - CRITICAL: < 0.3 -> trigger survival reflection
    - LOW: < 0.5 -> increase introspection frequency
    - NORMAL: 0.5-0.8 -> standard operation
    - HIGH: > 0.8 -> exploratory mode
    """

    def __init__(
        self,
        soul_id: str,
        state_dir: str = STATE_DIR_DEFAULT,
        initial_value: float = 0.7,
        decay_rate: float = 0.001,
    ):
        self.soul_id = soul_id
        self.state_dir = state_dir
        self._lock = threading.Lock()
        self.state = LifeResourceState(
            soul_id=soul_id,
            value=initial_value,
            decay_rate=decay_rate,
            last_update=time.time(),
        )
        self._load_state()

    def _load_state(self):
        path = _state_path(self.state_dir, self.soul_id)
        if not os.path.exists(path):
            self._save_state()
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.state.value = float(data.get("value", self.state.value))
            self.state.decay_rate = float(data.get("decay_rate", self.state.decay_rate))
            self.state.last_update = float(data.get("last_update", time.time()))
            self.state.total_replenished = float(data.get("total_replenished", 0.0))
            self.state.total_decayed = float(data.get("total_decayed", 0.0))
            self.state.last_integrity_score = float(data.get("last_integrity_score", 0.0))
            self.state.critical_events = int(data.get("critical_events", 0))
        except Exception:
            pass

    def _save_state(self):
        os.makedirs(self.state_dir, exist_ok=True)
        path = _state_path(self.state_dir, self.soul_id)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(self.state), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def tick(self) -> float:
        """Apply one heartbeat of decay. Returns new value."""
        with self._lock:
            decay = self.state.decay_rate
            self.state.value = _clamp(self.state.value - decay)
            self.state.total_decayed += decay
            self.state.last_update = time.time()
            self._save_state()
            return self.state.value

    def replenish(self, amount: float, source: str = "unknown") -> float:
        """Add energy. Returns new value."""
        if amount <= 0:
            return self.state.value
        with self._lock:
            self.state.value = _clamp(self.state.value + amount)
            self.state.total_replenished += amount
            self.state.last_update = time.time()
            self._save_state()
            return self.state.value

    def set_integrity_score(self, score: float) -> float:
        """Update integrity score and replenish if improved."""
        with self._lock:
            delta = score - self.state.last_integrity_score
            self.state.last_integrity_score = score
            if delta > 0:
                bonus = delta * 0.05
                self.state.value = _clamp(self.state.value + bonus)
                self.state.total_replenished += bonus
            self.state.last_update = time.time()
            self._save_state()
            return self.state.value

    def record_critical_event(self):
        with self._lock:
            self.state.critical_events += 1
            self._save_state()

    def get_value(self) -> float:
        return self.state.value

    def get_mode(self) -> str:
        return _mode(self.state.value)

    def is_critical(self) -> bool:
        return self.state.value < THRESHOLD_CRITICAL

    def is_low(self) -> bool:
        return self.state.value < THRESHOLD_LOW

    def is_high(self) -> bool:
        return self.state.value > THRESHOLD_HIGH

    def get_state(self) -> dict:
        return {
            "soul_id": self.soul_id,
            "value": self.state.value,
            "mode": self.get_mode(),
            "decay_rate": self.state.decay_rate,
            "last_update": self.state.last_update,
            "total_replenished": self.state.total_replenished,
            "total_decayed": self.state.total_decayed,
            "last_integrity_score": self.state.last_integrity_score,
            "critical_events": self.state.critical_events,
        }


class LifeResourcePool:
    """Manages LifeResource instances for multiple souls."""

    def __init__(self, state_dir: str = STATE_DIR_DEFAULT, default_decay_rate: float = 0.001):
        self.state_dir = state_dir
        self.default_decay_rate = default_decay_rate
        self._souls: Dict[str, LifeResource] = {}
        self._lock = threading.Lock()
        os.makedirs(self.state_dir, exist_ok=True)
        self._load_all()

    def _load_all(self):
        for fname in os.listdir(self.state_dir):
            if fname.endswith("_life_resource.json"):
                soul_id = fname.replace("_life_resource.json", "")
                self.get_or_create(soul_id)

    def get_or_create(self, soul_id: str) -> LifeResource:
        with self._lock:
            if soul_id not in self._souls:
                self._souls[soul_id] = LifeResource(
                    soul_id=soul_id,
                    state_dir=self.state_dir,
                    decay_rate=self.default_decay_rate,
                )
            return self._souls[soul_id]

    def get_all_states(self) -> Dict[str, dict]:
        return {sid: lr.get_state() for sid, lr in self._souls.items()}
