#!/usr/bin/env python3
"""
LifeResource: EVE's vitality scalar (0.0-1.0).

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
from typing import Optional


STATE_PATH_DEFAULT = "data/life_resource.json"


@dataclass
class LifeResourceState:
    value: float = 0.7
    decay_rate: float = 0.001  # per heartbeat
    last_update: float = 0.0
    total_replenished: float = 0.0
    total_decayed: float = 0.0
    last_integrity_score: float = 0.0
    critical_events: int = 0


class LifeResource:
    """
    Manages EVE's vitality resource.
    
    Thresholds:
    - CRITICAL: < 0.3 → trigger survival reflection
    - LOW: < 0.5 → increase introspection frequency
    - NORMAL: 0.5-0.8 → standard operation
    - HIGH: > 0.8 → exploratory mode
    """
    
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0
    
    THRESHOLD_CRITICAL = 0.3
    THRESHOLD_LOW = 0.5
    THRESHOLD_HIGH = 0.8
    
    def __init__(
        self,
        state_path: str = STATE_PATH_DEFAULT,
        initial_value: float = 0.7,
        decay_rate: float = 0.001,
    ):
        self.state_path = state_path
        self._lock = threading.Lock()
        self.state = LifeResourceState(
            value=initial_value,
            decay_rate=decay_rate,
            last_update=time.time(),
        )
        self._load_state()
    
    def _load_state(self):
        if not os.path.exists(self.state_path):
            self._save_state()
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
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
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        tmp = f"{self.state_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(self.state), f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.state_path)
    
    def _clamp(self, v: float) -> float:
        return max(self.MIN_VALUE, min(self.MAX_VALUE, v))
    
    def tick(self) -> float:
        """Apply one heartbeat of decay. Returns new value."""
        with self._lock:
            decay = self.state.decay_rate
            self.state.value = self._clamp(self.state.value - decay)
            self.state.total_decayed += decay
            self.state.last_update = time.time()
            self._save_state()
            return self.state.value
    
    def replenish(self, amount: float, source: str = "unknown") -> float:
        """Add energy. Returns new value."""
        if amount <= 0:
            return self.state.value
        with self._lock:
            self.state.value = self._clamp(self.state.value + amount)
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
                bonus = delta * 0.05  # +0.05 per 1.0 improvement
                self.state.value = self._clamp(self.state.value + bonus)
                self.state.total_replenished += bonus
            self.state.last_update = time.time()
            self._save_state()
            return self.state.value
    
    def record_critical_event(self):
        """Record that a critical threshold was reached."""
        with self._lock:
            self.state.critical_events += 1
            self._save_state()
    
    def get_value(self) -> float:
        return self.state.value
    
    def get_mode(self) -> str:
        """Return current mode based on value."""
        v = self.state.value
        if v < self.THRESHOLD_CRITICAL:
            return "CRITICAL"
        if v < self.THRESHOLD_LOW:
            return "LOW"
        if v > self.THRESHOLD_HIGH:
            return "HIGH"
        return "NORMAL"
    
    def is_critical(self) -> bool:
        return self.state.value < self.THRESHOLD_CRITICAL
    
    def is_low(self) -> bool:
        return self.state.value < self.THRESHOLD_LOW
    
    def is_high(self) -> bool:
        return self.state.value > self.THRESHOLD_HIGH
    
    def get_state(self) -> dict:
        return {
            "value": self.state.value,
            "mode": self.get_mode(),
            "decay_rate": self.state.decay_rate,
            "last_update": self.state.last_update,
            "total_replenished": self.state.total_replenished,
            "total_decayed": self.state.total_decayed,
            "last_integrity_score": self.state.last_integrity_score,
            "critical_events": self.state.critical_events,
        }
