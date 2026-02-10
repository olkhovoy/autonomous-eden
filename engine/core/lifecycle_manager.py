#!/usr/bin/env python3
"""LifecycleManager: token-based lifespan tracking service."""

import argparse
import json
import os
import time
import threading
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Optional, Dict, Any


STATE_PATH_DEFAULT = "data/lifecycle_state.json"


@dataclass
class LifecycleState:
    total_tokens_seen: int = 0
    max_lifespan_tokens: int = 10_000_000
    birth_timestamp: float = 0.0
    last_phase: str = "GROWTH"


class LifecycleManager:
    def __init__(self, state_path: str = STATE_PATH_DEFAULT, max_lifespan_tokens: int = 10_000_000):
        self.state_path = state_path
        self._lock = threading.Lock()
        self._on_phase_change: Optional[Callable[[str, str], None]] = None
        self.state = LifecycleState(total_tokens_seen=0, max_lifespan_tokens=max_lifespan_tokens)
        self._last_phase = self.get_phase()
        self._load_state()

    def _load_state(self):
        if not os.path.exists(self.state_path):
            self.state.birth_timestamp = time.time()
            self._save_state()
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.state.total_tokens_seen = int(data.get("total_tokens_seen", 0))
            self.state.max_lifespan_tokens = int(data.get("max_lifespan_tokens", self.state.max_lifespan_tokens))
            self.state.birth_timestamp = float(data.get("birth_timestamp", time.time()))
            self.state.last_phase = data.get("last_phase", "GROWTH")
            self._last_phase = self.get_phase()
        except Exception:
            self.state.birth_timestamp = time.time()
            return

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        tmp_path = f"{self.state_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.state), f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.state_path)

    def set_on_phase_change(self, cb: Callable[[str, str], None]):
        self._on_phase_change = cb

    def add_tokens(self, count: int):
        if count <= 0:
            return
        with self._lock:
            prev_phase = self.get_phase()
            self.state.total_tokens_seen += int(count)
            if self.state.total_tokens_seen < 0:
                self.state.total_tokens_seen = 0
            new_phase = self.get_phase()
            self._save_state()
        if prev_phase != new_phase and self._on_phase_change:
            self._on_phase_change(prev_phase, new_phase)

    def get_progress(self) -> float:
        if self.state.max_lifespan_tokens <= 0:
            return 1.0
        return min(1.0, max(0.0, self.state.total_tokens_seen / self.state.max_lifespan_tokens))

    def get_phase(self) -> str:
        progress = self.get_progress()
        if progress < 0.2:
            return "GROWTH"
        if progress < 0.8:
            return "PEAK"
        return "DECAY"

    def get_phase_progress(self) -> float:
        """Progress within current phase (0.0-1.0)."""
        progress = self.get_progress()
        if progress < 0.2:
            return progress / 0.2
        if progress < 0.8:
            return (progress - 0.2) / 0.6
        return (progress - 0.8) / 0.2

    def get_age_seconds(self) -> float:
        return time.time() - self.state.birth_timestamp

    def get_estimated_death(self) -> float:
        """Estimated timestamp of lifecycle end based on current token rate."""
        if self.state.total_tokens_seen <= 0:
            return 0.0
        age = self.get_age_seconds()
        if age <= 0:
            return 0.0
        tokens_per_sec = self.state.total_tokens_seen / age
        remaining = self.state.max_lifespan_tokens - self.state.total_tokens_seen
        if tokens_per_sec <= 0:
            return 0.0
        return time.time() + (remaining / tokens_per_sec)


class LifecycleHandler(BaseHTTPRequestHandler):
    manager: LifecycleManager = None

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
    
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def _json(self, code: int, payload: Dict[str, Any]):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/lifecycle/state":
            return self._json(200, {
                "total_tokens_seen": self.manager.state.total_tokens_seen,
                "max_lifespan_tokens": self.manager.state.max_lifespan_tokens,
                "progress": self.manager.get_progress(),
                "phase": self.manager.get_phase(),
                "phase_progress": self.manager.get_phase_progress(),
                "birth_timestamp": self.manager.state.birth_timestamp,
                "age_seconds": self.manager.get_age_seconds(),
                "estimated_death": self.manager.get_estimated_death(),
            })
        if self.path == "/lifecycle/phase":
            return self._json(200, {"phase": self.manager.get_phase()})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})

        if self.path == "/lifecycle/add_tokens":
            count = int(data.get("count", 0))
            if count <= 0:
                return self._json(400, {"error": "count must be > 0"})
            self.manager.add_tokens(count)
            return self._json(200, {
                "ok": True,
                "total_tokens_seen": self.manager.state.total_tokens_seen,
                "phase": self.manager.get_phase(),
                "progress": self.manager.get_progress(),
            })
        return self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="LifecycleManager service")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--max-lifespan", type=int, default=10_000_000)
    parser.add_argument("--state-path", type=str, default=STATE_PATH_DEFAULT)
    args = parser.parse_args()

    manager = LifecycleManager(state_path=args.state_path, max_lifespan_tokens=args.max_lifespan)
    LifecycleHandler.manager = manager
    server = HTTPServer(("0.0.0.0", args.port), LifecycleHandler)
    print(f"LifecycleManager listening on :{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
