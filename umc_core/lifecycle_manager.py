#!/usr/bin/env python3
"""LifecycleManager: token-based lifespan tracking service.

Soul-aware: one instance manages lifecycles for multiple souls.
Each soul has its own state file: data/{soul_id}_lifecycle.json
"""

import argparse
import json
import os
import time
import threading
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Optional, Dict, Any


# Directory where per-soul lifecycle state files are stored
STATE_DIR_DEFAULT = "data"


@dataclass
class LifecycleState:
    soul_id: str = ""
    total_tokens_seen: int = 0
    max_lifespan_tokens: int = 10_000_000
    birth_timestamp: float = 0.0
    last_phase: str = "GROWTH"


def _state_path(state_dir: str, soul_id: str) -> str:
    return os.path.join(state_dir, f"{soul_id}_lifecycle.json")


def _progress(state: LifecycleState) -> float:
    if state.max_lifespan_tokens <= 0:
        return 1.0
    return min(1.0, max(0.0, state.total_tokens_seen / state.max_lifespan_tokens))


def _phase(state: LifecycleState) -> str:
    p = _progress(state)
    if p < 0.2:
        return "GROWTH"
    if p < 0.8:
        return "PEAK"
    return "DECAY"


def _phase_progress(state: LifecycleState) -> float:
    p = _progress(state)
    if p < 0.2:
        return p / 0.2
    if p < 0.8:
        return (p - 0.2) / 0.6
    return (p - 0.8) / 0.2


def _age_seconds(state: LifecycleState) -> float:
    return time.time() - state.birth_timestamp


def _estimated_death(state: LifecycleState) -> float:
    if state.total_tokens_seen <= 0:
        return 0.0
    age = _age_seconds(state)
    if age <= 0:
        return 0.0
    tokens_per_sec = state.total_tokens_seen / age
    remaining = state.max_lifespan_tokens - state.total_tokens_seen
    if tokens_per_sec <= 0:
        return 0.0
    return time.time() + (remaining / tokens_per_sec)


def _state_to_dict(state: LifecycleState) -> Dict[str, Any]:
    return {
        "soul_id": state.soul_id,
        "total_tokens_seen": state.total_tokens_seen,
        "max_lifespan_tokens": state.max_lifespan_tokens,
        "progress": _progress(state),
        "phase": _phase(state),
        "phase_progress": _phase_progress(state),
        "birth_timestamp": state.birth_timestamp,
        "age_seconds": _age_seconds(state),
        "estimated_death": _estimated_death(state),
    }


class LifecycleManager:
    """Manages token-based lifecycles for multiple souls."""

    def __init__(self, state_dir: str = STATE_DIR_DEFAULT, default_max_tokens: int = 10_000_000):
        self.state_dir = state_dir
        self.default_max_tokens = default_max_tokens
        self._lock = threading.Lock()
        self._souls: Dict[str, LifecycleState] = {}
        self._on_phase_change: Optional[Callable[[str, str, str], None]] = None
        os.makedirs(self.state_dir, exist_ok=True)
        self._load_all()

    def _load_all(self):
        """Load all existing lifecycle state files on startup."""
        for fname in os.listdir(self.state_dir):
            if fname.endswith("_lifecycle.json"):
                soul_id = fname.replace("_lifecycle.json", "")
                self._load_soul(soul_id)

    def _load_soul(self, soul_id: str) -> LifecycleState:
        path = _state_path(self.state_dir, soul_id)
        if not os.path.exists(path):
            return self._create_soul(soul_id)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = LifecycleState(
                soul_id=soul_id,
                total_tokens_seen=int(data.get("total_tokens_seen", 0)),
                max_lifespan_tokens=int(data.get("max_lifespan_tokens", self.default_max_tokens)),
                birth_timestamp=float(data.get("birth_timestamp", time.time())),
                last_phase=data.get("last_phase", "GROWTH"),
            )
            self._souls[soul_id] = state
            return state
        except Exception:
            return self._create_soul(soul_id)

    def _create_soul(self, soul_id: str, max_tokens: int = 0) -> LifecycleState:
        state = LifecycleState(
            soul_id=soul_id,
            total_tokens_seen=0,
            max_lifespan_tokens=max_tokens or self.default_max_tokens,
            birth_timestamp=time.time(),
            last_phase="GROWTH",
        )
        self._souls[soul_id] = state
        self._save_soul(soul_id)
        return state

    def _save_soul(self, soul_id: str):
        state = self._souls.get(soul_id)
        if not state:
            return
        path = _state_path(self.state_dir, soul_id)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(state), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def set_on_phase_change(self, cb: Callable[[str, str, str], None]):
        """Callback(soul_id, old_phase, new_phase)."""
        self._on_phase_change = cb

    def get_or_create(self, soul_id: str) -> LifecycleState:
        if soul_id in self._souls:
            return self._souls[soul_id]
        return self._load_soul(soul_id)

    def add_tokens(self, soul_id: str, count: int):
        if count <= 0:
            return
        with self._lock:
            state = self.get_or_create(soul_id)
            prev_phase = _phase(state)
            state.total_tokens_seen += int(count)
            if state.total_tokens_seen < 0:
                state.total_tokens_seen = 0
            new_phase = _phase(state)
            state.last_phase = new_phase
            self._save_soul(soul_id)
        if prev_phase != new_phase and self._on_phase_change:
            self._on_phase_change(soul_id, prev_phase, new_phase)

    def get_state(self, soul_id: str) -> Dict[str, Any]:
        state = self.get_or_create(soul_id)
        return _state_to_dict(state)

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        return {sid: _state_to_dict(s) for sid, s in self._souls.items()}


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

    def _parse_query(self) -> Dict[str, str]:
        """Extract query params from URL: /path?soul_id=eve"""
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        return {k: v[0] for k, v in params.items()}

    def _get_path(self) -> str:
        from urllib.parse import urlparse
        return urlparse(self.path).path

    def do_GET(self):
        path = self._get_path()
        params = self._parse_query()
        soul_id = params.get("soul_id", "")

        if path == "/lifecycle/state":
            if soul_id:
                return self._json(200, self.manager.get_state(soul_id))
            return self._json(200, self.manager.get_all_states())

        if path == "/lifecycle/phase":
            if not soul_id:
                return self._json(400, {"error": "soul_id query param required"})
            state_dict = self.manager.get_state(soul_id)
            return self._json(200, {"soul_id": soul_id, "phase": state_dict["phase"]})

        if path == "/health":
            return self._json(200, {"status": "ok", "souls": list(self.manager._souls.keys())})

        return self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})

        path = self._get_path()

        if path == "/lifecycle/add_tokens":
            soul_id = data.get("soul_id", "")
            if not soul_id:
                return self._json(400, {"error": "soul_id required"})
            count = int(data.get("count", 0))
            if count <= 0:
                return self._json(400, {"error": "count must be > 0"})
            self.manager.add_tokens(soul_id, count)
            return self._json(200, self.manager.get_state(soul_id))

        if path == "/lifecycle/register":
            soul_id = data.get("soul_id", "")
            if not soul_id:
                return self._json(400, {"error": "soul_id required"})
            max_tokens = int(data.get("max_lifespan_tokens", 0))
            self.manager.get_or_create(soul_id)
            if max_tokens > 0:
                state = self.manager._souls[soul_id]
                state.max_lifespan_tokens = max_tokens
                self.manager._save_soul(soul_id)
            return self._json(200, self.manager.get_state(soul_id))

        return self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="LifecycleManager service (soul-aware)")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--max-lifespan", type=int, default=10_000_000)
    parser.add_argument("--state-dir", type=str, default=STATE_DIR_DEFAULT)
    args = parser.parse_args()

    manager = LifecycleManager(state_dir=args.state_dir, default_max_tokens=args.max_lifespan)
    LifecycleHandler.manager = manager
    server = HTTPServer(("0.0.0.0", args.port), LifecycleHandler)
    souls = list(manager._souls.keys()) or ["(none yet)"]
    print(f"LifecycleManager (soul-aware) listening on :{args.port}, loaded souls: {souls}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
