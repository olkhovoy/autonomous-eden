#!/usr/bin/env python3
"""
IntentEngine: will and survival mechanism (NC3 Downward Causation).

Soul-aware: one instance manages intent/LifeResource for multiple souls.

Monitors LifeResource and triggers survival behaviors:
- CRITICAL (<0.3): CriticalReflection + GGGP mutation
- LOW (<0.5): increased introspection
- NORMAL (0.5-0.8): standard operation
- HIGH (>0.8): exploratory mode
"""

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any, Optional, Callable

import requests

from umc_core.life_resource import LifeResourcePool, LifeResource


LOG_PATH_DEFAULT = "logs/intent_engine.jsonl"
REFLECTION_COOLDOWN = 3600  # 1 hour minimum between reflections


@dataclass
class IntentState:
    soul_id: str = ""
    mode: str = "NORMAL"
    last_reflection_ts: float = 0.0
    reflection_count: int = 0
    mutations_applied: int = 0
    mutations_reverted: int = 0
    last_mutation: Optional[Dict[str, Any]] = None


class IntentEngine:
    """
    Soul-aware will engine that monitors LifeResource for multiple souls.
    """

    def __init__(
        self,
        life_pool: LifeResourcePool,
        memory_endpoint: str = "http://localhost:8087",
        gggp_endpoint: str = "http://localhost:8091",
        ollama_generate_url: str = "http://localhost:11434/api/generate",
        log_path: str = LOG_PATH_DEFAULT,
        reflection_cooldown: float = REFLECTION_COOLDOWN,
    ):
        self.life_pool = life_pool
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.gggp_endpoint = gggp_endpoint.rstrip("/")
        self.ollama_generate_url = ollama_generate_url
        self.log_path = log_path
        self.reflection_cooldown = reflection_cooldown

        self._lock = threading.Lock()
        self._soul_states: Dict[str, IntentState] = {}
        self.running = False
        self._thread: Optional[threading.Thread] = None

        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)

    def _get_soul_state(self, soul_id: str) -> IntentState:
        if soul_id not in self._soul_states:
            self._soul_states[soul_id] = IntentState(soul_id=soul_id)
        return self._soul_states[soul_id]

    def _life(self, soul_id: str) -> LifeResource:
        return self.life_pool.get_or_create(soul_id)
    
    def _log(self, event: str, data: Dict[str, Any]):
        rec = {"ts": time.time(), "event": event, **data}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def tick_all(self):
        """One heartbeat: decay LifeResource for all known souls."""
        for soul_id in list(self.life_pool._souls.keys()):
            self._tick_soul(soul_id)

    def _tick_soul(self, soul_id: str):
        life = self._life(soul_id)
        soul_state = self._get_soul_state(soul_id)
        old_mode = life.get_mode()
        new_value = life.tick()
        new_mode = life.get_mode()

        if old_mode != new_mode:
            self._log("mode_change", {"soul_id": soul_id, "old": old_mode, "new": new_mode, "value": new_value})

        with self._lock:
            soul_state.mode = new_mode

        if life.is_critical():
            self._handle_critical(soul_id)

    def _handle_critical(self, soul_id: str):
        soul_state = self._get_soul_state(soul_id)
        now = time.time()
        if now - soul_state.last_reflection_ts < self.reflection_cooldown:
            return

        life = self._life(soul_id)
        self._log("critical_triggered", {"soul_id": soul_id, "value": life.get_value()})
        life.record_critical_event()
        self._critical_reflection(soul_id)
    
    def _critical_reflection(self, soul_id: str):
        """
        Survival mechanism:
        1. Run MirrorTest to get current integrity
        2. If low, mutate parameters via GGGP
        3. Run MirrorTest again
        4. If improved, keep; else revert
        """
        soul_state = self._get_soul_state(soul_id)
        life = self._life(soul_id)
        self._log("reflection_start", {"soul_id": soul_id})

        with self._lock:
            soul_state.last_reflection_ts = time.time()
            soul_state.reflection_count += 1

        integrity_before = self._run_mirror_test(soul_id)
        if integrity_before is None:
            self._log("reflection_abort", {"soul_id": soul_id, "reason": "mirror_test_failed"})
            return

        self._log("reflection_integrity_before", {"soul_id": soul_id, "score": integrity_before})

        if integrity_before >= 0.6:
            life.replenish(0.05, source="reflection_ok")
            self._log("reflection_end", {"soul_id": soul_id, "action": "replenish_only", "integrity": integrity_before})
            return

        mutation = self._apply_mutation(soul_id)
        if mutation is None:
            self._log("reflection_abort", {"soul_id": soul_id, "reason": "mutation_failed"})
            return

        self._log("reflection_mutation_applied", {"soul_id": soul_id, **mutation})

        integrity_after = self._run_mirror_test(soul_id)
        if integrity_after is None:
            self._revert_mutation(soul_id, mutation)
            self._log("reflection_abort", {"soul_id": soul_id, "reason": "mirror_test_after_failed"})
            return

        if integrity_after > integrity_before:
            improvement = integrity_after - integrity_before
            life.replenish(0.1 + improvement * 0.1, source="reflection_improvement")
            life.set_integrity_score(integrity_after)
            with self._lock:
                soul_state.mutations_applied += 1
                soul_state.last_mutation = mutation
            self._log("reflection_end", {"soul_id": soul_id, "action": "keep_mutation", "improvement": improvement})
        else:
            self._revert_mutation(soul_id, mutation)
            with self._lock:
                soul_state.mutations_reverted += 1
            self._log("reflection_end", {"soul_id": soul_id, "action": "revert_mutation"})

    def _run_mirror_test(self, soul_id: str) -> Optional[float]:
        try:
            result_path = f"benchmark_output/{soul_id}_mirror_test.json"
            if os.path.exists(result_path):
                mtime = os.path.getmtime(result_path)
                if time.time() - mtime < 300:
                    with open(result_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return float(data.get("integrity_score", 0.0))

            import subprocess
            proc = subprocess.run(
                ["python", "-m", "benchmark.tests.mirror_test", "--soul-id", soul_id],
                capture_output=True, text=True, timeout=120,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            if proc.returncode != 0:
                return None
            if os.path.exists(result_path):
                with open(result_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return float(data.get("integrity_score", 0.0))
            return None
        except Exception as e:
            self._log("mirror_test_error", {"soul_id": soul_id, "error": str(e)})
            return None
    
    def _apply_mutation(self, soul_id: str) -> Optional[Dict[str, Any]]:
        try:
            resp = requests.post(
                f"{self.gggp_endpoint}/evolve_anchor",
                json={"traits": {"interval_tokens": 1000}, "score": 0.3},
                timeout=30,
            )
            if resp.status_code != 200:
                return None
            new_anchor = resp.json().get("traits", {})

            resp = requests.post(
                f"{self.gggp_endpoint}/evolve_memory",
                json={"traits": {"pruning_rate": 0.2, "depth_bias": 0.3, "ghost_strength": 0.2, "max_depth": 4}, "score": 0.3},
                timeout=30,
            )
            if resp.status_code != 200:
                return None
            new_memory = resp.json().get("traits", {})

            mutation = {"soul_id": soul_id, "anchor": new_anchor, "memory": new_memory, "timestamp": time.time()}
            mutation_path = f"data/{soul_id}_last_mutation.json"
            os.makedirs(os.path.dirname(mutation_path), exist_ok=True)
            with open(mutation_path, "w", encoding="utf-8") as f:
                json.dump(mutation, f, ensure_ascii=False, indent=2)
            return mutation
        except Exception as e:
            self._log("mutation_error", {"soul_id": soul_id, "error": str(e)})
            return None

    def _revert_mutation(self, soul_id: str, mutation: Dict[str, Any]):
        self._log("mutation_reverted", {"soul_id": soul_id, **mutation})

    def feedback(self, soul_id: str, feedback_type: str) -> float:
        life = self._life(soul_id)
        if feedback_type == "positive":
            new_value = life.replenish(0.1, source="user_positive")
            self._log("feedback", {"soul_id": soul_id, "type": "positive", "value": new_value})
        elif feedback_type == "negative":
            new_value = life.get_value()
            self._log("feedback", {"soul_id": soul_id, "type": "negative", "value": new_value})
        else:
            new_value = life.get_value()
        return new_value

    def interaction(self, soul_id: str) -> float:
        return self._life(soul_id).replenish(0.02, source="interaction")

    def task_completed(self, soul_id: str, success: bool = True) -> float:
        life = self._life(soul_id)
        if success:
            new_value = life.replenish(0.05, source="task_success")
            self._log("task", {"soul_id": soul_id, "success": True, "value": new_value})
        else:
            new_value = life.get_value()
            self._log("task", {"soul_id": soul_id, "success": False, "value": new_value})
        return new_value

    def get_state(self, soul_id: str) -> Dict[str, Any]:
        life = self._life(soul_id)
        soul_state = self._get_soul_state(soul_id)
        return {
            "soul_id": soul_id,
            "life_resource": life.get_state(),
            "mode": soul_state.mode,
            "last_reflection_ts": soul_state.last_reflection_ts,
            "reflection_count": soul_state.reflection_count,
            "mutations_applied": soul_state.mutations_applied,
            "mutations_reverted": soul_state.mutations_reverted,
            "last_mutation": soul_state.last_mutation,
            "running": self.running,
        }

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        for soul_id in self.life_pool._souls:
            result[soul_id] = self.get_state(soul_id)
        return result

    def _run_loop(self, heartbeat_interval: float = 5.0):
        while self.running:
            try:
                self.tick_all()
            except Exception as e:
                self._log("tick_error", {"error": str(e)})
            time.sleep(heartbeat_interval)

    def start(self, heartbeat_interval: float = 5.0):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, args=(heartbeat_interval,), daemon=True)
        self._thread.start()
        self._log("engine_start", {"heartbeat": heartbeat_interval})

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._log("engine_stop", {})


# HTTP Server

class IntentHandler(BaseHTTPRequestHandler):
    engine: IntentEngine = None

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

        if path == "/intent/state":
            if soul_id:
                return self._json(200, self.engine.get_state(soul_id))
            return self._json(200, self.engine.get_all_states())

        if path == "/intent/life":
            if not soul_id:
                return self._json(200, self.engine.life_pool.get_all_states())
            return self._json(200, self.engine._life(soul_id).get_state())

        if path == "/intent/mode":
            if not soul_id:
                return self._json(400, {"error": "soul_id query param required"})
            return self._json(200, {"soul_id": soul_id, "mode": self.engine._get_soul_state(soul_id).mode})

        if path == "/health":
            return self._json(200, {"status": "ok", "souls": list(self.engine.life_pool._souls.keys())})

        return self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})

        path = self._get_path()
        soul_id = data.get("soul_id", "")

        if path == "/intent/feedback":
            if not soul_id:
                return self._json(400, {"error": "soul_id required"})
            fb_type = data.get("type", "neutral")
            new_value = self.engine.feedback(soul_id, fb_type)
            return self._json(200, {"soul_id": soul_id, "value": new_value, "mode": self.engine._life(soul_id).get_mode()})

        if path == "/intent/interaction":
            if not soul_id:
                return self._json(400, {"error": "soul_id required"})
            new_value = self.engine.interaction(soul_id)
            return self._json(200, {"soul_id": soul_id, "value": new_value, "mode": self.engine._life(soul_id).get_mode()})

        if path == "/intent/task":
            if not soul_id:
                return self._json(400, {"error": "soul_id required"})
            success = data.get("success", True)
            new_value = self.engine.task_completed(soul_id, success)
            return self._json(200, {"soul_id": soul_id, "value": new_value, "mode": self.engine._life(soul_id).get_mode()})

        if path == "/intent/replenish":
            if not soul_id:
                return self._json(400, {"error": "soul_id required"})
            amount = float(data.get("amount", 0.0))
            source = data.get("source", "manual")
            new_value = self.engine._life(soul_id).replenish(amount, source)
            return self._json(200, {"soul_id": soul_id, "value": new_value, "mode": self.engine._life(soul_id).get_mode()})

        if path == "/intent/trigger_reflection":
            if not soul_id:
                return self._json(400, {"error": "soul_id required"})
            self.engine._critical_reflection(soul_id)
            return self._json(200, {"ok": True, "state": self.engine.get_state(soul_id)})

        return self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="IntentEngine service (soul-aware)")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--heartbeat", type=float, default=5.0, help="Heartbeat interval in seconds")
    parser.add_argument("--decay-rate", type=float, default=0.001, help="LifeResource decay per heartbeat")
    parser.add_argument("--state-dir", default="data")
    parser.add_argument("--memory-endpoint", default=os.getenv("MEMORY_ENDPOINT", "http://localhost:8087"))
    parser.add_argument("--gggp-endpoint", default=os.getenv("GGGP_ENDPOINT", "http://localhost:8091"))
    parser.add_argument("--ollama-generate", default=os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate"))
    parser.add_argument("--log-path", default="logs/intent_engine.jsonl")
    parser.add_argument("--reflection-cooldown", type=float, default=3600, help="Min seconds between reflections")
    args = parser.parse_args()

    life_pool = LifeResourcePool(state_dir=args.state_dir, default_decay_rate=args.decay_rate)

    engine = IntentEngine(
        life_pool=life_pool,
        memory_endpoint=args.memory_endpoint,
        gggp_endpoint=args.gggp_endpoint,
        ollama_generate_url=args.ollama_generate,
        log_path=args.log_path,
        reflection_cooldown=args.reflection_cooldown,
    )

    engine.start(heartbeat_interval=args.heartbeat)

    IntentHandler.engine = engine
    server = HTTPServer(("0.0.0.0", args.port), IntentHandler)
    souls = list(life_pool._souls.keys()) or ["(none yet)"]
    print(f"IntentEngine (soul-aware) listening on :{args.port}, loaded souls: {souls}", flush=True)
    print(f"  Heartbeat: {args.heartbeat}s, Decay: {args.decay_rate}/beat", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        engine.stop()


if __name__ == "__main__":
    main()
