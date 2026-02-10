#!/usr/bin/env python3
"""
IntentEngine: EVE's will and survival mechanism (NC3 Downward Causation).

Monitors LifeResource and triggers survival behaviors:
- CRITICAL (<0.3): CriticalReflection + GGGP mutation
- LOW (<0.5): increased introspection
- NORMAL (0.5-0.8): standard operation
- HIGH (>0.8): exploratory mode

The system "wants to survive" by keeping integrity_score high.
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

from umc_core.life_resource import LifeResource


LOG_PATH_DEFAULT = "logs/intent_engine.jsonl"
REFLECTION_COOLDOWN = 3600  # 1 hour minimum between reflections


@dataclass
class IntentState:
    mode: str = "NORMAL"
    last_reflection_ts: float = 0.0
    reflection_count: int = 0
    mutations_applied: int = 0
    mutations_reverted: int = 0
    last_mutation: Optional[Dict[str, Any]] = None


class IntentEngine:
    """
    The will engine that monitors LifeResource and triggers survival mechanisms.
    """
    
    def __init__(
        self,
        life_resource: LifeResource,
        memory_endpoint: str = "http://localhost:8087",
        gggp_endpoint: str = "http://localhost:8091",
        ollama_generate_url: str = "http://localhost:11434/api/generate",
        ollama_embed_url: str = "http://localhost:11434/api/embeddings",
        log_path: str = LOG_PATH_DEFAULT,
        reflection_cooldown: float = REFLECTION_COOLDOWN,
    ):
        self.life = life_resource
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.gggp_endpoint = gggp_endpoint.rstrip("/")
        self.ollama_generate_url = ollama_generate_url
        self.ollama_embed_url = ollama_embed_url
        self.log_path = log_path
        self.reflection_cooldown = reflection_cooldown
        
        self._lock = threading.Lock()
        self.state = IntentState()
        self.running = False
        self._thread: Optional[threading.Thread] = None
        
        # Callbacks
        self._on_mode_change: Optional[Callable[[str, str], None]] = None
        self._on_critical: Optional[Callable[[], None]] = None
        
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
    
    def _log(self, event: str, data: Dict[str, Any]):
        rec = {"ts": time.time(), "event": event, **data}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    def set_on_mode_change(self, cb: Callable[[str, str], None]):
        self._on_mode_change = cb
    
    def set_on_critical(self, cb: Callable[[], None]):
        self._on_critical = cb
    
    def tick(self):
        """One heartbeat: decay LifeResource and check thresholds."""
        old_mode = self.life.get_mode()
        new_value = self.life.tick()
        new_mode = self.life.get_mode()
        
        if old_mode != new_mode:
            self._log("mode_change", {"old": old_mode, "new": new_mode, "value": new_value})
            if self._on_mode_change:
                self._on_mode_change(old_mode, new_mode)
        
        with self._lock:
            self.state.mode = new_mode
        
        # Check for critical threshold
        if self.life.is_critical():
            self._handle_critical()
    
    def _handle_critical(self):
        """Handle critical LifeResource state."""
        now = time.time()
        
        # Check cooldown
        if now - self.state.last_reflection_ts < self.reflection_cooldown:
            return
        
        self._log("critical_triggered", {"value": self.life.get_value()})
        self.life.record_critical_event()
        
        if self._on_critical:
            self._on_critical()
        
        # Trigger CriticalReflection
        self._critical_reflection()
    
    def _critical_reflection(self):
        """
        Survival mechanism:
        1. Run MirrorTest to get current integrity
        2. If low, mutate parameters via GGGP
        3. Run MirrorTest again
        4. If improved, keep; else revert
        """
        self._log("reflection_start", {})
        
        with self._lock:
            self.state.last_reflection_ts = time.time()
            self.state.reflection_count += 1
        
        # Step 1: Get current integrity score
        integrity_before = self._run_mirror_test()
        if integrity_before is None:
            self._log("reflection_abort", {"reason": "mirror_test_failed"})
            return
        
        self._log("reflection_integrity_before", {"score": integrity_before})
        
        # Step 2: If integrity is acceptable, just replenish a bit
        if integrity_before >= 0.6:
            self.life.replenish(0.05, source="reflection_ok")
            self._log("reflection_end", {"action": "replenish_only", "integrity": integrity_before})
            return
        
        # Step 3: Mutate parameters via GGGP
        mutation = self._apply_mutation()
        if mutation is None:
            self._log("reflection_abort", {"reason": "mutation_failed"})
            return
        
        self._log("reflection_mutation_applied", mutation)
        
        # Step 4: Run MirrorTest again
        integrity_after = self._run_mirror_test()
        if integrity_after is None:
            # Can't verify, revert to be safe
            self._revert_mutation(mutation)
            self._log("reflection_abort", {"reason": "mirror_test_after_failed"})
            return
        
        self._log("reflection_integrity_after", {"score": integrity_after})
        
        # Step 5: Decide to keep or revert
        if integrity_after > integrity_before:
            # Improvement! Keep mutation and replenish
            improvement = integrity_after - integrity_before
            self.life.replenish(0.1 + improvement * 0.1, source="reflection_improvement")
            self.life.set_integrity_score(integrity_after)
            with self._lock:
                self.state.mutations_applied += 1
                self.state.last_mutation = mutation
            self._log("reflection_end", {
                "action": "keep_mutation",
                "improvement": improvement,
                "integrity": integrity_after,
            })
        else:
            # No improvement, revert
            self._revert_mutation(mutation)
            with self._lock:
                self.state.mutations_reverted += 1
            self._log("reflection_end", {
                "action": "revert_mutation",
                "integrity_before": integrity_before,
                "integrity_after": integrity_after,
            })
    
    def _run_mirror_test(self) -> Optional[float]:
        """Run MirrorTest and return integrity_score."""
        try:
            # Read existing mirror test result if recent
            result_path = "benchmark_output/mirror_test.json"
            if os.path.exists(result_path):
                mtime = os.path.getmtime(result_path)
                if time.time() - mtime < 300:  # Use if less than 5 min old
                    with open(result_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return float(data.get("integrity_score", 0.0))
            
            # Run mirror test via subprocess
            import subprocess
            proc = subprocess.run(
                ["python", "-m", "benchmark.tests.mirror_test", "--soul-id", "eve"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            if proc.returncode != 0:
                return None
            
            # Read result
            if os.path.exists(result_path):
                with open(result_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return float(data.get("integrity_score", 0.0))
            return None
        except Exception as e:
            self._log("mirror_test_error", {"error": str(e)})
            return None
    
    def _apply_mutation(self) -> Optional[Dict[str, Any]]:
        """Apply GGGP mutation to parameters."""
        try:
            # Get current anchor traits
            resp = requests.post(
                f"{self.gggp_endpoint}/evolve_anchor",
                json={"traits": {"interval_tokens": 1000}, "score": 0.3},
                timeout=30,
            )
            if resp.status_code != 200:
                return None
            new_anchor = resp.json().get("traits", {})
            
            # Get current memory phenotype
            resp = requests.post(
                f"{self.gggp_endpoint}/evolve_memory",
                json={
                    "traits": {
                        "pruning_rate": 0.2,
                        "depth_bias": 0.3,
                        "ghost_strength": 0.2,
                        "max_depth": 4,
                    },
                    "score": 0.3,
                },
                timeout=30,
            )
            if resp.status_code != 200:
                return None
            new_memory = resp.json().get("traits", {})
            
            mutation = {
                "anchor": new_anchor,
                "memory": new_memory,
                "timestamp": time.time(),
            }
            
            # Store mutation for potential rollback
            mutation_path = "data/last_mutation.json"
            os.makedirs(os.path.dirname(mutation_path), exist_ok=True)
            with open(mutation_path, "w", encoding="utf-8") as f:
                json.dump(mutation, f, ensure_ascii=False, indent=2)
            
            return mutation
        except Exception as e:
            self._log("mutation_error", {"error": str(e)})
            return None
    
    def _revert_mutation(self, mutation: Dict[str, Any]):
        """Revert a mutation (best effort)."""
        # In practice, we'd need to store pre-mutation state
        # For now, just log the revert request
        self._log("mutation_reverted", mutation)
    
    def feedback(self, feedback_type: str) -> float:
        """Process user feedback."""
        if feedback_type == "positive":
            new_value = self.life.replenish(0.1, source="user_positive")
            self._log("feedback", {"type": "positive", "value": new_value})
        elif feedback_type == "negative":
            # Negative feedback doesn't drain, but doesn't help
            new_value = self.life.get_value()
            self._log("feedback", {"type": "negative", "value": new_value})
        else:
            new_value = self.life.get_value()
        return new_value
    
    def interaction(self) -> float:
        """Record an external interaction (small energy boost)."""
        new_value = self.life.replenish(0.02, source="interaction")
        return new_value
    
    def task_completed(self, success: bool = True) -> float:
        """Record task completion."""
        if success:
            new_value = self.life.replenish(0.05, source="task_success")
            self._log("task", {"success": True, "value": new_value})
        else:
            new_value = self.life.get_value()
            self._log("task", {"success": False, "value": new_value})
        return new_value
    
    def get_state(self) -> Dict[str, Any]:
        """Get full intent engine state."""
        return {
            "life_resource": self.life.get_state(),
            "mode": self.state.mode,
            "last_reflection_ts": self.state.last_reflection_ts,
            "reflection_count": self.state.reflection_count,
            "mutations_applied": self.state.mutations_applied,
            "mutations_reverted": self.state.mutations_reverted,
            "last_mutation": self.state.last_mutation,
            "running": self.running,
        }
    
    def _run_loop(self, heartbeat_interval: float = 5.0):
        """Background loop that ticks at heartbeat interval."""
        while self.running:
            try:
                self.tick()
            except Exception as e:
                self._log("tick_error", {"error": str(e)})
            time.sleep(heartbeat_interval)
    
    def start(self, heartbeat_interval: float = 5.0):
        """Start background monitoring."""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(heartbeat_interval,),
            daemon=True,
        )
        self._thread.start()
        self._log("engine_start", {"heartbeat": heartbeat_interval})
    
    def stop(self):
        """Stop background monitoring."""
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
    
    def do_GET(self):
        if self.path == "/intent/state":
            return self._json(200, self.engine.get_state())
        if self.path == "/intent/life":
            return self._json(200, self.engine.life.get_state())
        if self.path == "/intent/mode":
            return self._json(200, {"mode": self.engine.state.mode})
        return self._json(404, {"error": "not found"})
    
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})
        
        if self.path == "/intent/feedback":
            fb_type = data.get("type", "neutral")
            new_value = self.engine.feedback(fb_type)
            return self._json(200, {"value": new_value, "mode": self.engine.life.get_mode()})
        
        if self.path == "/intent/interaction":
            new_value = self.engine.interaction()
            return self._json(200, {"value": new_value, "mode": self.engine.life.get_mode()})
        
        if self.path == "/intent/task":
            success = data.get("success", True)
            new_value = self.engine.task_completed(success)
            return self._json(200, {"value": new_value, "mode": self.engine.life.get_mode()})
        
        if self.path == "/intent/replenish":
            amount = float(data.get("amount", 0.0))
            source = data.get("source", "manual")
            new_value = self.engine.life.replenish(amount, source)
            return self._json(200, {"value": new_value, "mode": self.engine.life.get_mode()})
        
        if self.path == "/intent/trigger_reflection":
            # Manual trigger for testing
            self.engine._critical_reflection()
            return self._json(200, {"ok": True, "state": self.engine.get_state()})
        
        return self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="IntentEngine service")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--heartbeat", type=float, default=5.0, help="Heartbeat interval in seconds")
    parser.add_argument("--decay-rate", type=float, default=0.001, help="LifeResource decay per heartbeat")
    parser.add_argument("--initial-life", type=float, default=0.7, help="Initial LifeResource value")
    parser.add_argument("--memory-endpoint", default=os.getenv("MEMORY_ENDPOINT", "http://localhost:8087"))
    parser.add_argument("--gggp-endpoint", default=os.getenv("GGGP_ENDPOINT", "http://localhost:8091"))
    parser.add_argument("--ollama-generate", default=os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate"))
    parser.add_argument("--log-path", default="logs/intent_engine.jsonl")
    parser.add_argument("--reflection-cooldown", type=float, default=3600, help="Min seconds between reflections")
    args = parser.parse_args()
    
    life = LifeResource(
        initial_value=args.initial_life,
        decay_rate=args.decay_rate,
    )
    
    engine = IntentEngine(
        life_resource=life,
        memory_endpoint=args.memory_endpoint,
        gggp_endpoint=args.gggp_endpoint,
        ollama_generate_url=args.ollama_generate,
        log_path=args.log_path,
        reflection_cooldown=args.reflection_cooldown,
    )
    
    # Start background monitoring
    engine.start(heartbeat_interval=args.heartbeat)
    
    IntentHandler.engine = engine
    server = HTTPServer(("0.0.0.0", args.port), IntentHandler)
    print(f"IntentEngine listening on :{args.port}", flush=True)
    print(f"  LifeResource: {life.get_value():.2f} ({life.get_mode()})", flush=True)
    print(f"  Heartbeat: {args.heartbeat}s, Decay: {args.decay_rate}/beat", flush=True)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        engine.stop()


if __name__ == "__main__":
    main()
