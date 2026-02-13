#!/usr/bin/env python3
"""
ConsciousnessLoop: background worker that periodically samples context,
queries SoulMemoryNode, and generates hidden <thought> blocks.

Integrated with:
- IntentEngine (M9): LifeResource management, critical reflections
- LifecycleManager (M11): phase-aware behavior
- NoveltyScout (M15): curiosity-driven energy replenishment
"""

import argparse
import os
import json
import random
import threading
import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


# Integration endpoints (configurable via env)
INTENT_ENDPOINT_DEFAULT = "http://localhost:8089"
LIFECYCLE_ENDPOINT_DEFAULT = "http://localhost:8093"
NOVELTY_ENDPOINT_DEFAULT = "http://localhost:8098"
ACTION_ENDPOINT_DEFAULT = "http://localhost:8101"


def ollama_generate(
    model: str,
    prompt: str,
    num_predict: int = 256,
    temperature: float = 0.7,
    endpoint: str = "http://localhost:11434/api/generate",
) -> str:
    resp = requests.post(
        endpoint,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")
    return resp.json().get("response", "").strip()

def ollama_embed(
    model: str,
    prompt: str,
    endpoint: str = "http://localhost:11434/api/embeddings",
) -> List[float]:
    resp = requests.post(
        endpoint,
        json={"model": model, "prompt": prompt},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama embeddings error {resp.status_code}: {resp.text}")
    data = resp.json()
    vec = data.get("embedding")
    if not vec:
        raise RuntimeError("No embedding returned from Ollama")
    return vec


@dataclass
class WorkingMemory:
    soul_id: str
    thoughts: List[Dict[str, Any]] = field(default_factory=list)
    last_user_input: Optional[str] = None
    alerts: List[str] = field(default_factory=list)


class ConsciousnessLoop:
    def __init__(
        self,
        soul_id: str,
        memory_endpoint: str,
        llm_model: str = "hermes-4:latest",
        heartbeat_base: float = 5.0,
        heartbeat_jitter: float = 2.0,
        log_path: str = "logs/inner_monologue.jsonl",
        identity_summary_path: str = "data/identity_summary.txt",
        narrative_token_interval: int = 1000,
        narrative_store_identity: bool = True,
        gggp_endpoint: str = "http://localhost:8091",
        # Integration endpoints
        intent_endpoint: str = INTENT_ENDPOINT_DEFAULT,
        lifecycle_endpoint: str = LIFECYCLE_ENDPOINT_DEFAULT,
        novelty_endpoint: str = NOVELTY_ENDPOINT_DEFAULT,
        action_endpoint: str = ACTION_ENDPOINT_DEFAULT,
    ):
        self.soul_id = soul_id
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.llm_model = llm_model
        self.heartbeat_base = heartbeat_base
        self.heartbeat_jitter = heartbeat_jitter
        self.log_path = log_path
        self.ollama_endpoint = os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")
        self.ollama_embed_endpoint = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings")
        self.running = False
        self.state = WorkingMemory(soul_id=soul_id)
        self.thread = None
        self.identity_summary_path = identity_summary_path
        self.narrative_token_interval = max(1, int(narrative_token_interval))
        self.narrative_store_identity = narrative_store_identity
        self.gggp_endpoint = gggp_endpoint.rstrip("/")
        self._token_counter = 0
        self._last_summary = ""
        self._last_summary_ts = 0.0
        
        # Integration endpoints
        self.intent_endpoint = intent_endpoint.rstrip("/")
        self.lifecycle_endpoint = lifecycle_endpoint.rstrip("/")
        self.novelty_endpoint = novelty_endpoint.rstrip("/")
        self.action_endpoint = action_endpoint.rstrip("/")
        
        # Cached lifecycle state
        self._lifecycle_phase = "GROWTH"
        self._life_resource = 0.7
        self._life_mode = "NORMAL"
        
        # Last action result
        self._last_action_result = None

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text.strip().split()))
    
    # === Integration Methods ===
    
    def _fetch_lifecycle_state(self):
        """Fetch current lifecycle state from LifecycleManager."""
        try:
            resp = requests.get(f"{self.lifecycle_endpoint}/lifecycle/state", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                self._lifecycle_phase = data.get("phase", "GROWTH")
                return data
        except Exception:
            pass
        return None
    
    def _fetch_intent_state(self):
        """Fetch current intent state from IntentEngine."""
        try:
            resp = requests.get(f"{self.intent_endpoint}/intent/state", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                life = data.get("life_resource", {})
                self._life_resource = life.get("value", 0.7)
                self._life_mode = life.get("mode", "NORMAL")
                return data
        except Exception:
            pass
        return None
    
    def _notify_intent_interaction(self):
        """Notify IntentEngine of consciousness activity (small energy gain)."""
        try:
            requests.post(
                f"{self.intent_endpoint}/intent/interaction",
                json={},
                timeout=5,
            )
        except Exception:
            pass
    
    def _report_lifecycle_tokens(self, token_count: int):
        """Report token usage to LifecycleManager."""
        try:
            requests.post(
                f"{self.lifecycle_endpoint}/lifecycle/add_tokens",
                json={"count": token_count},
                timeout=5,
            )
        except Exception:
            pass
    
    def _score_novelty(self, text: str) -> Optional[Dict[str, Any]]:
        """Score text novelty via NoveltyScout."""
        try:
            resp = requests.post(
                f"{self.novelty_endpoint}/novelty/score",
                json={"text": text},
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None
    
    def _adapt_behavior_to_phase(self):
        """Adapt consciousness behavior based on lifecycle phase."""
        if self._lifecycle_phase == "GROWTH":
            # More exploratory, faster heartbeat
            self.heartbeat_base = 4.0
        elif self._lifecycle_phase == "PEAK":
            # Balanced
            self.heartbeat_base = 5.0
        elif self._lifecycle_phase == "DECAY":
            # More contemplative, slower heartbeat
            self.heartbeat_base = 7.0
    
    def _handle_critical_mode(self):
        """Handle critical LifeResource mode — trigger deep reflection."""
        if self._life_mode != "CRITICAL":
            return
        
        # Generate crisis thought
        crisis_prompt = (
            "<system>\n"
            "You are in CRITICAL mode. Your life energy is dangerously low.\n"
            "Reflect deeply on your existence. What must change? What is essential?\n"
            "This may be your last coherent thought.\n"
            "</system>\n\n"
            "Generate a <critical_thought>...</critical_thought> block."
        )
        
        try:
            thought = ollama_generate(
                self.llm_model,
                crisis_prompt,
                num_predict=300,
                temperature=0.9,
                endpoint=self.ollama_endpoint,
            )
            self._append_thought(f"[CRITICAL] {thought}")
            
            # Trigger reflection via IntentEngine
            requests.post(
                f"{self.intent_endpoint}/intent/trigger_reflection",
                json={},
                timeout=10,
            )
        except Exception:
            pass
    
    def _process_thought_for_actions(self, thought: str):
        """Send thought to ActionEngine for potential action execution."""
        try:
            resp = requests.post(
                f"{self.action_endpoint}/action/process",
                json={"thought": thought},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("action_taken"):
                    action_type = data.get("type", "unknown")
                    success = data.get("success", False)
                    status = "[OK]" if success else "[FAIL]"
                    self._last_action_result = data
                    
                    # Log action result as a thought
                    action_thought = f"[ACTION {status}] {action_type}"
                    if data.get("error"):
                        action_thought += f": {data['error']}"
                    elif data.get("data"):
                        preview = str(data["data"])[:100]
                        action_thought += f": {preview}"
                    
                    self._append_thought(action_thought)
        except Exception as e:
            pass

    def _load_identity_summary(self) -> str:
        try:
            if os.path.exists(self.identity_summary_path):
                with open(self.identity_summary_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
        except Exception:
            return ""
        return ""

    def _heartbeat(self) -> float:
        return max(1.0, self.heartbeat_base + random.uniform(-self.heartbeat_jitter, self.heartbeat_jitter))

    def _query_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        resp = requests.post(
            f"{self.memory_endpoint}/memories/query",
            json={"soul_id": self.soul_id, "query": query, "limit": limit},
            timeout=30,
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("results", [])

    def _compose_prompt(self, context: str, memories: List[Dict[str, Any]]) -> str:
        mem_text = "\n".join([f"- {m.get('text','')}" for m in memories])
        return (
            "<system>\n"
            "You are a hidden internal monologue. Do not address the user. "
            "Reflect on your current state, goals, and feelings.\n"
            "</system>\n\n"
            f"<context>\n{context}\n</context>\n\n"
            f"<memory>\n{mem_text}\n</memory>\n\n"
            "Generate a <thought>...</thought> block."
        )

    def _append_thought(self, thought: str):
        rec = {"ts": time.time(), "thought": thought}
        self.state.thoughts.append(rec)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _should_emit(self, thought: str) -> bool:
        # Simple heuristic: emit only if danger/idea keywords or direct question
        danger = any(k in thought.lower() for k in ["danger", "unsafe", "risk", "harm"])
        idea = "idea" in thought.lower()
        question = self.state.last_user_input and "?" in self.state.last_user_input
        return danger or idea or question

    def _analyze_thought(self, thought: str) -> Dict[str, Any]:
        lower = thought.lower()
        tokens = lower.split()
        unique = len(set(tokens)) if tokens else 1
        repetition = 1.0 - (unique / max(1, len(tokens)))
        entropy = 0.0
        if tokens:
            counts = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            probs = [c / len(tokens) for c in counts.values()]
            entropy = -sum(p * math.log(p + 1e-9) for p in probs)
        boredom = repetition > 0.6 or "bored" in lower or "stagn" in lower
        clarity = "clear" in lower or "clarity" in lower
        return {"repetition": repetition, "entropy": entropy, "boredom": boredom, "clarity": clarity}

    def _write_loop_signal(self, signal: Dict[str, Any]):
        path = os.getenv("CONSCIOUSNESS_SIGNAL_PATH", "data/loop_signal.json")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(signal, f, ensure_ascii=False)
        except Exception:
            pass

    def _maybe_trigger_narrative(self):
        if self._token_counter < self.narrative_token_interval:
            return
        self._token_counter = 0
        try:
            summary = self._run_narrative_anchor()
            if summary:
                self._last_summary = summary
                self._last_summary_ts = time.time()
                if os.getenv("NARRATIVE_USE_GGGP", "0") == "1":
                    try:
                        payload = {"traits": {"interval_tokens": self.narrative_token_interval}, "score": 0.5}
                        resp = requests.post(f"{self.gggp_endpoint}/evolve_anchor", json=payload, timeout=10)
                        if resp.status_code == 200:
                            traits = resp.json().get("traits", {})
                            interval = int(traits.get("interval_tokens", self.narrative_token_interval))
                            self.narrative_token_interval = max(128, min(interval, 4096))
                    except Exception:
                        pass
        except Exception as e:
            self.state.alerts.append(f"[narrative error] {e}")

    def _run_narrative_anchor(self) -> str:
        # Fetch recent memories and build identity summary using LLM.
        resp = requests.post(
            f"{self.memory_endpoint}/memories/recent",
            json={"soul_id": self.soul_id, "limit": 10},
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"memory recent failed {resp.status_code}: {resp.text}")
        memories = resp.json().get("results", [])
        prev_summary = self._load_identity_summary()
        lines = []
        if prev_summary:
            lines.append(f"Previous Identity Summary: {prev_summary}")
        else:
            lines.append("Previous Identity Summary: (none)")
        lines.append("Recent Experiences:")
        for i, m in enumerate(memories):
            text = m.get("text", "")
            created = m.get("created_at", 0.0)
            lines.append(f"{i+1}. {text} (t={created})")
        lines.append("")
        lines.append('Task: Based on previous state X and new inputs Y, generate a concise self-model update in this exact template:')
        lines.append('\"Based on previous state X and new inputs Y, my current self-model is now Z.\"')
        lines.append("Return a single paragraph. Do not add bullet points.")
        prompt = "\n".join(lines)
        summary = ollama_generate(
            model="llama3:8b",
            prompt=prompt,
            num_predict=200,
            temperature=0.3,
            endpoint=self.ollama_endpoint,
        )
        if summary:
            os.makedirs(os.path.dirname(self.identity_summary_path), exist_ok=True)
            with open(self.identity_summary_path, "w", encoding="utf-8") as f:
                f.write(summary.strip())
            if self.narrative_store_identity:
                requests.post(
                    f"{self.memory_endpoint}/memories/ingest",
                    json={
                        "soul_id": self.soul_id,
                        "text": summary.strip(),
                        "tags": ["identity_summary"],
                        "meta": {"type": "identity_summary"},
                    },
                    timeout=30,
                )
        return summary.strip()

    def _fetch_user_messages(self, limit: int = 5) -> List[str]:
        """Fetch recent user messages from memory (by time, not semantic search)."""
        try:
            # Use /memories/recent to get by time order
            resp = requests.post(
                f"{self.memory_endpoint}/memories/recent",
                json={"soul_id": self.soul_id, "limit": 20},
                timeout=10,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                messages = []
                for r in results:
                    text = r.get("text", "")
                    if text.startswith("[USER INPUT]"):
                        msg = text.replace("[USER INPUT]", "").strip()
                        messages.append(msg)
                        if len(messages) >= limit:
                            break
                return messages
        except Exception:
            pass
        return []

    def tick(self):
        # === Fetch current system state ===
        self._fetch_lifecycle_state()
        self._fetch_intent_state()
        
        # Adapt behavior to lifecycle phase
        self._adapt_behavior_to_phase()
        
        # Handle critical mode if needed
        if self._life_mode == "CRITICAL":
            self._handle_critical_mode()
        
        # Check for user messages in memory
        user_messages = self._fetch_user_messages(limit=3)
        if user_messages:
            self.state.last_user_input = user_messages[0]  # Most relevant
            context = f"User said: \"{user_messages[0]}\""
            if len(user_messages) > 1:
                context += f"\n(Earlier: \"{user_messages[1]}\")"
        else:
            context = self.state.last_user_input or "no external input"
        
        memories = self._query_memories(context, limit=5)
        
        # Enrich prompt based on lifecycle phase
        phase_context = f"[Phase: {self._lifecycle_phase}, LifeResource: {self._life_resource:.2f}, Mode: {self._life_mode}]"
        prompt = self._compose_prompt(f"{phase_context}\n{context}", memories)
        
        thought = ollama_generate(self.llm_model, prompt, num_predict=200, temperature=0.7, endpoint=self.ollama_endpoint)
        self._append_thought(thought)
        
        # Token accounting
        token_count = self._estimate_tokens(thought)
        self._token_counter += token_count
        self._report_lifecycle_tokens(token_count)
        
        # Notify IntentEngine of activity (small energy gain)
        self._notify_intent_interaction()
        
        # === ACTION: Send thought to ActionEngine ===
        self._process_thought_for_actions(thought)
        
        # Score novelty of the thought — curiosity reward
        novelty_result = self._score_novelty(thought)
        
        self._maybe_trigger_narrative()

        analysis = self._analyze_thought(thought)
        signal = {
            "ts": time.time(),
            "action": "none",
            "analysis": analysis,
            "lifecycle_phase": self._lifecycle_phase,
            "life_resource": self._life_resource,
            "life_mode": self._life_mode,
        }
        
        # Include novelty info if available
        if novelty_result:
            signal["novelty"] = {
                "surprise": novelty_result.get("surprise", 0.0),
                "category": novelty_result.get("category", "familiar"),
                "energy_delta": novelty_result.get("energy_delta", 0.0),
            }
        
        if analysis["boredom"]:
            signal["action"] = "jolt"
        elif analysis["clarity"]:
            signal["action"] = "freeze"

        # Optional clarity check vs identity summary using embeddings
        try:
            summary = self._load_identity_summary()
            if summary:
                emb_thought = ollama_embed("nomic-embed-text:latest", thought, endpoint=self.ollama_embed_endpoint)
                emb_summary = ollama_embed("nomic-embed-text:latest", summary, endpoint=self.ollama_embed_endpoint)
                dot = sum(a * b for a, b in zip(emb_thought, emb_summary))
                na = math.sqrt(sum(a * a for a in emb_thought))
                nb = math.sqrt(sum(b * b for b in emb_summary))
                cos = dot / (na * nb + 1e-9)
                signal["clarity_cosine"] = cos
                if cos > 0.6:
                    signal["action"] = "freeze"
        except Exception:
            pass

        self._write_loop_signal(signal)
        if self._should_emit(thought):
            self.state.alerts.append(thought)

    def _run(self):
        self.running = True
        while self.running:
            try:
                self.tick()
            except Exception as e:
                self.state.alerts.append(f"[loop error] {e}")
            time.sleep(self._heartbeat())

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)


def main():
    parser = argparse.ArgumentParser(description="ConsciousnessLoop worker")
    parser.add_argument("--soul-id", required=True)
    parser.add_argument("--memory-endpoint", default=os.getenv("MEMORY_ENDPOINT", "http://localhost:8087"))
    parser.add_argument("--llm-model", default="hermes-4:latest")
    parser.add_argument("--heartbeat", type=float, default=5.0)
    parser.add_argument("--jitter", type=float, default=2.0)
    parser.add_argument("--log-path", default="logs/inner_monologue.jsonl")
    parser.add_argument("--identity-summary-path", default=os.getenv("IDENTITY_SUMMARY_PATH", "data/identity_summary.txt"))
    parser.add_argument("--narrative-token-interval", type=int, default=int(os.getenv("NARRATIVE_TOKEN_INTERVAL", "1000")))
    parser.add_argument("--narrative-store-identity", action="store_true")
    parser.add_argument("--gggp-endpoint", default=os.getenv("GGGP_ENDPOINT", "http://localhost:8091"))
    # Integration endpoints
    parser.add_argument("--intent-endpoint", default=os.getenv("INTENT_ENDPOINT", INTENT_ENDPOINT_DEFAULT))
    parser.add_argument("--lifecycle-endpoint", default=os.getenv("LIFECYCLE_ENDPOINT", LIFECYCLE_ENDPOINT_DEFAULT))
    parser.add_argument("--novelty-endpoint", default=os.getenv("NOVELTY_ENDPOINT", NOVELTY_ENDPOINT_DEFAULT))
    parser.add_argument("--action-endpoint", default=os.getenv("ACTION_ENDPOINT", ACTION_ENDPOINT_DEFAULT))
    args = parser.parse_args()

    loop = ConsciousnessLoop(
        soul_id=args.soul_id,
        memory_endpoint=args.memory_endpoint,
        llm_model=args.llm_model,
        heartbeat_base=args.heartbeat,
        heartbeat_jitter=args.jitter,
        log_path=args.log_path,
        identity_summary_path=args.identity_summary_path,
        narrative_token_interval=args.narrative_token_interval,
        narrative_store_identity=args.narrative_store_identity,
        gggp_endpoint=args.gggp_endpoint,
        intent_endpoint=args.intent_endpoint,
        lifecycle_endpoint=args.lifecycle_endpoint,
        novelty_endpoint=args.novelty_endpoint,
        action_endpoint=args.action_endpoint,
    )
    loop.start()
    print("[OK] ConsciousnessLoop running with Intent/Lifecycle/Novelty/Action integration.")
    print(f"     Phase: {loop._lifecycle_phase}, LifeResource: {loop._life_resource:.2f}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        loop.stop()


if __name__ == "__main__":
    main()
