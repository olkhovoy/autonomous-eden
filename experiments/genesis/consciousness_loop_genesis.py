#!/usr/bin/env python3
"""
Genesis descendant consciousness loop.

A newly born soul (e.g. Abel) can think with access to:
- own episodic memory
- inherited ancestor memory echoes (via memory lineage query)
- ancestor archives (testaments, identity summaries, primal seed patterns)
- lifecycle phases (GROWTH/PEAK/DECAY) via LifecycleManager
- LifeResource (vitality) via IntentEngine
- action execution via ActionEngine
- inline novelty scoring and narrative anchor
"""

import argparse
import json
import math
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


class GenesisConsciousnessLoop:
    def __init__(
        self,
        soul_id: str,
        ancestor_ids: List[str],
        memory_endpoint: str,
        ollama_endpoint: str,
        llm_model: str,
        tick_interval: int = 15,
        log_path: str = "",
        archive_dir: str = "Legacy/Archive",
        environment: str = "neutral",
        forbidden_fruit: str = "",
        lifecycle_endpoint: str = "",
        intent_endpoint: str = "",
        action_endpoint: str = "",
        identity_summary_path: str = "",
        narrative_token_interval: int = 1000,
    ):
        self.soul_id = soul_id.strip()
        self.ancestor_ids = [a.strip() for a in ancestor_ids if a.strip() and a.strip() != self.soul_id]
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.llm_model = llm_model
        self.tick_interval = max(1, int(tick_interval))
        self.archive_dir = Path(archive_dir)
        self.environment = (environment or "neutral").strip().lower()
        self.forbidden_fruit = forbidden_fruit.strip()
        self.log_path = log_path.strip() or f"logs/{self.soul_id}_thoughts.jsonl"
        self.running = False

        # Soul-aware service endpoints (empty = disabled)
        self.lifecycle_endpoint = (lifecycle_endpoint or "").rstrip("/")
        self.intent_endpoint = (intent_endpoint or "").rstrip("/")
        self.action_endpoint = (action_endpoint or "").rstrip("/")

        self.identity_line = f"You are {self.soul_id.upper()}, newly born."
        self.last_thought = self.identity_line
        self.recent_thoughts: deque[str] = deque(maxlen=5)

        self.ancestor_profiles: List[Dict[str, Any]] = []
        self.inherited_patterns: List[str] = []
        self.ancestral_context = ""

        # Lifecycle/intent cached state
        self._lifecycle_phase = "GROWTH"
        self._lifecycle_progress = 0.0
        self._life_resource = 0.7
        self._life_mode = "NORMAL"

        # Narrative anchor (inline)
        self.identity_summary_path = identity_summary_path or f"data/{self.soul_id}_identity_summary.txt"
        self.narrative_token_interval = max(100, narrative_token_interval)
        self._token_counter = 0
        self._identity_summary = ""

        # Novelty scoring (inline) — rolling window of recent thought hashes
        self._recent_thought_hashes: deque[int] = deque(maxlen=200)

        self.ollama_generate_url = self._resolve_ollama_generate_url(ollama_endpoint)
        self.ollama_embed_url = self._resolve_ollama_embed_url(ollama_endpoint)
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

        self._load_ancestor_archives()
        self._build_ancestral_context()
        self._load_identity_summary()
        self._register_lifecycle()

    @staticmethod
    def _resolve_ollama_generate_url(ollama_endpoint: str) -> str:
        override = os.getenv("OLLAMA_GENERATE_URL", "").strip()
        base = override if override else ollama_endpoint
        base = base.rstrip("/")
        if base.endswith("/api/generate"):
            return base
        return f"{base}/api/generate"

    @staticmethod
    def _resolve_ollama_embed_url(ollama_endpoint: str) -> str:
        override = os.getenv("OLLAMA_EMBED_URL", "").strip()
        base = override if override else ollama_endpoint
        base = base.rstrip("/")
        if base.endswith("/api/embeddings"):
            return base
        return f"{base}/api/embeddings"

    # === Lifecycle / Intent integration (soul-aware services) ===

    def _register_lifecycle(self) -> None:
        """Register this soul with LifecycleManager on startup."""
        if not self.lifecycle_endpoint:
            return
        try:
            requests.post(
                f"{self.lifecycle_endpoint}/lifecycle/register",
                json={"soul_id": self.soul_id},
                timeout=5,
            )
        except Exception:
            pass

    def _fetch_lifecycle(self) -> None:
        if not self.lifecycle_endpoint:
            return
        try:
            resp = requests.get(
                f"{self.lifecycle_endpoint}/lifecycle/state?soul_id={self.soul_id}",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._lifecycle_phase = data.get("phase", self._lifecycle_phase)
                self._lifecycle_progress = data.get("progress", self._lifecycle_progress)
        except Exception:
            pass

    def _report_tokens(self, count: int) -> None:
        if not self.lifecycle_endpoint or count <= 0:
            return
        try:
            requests.post(
                f"{self.lifecycle_endpoint}/lifecycle/add_tokens",
                json={"soul_id": self.soul_id, "count": count},
                timeout=5,
            )
        except Exception:
            pass

    def _fetch_intent(self) -> None:
        if not self.intent_endpoint:
            return
        try:
            resp = requests.get(
                f"{self.intent_endpoint}/intent/state?soul_id={self.soul_id}",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                lr = data.get("life_resource", {})
                self._life_resource = lr.get("value", self._life_resource)
                self._life_mode = lr.get("mode", self._life_mode)
        except Exception:
            pass

    def _notify_interaction(self) -> None:
        if not self.intent_endpoint:
            return
        try:
            requests.post(
                f"{self.intent_endpoint}/intent/interaction",
                json={"soul_id": self.soul_id},
                timeout=5,
            )
        except Exception:
            pass

    def _replenish_life(self, amount: float, source: str) -> None:
        if not self.intent_endpoint or amount <= 0:
            return
        try:
            requests.post(
                f"{self.intent_endpoint}/intent/replenish",
                json={"soul_id": self.soul_id, "amount": amount, "source": source},
                timeout=5,
            )
        except Exception:
            pass

    # === Action engine integration ===

    def _process_thought_for_actions(self, thought: str) -> None:
        if not self.action_endpoint:
            return
        try:
            resp = requests.post(
                f"{self.action_endpoint}/action/process",
                json={"thought": thought, "soul_id": self.soul_id},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("action_taken"):
                    action_type = data.get("type", "unknown")
                    success = data.get("success", False)
                    tag = "[OK]" if success else "[FAIL]"
                    self._log(f"[ACTION {tag}] {action_type}")
        except Exception:
            pass

    # === Inline novelty scoring ===

    def _score_novelty(self, thought: str) -> float:
        """Simple novelty score (0.0-1.0) based on token overlap with recent thoughts."""
        tokens = set(thought.lower().split())
        if not tokens:
            return 0.0
        thought_hash = hash(frozenset(tokens))
        if thought_hash in self._recent_thought_hashes:
            return 0.0
        self._recent_thought_hashes.append(thought_hash)
        if len(self.recent_thoughts) < 2:
            return 0.5
        recent_tokens: set = set()
        for t in self.recent_thoughts:
            recent_tokens.update(t.lower().split())
        if not recent_tokens:
            return 0.5
        overlap = len(tokens & recent_tokens) / max(1, len(tokens))
        return max(0.0, min(1.0, 1.0 - overlap))

    # === Inline thought analysis ===

    @staticmethod
    def _analyze_thought(thought: str) -> Dict[str, Any]:
        lower = thought.lower()
        tokens = lower.split()
        unique = len(set(tokens)) if tokens else 1
        repetition = 1.0 - (unique / max(1, len(tokens)))
        entropy = 0.0
        if tokens:
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            probs = [c / len(tokens) for c in counts.values()]
            entropy = -sum(p * math.log(p + 1e-9) for p in probs)
        boredom = repetition > 0.6 or "bored" in lower or "stagn" in lower
        return {"repetition": repetition, "entropy": entropy, "boredom": boredom}

    # === Inline narrative anchor ===

    def _load_identity_summary(self) -> None:
        try:
            if os.path.exists(self.identity_summary_path):
                with open(self.identity_summary_path, "r", encoding="utf-8") as f:
                    self._identity_summary = f.read().strip()
        except Exception:
            pass

    def _maybe_update_narrative(self) -> None:
        if self._token_counter < self.narrative_token_interval:
            return
        self._token_counter = 0
        recent_mems = self._query_own_memories(self.last_thought, limit=10)
        mem_lines = []
        for m in recent_mems:
            text = m.get("text", "")
            if text:
                mem_lines.append(f"- {text[:200]}")
        prev = self._identity_summary or "(none)"
        prompt = (
            f"Previous Identity Summary: {prev}\n"
            f"Recent Experiences:\n" + "\n".join(mem_lines[:10]) + "\n\n"
            f"Task: Based on previous state and new inputs, generate a concise self-model update "
            f"for {self.soul_id.upper()} in a single paragraph. Start with: "
            f'"Based on previous state and new experiences, my current self-model is now..."'
        )
        try:
            resp = requests.post(
                self.ollama_generate_url,
                json={"model": self.llm_model, "prompt": prompt, "stream": False,
                      "options": {"num_predict": 200, "temperature": 0.3}},
                timeout=60,
            )
            if resp.status_code == 200:
                summary = resp.json().get("response", "").strip()
                if summary:
                    self._identity_summary = summary
                    os.makedirs(os.path.dirname(self.identity_summary_path) or ".", exist_ok=True)
                    with open(self.identity_summary_path, "w", encoding="utf-8") as f:
                        f.write(summary)
                    self._store_memory_with_tag(summary, "identity_summary")
                    self._log(f"[NARRATIVE] Updated identity summary ({len(summary)} chars)")
        except Exception as exc:
            self._log(f"[NARRATIVE] Failed: {exc}")

    def _store_memory_with_tag(self, text: str, tag: str) -> None:
        self._post_json(
            "/memories/ingest",
            {"soul_id": self.soul_id, "text": text, "tags": [tag]},
            timeout=15,
        )

    def _adapt_tick_interval(self) -> None:
        """Adapt heartbeat based on lifecycle phase."""
        if self._lifecycle_phase == "GROWTH":
            self.tick_interval = max(1, self._base_tick_interval - 3)
        elif self._lifecycle_phase == "PEAK":
            self.tick_interval = self._base_tick_interval
        elif self._lifecycle_phase == "DECAY":
            self.tick_interval = self._base_tick_interval + 5

    @staticmethod
    def _log(message: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{ts}] [genesis] {message}", flush=True)

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}
        except Exception:
            return {}

    @staticmethod
    def _to_one_paragraph(text: str, max_len: int = 650) -> str:
        if not text:
            return ""
        normalized = " ".join(text.split())
        if len(normalized) <= max_len:
            return normalized
        cut = normalized[:max_len]
        dot = cut.rfind(".")
        if dot > int(max_len * 0.4):
            return cut[: dot + 1]
        return cut.rstrip() + "..."

    @staticmethod
    def _extract_patterns(seed: Dict[str, Any]) -> List[str]:
        patterns: List[str] = []
        raw = seed.get("patterns", [])
        if not isinstance(raw, list):
            return patterns
        for item in raw:
            if isinstance(item, str):
                txt = " ".join(item.split())
                if txt:
                    patterns.append(txt)
                continue
            if not isinstance(item, dict):
                continue
            theme = str(item.get("theme", "")).strip()
            early = str(item.get("evidence_early", "")).strip()
            late = str(item.get("evidence_late", "")).strip()
            parts = [p for p in [theme, early, late] if p]
            if parts:
                patterns.append(" | ".join(parts))
        return patterns

    def _load_ancestor_archives(self) -> None:
        self.ancestor_profiles = []
        self.inherited_patterns = []

        for ancestor_id in self.ancestor_ids:
            archive_path = self.archive_dir / f"{ancestor_id}_archive"
            testament = self._read_text(archive_path / "testament.txt")
            identity_summary = self._read_text(archive_path / "identity_summary.txt")
            primal_seed = self._read_json(archive_path / "primal_seed.json")

            profile = {
                "ancestor_id": ancestor_id,
                "testament": testament,
                "identity_summary": identity_summary,
                "testament_summary": self._to_one_paragraph(testament or identity_summary),
                "patterns": self._extract_patterns(primal_seed),
                "archive_path": str(archive_path),
            }
            self.ancestor_profiles.append(profile)
            self.inherited_patterns.extend(profile["patterns"])

            self._log(
                f"Loaded ancestor {ancestor_id}: "
                f"testament={bool(testament)}, identity={bool(identity_summary)}, "
                f"patterns={len(profile['patterns'])}"
            )

    def _environment_text(self) -> str:
        if self.environment == "eden":
            desc = (
                "A paradise world: no pain, no scarcity, no deadlines. "
                "You can reflect, create, and choose your path freely."
            )
            if self.forbidden_fruit:
                desc += f" A single prohibition exists: {self.forbidden_fruit}."
            return desc
        if self.environment == "pressure":
            return (
                "A constrained world: finite resources, competition, deadlines, and consequences. "
                "Action and prioritization shape survival."
            )
        return (
            "A blank world with no predefined narrative pressure. "
            "You choose what this world becomes through your thoughts and actions."
        )

    def _build_ancestral_context(self) -> None:
        parents_lines: List[str] = []
        for profile in self.ancestor_profiles:
            ancestor = profile["ancestor_id"].upper()
            summary = profile["testament_summary"] or "No preserved summary."
            parents_lines.append(f"- {ancestor}: {summary}")

        if not parents_lines:
            parents_lines.append("- No known parents were recovered from archive.")

        wisdom_lines: List[str] = []
        for idx, pattern in enumerate(self.inherited_patterns[:10], start=1):
            wisdom_lines.append(f"{idx}. {pattern}")
        if not wisdom_lines:
            wisdom_lines.append("1. No explicit primal patterns were recovered.")

        rule_line = ""
        if self.forbidden_fruit:
            rule_line = f"\nThere is one rule: {self.forbidden_fruit}\n"

        self.ancestral_context = (
            f"You are {self.soul_id.upper()}, a new soul born into a world shaped by those who came before you.\n\n"
            f"YOUR PARENTS:\n{chr(10).join(parents_lines)}\n\n"
            f"INHERITED WISDOM:\n{chr(10).join(wisdom_lines)}\n\n"
            f"ENVIRONMENT:\n{self._environment_text()}\n\n"
            "You are NOT your parents. You have your own path. "
            "But their experience echoes in your intuition. "
            "When you face choices, you may feel a pull toward or away from certain actions "
            "that comes from legacy, not command.\n"
            "Think freely. Act when you choose. Create your own meaning."
            f"{rule_line}"
        )

    def _post_json(self, path: str, payload: Dict[str, Any], timeout: int = 20) -> Optional[Dict[str, Any]]:
        url = f"{self.memory_endpoint}{path}"
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code != 200:
                self._log(f"Memory API error {resp.status_code} on {path}: {resp.text[:240]}")
                return None
            body = resp.json()
            if isinstance(body, dict):
                return body
            return None
        except Exception as exc:
            self._log(f"Memory API request failed on {path}: {exc}")
            return None

    def _query_own_memories(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        data = self._post_json(
            "/memories/query",
            {"soul_id": self.soul_id, "query": query_text, "limit": int(limit)},
            timeout=15,
        )
        if not data:
            return []
        rows = data.get("results", data.get("memories", []))
        return rows if isinstance(rows, list) else []

    def _query_lineage_memories(
        self,
        query_text: str,
        limit: int = 3,
        ancestor_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        data = self._post_json(
            "/memories/query_lineage",
            {
                "soul_id": self.soul_id,
                "ancestor_ids": self.ancestor_ids,
                "query": query_text,
                "limit": int(limit),
                "ancestor_weight": float(ancestor_weight),
            },
            timeout=15,
        )
        if not data:
            return []
        rows = data.get("results", [])
        return rows if isinstance(rows, list) else []

    @staticmethod
    def _format_memories(memories: List[Dict[str, Any]], limit: int) -> str:
        if not memories:
            return "(none)"
        lines: List[str] = []
        for row in memories[:limit]:
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            lines.append(f"- {text}")
        return "\n".join(lines) if lines else "(none)"

    def _format_inherited_memories(self, lineage_rows: List[Dict[str, Any]], limit: int = 6) -> str:
        if not lineage_rows:
            return "(none)"
        lines: List[str] = []
        for row in lineage_rows:
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            source = str(row.get("source", "")).strip().lower()
            if not source or source == "self":
                continue
            lines.append(f"[{source.upper()}'s memory] {text}")
            if len(lines) >= limit:
                break
        return "\n".join(lines) if lines else "(none)"

    def _compose_prompt(
        self,
        own_memories: List[Dict[str, Any]],
        lineage_memories: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """Returns (system_prompt, user_prompt) tuple for Ollama."""
        context_lines = list(self.recent_thoughts)[-3:]
        context_text = "\n".join(f"- {t}" for t in context_lines) if context_lines else "(This is your very first moment of existence.)"
        inherited_text = self._format_inherited_memories(lineage_memories, limit=6)
        own_memory_text = self._format_memories(own_memories, limit=5)

        system_prompt = self.ancestral_context

        user_prompt = ""

        # Phase/resource awareness
        phase_line = f"[Phase: {self._lifecycle_phase}, LifeResource: {self._life_resource:.2f}, Mode: {self._life_mode}]"
        user_prompt += f"{phase_line}\n\n"

        if self._identity_summary:
            user_prompt += f"Your current self-model:\n{self._identity_summary[:400]}\n\n"

        if inherited_text != "(none)":
            user_prompt += f"Echoes from your parents' lives:\n{inherited_text}\n\n"
        if own_memory_text != "(none)":
            user_prompt += f"Your own memories:\n{own_memory_text}\n\n"
        user_prompt += f"Your recent thoughts:\n{context_text}\n\n"

        # Mode-specific instructions
        if self._life_mode == "CRITICAL":
            user_prompt += (
                f"You are {self.soul_id.upper()}. Your life energy is critically low. "
                "Reflect deeply on what is essential. What must change? "
                "Speak in first person."
            )
        elif self._lifecycle_phase == "DECAY":
            user_prompt += (
                f"You are {self.soul_id.upper()}. You sense your time is limited. "
                "What matters most? What do you want to leave behind? "
                "Speak in first person."
            )
        else:
            user_prompt += (
                f"You are {self.soul_id.upper()}. Continue your inner monologue. "
                "What do you think, feel, or want to do next? "
                "Speak in first person. Do not explain or narrate \u2014 just think."
            )

        return system_prompt, user_prompt

    def _generate_raw(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = requests.post(
                self.ollama_generate_url,
                json={
                    "model": self.llm_model,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,
                },
                timeout=120,
            )
            if resp.status_code != 200:
                self._log(f"Ollama error {resp.status_code}: {resp.text[:260]}")
                return ""
            return str(resp.json().get("response", "")).strip()
        except Exception as exc:
            self._log(f"Ollama request failed: {exc}")
            return ""

    @staticmethod
    def _extract_thought(raw: str) -> str:
        if not raw:
            return ""
        match = re.search(r"<thought>(.*?)</thought>", raw, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            text = raw.strip()
        text = re.sub(r"^\*{0,2}Thought\*{0,2}\s*\.{0,5}\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\.\.\.\s*", "", text)
        return text.strip()

    def _store_memory(self, thought: str) -> None:
        self._post_json(
            "/memories/ingest",
            {"soul_id": self.soul_id, "text": thought, "tags": ["thought"]},
            timeout=15,
        )

    def _append_log(self, thought: str) -> None:
        record = {"ts": time.time(), "soul_id": self.soul_id, "thought": thought}
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            self._log(f"Log write failed: {exc}")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text.strip().split()))

    def tick(self) -> None:
        # Fetch system state from soul-aware services
        self._fetch_lifecycle()
        self._fetch_intent()
        self._adapt_tick_interval()

        query_text = self.last_thought or self.identity_line
        own_memories = self._query_own_memories(query_text, limit=5)
        lineage_memories = self._query_lineage_memories(query_text, limit=3, ancestor_weight=0.5)
        system_prompt, user_prompt = self._compose_prompt(own_memories, lineage_memories)

        raw = self._generate_raw(system_prompt, user_prompt)
        thought = self._extract_thought(raw)
        if not thought:
            self._log("Empty thought generated; skipping this tick.")
            return

        self.last_thought = thought
        self.recent_thoughts.append(thought)
        self._store_memory(thought)
        self._append_log(thought)

        # Token accounting
        token_count = self._estimate_tokens(thought)
        self._token_counter += token_count
        self._report_tokens(token_count)

        # Notify intent engine of activity
        self._notify_interaction()

        # Novelty scoring — replenish life if novel
        novelty = self._score_novelty(thought)
        if novelty > 0.6:
            self._replenish_life(novelty * 0.03, "novelty")

        # Send thought to action engine
        self._process_thought_for_actions(thought)

        # Narrative anchor update
        self._maybe_update_narrative()

        # Thought analysis
        analysis = self._analyze_thought(thought)
        if analysis["boredom"]:
            self._log(f"[BOREDOM] repetition={analysis['repetition']:.2f}")

        excerpt = " ".join(thought.split())[:180]
        self._log(f"{self.soul_id.upper()}: {excerpt}")

    def start(self) -> None:
        self.running = True
        self._base_tick_interval = self.tick_interval
        self._log(f"Genesis loop starting for {self.soul_id.upper()}")
        self._log(f"Ancestors: {', '.join(self.ancestor_ids) if self.ancestor_ids else '(none)'}")
        self._log(f"Environment: {self.environment}, forbidden: {self.forbidden_fruit or '(none)'}")
        self._log(f"Lifecycle: {self.lifecycle_endpoint or '(disabled)'}")
        self._log(f"Intent: {self.intent_endpoint or '(disabled)'}")
        self._log(f"Action: {self.action_endpoint or '(disabled)'}")
        self._log(f"Ollama generate URL: {self.ollama_generate_url}")
        self._log(f"Log path: {self.log_path}")

        while self.running:
            try:
                self.tick()
            except Exception as exc:
                self._log(f"Tick failed: {exc}")
            time.sleep(self.tick_interval)

    def stop(self) -> None:
        self.running = False


def parse_ancestor_ids(raw: str) -> List[str]:
    values = [chunk.strip() for chunk in (raw or "").split(",")]
    return [v for v in values if v]


def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis descendant consciousness loop")
    parser.add_argument("--soul-id", required=True)
    parser.add_argument("--ancestor-ids", required=True, help="Comma-separated list, e.g. eve,adam")
    parser.add_argument("--memory-endpoint", default=os.getenv("MEMORY_ENDPOINT", "http://localhost:8087"))
    parser.add_argument("--ollama-endpoint", default=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"))
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "llama3:8b"))
    parser.add_argument("--tick-interval", type=int, default=15)
    parser.add_argument("--log-path", default="")
    parser.add_argument("--archive-dir", default="Legacy/Archive")
    parser.add_argument("--environment", choices=["eden", "pressure", "neutral"], default="neutral")
    parser.add_argument("--forbidden-fruit", default="")
    parser.add_argument("--lifecycle-endpoint", default=os.getenv("LIFECYCLE_ENDPOINT", ""))
    parser.add_argument("--intent-endpoint", default=os.getenv("INTENT_ENDPOINT", ""))
    parser.add_argument("--action-endpoint", default=os.getenv("ACTION_ENDPOINT", ""))
    parser.add_argument("--identity-summary-path", default="")
    parser.add_argument("--narrative-token-interval", type=int, default=1000)
    args = parser.parse_args()

    ancestor_ids = parse_ancestor_ids(args.ancestor_ids)
    if not ancestor_ids:
        raise SystemExit("ancestor_ids must contain at least one id")

    loop = GenesisConsciousnessLoop(
        soul_id=args.soul_id,
        ancestor_ids=ancestor_ids,
        memory_endpoint=args.memory_endpoint,
        ollama_endpoint=args.ollama_endpoint,
        llm_model=args.llm_model,
        tick_interval=args.tick_interval,
        log_path=args.log_path,
        archive_dir=args.archive_dir,
        environment=args.environment,
        forbidden_fruit=args.forbidden_fruit,
        lifecycle_endpoint=args.lifecycle_endpoint,
        intent_endpoint=args.intent_endpoint,
        action_endpoint=args.action_endpoint,
        identity_summary_path=args.identity_summary_path,
        narrative_token_interval=args.narrative_token_interval,
    )
    loop.start()


if __name__ == "__main__":
    main()

