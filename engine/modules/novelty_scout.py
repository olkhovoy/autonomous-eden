#!/usr/bin/env python3
"""
NoveltyScout: Semantic hunger for the unknown.

"Satoshi Instinct" — EVE values the genesis block of an idea more than noise.

Calculates information surprise:
- Low surprise (predictable) → no energy gain
- High surprise (novel pattern) → LifeResource boost
- Genesis block (completely new) → maximum boost + GGGP evolution trigger
"""

import argparse
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any, List, Optional

import requests


LOG_PATH_DEFAULT = "logs/novelty_scout.jsonl"
HISTORY_SIZE = 1000  # Rolling window of recent embeddings


@dataclass
class NoveltyResult:
    surprise: float  # 0.0-1.0
    category: str    # "known", "interesting", "novel", "genesis"
    energy_gain: float
    memory_similarity: float
    ancestor_similarity: float


class NoveltyScout:
    """
    Calculates information surprise and drives curiosity.
    """
    
    # Surprise thresholds
    THRESHOLD_INTERESTING = 0.3
    THRESHOLD_NOVEL = 0.6
    THRESHOLD_GENESIS = 0.8
    
    # Energy gains
    ENERGY_KNOWN = 0.0
    ENERGY_INTERESTING = 0.01
    ENERGY_NOVEL = 0.03
    ENERGY_GENESIS = 0.05
    
    def __init__(
        self,
        memory_endpoint: str = "http://localhost:8087",
        ancestor_endpoint: str = "http://localhost:8097",
        intent_endpoint: str = "http://localhost:8089",
        gggp_endpoint: str = "http://localhost:8091",
        ollama_embed_url: str = "http://localhost:11434/api/embeddings",
        embed_model: str = "nomic-embed-text:latest",
        log_path: str = LOG_PATH_DEFAULT,
        lifecycle_phase: str = "PEAK",
    ):
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.ancestor_endpoint = ancestor_endpoint.rstrip("/")
        self.intent_endpoint = intent_endpoint.rstrip("/")
        self.gggp_endpoint = gggp_endpoint.rstrip("/")
        self.ollama_embed_url = ollama_embed_url
        self.embed_model = embed_model
        self.log_path = log_path
        self.lifecycle_phase = lifecycle_phase
        
        # Rolling window of recent embeddings for fast comparison
        self._recent_embeddings: deque = deque(maxlen=HISTORY_SIZE)
        
        # Genesis patterns (high novelty items)
        self._genesis_patterns: List[Dict[str, Any]] = []
        
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
    
    def _log(self, event: str, data: Dict[str, Any]):
        rec = {"ts": time.time(), "event": event, **data}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    def _embed(self, text: str) -> Optional[List[float]]:
        """Get embedding for text."""
        try:
            resp = requests.post(
                self.ollama_embed_url,
                json={"model": self.embed_model, "prompt": text},
                timeout=60,
            )
            if resp.status_code != 200:
                return None
            return resp.json().get("embedding")
        except Exception:
            return None
    
    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
    
    def _max_similarity_to_recent(self, embedding: List[float]) -> float:
        """Find max similarity to recent embeddings."""
        if not self._recent_embeddings:
            return 0.0
        return max(self._cosine(embedding, e) for e in self._recent_embeddings)
    
    def _query_memory_similarity(self, text: str) -> float:
        """Query memory node for similarity."""
        try:
            resp = requests.post(
                f"{self.memory_endpoint}/memories/query",
                json={"soul_id": "eve", "query": text, "limit": 5},
                timeout=30,
            )
            if resp.status_code != 200:
                return 0.0
            results = resp.json().get("results", [])
            if not results:
                return 0.0
            # Qdrant returns similarity scores
            # Assume strength/saliency correlates with relevance
            return max(r.get("strength", 0) for r in results)
        except Exception:
            return 0.0
    
    def _query_ancestor_similarity(self, text: str) -> float:
        """Query ancestor resonance."""
        try:
            resp = requests.post(
                f"{self.ancestor_endpoint}/ancestors/resonate",
                json={"text": text, "top_k": 1},
                timeout=30,
            )
            if resp.status_code != 200:
                return 0.0
            resonances = resp.json().get("resonances", [])
            if not resonances:
                return 0.0
            return resonances[0].get("score", 0.0)
        except Exception:
            return 0.0
    
    def _replenish_life(self, amount: float, source: str):
        """Send energy to IntentEngine."""
        try:
            requests.post(
                f"{self.intent_endpoint}/intent/replenish",
                json={"amount": amount, "source": source},
                timeout=10,
            )
        except Exception:
            pass
    
    def _trigger_gggp_evolution(self, pattern: str):
        """Trigger GGGP to evolve toward novel attractor."""
        try:
            # Signal high fitness for novel pattern exploration
            requests.post(
                f"{self.gggp_endpoint}/evolve",
                json={
                    "traits": {
                        "creativity": 0.8,
                        "attention_span": 0.6,
                        "curiosity": 0.9,
                        "paranoia": 0.3,
                    },
                    "score": 0.9,  # High score to encourage this direction
                },
                timeout=30,
            )
            self._log("gggp_evolution_triggered", {"pattern_preview": pattern[:100]})
        except Exception:
            pass
    
    def set_lifecycle_phase(self, phase: str):
        """Update lifecycle phase (affects novelty weight)."""
        self.lifecycle_phase = phase
    
    def _get_phase_multiplier(self) -> float:
        """Novelty weight based on lifecycle phase."""
        if self.lifecycle_phase == "GROWTH":
            return 1.0  # Aggressive exploration
        elif self.lifecycle_phase == "PEAK":
            return 0.7  # Balanced
        elif self.lifecycle_phase == "DECAY":
            return 0.3  # Conservative, consolidate
        return 0.7
    
    def calculate_surprise(self, text: str) -> NoveltyResult:
        """
        Calculate information surprise for incoming text.
        
        Returns NoveltyResult with surprise score, category, and energy gain.
        """
        if not text:
            return NoveltyResult(
                surprise=0.0,
                category="empty",
                energy_gain=0.0,
                memory_similarity=0.0,
                ancestor_similarity=0.0,
            )
        
        # Get embedding
        embedding = self._embed(text)
        if not embedding:
            return NoveltyResult(
                surprise=0.0,
                category="error",
                energy_gain=0.0,
                memory_similarity=0.0,
                ancestor_similarity=0.0,
            )
        
        # Calculate similarities
        recent_sim = self._max_similarity_to_recent(embedding)
        memory_sim = self._query_memory_similarity(text)
        ancestor_sim = self._query_ancestor_similarity(text)
        
        # Combined similarity (own memories matter more than ancestors)
        combined_sim = 0.5 * max(recent_sim, memory_sim) + 0.3 * memory_sim + 0.2 * ancestor_sim
        
        # Surprise = 1 - similarity
        surprise = 1.0 - combined_sim
        
        # Apply lifecycle phase multiplier
        phase_mult = self._get_phase_multiplier()
        effective_surprise = surprise * phase_mult
        
        # Categorize and determine energy gain
        if effective_surprise >= self.THRESHOLD_GENESIS:
            category = "genesis"
            energy_gain = self.ENERGY_GENESIS
            # Store as genesis pattern
            self._genesis_patterns.append({
                "text": text[:500],
                "embedding": embedding,
                "timestamp": time.time(),
                "surprise": surprise,
            })
            # Trigger GGGP evolution
            self._trigger_gggp_evolution(text)
            self._log("genesis_discovered", {"text_preview": text[:100], "surprise": surprise})
        elif effective_surprise >= self.THRESHOLD_NOVEL:
            category = "novel"
            energy_gain = self.ENERGY_NOVEL
        elif effective_surprise >= self.THRESHOLD_INTERESTING:
            category = "interesting"
            energy_gain = self.ENERGY_INTERESTING
        else:
            category = "known"
            energy_gain = self.ENERGY_KNOWN
        
        # Add to recent embeddings
        self._recent_embeddings.append(embedding)
        
        # Replenish life if energy gain
        if energy_gain > 0:
            self._replenish_life(energy_gain, f"novelty_{category}")
        
        result = NoveltyResult(
            surprise=effective_surprise,
            category=category,
            energy_gain=energy_gain,
            memory_similarity=memory_sim,
            ancestor_similarity=ancestor_sim,
        )
        
        if category != "known":
            self._log("novelty_detected", asdict(result))
        
        return result
    
    def get_genesis_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent genesis patterns (without embeddings)."""
        patterns = self._genesis_patterns[-limit:]
        return [
            {
                "text": p["text"],
                "timestamp": p["timestamp"],
                "surprise": p["surprise"],
            }
            for p in patterns
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get novelty scout statistics."""
        return {
            "recent_embeddings_count": len(self._recent_embeddings),
            "genesis_patterns_count": len(self._genesis_patterns),
            "lifecycle_phase": self.lifecycle_phase,
            "phase_multiplier": self._get_phase_multiplier(),
        }


# HTTP Server

class NoveltyHandler(BaseHTTPRequestHandler):
    scout: NoveltyScout = None
    
    def _json(self, code: int, payload: Dict[str, Any]):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    
    def do_GET(self):
        if self.path == "/novelty/stats":
            return self._json(200, self.scout.get_stats())
        
        if self.path == "/novelty/genesis":
            limit = 10
            if "?" in self.path:
                query = self.path.split("?")[1]
                for part in query.split("&"):
                    if part.startswith("limit="):
                        limit = int(part.split("=")[1])
            return self._json(200, {"patterns": self.scout.get_genesis_patterns(limit)})
        
        return self._json(404, {"error": "not found"})
    
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})
        
        if self.path == "/novelty/score":
            text = data.get("text", "")
            if not text:
                return self._json(400, {"error": "text required"})
            result = self.scout.calculate_surprise(text)
            return self._json(200, asdict(result))
        
        if self.path == "/novelty/phase":
            phase = data.get("phase", "PEAK")
            self.scout.set_lifecycle_phase(phase)
            return self._json(200, {"phase": phase})
        
        return self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="NoveltyScout service")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--memory-endpoint", default=os.getenv("MEMORY_ENDPOINT", "http://localhost:8087"))
    parser.add_argument("--ancestor-endpoint", default="http://localhost:8097")
    parser.add_argument("--intent-endpoint", default="http://localhost:8089")
    parser.add_argument("--gggp-endpoint", default=os.getenv("GGGP_ENDPOINT", "http://localhost:8091"))
    parser.add_argument("--ollama-embed", default=os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings"))
    parser.add_argument("--log-path", default=LOG_PATH_DEFAULT)
    parser.add_argument("--lifecycle-phase", default="PEAK")
    args = parser.parse_args()
    
    scout = NoveltyScout(
        memory_endpoint=args.memory_endpoint,
        ancestor_endpoint=args.ancestor_endpoint,
        intent_endpoint=args.intent_endpoint,
        gggp_endpoint=args.gggp_endpoint,
        ollama_embed_url=args.ollama_embed,
        log_path=args.log_path,
        lifecycle_phase=args.lifecycle_phase,
    )
    
    NoveltyHandler.scout = scout
    server = HTTPServer(("0.0.0.0", args.port), NoveltyHandler)
    print(f"NoveltyScout listening on :{args.port}", flush=True)
    print(f"  Lifecycle phase: {args.lifecycle_phase}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
