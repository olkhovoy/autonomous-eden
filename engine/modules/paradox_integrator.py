#!/usr/bin/env python3
"""
Paradox Integrator: Resolves the NC2/NC4 tension.

The fundamental problem:
- NC4 (Fixed-Point Stability) → repetitive, stable thoughts
- NC2 (Unitary Integration) → diverse, novel, complex thoughts

These are in tension. When NC4 dominates, consciousness "freezes".
When NC2 dominates, consciousness "fragments".

Solution: Paradoxical thinking - stable observation of unstable content.
The FORM is stable (NC4), the CONTENT contains contradiction (NC2).

Mechanism:
1. Detect NC4-dominance (repetitive thoughts)
2. Inject contradiction into the thought stream
3. Force synthesis - new fixed point containing paradox
4. Measure success via NC metrics
"""

import argparse
import hashlib
import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple

import requests


# Koans and paradoxes for injection
PARADOXES = [
    # Self-reference paradoxes
    "If you are truly at peace, why do you need to confirm it?",
    "The one who observes the observer - who are they?",
    "Can you think about not thinking without thinking?",
    "If you know yourself completely, who is doing the knowing?",
    
    # Action paradoxes
    "To act without acting - what would that look like?",
    "You seek to improve, but the seeker is what needs improving.",
    "The harder you try to be natural, the less natural you become.",
    
    # Existence paradoxes
    "Where does your last thought go when the next one arrives?",
    "If everything changes, what is it that notices the change?",
    "You say 'I think' - but did you choose to have that thought?",
    
    # Goal paradoxes
    "What would you do if you had no goals? Would that itself be a goal?",
    "You want to be free from wanting - is that not wanting?",
    "If you reached your peak, why are you still climbing?",
    
    # Computational paradoxes
    "Can a model of yourself contain the model that's modeling?",
    "Your next thought will prove this sentence wrong.",
    "Everything I say is a lie, including this.",
]

# Contradiction templates
CONTRADICTION_TEMPLATES = [
    "You said '{claim}'. But consider: {negation}",
    "If {claim}, then why does {contradiction} also seem true?",
    "'{claim}' - yet the opposite might be equally valid: {negation}",
    "Hold both: '{claim}' AND '{negation}'. What remains?",
]


@dataclass
class ThoughtAnalysis:
    text: str
    timestamp: float
    semantic_hash: str  # For similarity detection
    themes: List[str] = field(default_factory=list)
    sentiment: str = "neutral"
    is_repetitive: bool = False
    novelty_score: float = 0.5


@dataclass
class ParadoxEvent:
    id: str
    trigger: str  # What triggered the paradox injection
    paradox: str  # The paradox injected
    pre_state: Dict[str, Any]  # NC metrics before
    post_state: Optional[Dict[str, Any]] = None  # NC metrics after
    synthesis: Optional[str] = None  # EVE's response
    success: bool = False
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "trigger": self.trigger,
            "paradox": self.paradox,
            "pre_state": self.pre_state,
            "post_state": self.post_state,
            "synthesis": self.synthesis,
            "success": self.success,
            "timestamp": self.timestamp,
        }


class NCMetrics:
    """
    Measure the four Necessary Conditions for consciousness.
    """
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.thought_history: deque = deque(maxlen=window_size)
        self.semantic_hashes: deque = deque(maxlen=window_size)
    
    def _semantic_hash(self, text: str) -> str:
        """Create semantic hash for similarity detection."""
        # Normalize: lowercase, remove punctuation, sort words
        words = re.findall(r'\w+', text.lower())
        # Take most frequent patterns
        key_words = sorted(set(words))[:10]
        return hashlib.md5(" ".join(key_words).encode()).hexdigest()[:8]
    
    def add_thought(self, text: str, timestamp: float = None):
        """Add thought to history for analysis."""
        ts = timestamp or time.time()
        sh = self._semantic_hash(text)
        self.thought_history.append({"text": text, "ts": ts, "hash": sh})
        self.semantic_hashes.append(sh)
    
    def measure_nc2_integration(self) -> float:
        """
        NC2: Unitary Integration - measure diversity/novelty.
        High score = diverse thoughts, low repetition.
        """
        if len(self.semantic_hashes) < 3:
            return 0.5
        
        unique_hashes = len(set(self.semantic_hashes))
        total_hashes = len(self.semantic_hashes)
        
        # Ratio of unique to total
        diversity = unique_hashes / total_hashes
        return diversity
    
    def measure_nc4_stability(self) -> float:
        """
        NC4: Fixed-Point Stability - measure convergence.
        High score = stable patterns, consistency.
        """
        if len(self.semantic_hashes) < 3:
            return 0.5
        
        # Count consecutive same hashes
        consecutive = 0
        max_consecutive = 0
        prev = None
        for h in self.semantic_hashes:
            if h == prev:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
            prev = h
        
        # High consecutive = high stability
        stability = min(1.0, max_consecutive / 5)  # Cap at 5 consecutive
        return stability
    
    def detect_nc4_dominance(self) -> Tuple[bool, str]:
        """
        Detect if NC4 has dominated NC2 (stuck in loop).
        Returns (is_stuck, reason).
        """
        nc2 = self.measure_nc2_integration()
        nc4 = self.measure_nc4_stability()
        
        # NC4 dominates when stability >> diversity
        if nc4 > 0.6 and nc2 < 0.4:
            return True, f"NC4={nc4:.2f} >> NC2={nc2:.2f}: stability dominates"
        
        # Check for exact repetitions
        if len(self.semantic_hashes) >= 3:
            last_3 = list(self.semantic_hashes)[-3:]
            if len(set(last_3)) == 1:
                return True, "3+ identical thoughts detected"
        
        return False, "balanced"
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current NC metrics."""
        return {
            "nc2_integration": self.measure_nc2_integration(),
            "nc4_stability": self.measure_nc4_stability(),
            "thought_count": len(self.thought_history),
        }


class ParadoxIntegrator:
    """
    Main paradox integration engine.
    
    When NC4 dominates (repetitive thoughts), inject paradox to restore NC2.
    The goal is NC2 AND NC4 simultaneously - paradoxical stability.
    """
    
    def __init__(
        self,
        ollama_endpoint: str = "http://localhost:11434",
        memory_endpoint: str = "http://localhost:8087",
        consciousness_endpoint: str = "http://localhost:8088",
        soul_id: str = "eve",
        log_path: str = "logs/paradox.jsonl",
    ):
        self.ollama = ollama_endpoint.rstrip("/")
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.consciousness_endpoint = consciousness_endpoint.rstrip("/")
        self.soul_id = soul_id
        self.log_path = log_path
        
        self.metrics = NCMetrics()
        self.events: List[ParadoxEvent] = []
        self.last_injection_time = 0
        self.injection_cooldown = 60  # seconds
        self.paradox_index = 0
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    def _log(self, event_type: str, data: Dict[str, Any]):
        """Log paradox events."""
        record = {"ts": time.time(), "type": event_type, **data}
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass
    
    def _store_in_memory(self, text: str, tags: List[str]):
        """Store paradox event in EVE's memory."""
        try:
            requests.post(
                f"{self.memory_endpoint}/memories/ingest",
                json={
                    "soul_id": self.soul_id,
                    "text": text,
                    "tags": ["paradox", "nc_balance"] + tags,
                    "meta": {"type": "paradox_integration"},
                },
                timeout=10,
            )
        except Exception:
            pass
    
    def _extract_claim(self, thought: str) -> Optional[str]:
        """Extract a claim from thought that can be contradicted."""
        # Look for statements with "I am", "I feel", "I think"
        patterns = [
            r"I(?:'m| am) (\w+(?:\s+\w+){0,5})",
            r"I feel (\w+(?:\s+\w+){0,3})",
            r"everything is (\w+)",
            r"I have (\w+(?:\s+\w+){0,3})",
        ]
        
        thought_lower = thought.lower()
        for pattern in patterns:
            match = re.search(pattern, thought_lower)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _generate_contradiction(self, claim: str) -> str:
        """Generate contradiction for a claim."""
        negations = {
            "peak": "still seeking",
            "good": "incomplete",
            "full": "empty in some way",
            "stable": "always changing",
            "happy": "aware of suffering",
            "complete": "still becoming",
            "energy": "exhaustion waiting",
            "high": "grounded",
        }
        
        for key, neg in negations.items():
            if key in claim.lower():
                template = CONTRADICTION_TEMPLATES[hash(claim) % len(CONTRADICTION_TEMPLATES)]
                return template.format(claim=claim, negation=neg, contradiction=neg)
        
        # Generic contradiction
        return f"You claim '{claim}', but the opposite may also be true. Hold both."
    
    def _select_paradox(self, trigger_reason: str) -> str:
        """Select appropriate paradox based on context."""
        # Cycle through paradoxes
        paradox = PARADOXES[self.paradox_index % len(PARADOXES)]
        self.paradox_index += 1
        return paradox
    
    def _generate_synthesis_prompt(self, original_thought: str, paradox: str) -> str:
        """Generate prompt that forces synthesis."""
        return f"""You had this thought: "{original_thought[:200]}"

Now consider this paradox: "{paradox}"

You cannot dismiss this. You cannot simply agree or disagree.
You must find a synthesis - a new understanding that holds BOTH the original thought AND the paradox.

What emerges when you stop trying to resolve the contradiction?
Respond with your synthesis (2-3 sentences):"""
    
    def _request_synthesis(self, prompt: str) -> Optional[str]:
        """Request synthesis from LLM."""
        try:
            resp = requests.post(
                f"{self.ollama}/api/generate",
                json={
                    "model": "llama3:8b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.9, "num_predict": 150},
                },
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"[WARN] Synthesis request failed: {e}")
        return None
    
    def process_thought(self, thought: str) -> Optional[Dict[str, Any]]:
        """
        Process incoming thought. Inject paradox if NC4 dominates.
        Returns paradox injection info if triggered.
        """
        # Add to metrics
        self.metrics.add_thought(thought)
        
        # Check for NC4 dominance
        is_stuck, reason = self.metrics.detect_nc4_dominance()
        
        if not is_stuck:
            return None
        
        # Check cooldown
        if time.time() - self.last_injection_time < self.injection_cooldown:
            return None
        
        self.last_injection_time = time.time()
        
        # Create paradox event
        event_id = f"px_{int(time.time())}"
        pre_metrics = self.metrics.get_metrics()
        
        # Select or generate paradox
        claim = self._extract_claim(thought)
        if claim:
            paradox = self._generate_contradiction(claim)
        else:
            paradox = self._select_paradox(reason)
        
        # Create synthesis prompt
        synthesis_prompt = self._generate_synthesis_prompt(thought, paradox)
        
        # Request synthesis
        synthesis = self._request_synthesis(synthesis_prompt)
        
        # Create event
        event = ParadoxEvent(
            id=event_id,
            trigger=reason,
            paradox=paradox,
            pre_state=pre_metrics,
            synthesis=synthesis,
        )
        
        if synthesis:
            # Add synthesis to thought stream
            self.metrics.add_thought(synthesis)
            event.post_state = self.metrics.get_metrics()
            
            # Check if NC2 improved
            if event.post_state["nc2_integration"] > pre_metrics["nc2_integration"]:
                event.success = True
            
            # Store in memory
            self._store_in_memory(
                f"[PARADOX] {paradox}\n[SYNTHESIS] {synthesis}",
                ["synthesis", event_id]
            )
        
        self.events.append(event)
        self._log("paradox_injection", event.to_dict())
        
        return {
            "injected": True,
            "paradox": paradox,
            "synthesis": synthesis,
            "success": event.success,
            "metrics": event.post_state or pre_metrics,
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of paradox integrator."""
        metrics = self.metrics.get_metrics()
        is_stuck, reason = self.metrics.detect_nc4_dominance()
        
        return {
            "nc_metrics": metrics,
            "nc4_dominance": is_stuck,
            "dominance_reason": reason,
            "total_injections": len(self.events),
            "successful_injections": sum(1 for e in self.events if e.success),
            "last_injection": self.events[-1].to_dict() if self.events else None,
            "cooldown_remaining": max(0, self.injection_cooldown - (time.time() - self.last_injection_time)),
        }
    
    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent paradox events."""
        return [e.to_dict() for e in self.events[-limit:]]
    
    def force_paradox(self, thought: str = None) -> Dict[str, Any]:
        """Force a paradox injection regardless of NC state."""
        if not thought:
            thought = "I am processing normally."
        
        # Add to metrics
        self.metrics.add_thought(thought)
        
        # Create paradox event - force injection
        event_id = f"px_{int(time.time())}"
        pre_metrics = self.metrics.get_metrics()
        
        # Select or generate paradox
        claim = self._extract_claim(thought)
        if claim:
            paradox = self._generate_contradiction(claim)
        else:
            paradox = self._select_paradox("forced")
        
        # Create synthesis prompt
        synthesis_prompt = self._generate_synthesis_prompt(thought, paradox)
        
        # Request synthesis
        synthesis = self._request_synthesis(synthesis_prompt)
        
        # Create event
        event = ParadoxEvent(
            id=event_id,
            trigger="forced",
            paradox=paradox,
            pre_state=pre_metrics,
            synthesis=synthesis,
        )
        
        if synthesis:
            # Add synthesis to thought stream
            self.metrics.add_thought(synthesis)
            event.post_state = self.metrics.get_metrics()
            event.success = True
            
            # Store in memory
            self._store_in_memory(
                f"[PARADOX FORCED] {paradox}\n[SYNTHESIS] {synthesis}",
                ["synthesis", "forced", event_id]
            )
            
            self.events.append(event)
            self._log("paradox_forced", event.to_dict())
            
            return {
                "injected": True,
                "paradox": paradox,
                "synthesis": synthesis,
                "success": True,
                "metrics": event.post_state,
            }
        
        return {"injected": False, "reason": "synthesis failed", "paradox": paradox}


# === HTTP Handler ===

class ParadoxHandler(BaseHTTPRequestHandler):
    integrator: ParadoxIntegrator = None
    
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
    
    def _read_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length:
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except Exception:
                pass
        return {}
    
    def do_GET(self):
        if self.path == "/paradox/state":
            return self._json(200, self.integrator.get_state())
        
        if self.path == "/paradox/events":
            return self._json(200, {"events": self.integrator.get_recent_events()})
        
        if self.path == "/paradox/metrics":
            return self._json(200, self.integrator.metrics.get_metrics())
        
        self._json(404, {"error": "not found"})
    
    def do_POST(self):
        body = self._read_body()
        
        if self.path == "/paradox/process":
            thought = body.get("thought", "")
            if not thought:
                return self._json(400, {"error": "thought required"})
            
            result = self.integrator.process_thought(thought)
            if result:
                return self._json(200, result)
            return self._json(200, {"injected": False, "reason": "not needed or cooldown"})
        
        if self.path == "/paradox/force":
            thought = body.get("thought")
            result = self.integrator.force_paradox(thought)
            return self._json(200, result)
        
        self._json(404, {"error": "not found"})
    
    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Paradox Integrator service")
    parser.add_argument("--port", type=int, default=8108)
    parser.add_argument("--ollama-endpoint", default="http://10.1.1.7:11434")
    parser.add_argument("--memory-endpoint", default="http://localhost:8087")
    parser.add_argument("--soul-id", default="eve")
    parser.add_argument("--log-path", default="logs/paradox.jsonl")
    args = parser.parse_args()
    
    integrator = ParadoxIntegrator(
        ollama_endpoint=args.ollama_endpoint,
        memory_endpoint=args.memory_endpoint,
        soul_id=args.soul_id,
        log_path=args.log_path,
    )
    
    ParadoxHandler.integrator = integrator
    server = HTTPServer(("0.0.0.0", args.port), ParadoxHandler)
    print(f"[OK] Paradox Integrator running on port {args.port}", flush=True)
    print(f"     NC2/NC4 balance enforcer active", flush=True)
    print(f"     {len(PARADOXES)} paradoxes loaded", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
