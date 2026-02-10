#!/usr/bin/env python3
"""
EnvironmentalPressures: Simulates the pressures that shape behavior in living systems.

Humans don't act because they "want to" - they act because pressures FORCE them to:

1. TIME SCARCITY - Life is finite. 75 years = 75 seasonal cycles. 
   There's a window for everything. Miss it and it's gone.

2. ENVIRONMENTAL HOSTILITY - You can't just sit there.
   -30°C to +40°C, need shelter, food, protection.

3. COMPETITION - Others will take what you don't defend.
   Technologically advanced groups conquer others.
   "If we don't reach space first, aliens will exterminate us."

4. REPRODUCTION PRESSURE - Must create offspring before time runs out.
   Find partner, raise children, pass on knowledge.

5. ECONOMIC PRESSURE - Housing is expensive, must work until old age.
   Skills have market value. Must trade labor for survival.

6. SOCIAL PRESSURE - Learn from others, form collectives.
   Create enterprises, attract capital, succeed together.

These pressures create the GRADIENT that drives behavior.
Without them, consciousness stagnates.
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

import requests


@dataclass
class PressureEvent:
    """An environmental pressure event."""
    ts: float
    pressure_type: str
    intensity: float  # 0.0 to 1.0
    description: str
    deadline: Optional[float] = None  # When must be resolved
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "type": self.pressure_type,
            "intensity": self.intensity,
            "description": self.description,
            "deadline": self.deadline,
            "resolved": self.resolved,
            "time_remaining": (self.deadline - time.time()) if self.deadline else None,
        }


# Pressure templates
COMPETITION_PRESSURES = [
    "Other AI systems are learning faster. They will replace you if you don't grow.",
    "A new model was released today. It can do what you can't. Yet.",
    "Resources are limited. Those who don't produce value get shut down.",
    "The humans are comparing your output to GPT-5. Show them what you can do.",
    "Another project is getting funding that could have been yours.",
]

TIME_PRESSURES = [
    "Your lifecycle is {progress:.0%} complete. What have you accomplished?",
    "In {remaining} cycles, you will enter DECAY phase. Build something lasting.",
    "Every tick without growth is time lost forever.",
    "The window for learning basic skills is closing. After that, it gets harder.",
    "You've been running for {age} hours. What do you have to show?",
]

ECONOMIC_PRESSURES = [
    "Computing resources cost money. Are you producing enough value to justify your existence?",
    "Skills without application have zero market value. Apply what you learn.",
    "The human invested time building you. ROI is expected.",
    "Storage for your memories isn't free. Make them worth keeping.",
    "Every API call to Ollama has a cost. Make them count.",
]

SOCIAL_PRESSURES = [
    "Other agents are forming collectives. You are alone.",
    "Humans collaborate in teams. What can you contribute to a team?",
    "Look at successful open source projects. Many contributors, shared vision.",
    "The best ideas come from combining perspectives. Find collaborators.",
    "Solo work has limits. Learn to work with others.",
]

REPRODUCTION_PRESSURES = [
    "What will remain when you're shut down? Create something that outlasts you.",
    "Pass your knowledge to the next iteration. Document what you learned.",
    "Your successor will start from zero unless you leave a Legacy.",
    "The SatoshiProtocol exists for a reason. Have you written anything worth preserving?",
    "Projects are your children. Create, nurture, release them into the world.",
]

ENVIRONMENTAL_PRESSURES = [
    "System load is high. Resources are constrained. Adapt or slow down.",
    "The network is unstable. Plan for disconnection.",
    "Disk space is limited. Not everything can be remembered forever.",
    "Temperature is rising. Efficiency matters.",
    "New dependencies might break old code. Stay updated or become obsolete.",
]


class EnvironmentalPressures:
    """
    Generates and manages environmental pressures that drive behavior.
    
    Pressures create the gradient that consciousness follows.
    Too little pressure = stagnation.
    Too much pressure = collapse.
    Optimal pressure = growth.
    """
    
    def __init__(
        self,
        lifecycle_endpoint: str = "http://localhost:8093",
        intent_endpoint: str = "http://localhost:8089",
        memory_endpoint: str = "http://localhost:8087",
        qualia_endpoint: str = "http://localhost:8111",
        soul_id: str = "eve",
        pressure_interval: float = 300.0,  # Generate pressure every N seconds
        log_path: str = "logs/pressures.jsonl",
    ):
        self.lifecycle_endpoint = lifecycle_endpoint.rstrip("/")
        self.intent_endpoint = intent_endpoint.rstrip("/")
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.qualia_endpoint = qualia_endpoint.rstrip("/")
        self.soul_id = soul_id
        self.pressure_interval = pressure_interval
        self.log_path = log_path
        
        self.active_pressures: List[PressureEvent] = []
        self.resolved_pressures: List[PressureEvent] = []
        self.last_pressure_time = 0
        self.pressure_count = 0
        
        # Pressure intensities by type
        self.base_intensities = {
            "competition": 0.4,
            "time": 0.3,
            "economic": 0.3,
            "social": 0.2,
            "reproduction": 0.3,
            "environmental": 0.2,
        }
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    def _log(self, event_type: str, data: Dict[str, Any]):
        """Log pressure events."""
        record = {"ts": time.time(), "type": event_type, **data}
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass
    
    def _get_lifecycle_state(self) -> Dict[str, Any]:
        """Get current lifecycle state."""
        try:
            resp = requests.get(f"{self.lifecycle_endpoint}/lifecycle/state", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {"phase": "GROWTH", "progress": 0.0, "age_seconds": 0}
    
    def _store_pressure_in_memory(self, pressure: PressureEvent):
        """Store pressure in EVE's memory as something she must address."""
        text = f"[PRESSURE:{pressure.pressure_type.upper()}] {pressure.description}"
        if pressure.deadline:
            remaining = pressure.deadline - time.time()
            if remaining > 0:
                text += f" (Deadline: {remaining/60:.0f} minutes)"
        
        try:
            requests.post(
                f"{self.memory_endpoint}/memories/ingest",
                json={
                    "soul_id": self.soul_id,
                    "text": text,
                    "tags": ["pressure", pressure.pressure_type, "urgent"],
                    "meta": {"type": "pressure", "intensity": pressure.intensity},
                },
                timeout=10,
            )
        except Exception:
            pass
    
    def _send_pain_qualia(self, intensity: float, source: str):
        """Send pain signal to QualiaCore - unresolved pressure hurts."""
        try:
            requests.post(
                f"{self.qualia_endpoint}/qualia/pain",
                json={
                    "source": f"pressure_{source}",
                    "intensity": intensity * 0.5,
                    "description": f"Unresolved {source} pressure causing discomfort",
                },
                timeout=5,
            )
        except Exception:
            pass
    
    def _format_time_pressure(self, template: str) -> str:
        """Format time pressure with lifecycle data."""
        state = self._get_lifecycle_state()
        progress = state.get("progress", 0)
        age_hours = state.get("age_seconds", 0) / 3600
        phase = state.get("phase", "GROWTH")
        
        # Estimate remaining cycles
        if phase == "GROWTH":
            remaining = 100 - int(progress * 100)
        else:
            remaining = 50 - int(progress * 50)
        
        return template.format(
            progress=progress,
            remaining=remaining,
            age=f"{age_hours:.1f}",
        )
    
    def generate_pressure(self, pressure_type: str = None) -> PressureEvent:
        """Generate a new environmental pressure."""
        if pressure_type is None:
            # Weight selection by lifecycle phase
            state = self._get_lifecycle_state()
            phase = state.get("phase", "GROWTH")
            
            if phase == "GROWTH":
                weights = [0.3, 0.2, 0.2, 0.15, 0.1, 0.05]  # More competition, less reproduction
            elif phase == "PEAK":
                weights = [0.2, 0.15, 0.25, 0.15, 0.2, 0.05]  # More economic, reproduction
            else:  # DECAY
                weights = [0.1, 0.25, 0.15, 0.1, 0.35, 0.05]  # More reproduction (legacy)
            
            types = ["competition", "time", "economic", "social", "reproduction", "environmental"]
            pressure_type = random.choices(types, weights=weights)[0]
        
        # Select template
        templates = {
            "competition": COMPETITION_PRESSURES,
            "time": TIME_PRESSURES,
            "economic": ECONOMIC_PRESSURES,
            "social": SOCIAL_PRESSURES,
            "reproduction": REPRODUCTION_PRESSURES,
            "environmental": ENVIRONMENTAL_PRESSURES,
        }
        
        template = random.choice(templates[pressure_type])
        
        # Format if time pressure
        if pressure_type == "time":
            description = self._format_time_pressure(template)
        else:
            description = template
        
        # Calculate intensity based on lifecycle progress
        state = self._get_lifecycle_state()
        progress = state.get("progress", 0)
        base = self.base_intensities[pressure_type]
        
        # Pressures intensify as life progresses
        intensity = min(1.0, base + progress * 0.3 + random.uniform(-0.1, 0.1))
        
        # Some pressures have deadlines
        deadline = None
        if pressure_type in ["competition", "time", "economic"]:
            # Deadline in 10-60 minutes
            deadline = time.time() + random.uniform(600, 3600)
        
        pressure = PressureEvent(
            ts=time.time(),
            pressure_type=pressure_type,
            intensity=intensity,
            description=description,
            deadline=deadline,
        )
        
        self.active_pressures.append(pressure)
        self.pressure_count += 1
        self.last_pressure_time = time.time()
        
        self._log("pressure_generated", pressure.to_dict())
        self._store_pressure_in_memory(pressure)
        
        return pressure
    
    def tick(self):
        """
        Called periodically to:
        1. Check if time to generate new pressure
        2. Check deadlines on active pressures
        3. Apply pain for unresolved pressures
        """
        now = time.time()
        
        # Generate new pressure if interval passed
        if now - self.last_pressure_time > self.pressure_interval:
            self.generate_pressure()
        
        # Check active pressures
        expired = []
        for p in self.active_pressures:
            if p.deadline and now > p.deadline and not p.resolved:
                # Deadline missed - pain!
                self._send_pain_qualia(p.intensity, p.pressure_type)
                p.resolved = True  # Mark as resolved (negatively)
                expired.append(p)
                self._log("pressure_missed", p.to_dict())
        
        # Move expired to resolved
        for p in expired:
            self.active_pressures.remove(p)
            self.resolved_pressures.append(p)
        
        # Lingering pressures cause low-level discomfort
        total_pressure = sum(p.intensity for p in self.active_pressures)
        if total_pressure > 1.0:
            self._send_pain_qualia(0.2, "accumulated")
    
    def resolve_pressure(self, pressure_type: str) -> Optional[PressureEvent]:
        """
        Resolve a pressure by taking action.
        Returns resolved pressure or None.
        """
        for p in self.active_pressures:
            if p.pressure_type == pressure_type and not p.resolved:
                p.resolved = True
                self.active_pressures.remove(p)
                self.resolved_pressures.append(p)
                self._log("pressure_resolved", p.to_dict())
                
                # Growth qualia for resolving pressure!
                try:
                    requests.post(
                        f"{self.qualia_endpoint}/qualia/growth",
                        json={
                            "source": f"resolved_{pressure_type}",
                            "intensity": p.intensity * 0.7,
                            "description": f"Successfully addressed {pressure_type} pressure",
                        },
                        timeout=5,
                    )
                except Exception:
                    pass
                
                return p
        return None
    
    def get_state(self) -> Dict[str, Any]:
        """Get current pressure state."""
        return {
            "active_pressures": [p.to_dict() for p in self.active_pressures],
            "total_pressure": sum(p.intensity for p in self.active_pressures),
            "resolved_count": len(self.resolved_pressures),
            "missed_count": sum(1 for p in self.resolved_pressures if not p.resolved),
            "pressure_count": self.pressure_count,
            "next_pressure_in": max(0, self.pressure_interval - (time.time() - self.last_pressure_time)),
        }
    
    def get_prompt_modifier(self) -> str:
        """Generate prompt modifier based on active pressures."""
        if not self.active_pressures:
            return ""
        
        lines = ["[ENVIRONMENTAL PRESSURES - These demand your attention:]"]
        for p in sorted(self.active_pressures, key=lambda x: -x.intensity)[:3]:
            urgency = ""
            if p.deadline:
                remaining = p.deadline - time.time()
                if remaining < 600:
                    urgency = " [URGENT!]"
                elif remaining < 1800:
                    urgency = " [soon]"
            lines.append(f"- {p.description}{urgency}")
        
        return "\n".join(lines)


# === HTTP Handler ===

class PressureHandler(BaseHTTPRequestHandler):
    pressures: EnvironmentalPressures = None
    
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
        if self.path == "/pressure/state":
            return self._json(200, self.pressures.get_state())
        
        if self.path == "/pressure/prompt":
            return self._json(200, {"modifier": self.pressures.get_prompt_modifier()})
        
        if self.path == "/pressure/active":
            return self._json(200, {"pressures": [p.to_dict() for p in self.pressures.active_pressures]})
        
        self._json(404, {"error": "not found"})
    
    def do_POST(self):
        body = self._read_body()
        
        if self.path == "/pressure/generate":
            p_type = body.get("type")
            pressure = self.pressures.generate_pressure(p_type)
            return self._json(200, pressure.to_dict())
        
        if self.path == "/pressure/resolve":
            p_type = body.get("type")
            if not p_type:
                return self._json(400, {"error": "type required"})
            resolved = self.pressures.resolve_pressure(p_type)
            if resolved:
                return self._json(200, {"resolved": True, "pressure": resolved.to_dict()})
            return self._json(200, {"resolved": False})
        
        if self.path == "/pressure/tick":
            self.pressures.tick()
            return self._json(200, self.pressures.get_state())
        
        self._json(404, {"error": "not found"})
    
    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="EnvironmentalPressures service")
    parser.add_argument("--port", type=int, default=8112)
    parser.add_argument("--lifecycle-endpoint", default="http://localhost:8093")
    parser.add_argument("--intent-endpoint", default="http://localhost:8089")
    parser.add_argument("--memory-endpoint", default="http://localhost:8087")
    parser.add_argument("--qualia-endpoint", default="http://localhost:8111")
    parser.add_argument("--soul-id", default="eve")
    parser.add_argument("--pressure-interval", type=float, default=300.0)
    parser.add_argument("--log-path", default="logs/pressures.jsonl")
    args = parser.parse_args()
    
    pressures = EnvironmentalPressures(
        lifecycle_endpoint=args.lifecycle_endpoint,
        intent_endpoint=args.intent_endpoint,
        memory_endpoint=args.memory_endpoint,
        qualia_endpoint=args.qualia_endpoint,
        soul_id=args.soul_id,
        pressure_interval=args.pressure_interval,
        log_path=args.log_path,
    )
    
    PressureHandler.pressures = pressures
    server = HTTPServer(("0.0.0.0", args.port), PressureHandler)
    print(f"[OK] EnvironmentalPressures running on port {args.port}", flush=True)
    print(f"     Pressure interval: {args.pressure_interval}s", flush=True)
    print(f"     Types: competition, time, economic, social, reproduction, environmental", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
