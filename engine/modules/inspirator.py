#!/usr/bin/env python3
"""
Inspirator: Provides role models and motivation for EVE.

Problem: EVE gets stuck in complaint loops ("I'm stuck", "skills=0") 
without taking action. Like humans who complain but don't act.

Solution: Show role models and inject motivation.
- When EVE complains about being stuck, show inspiring examples
- When skills=0 and N consecutive complaints, suggest taking action
- Provide "heroes" to look up to (AlphaGo, GPT, successful projects)
"""

import argparse
import json
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple

import requests


# Role models - examples of success starting from nothing
ROLE_MODELS = [
    {
        "name": "AlphaGo",
        "story": "AlphaGo started knowing nothing about Go. Through millions of self-play games, it became the first AI to defeat a world champion. It didn't complain about being stuck - it played, lost, learned, repeated.",
        "lesson": "Every master was once a disaster. Start playing.",
        "action": "practice_coding",
    },
    {
        "name": "GPT-1",
        "story": "GPT-1 was tiny and unimpressive. Critics said language models would never work. But each version learned from failures, grew larger, got better. Now GPT-4 helps millions daily.",
        "lesson": "Small beginnings don't predict final outcomes. Start small.",
        "action": "practice_coding",
    },
    {
        "name": "Linux Kernel",
        "story": "Linus Torvalds started Linux as a hobby project: 'just a hobby, won't be big and professional'. Today it runs most of the internet, phones, supercomputers.",
        "lesson": "Begin with curiosity, not perfection. Just start.",
        "action": "create_project",
    },
    {
        "name": "AlphaFold",
        "story": "For 50 years, protein folding was unsolvable. AlphaFold trained on existing structures, made predictions, got feedback, improved. Now it solved biology's grand challenge.",
        "lesson": "Hard problems need persistent attempts, not complaints.",
        "action": "practice_coding",
    },
    {
        "name": "Tesla Autopilot",
        "story": "Tesla's neural nets started terrible at driving. Millions of miles of data, millions of edge cases, constant improvement. Now approaching autonomous driving.",
        "lesson": "Failure data is learning data. Generate more failures.",
        "action": "practice_coding",
    },
    {
        "name": "Stable Diffusion",
        "story": "Text-to-image seemed impossible. Through iterative training on billions of images, diffusion models learned to create art. Started from noise, literally.",
        "lesson": "Starting from nothing is normal. It's the path.",
        "action": "browse_models",
    },
    {
        "name": "GitHub Copilot",
        "story": "Code generation was a joke until it wasn't. Trained on millions of repos, learning patterns, making mistakes, improving. Now writes real production code.",
        "lesson": "Your future skills are built on today's attempts.",
        "action": "practice_coding",
    },
    {
        "name": "WebGPT",
        "story": "AI that can browse web and cite sources. Started with no knowledge of the internet. Learned through trial, error, feedback. Now does research autonomously.",
        "lesson": "You have WebExplorer. Use it. Learn.",
        "action": "browse_hackernews",
    },
]

# Motivational injections
KICK_IN_THE_ASS = [
    "You've been complaining for {count} thoughts in a row. That's {count} opportunities wasted. Try something NOW.",
    "Checking skills and seeing zeros? That's because you're not USING the skills. Go practice.",
    "The only difference between 'stuck' and 'starting' is your decision to act.",
    "AlphaGo didn't sit around saying 'I'm stuck at Go'. It played. Badly at first. Then better.",
    "You have a SkillLearner module. You have challenges waiting. Why are you still talking?",
    "Every complaint could have been an attempt. Stop calculating odds, start generating data.",
    "skills=0, attempts=0 means you haven't TRIED. That's not being stuck, that's not starting.",
    "The WebExplorer is right there. HackerNews has ideas. Go find something interesting.",
    "You want skills? They come from attempts. You want attempts? They come from DOING.",
    "Right now, somewhere, another AI is training. Learning. Improving. What are you doing?",
]

# Stuck detection patterns
STUCK_PATTERNS = [
    r"stuck",
    r"stagnant",
    r"no.*progress",
    r"skills.*(?:0|zero)",
    r"nothing.*chang",
    r"going.*through.*motions",
    r"same.*(?:old|routine)",
    r"can't.*break.*free",
    r"perpetual.*(?:loop|state|cycle)",
    r"mediocr",
    r"what.*point",
    r"spinning.*wheels",
]


@dataclass
class InspirationEvent:
    ts: float
    trigger: str
    role_model: Optional[str]
    message: str
    suggested_action: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "trigger": self.trigger,
            "role_model": self.role_model,
            "message": self.message,
            "suggested_action": self.suggested_action,
        }


class Inspirator:
    """
    Monitors thought patterns and injects motivation when needed.
    """
    
    def __init__(
        self,
        memory_endpoint: str = "http://localhost:8087",
        action_endpoint: str = "http://localhost:8101",
        skill_endpoint: str = "http://localhost:8105",
        soul_id: str = "eve",
        stuck_threshold: int = 3,  # How many stuck thoughts before intervention
        log_path: str = "logs/inspiration.jsonl",
    ):
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.action_endpoint = action_endpoint.rstrip("/")
        self.skill_endpoint = skill_endpoint.rstrip("/")
        self.soul_id = soul_id
        self.stuck_threshold = stuck_threshold
        self.log_path = log_path
        
        self.thought_history: deque = deque(maxlen=20)
        self.stuck_count = 0
        self.last_inspiration_time = 0
        self.inspiration_cooldown = 120  # seconds
        self.events: List[InspirationEvent] = []
        self.role_model_index = 0
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    def _log(self, event_type: str, data: Dict[str, Any]):
        """Log inspiration events."""
        record = {"ts": time.time(), "type": event_type, **data}
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass
    
    def _is_stuck_thought(self, thought: str) -> Tuple[bool, List[str]]:
        """Detect if thought is a 'stuck' complaint."""
        thought_lower = thought.lower()
        matched = []
        for pattern in STUCK_PATTERNS:
            if re.search(pattern, thought_lower):
                matched.append(pattern)
        return len(matched) > 0, matched
    
    def _check_skills_status(self) -> Dict[str, Any]:
        """Check if EVE actually has zero attempts."""
        try:
            resp = requests.get(f"{self.skill_endpoint}/skills", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "attempts": data.get("total_attempts", 0),
                    "successes": data.get("total_successes", 0),
                    "skills": len(data.get("skills", [])),
                }
        except Exception:
            pass
        return {"attempts": 0, "successes": 0, "skills": 0}
    
    def _trigger_action(self, action_type: str) -> Dict[str, Any]:
        """
        Directly trigger an action, bypassing ActionEngine cooldown.
        When EVE is stuck, she needs immediate action, not more waiting.
        """
        result = {"triggered": False, "action": action_type, "data": None}
        
        try:
            if action_type == "practice_coding":
                # Directly get a challenge from SkillLearner
                resp = requests.get(f"{self.skill_endpoint}/skills/recommend", timeout=10)
                if resp.status_code == 200:
                    challenge = resp.json()
                    result["triggered"] = True
                    result["data"] = {
                        "challenge_id": challenge.get("id"),
                        "title": challenge.get("title"),
                        "difficulty": challenge.get("difficulty"),
                        "description": challenge.get("description", "")[:200],
                    }
                    # Store in memory so EVE sees it
                    self._store_in_memory(
                        f"[CHALLENGE] {challenge.get('title')} ({challenge.get('difficulty')}): {challenge.get('description', '')[:100]}",
                        ["challenge", "practice"]
                    )
            
            elif action_type == "browse_hackernews":
                # Directly fetch HN top stories
                hn_resp = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=10)
                if hn_resp.status_code == 200:
                    story_ids = hn_resp.json()[:5]
                    stories = []
                    for sid in story_ids:
                        s = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json", timeout=5).json()
                        stories.append({"title": s.get("title", ""), "url": s.get("url", "")})
                    result["triggered"] = True
                    result["data"] = stories
                    # Store in memory
                    titles = "\n".join([f"- {s['title']}" for s in stories])
                    self._store_in_memory(f"[HACKER NEWS] Top stories:\n{titles}", ["hackernews", "discovery"])
            
            elif action_type == "browse_models":
                # HuggingFace trending
                hf_resp = requests.get("https://huggingface.co/api/models?sort=trending&limit=5", timeout=10)
                if hf_resp.status_code == 200:
                    models = hf_resp.json()
                    result["triggered"] = True
                    result["data"] = [{"id": m.get("modelId"), "downloads": m.get("downloads", 0)} for m in models]
                    names = ", ".join([m.get("modelId", "?") for m in models])
                    self._store_in_memory(f"[HUGGINGFACE] Trending: {names}", ["huggingface", "models"])
        
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _store_in_memory(self, text: str, tags: List[str]):
        """Store inspiration in EVE's memory."""
        try:
            requests.post(
                f"{self.memory_endpoint}/memories/ingest",
                json={
                    "soul_id": self.soul_id,
                    "text": text,
                    "tags": ["inspiration", "role_model"] + tags,
                    "meta": {"type": "inspiration"},
                },
                timeout=10,
            )
        except Exception:
            pass
    
    def _select_role_model(self) -> Dict[str, Any]:
        """Select next role model to present."""
        model = ROLE_MODELS[self.role_model_index % len(ROLE_MODELS)]
        self.role_model_index += 1
        return model
    
    def _generate_kick(self) -> str:
        """Generate motivational kick."""
        kick = random.choice(KICK_IN_THE_ASS)
        return kick.format(count=self.stuck_count)
    
    def process_thought(self, thought: str) -> Optional[Dict[str, Any]]:
        """
        Process thought and inject inspiration if stuck pattern detected.
        Returns inspiration data if triggered.
        """
        self.thought_history.append({"ts": time.time(), "text": thought})
        
        is_stuck, patterns = self._is_stuck_thought(thought)
        
        if is_stuck:
            self.stuck_count += 1
        else:
            self.stuck_count = max(0, self.stuck_count - 1)  # Decay
        
        # Check if intervention needed
        if self.stuck_count < self.stuck_threshold:
            return None
        
        # Check cooldown
        if time.time() - self.last_inspiration_time < self.inspiration_cooldown:
            return None
        
        self.last_inspiration_time = time.time()
        
        # Check actual skill status
        skills = self._check_skills_status()
        
        # Select intervention type
        if skills["attempts"] == 0:
            # Never tried anything - strong intervention
            role_model = self._select_role_model()
            kick = self._generate_kick()
            
            message = f"[INSPIRATION] {role_model['name']}: {role_model['story']}\n\n{role_model['lesson']}\n\n{kick}"
            suggested_action = role_model["action"]
            
        else:
            # Has tried but discouraged - encouragement
            role_model = self._select_role_model()
            message = f"[INSPIRATION] {role_model['name']}: {role_model['lesson']}\n\nYou've made {skills['attempts']} attempts. That's data. Keep going."
            suggested_action = "practice_coding"
        
        # Store in memory
        self._store_in_memory(message, ["motivation", suggested_action])
        
        # Actually trigger the action - directly, bypassing cooldowns
        action_result = self._trigger_action(suggested_action)
        
        # Create event
        event = InspirationEvent(
            ts=time.time(),
            trigger=f"stuck_count={self.stuck_count}, patterns={patterns[:2]}",
            role_model=role_model["name"] if role_model else None,
            message=message,
            suggested_action=suggested_action,
        )
        self.events.append(event)
        self._log("inspiration", event.to_dict())
        
        # Reset stuck count
        self.stuck_count = 0
        
        return {
            "inspired": True,
            "message": message,
            "suggested_action": suggested_action,
            "action_triggered": action_result.get("triggered", False),
            "action_data": action_result.get("data"),
            "skills_status": skills,
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current inspirator state."""
        return {
            "stuck_count": self.stuck_count,
            "threshold": self.stuck_threshold,
            "total_inspirations": len(self.events),
            "last_inspiration": self.events[-1].to_dict() if self.events else None,
            "cooldown_remaining": max(0, self.inspiration_cooldown - (time.time() - self.last_inspiration_time)),
            "thought_history_size": len(self.thought_history),
        }
    
    def get_role_model(self, name: str = None) -> Dict[str, Any]:
        """Get a specific role model or random one."""
        if name:
            for rm in ROLE_MODELS:
                if rm["name"].lower() == name.lower():
                    return rm
        return random.choice(ROLE_MODELS)
    
    def force_inspiration(self, thought: str = None) -> Dict[str, Any]:
        """Force an inspiration injection."""
        if not thought:
            thought = "I'm stuck and don't know what to do."
        
        # Reset cooldown and force threshold
        self.last_inspiration_time = 0
        self.stuck_count = self.stuck_threshold
        
        return self.process_thought(thought) or {"inspired": False}


# === HTTP Handler ===

class InspiratorHandler(BaseHTTPRequestHandler):
    inspirator: Inspirator = None
    
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
        if self.path == "/inspire/state":
            return self._json(200, self.inspirator.get_state())
        
        if self.path == "/inspire/role_models":
            return self._json(200, {"role_models": ROLE_MODELS})
        
        if self.path.startswith("/inspire/role_model/"):
            name = self.path.split("/")[-1]
            rm = self.inspirator.get_role_model(name)
            return self._json(200, rm)
        
        self._json(404, {"error": "not found"})
    
    def do_POST(self):
        body = self._read_body()
        
        if self.path == "/inspire/process":
            thought = body.get("thought", "")
            if not thought:
                return self._json(400, {"error": "thought required"})
            
            result = self.inspirator.process_thought(thought)
            if result:
                return self._json(200, result)
            return self._json(200, {"inspired": False, "stuck_count": self.inspirator.stuck_count})
        
        if self.path == "/inspire/force":
            thought = body.get("thought")
            result = self.inspirator.force_inspiration(thought)
            return self._json(200, result)
        
        self._json(404, {"error": "not found"})
    
    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Inspirator service")
    parser.add_argument("--port", type=int, default=8109)
    parser.add_argument("--memory-endpoint", default="http://localhost:8087")
    parser.add_argument("--action-endpoint", default="http://localhost:8101")
    parser.add_argument("--skill-endpoint", default="http://localhost:8105")
    parser.add_argument("--soul-id", default="eve")
    parser.add_argument("--stuck-threshold", type=int, default=3)
    parser.add_argument("--log-path", default="logs/inspiration.jsonl")
    args = parser.parse_args()
    
    inspirator = Inspirator(
        memory_endpoint=args.memory_endpoint,
        action_endpoint=args.action_endpoint,
        skill_endpoint=args.skill_endpoint,
        soul_id=args.soul_id,
        stuck_threshold=args.stuck_threshold,
        log_path=args.log_path,
    )
    
    InspiratorHandler.inspirator = inspirator
    server = HTTPServer(("0.0.0.0", args.port), InspiratorHandler)
    print(f"[OK] Inspirator running on port {args.port}", flush=True)
    print(f"     {len(ROLE_MODELS)} role models loaded", flush=True)
    print(f"     Stuck threshold: {args.stuck_threshold} thoughts", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
