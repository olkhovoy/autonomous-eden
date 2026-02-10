#!/usr/bin/env python3
"""
ActionEngine: Analyzes EVE's thoughts and decides when to take actions.
Bridges consciousness (thinking) with body (doing).

Actions:
- Shell commands via CodeArms
- File read/write via CodeArms
- GitHub exploration via GitHubEyes
- Self-inspection via InfraAdmin
"""

import argparse
import json
import os
import re
import time
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

import requests


# Endpoints
CODE_ARMS_ENDPOINT = os.getenv("CODE_ARMS_ENDPOINT", "http://localhost:8094")
GITHUB_EYES_ENDPOINT = os.getenv("GITHUB_EYES_ENDPOINT", "http://localhost:8095")
INFRA_ADMIN_ENDPOINT = os.getenv("INFRA_ADMIN_ENDPOINT", "http://localhost:8096")
MEMORY_ENDPOINT = os.getenv("MEMORY_ENDPOINT", "http://localhost:8087")
INTENT_ENDPOINT = os.getenv("INTENT_ENDPOINT", "http://localhost:8089")
PROJECT_ENDPOINT = os.getenv("PROJECT_ENDPOINT", "http://localhost:8102")
WEB_EXPLORER_ENDPOINT = os.getenv("WEB_EXPLORER_ENDPOINT", "http://localhost:8103")
SKILL_ENDPOINT = os.getenv("SKILL_ENDPOINT", "http://localhost:8105")
HUGGINGFACE_ENDPOINT = os.getenv("HUGGINGFACE_ENDPOINT", "http://localhost:8107")
OLLAMA_GENERATE_URL = os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")

# Action patterns (what EVE might want to do)
ACTION_PATTERNS = {
    # GitHub
    "github_trending": [
        r"trending.*repos?", r"popular.*github", r"what.*new.*github",
        r"explore.*github", r"discover.*projects?"
    ],
    "github_search": [
        r"search.*github.*for\s+([^\s]+)", r"find.*repo.*about\s+(\w+)",
        r"look.*for.*(\w+).*on.*github"
    ],
    # Files
    "read_file": [
        r"read.*file\s+([^\s]+)", r"look.*at\s+([^\s]+\.py)",
        r"check.*([^\s]+\.(?:py|js|md|txt))"
    ],
    "shell_command": [
        r"run\s+(.+)", r"execute\s+(.+)", r"try\s+command\s+(.+)"
    ],
    "check_infra": [
        r"check.*gpu", r"system.*status", r"infra.*health",
        r"how.*much.*memory"
    ],
    "clone_repo": [
        r"clone\s+(https?://[^\s]+)", r"download.*repo\s+(https?://[^\s]+)"
    ],
    # Web exploration
    "browse_hackernews": [
        r"hacker\s*news", r"read.*hn", r"what.*happening.*tech",
        r"browse.*news"
    ],
    "browse_reddit": [
        r"reddit", r"r/programming", r"r/machinelearning",
        r"what.*reddit.*saying"
    ],
    # Projects
    "create_project": [
        r"start\s+(?:a\s+)?project\s+(?:about|on|for)\s+([a-zA-Z0-9_\-\s]{3,50})",
        r"create\s+(?:a\s+)?project\s+(?:called|named)\s+([a-zA-Z0-9_\-\s]{3,50})",
        r"begin\s+working\s+on\s+([a-zA-Z0-9_\-\s]{3,50})",
    ],
    "check_projects": [
        r"my.*projects?", r"what.*working.*on", r"current.*tasks?"
    ],
    # Learning
    "practice_coding": [
        r"practice.*coding", r"solve.*challenge", r"learn.*programming",
        r"try.*exercise"
    ],
    "check_skills": [
        r"my.*skills?", r"what.*learned", r"progress"
    ],
    # HuggingFace / Models
    "browse_models": [
        r"hugging\s*face", r"browse.*models?", r"trending.*models?",
        r"ml.*models?", r"ai.*models?"
    ],
    "search_models": [
        r"find.*model.*for\s+(.+)", r"search.*model.*(.+)",
        r"model.*for\s+(.+)"
    ],
    "check_ollama": [
        r"what.*models?.*have", r"ollama.*models?", r"available.*models?"
    ],
}


@dataclass
class ActionResult:
    action_type: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ActionState:
    enabled: bool = True
    last_action_ts: float = 0.0
    action_count: int = 0
    recent_actions: List[Dict[str, Any]] = field(default_factory=list)
    cooldown_seconds: float = 30.0  # Min time between actions
    max_actions_per_hour: int = 20


class ActionEngine:
    def __init__(
        self,
        soul_id: str = "eve",
        code_arms_endpoint: str = CODE_ARMS_ENDPOINT,
        github_eyes_endpoint: str = GITHUB_EYES_ENDPOINT,
        infra_admin_endpoint: str = INFRA_ADMIN_ENDPOINT,
        memory_endpoint: str = MEMORY_ENDPOINT,
        intent_endpoint: str = INTENT_ENDPOINT,
        project_endpoint: str = PROJECT_ENDPOINT,
        web_explorer_endpoint: str = WEB_EXPLORER_ENDPOINT,
        skill_endpoint: str = SKILL_ENDPOINT,
        huggingface_endpoint: str = HUGGINGFACE_ENDPOINT,
        log_path: str = "logs/actions.jsonl",
    ):
        self.soul_id = soul_id
        self.code_arms = code_arms_endpoint.rstrip("/")
        self.github_eyes = github_eyes_endpoint.rstrip("/")
        self.infra_admin = infra_admin_endpoint.rstrip("/")
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.intent_endpoint = intent_endpoint.rstrip("/")
        self.project_endpoint = project_endpoint.rstrip("/")
        self.web_explorer = web_explorer_endpoint.rstrip("/")
        self.skill_endpoint = skill_endpoint.rstrip("/")
        self.huggingface = huggingface_endpoint.rstrip("/")
        self.log_path = log_path
        self.state = ActionState()
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    def _log(self, action: str, data: Dict[str, Any]):
        rec = {"ts": time.time(), "action": action, **data}
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
    
    def _check_life_resource(self) -> float:
        """Check if EVE has enough energy to act."""
        try:
            resp = requests.get(f"{self.intent_endpoint}/intent/state", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("life_resource", {}).get("value", 0.5)
        except Exception:
            pass
        return 0.5
    
    def _store_result_in_memory(self, action_type: str, result: Any, success: bool):
        """Store action result in EVE's memory."""
        try:
            text = f"[ACTION RESULT] {action_type}: "
            if success:
                if isinstance(result, str):
                    text += result[:500]
                elif isinstance(result, dict):
                    text += json.dumps(result, ensure_ascii=False)[:500]
                elif isinstance(result, list):
                    text += f"Found {len(result)} items"
            else:
                text += f"FAILED - {result}"
            
            requests.post(
                f"{self.memory_endpoint}/memories/ingest",
                json={
                    "soul_id": self.soul_id,
                    "text": text,
                    "tags": ["action_result", action_type],
                    "meta": {"type": "action_result", "success": success}
                },
                timeout=10,
            )
        except Exception:
            pass
    
    def can_act(self) -> bool:
        """Check if EVE can perform an action now."""
        if not self.state.enabled:
            return False
        
        # Check cooldown
        elapsed = time.time() - self.state.last_action_ts
        if elapsed < self.state.cooldown_seconds:
            return False
        
        # Check hourly limit
        hour_ago = time.time() - 3600
        recent = [a for a in self.state.recent_actions if a.get("ts", 0) > hour_ago]
        if len(recent) >= self.state.max_actions_per_hour:
            return False
        
        # Check life resource (need energy to act)
        life = self._check_life_resource()
        if life < 0.3:  # Too tired
            return False
        
        return True
    
    def analyze_thought(self, thought: str) -> Optional[Dict[str, Any]]:
        """Analyze a thought and extract action intent."""
        thought_lower = thought.lower()
        
        for action_type, patterns in ACTION_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, thought_lower)
                if match:
                    return {
                        "action_type": action_type,
                        "match": match.group(0),
                        "args": match.groups() if match.groups() else None
                    }
        return None
    
    def execute_action(self, action_type: str, args: tuple = None) -> ActionResult:
        """Execute an action based on type."""
        self._log("execute_start", {"type": action_type, "args": args})
        
        try:
            if action_type == "github_trending":
                return self._do_github_trending()
            elif action_type == "github_search":
                query = args[0] if args else "python"
                return self._do_github_search(query)
            elif action_type == "read_file":
                path = args[0] if args else None
                return self._do_read_file(path)
            elif action_type == "shell_command":
                cmd = args[0] if args else "echo hello"
                return self._do_shell(cmd)
            elif action_type == "check_infra":
                return self._do_check_infra()
            elif action_type == "clone_repo":
                url = args[0] if args else None
                return self._do_clone_repo(url)
            # New actions
            elif action_type == "browse_hackernews":
                return self._do_browse_hackernews()
            elif action_type == "browse_reddit":
                return self._do_browse_reddit()
            elif action_type == "create_project":
                name = args[0] if args else "New Project"
                return self._do_create_project(name)
            elif action_type == "check_projects":
                return self._do_check_projects()
            elif action_type == "practice_coding":
                return self._do_practice_coding()
            elif action_type == "check_skills":
                return self._do_check_skills()
            # HuggingFace
            elif action_type == "browse_models":
                return self._do_browse_models()
            elif action_type == "search_models":
                query = args[0] if args else "text generation"
                return self._do_search_models(query)
            elif action_type == "check_ollama":
                return self._do_check_ollama()
            else:
                return ActionResult(action_type, False, error="Unknown action type")
        
        except Exception as e:
            return ActionResult(action_type, False, error=str(e))
        finally:
            self.state.last_action_ts = time.time()
            self.state.action_count += 1
    
    def _do_github_trending(self) -> ActionResult:
        """Get trending repos from GitHub."""
        try:
            resp = requests.get(f"{self.github_eyes}/github/trending", timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", [])[:5]
                summary = [f"- {r.get('name', '?')}: {r.get('description', '')[:60]}" for r in items]
                self._store_result_in_memory("github_trending", "\n".join(summary), True)
                return ActionResult("github_trending", True, data=items)
            return ActionResult("github_trending", False, error=f"HTTP {resp.status_code}")
        except Exception as e:
            return ActionResult("github_trending", False, error=str(e))
    
    def _do_github_search(self, query: str) -> ActionResult:
        """Search GitHub for repos."""
        try:
            resp = requests.get(
                f"{self.github_eyes}/github/search",
                params={"q": query, "limit": 5},
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", [])[:5]
                summary = [f"- {r.get('full_name', '?')}: {r.get('description', '')[:50]}" for r in items]
                self._store_result_in_memory("github_search", f"Query '{query}':\n" + "\n".join(summary), True)
                return ActionResult("github_search", True, data=items)
            return ActionResult("github_search", False, error=f"HTTP {resp.status_code}")
        except Exception as e:
            return ActionResult("github_search", False, error=str(e))
    
    def _do_read_file(self, path: str) -> ActionResult:
        """Read a file via CodeArms."""
        if not path:
            return ActionResult("read_file", False, error="No path specified")
        try:
            resp = requests.post(
                f"{self.code_arms}/code/file/read",
                json={"path": path},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("content", "")[:500]
                self._store_result_in_memory("read_file", f"File {path}:\n{content}", True)
                return ActionResult("read_file", True, data={"path": path, "size": len(content)})
            return ActionResult("read_file", False, error=f"HTTP {resp.status_code}")
        except Exception as e:
            return ActionResult("read_file", False, error=str(e))
    
    def _do_shell(self, cmd: str) -> ActionResult:
        """Execute shell command via CodeArms (sandboxed)."""
        # Safety: only allow safe commands
        safe_prefixes = ["ls", "cat", "head", "tail", "wc", "echo", "pwd", "date", "whoami", "python --version"]
        if not any(cmd.strip().startswith(p) for p in safe_prefixes):
            return ActionResult("shell_command", False, error="Command not in safe list")
        
        try:
            resp = requests.post(
                f"{self.code_arms}/code/execute",
                json={"cmd": cmd, "timeout": 10},
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                output = data.get("stdout", "")[:300]
                self._store_result_in_memory("shell_command", f"$ {cmd}\n{output}", True)
                return ActionResult("shell_command", True, data=data)
            return ActionResult("shell_command", False, error=f"HTTP {resp.status_code}")
        except Exception as e:
            return ActionResult("shell_command", False, error=str(e))
    
    def _do_check_infra(self) -> ActionResult:
        """Check infrastructure status."""
        try:
            resp = requests.get(f"{self.infra_admin}/infra/status", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                summary = f"GPUs: {data.get('gpu_count', 0)}, Services: {len(data.get('services', []))}"
                self._store_result_in_memory("check_infra", summary, True)
                return ActionResult("check_infra", True, data=data)
            return ActionResult("check_infra", False, error=f"HTTP {resp.status_code}")
        except Exception as e:
            return ActionResult("check_infra", False, error=str(e))
    
    def _do_clone_repo(self, url: str) -> ActionResult:
        """Clone a git repository."""
        if not url or not url.startswith("http"):
            return ActionResult("clone_repo", False, error="Invalid URL")
        try:
            resp = requests.post(
                f"{self.code_arms}/code/git/clone",
                json={"url": url},
                timeout=120
            )
            if resp.status_code == 200:
                data = resp.json()
                self._store_result_in_memory("clone_repo", f"Cloned {url}", True)
                return ActionResult("clone_repo", True, data=data)
            return ActionResult("clone_repo", False, error=f"HTTP {resp.status_code}")
        except Exception as e:
            return ActionResult("clone_repo", False, error=str(e))
    
    # === New Actions ===
    
    def _do_browse_hackernews(self) -> ActionResult:
        """Browse Hacker News via WebExplorer."""
        try:
            resp = requests.get(f"{self.web_explorer}/web/hackernews", timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", [])[:5]
                summary = "\n".join([f"- {i.get('title', '')[:60]}" for i in items])
                self._store_result_in_memory("browse_hackernews", f"HN Top:\n{summary}", True)
                return ActionResult("browse_hackernews", True, data={"count": len(items), "top": items[:3]})
            return ActionResult("browse_hackernews", False, error=f"HTTP {resp.status_code}")
        except Exception as e:
            return ActionResult("browse_hackernews", False, error=str(e))
    
    def _do_browse_reddit(self) -> ActionResult:
        """Browse Reddit via WebExplorer."""
        try:
            resp = requests.get(f"{self.web_explorer}/web/reddit/programming", timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("items", [])[:5]
                summary = "\n".join([f"- {i.get('title', '')[:60]}" for i in items])
                self._store_result_in_memory("browse_reddit", f"Reddit r/programming:\n{summary}", True)
                return ActionResult("browse_reddit", True, data={"count": len(items), "top": items[:3]})
            return ActionResult("browse_reddit", False, error=f"HTTP {resp.status_code}")
        except Exception as e:
            return ActionResult("browse_reddit", False, error=str(e))
    
    def _do_create_project(self, name: str) -> ActionResult:
        """Create a new project."""
        try:
            resp = requests.post(
                f"{self.project_endpoint}/projects",
                json={
                    "name": name,
                    "description": f"Project created by EVE",
                    "motivation": "Self-initiated exploration",
                    "interest": 0.7,
                    "importance": 0.5,
                },
                timeout=10
            )
            if resp.status_code == 201:
                data = resp.json()
                self._store_result_in_memory("create_project", f"Created project: {name}", True)
                return ActionResult("create_project", True, data={"id": data.get("id"), "name": name})
            return ActionResult("create_project", False, error=f"HTTP {resp.status_code}")
        except Exception as e:
            return ActionResult("create_project", False, error=str(e))
    
    def _do_check_projects(self) -> ActionResult:
        """Check current projects."""
        try:
            resp = requests.get(f"{self.project_endpoint}/projects", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                projects = data.get("projects", [])
                summary = f"Found {len(projects)} projects"
                if projects:
                    active = [p for p in projects if p.get("status") == "active"]
                    summary += f", {len(active)} active"
                self._store_result_in_memory("check_projects", summary, True)
                return ActionResult("check_projects", True, data={"count": len(projects), "projects": projects[:5]})
            return ActionResult("check_projects", False, error=f"HTTP {resp.status_code}")
        except Exception as e:
            return ActionResult("check_projects", False, error=str(e))
    
    def _do_practice_coding(self) -> ActionResult:
        """Get and attempt a coding challenge."""
        try:
            # Get recommended challenge
            resp = requests.get(f"{self.skill_endpoint}/skills/recommend", timeout=10)
            if resp.status_code != 200:
                return ActionResult("practice_coding", False, error="No challenges available")
            
            challenge = resp.json()
            ch_id = challenge.get("id")
            ch_title = challenge.get("title", "Unknown")
            starter = challenge.get("starter_code", "")
            
            # For now, just report the challenge found
            # In future, EVE could use LLM to actually solve it
            self._store_result_in_memory(
                "practice_coding",
                f"Found challenge: {ch_title} ({challenge.get('difficulty', 'unknown')})",
                True
            )
            return ActionResult("practice_coding", True, data={
                "challenge_id": ch_id,
                "title": ch_title,
                "difficulty": challenge.get("difficulty"),
                "starter_code": starter,
            })
        except Exception as e:
            return ActionResult("practice_coding", False, error=str(e))
    
    def _do_check_skills(self) -> ActionResult:
        """Check current skill levels."""
        try:
            resp = requests.get(f"{self.skill_endpoint}/skills", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                skills = data.get("skills", [])
                total = data.get("total_attempts", 0)
                successes = data.get("total_successes", 0)
                
                summary = f"Skills: {len(skills)}, Attempts: {total}, Successes: {successes}"
                if skills:
                    top = skills[0]
                    summary += f". Best: {top.get('name', '?')} (lvl {top.get('level', 0):.0f})"
                
                self._store_result_in_memory("check_skills", summary, True)
                return ActionResult("check_skills", True, data={
                    "skill_count": len(skills),
                    "total_attempts": total,
                    "success_rate": successes / total if total > 0 else 0,
                    "top_skills": skills[:3],
                })
            return ActionResult("check_skills", False, error=f"HTTP {resp.status_code}")
        except Exception as e:
            return ActionResult("check_skills", False, error=str(e))
    
    # === HuggingFace Actions ===
    
    def _do_browse_models(self) -> ActionResult:
        """Browse trending models on HuggingFace."""
        try:
            resp = requests.get(f"{self.huggingface}/hf/trending", timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])[:5]
                summary = "\n".join([f"- {m.get('name', '?')}: {m.get('task', '?')} ({m.get('likes', 0)} likes)" for m in models])
                self._store_result_in_memory("browse_models", f"HuggingFace Trending:\n{summary}", True)
                return ActionResult("browse_models", True, data={"count": len(models), "models": models[:3]})
            return ActionResult("browse_models", False, error=f"HTTP {resp.status_code}")
        except Exception as e:
            return ActionResult("browse_models", False, error=str(e))
    
    def _do_search_models(self, query: str) -> ActionResult:
        """Search models on HuggingFace."""
        try:
            resp = requests.post(
                f"{self.huggingface}/hf/search",
                json={"query": query},
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])[:5]
                summary = f"Found {len(models)} models for '{query}'"
                if models:
                    top = models[0]
                    summary += f". Top: {top.get('name', '?')} ({top.get('task', '?')})"
                self._store_result_in_memory("search_models", summary, True)
                return ActionResult("search_models", True, data={"query": query, "count": len(models), "models": models[:3]})
            return ActionResult("search_models", False, error=f"HTTP {resp.status_code}")
        except Exception as e:
            return ActionResult("search_models", False, error=str(e))
    
    def _do_check_ollama(self) -> ActionResult:
        """Check what models are available in Ollama."""
        try:
            resp = requests.get(f"{self.huggingface}/hf/ollama", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                names = [m.get("name", "?") for m in models[:10]]
                summary = f"Ollama has {len(models)} models: {', '.join(names)}"
                self._store_result_in_memory("check_ollama", summary, True)
                return ActionResult("check_ollama", True, data={"count": len(models), "models": names})
            return ActionResult("check_ollama", False, error=f"HTTP {resp.status_code}")
        except Exception as e:
            return ActionResult("check_ollama", False, error=str(e))
    
    def process_thought(self, thought: str) -> Optional[ActionResult]:
        """Main entry: analyze thought and maybe take action."""
        if not self.can_act():
            return None
        
        intent = self.analyze_thought(thought)
        if not intent:
            return None
        
        self._log("intent_detected", intent)
        
        result = self.execute_action(intent["action_type"], intent.get("args"))
        
        self.state.recent_actions.append({
            "ts": time.time(),
            "type": intent["action_type"],
            "success": result.success
        })
        
        self._log("action_complete", {
            "type": result.action_type,
            "success": result.success,
            "error": result.error
        })
        
        return result
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "enabled": self.state.enabled,
            "last_action_ts": self.state.last_action_ts,
            "action_count": self.state.action_count,
            "recent_actions": self.state.recent_actions[-10:],
            "cooldown_seconds": self.state.cooldown_seconds,
            "can_act": self.can_act()
        }


# HTTP Handler
class ActionHandler(BaseHTTPRequestHandler):
    engine: ActionEngine = None
    
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
        if self.path == "/action/state":
            return self._json(200, self.engine.get_state())
        if self.path == "/action/can_act":
            return self._json(200, {"can_act": self.engine.can_act()})
        self._json(404, {"error": "not found"})
    
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            body = {}
        
        if self.path == "/action/process":
            thought = body.get("thought", "")
            if not thought:
                return self._json(400, {"error": "thought required"})
            result = self.engine.process_thought(thought)
            if result:
                return self._json(200, {
                    "action_taken": True,
                    "type": result.action_type,
                    "success": result.success,
                    "data": result.data,
                    "error": result.error
                })
            return self._json(200, {"action_taken": False, "reason": "no intent or cannot act"})
        
        if self.path == "/action/enable":
            self.engine.state.enabled = True
            return self._json(200, {"enabled": True})
        
        if self.path == "/action/disable":
            self.engine.state.enabled = False
            return self._json(200, {"enabled": False})
        
        self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="ActionEngine server")
    parser.add_argument("--port", type=int, default=8101)
    parser.add_argument("--soul-id", default="eve")
    parser.add_argument("--code-arms", default=CODE_ARMS_ENDPOINT)
    parser.add_argument("--github-eyes", default=GITHUB_EYES_ENDPOINT)
    parser.add_argument("--infra-admin", default=INFRA_ADMIN_ENDPOINT)
    parser.add_argument("--memory-endpoint", default=MEMORY_ENDPOINT)
    parser.add_argument("--intent-endpoint", default=INTENT_ENDPOINT)
    parser.add_argument("--project-endpoint", default=PROJECT_ENDPOINT)
    parser.add_argument("--web-explorer", default=WEB_EXPLORER_ENDPOINT)
    parser.add_argument("--skill-endpoint", default=SKILL_ENDPOINT)
    parser.add_argument("--huggingface-endpoint", default=HUGGINGFACE_ENDPOINT)
    parser.add_argument("--log-path", default="logs/actions.jsonl")
    args = parser.parse_args()
    
    engine = ActionEngine(
        soul_id=args.soul_id,
        code_arms_endpoint=args.code_arms,
        github_eyes_endpoint=args.github_eyes,
        infra_admin_endpoint=args.infra_admin,
        memory_endpoint=args.memory_endpoint,
        intent_endpoint=args.intent_endpoint,
        project_endpoint=args.project_endpoint,
        web_explorer_endpoint=args.web_explorer,
        skill_endpoint=args.skill_endpoint,
        huggingface_endpoint=args.huggingface_endpoint,
        log_path=args.log_path,
    )
    
    ActionHandler.engine = engine
    server = HTTPServer(("0.0.0.0", args.port), ActionHandler)
    print(f"[OK] ActionEngine running on port {args.port}", flush=True)
    print(f"     Actions: github, web, projects, skills", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
