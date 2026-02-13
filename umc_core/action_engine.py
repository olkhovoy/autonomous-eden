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
OLLAMA_GENERATE_URL = os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")

# Action patterns (what EVE might want to do)
ACTION_PATTERNS = {
    "github_trending": [
        r"trending.*repos?", r"popular.*github", r"what.*new.*github",
        r"explore.*github", r"discover.*projects?"
    ],
    "github_search": [
        r"search.*github.*for\s+([^\s]+)", r"find.*repo.*about\s+(\w+)",
        r"look.*for.*(\w+).*on.*github"
    ],
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
    ]
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
        log_path: str = "logs/actions.jsonl",
    ):
        self.soul_id = soul_id
        self.code_arms = code_arms_endpoint.rstrip("/")
        self.github_eyes = github_eyes_endpoint.rstrip("/")
        self.infra_admin = infra_admin_endpoint.rstrip("/")
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.intent_endpoint = intent_endpoint.rstrip("/")
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
    parser.add_argument("--log-path", default="logs/actions.jsonl")
    args = parser.parse_args()
    
    engine = ActionEngine(
        soul_id=args.soul_id,
        code_arms_endpoint=args.code_arms,
        github_eyes_endpoint=args.github_eyes,
        infra_admin_endpoint=args.infra_admin,
        memory_endpoint=args.memory_endpoint,
        intent_endpoint=args.intent_endpoint,
        log_path=args.log_path,
    )
    
    ActionHandler.engine = engine
    server = HTTPServer(("0.0.0.0", args.port), ActionHandler)
    print(f"[OK] ActionEngine running on port {args.port}", flush=True)
    print(f"     Endpoints: /action/state, /action/process, /action/enable", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
