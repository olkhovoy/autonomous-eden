#!/usr/bin/env python3
"""CodeArms: controlled execution + file/git operations with audit log."""

import argparse
import json
import os
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any

from umc_core.sandbox import Sandbox


class RateLimiter:
    def __init__(self, max_per_minute: int = 10):
        self.max = max_per_minute
        self.events = deque()

    def allow(self) -> bool:
        now = time.time()
        while self.events and now - self.events[0] > 60:
            self.events.popleft()
        if len(self.events) >= self.max:
            return False
        self.events.append(now)
        return True


class CodeArms:
    def __init__(self, workspace_dir: str = "workspace", log_path: str = "logs/code_arms.jsonl"):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.log_path = log_path
        self.sandbox = Sandbox(workspace_dir=self.workspace_dir)
        self.limiter = RateLimiter(max_per_minute=10)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _log(self, action: str, payload: Dict[str, Any]):
        rec = {"ts": time.time(), "action": action, **payload}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _check_rate(self):
        if not self.limiter.allow():
            raise RuntimeError("rate limit exceeded")

    def execute_shell(self, cmd: str, timeout: int = 30) -> Dict[str, Any]:
        self._check_rate()
        result = self.sandbox.execute(cmd, timeout=timeout)
        self._log("execute_shell", {"cmd": cmd, **result})
        return result

    def read_file(self, path: str) -> str:
        self._check_rate()
        full = path if os.path.isabs(path) else os.path.join(self.workspace_dir, path)
        with open(full, "r", encoding="utf-8") as f:
            data = f.read()
        self._log("read_file", {"path": full, "bytes": len(data)})
        return data

    def write_file(self, path: str, content: str) -> bool:
        self._check_rate()
        full = path if os.path.isabs(path) else os.path.join(self.workspace_dir, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        self._log("write_file", {"path": full, "bytes": len(content)})
        return True

    def git_clone(self, repo_url: str) -> str:
        self._check_rate()
        name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        dest = os.path.join(self.workspace_dir, name)
        cmd = f"git clone {repo_url} {dest}"
        result = self.sandbox.execute(cmd, timeout=120)
        self._log("git_clone", {"url": repo_url, "dest": dest, **result})
        if result["exit_code"] != 0:
            raise RuntimeError(result["stderr"] or "git clone failed")
        return dest

    def git_status(self) -> Dict[str, Any]:
        self._check_rate()
        result = self.sandbox.execute(f"cd /workspace && git status --porcelain", timeout=30)
        self._log("git_status", result)
        return result

    def git_commit(self, message: str) -> str:
        self._check_rate()
        result = self.sandbox.execute(f"cd /workspace && git add -A && git commit -m {json.dumps(message)}", timeout=60)
        self._log("git_commit", {"message": message, **result})
        if result["exit_code"] != 0:
            raise RuntimeError(result["stderr"] or "git commit failed")
        return result["stdout"].strip()

    def run_tests(self, path: str) -> Dict[str, Any]:
        self._check_rate()
        full = path if os.path.isabs(path) else os.path.join("/workspace", path)
        cmd = f"cd {full} && python -m pytest"
        result = self.sandbox.execute(cmd, timeout=300)
        passed = result["exit_code"] == 0
        payload = {"passed": passed, "failed": not passed, "output": result["stdout"] + result["stderr"]}
        self._log("run_tests", {"path": full, **payload})
        return payload


class CodeArmsHandler(BaseHTTPRequestHandler):
    arms: CodeArms = None

    def _json(self, code: int, payload: Dict[str, Any]):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})

        try:
            if self.path == "/code/execute":
                cmd = data.get("cmd", "")
                return self._json(200, self.arms.execute_shell(cmd))
            if self.path == "/code/file/read":
                path = data.get("path", "")
                return self._json(200, {"content": self.arms.read_file(path)})
            if self.path == "/code/file/write":
                path = data.get("path", "")
                content = data.get("content", "")
                ok = self.arms.write_file(path, content)
                return self._json(200, {"ok": ok})
            if self.path == "/code/git/clone":
                url = data.get("url", "")
                dest = self.arms.git_clone(url)
                return self._json(200, {"path": dest})
        except Exception as e:
            return self._json(500, {"error": str(e)})

        return self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="CodeArms service")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--workspace", type=str, default="workspace")
    parser.add_argument("--log-path", type=str, default="logs/code_arms.jsonl")
    args = parser.parse_args()

    arms = CodeArms(workspace_dir=args.workspace, log_path=args.log_path)
    CodeArmsHandler.arms = arms
    server = HTTPServer(("0.0.0.0", args.port), CodeArmsHandler)
    print(f"CodeArms listening on :{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
