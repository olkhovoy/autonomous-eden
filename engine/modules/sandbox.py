#!/usr/bin/env python3
"""Sandbox: Docker-based isolated execution."""

import os
import subprocess
import uuid
from typing import Dict, Optional


class Sandbox:
    def __init__(self, workspace_dir: str = "workspace", image: str = "python:3.11-slim"):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.image = image
        self.container_name = f"umc_sandbox_{uuid.uuid4().hex[:8]}"
        self.running = False

    def start(self):
        os.makedirs(self.workspace_dir, exist_ok=True)
        if self.running:
            return
        # Best-effort network restriction: default to none. Allowlist not enforced here.
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", self.container_name,
            "--cpus", "2",
            "--memory", "4g",
            "--network", "none",
            "-v", f"{self.workspace_dir}:/workspace",
            self.image,
            "sleep", "infinity",
        ]
        subprocess.run(cmd, check=True)
        self.running = True

    def stop(self):
        if not self.running:
            return
        subprocess.run(["docker", "rm", "-f", self.container_name], check=False)
        self.running = False

    def execute(self, cmd: str, timeout: int = 30) -> Dict[str, Optional[str]]:
        if not self.running:
            self.start()
        exec_cmd = ["docker", "exec", self.container_name, "sh", "-lc", cmd]
        proc = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        }
