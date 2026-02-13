#!/usr/bin/env python3
"""InfraAdmin: Docker-compose + system resource monitor."""

import argparse
import json
import os
import shutil
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any

from umc_core.gpu_monitor import GPUMonitor


class InfraAdmin:
    def __init__(self, compose_dir: str = "."):
        self.compose_dir = compose_dir
        self.gpu = GPUMonitor()

    def _run_compose(self, args: list) -> str:
        cmd = ["docker", "compose"] + args
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=self.compose_dir, check=False)
        if proc.returncode != 0:
            return proc.stderr.strip()
        return proc.stdout.strip()

    def get_services_status(self) -> Dict[str, Any]:
        out = self._run_compose(["ps", "--format", "json"])
        try:
            data = json.loads(out)
            return {item.get("Service", "unknown"): item for item in data}
        except Exception:
            # fallback parse plain table
            out = self._run_compose(["ps"])
            lines = [l for l in out.splitlines() if l.strip()]
            status = {}
            if len(lines) >= 2:
                for line in lines[1:]:
                    parts = line.split()
                    if not parts:
                        continue
                    status[parts[0]] = {"raw": line}
            return status

    def restart_service(self, name: str) -> bool:
        out = self._run_compose(["restart", name])
        return "error" not in out.lower()

    def get_logs(self, name: str, lines: int = 100) -> str:
        return self._run_compose(["logs", "--tail", str(lines), name])

    def health_check(self) -> Dict[str, Any]:
        return {"services": self.get_services_status(), "gpu_healthy": self.gpu.is_healthy()}

    def get_system_resources(self) -> Dict[str, Any]:
        # CPU load
        load1, load5, load15 = os.getloadavg() if hasattr(os, "getloadavg") else (0, 0, 0)
        # RAM
        mem_total = 0
        mem_free = 0
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_total = int(line.split()[1]) // 1024
                    if line.startswith("MemAvailable:"):
                        mem_free = int(line.split()[1]) // 1024
        except Exception:
            pass
        # Disk
        disk = shutil.disk_usage(self.compose_dir)
        return {
            "load": {"1m": load1, "5m": load5, "15m": load15},
            "memory_mb": {"total": mem_total, "available": mem_free},
            "disk_gb": {
                "total": round(disk.total / 1e9, 2),
                "used": round(disk.used / 1e9, 2),
                "free": round(disk.free / 1e9, 2),
            },
        }


class InfraHandler(BaseHTTPRequestHandler):
    admin: InfraAdmin = None

    def _json(self, code: int, payload: Dict[str, Any]):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/infra/status":
            return self._json(200, self.admin.health_check())
        if self.path == "/infra/gpus":
            return self._json(200, {"gpus": self.admin.gpu.get_gpus()})
        if self.path.startswith("/infra/logs/"):
            service = self.path.split("/", 3)[-1]
            return self._json(200, {"logs": self.admin.get_logs(service)})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return self._json(400, {"error": "invalid JSON"})
        if self.path == "/infra/restart":
            service = data.get("service")
            if not service:
                return self._json(400, {"error": "service required"})
            ok = self.admin.restart_service(service)
            return self._json(200, {"ok": ok})
        return self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="InfraAdmin service")
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--compose-dir", type=str, default=".")
    args = parser.parse_args()

    admin = InfraAdmin(compose_dir=args.compose_dir)
    InfraHandler.admin = admin
    server = HTTPServer(("0.0.0.0", args.port), InfraHandler)
    print(f"InfraAdmin listening on :{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
