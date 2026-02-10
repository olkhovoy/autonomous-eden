#!/usr/bin/env python3
"""InfraAdmin: Service health monitor + system resources.

Works without docker access by checking services via HTTP.
"""

import argparse
import json
import os
import shutil
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any, List

import requests


# Services to monitor (name -> url for GET request)
SERVICES = {
    "soul_memory": "http://soul_memory:8087/",  # Just check if responding
    "intent_engine": "http://intent_engine:8089/intent/state",
    "gggp_bridge": "http://gggp_bridge:8091/gggp/state",
    "lifecycle_manager": "http://lifecycle_manager:8093/lifecycle/state",
    "code_arms": "http://code_arms:8094/code/status",
    "github_eyes": "http://github_eyes:8095/",
    "action_engine": "http://action_engine:8101/action/state",
    "project_manager": "http://project_manager:8102/projects",
    "web_explorer": "http://web_explorer:8103/web/state",
    "skill_learner": "http://skill_learner:8105/skills",
    "self_modifier": "http://self_modifier:8106/modify/state",
    "huggingface_explorer": "http://huggingface_explorer:8107/hf/state",
}


class InfraAdmin:
    def __init__(self, compose_dir: str = ".", ollama_url: str = "http://localhost:11434"):
        self.compose_dir = compose_dir
        self.ollama_url = ollama_url
        self.service_status_cache = {}
        self.last_check = 0
        self.check_interval = 30  # seconds

    def check_service(self, name: str, url: str) -> Dict[str, Any]:
        """Check if a service is responding."""
        try:
            start = time.time()
            resp = requests.get(url, timeout=5)
            latency = (time.time() - start) * 1000
            # 501 means server is running but doesn't support GET on that path
            # 404 means endpoint exists but path not found
            online = resp.status_code in [200, 404, 501]
            return {
                "name": name,
                "online": online,
                "status_code": resp.status_code,
                "latency_ms": round(latency, 1),
            }
        except requests.exceptions.Timeout:
            return {"name": name, "online": False, "error": "timeout"}
        except requests.exceptions.ConnectionError:
            return {"name": name, "online": False, "error": "connection refused"}
        except Exception as e:
            return {"name": name, "online": False, "error": str(e)[:50]}

    def get_services_status(self) -> Dict[str, Any]:
        """Get status of all monitored services."""
        now = time.time()
        if now - self.last_check < self.check_interval and self.service_status_cache:
            return self.service_status_cache
        
        status = {}
        for name, url in SERVICES.items():
            status[name] = self.check_service(name, url)
        
        self.service_status_cache = status
        self.last_check = now
        return status

    def get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU status from Ollama."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return {
                    "available": True,
                    "models_loaded": len(models),
                    "model_names": [m.get("name", "?") for m in models[:5]],
                }
        except Exception:
            pass
        return {"available": False}

    def get_system_resources(self) -> Dict[str, Any]:
        """Get system resource usage."""
        # CPU load
        load1, load5, load15 = os.getloadavg() if hasattr(os, "getloadavg") else (0, 0, 0)
        
        # RAM
        mem_total = 0
        mem_available = 0
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_total = int(line.split()[1]) // 1024
                    if line.startswith("MemAvailable:"):
                        mem_available = int(line.split()[1]) // 1024
        except Exception:
            pass
        
        # Disk
        try:
            disk = shutil.disk_usage(self.compose_dir)
            disk_info = {
                "total_gb": round(disk.total / 1e9, 2),
                "used_gb": round(disk.used / 1e9, 2),
                "free_gb": round(disk.free / 1e9, 2),
                "used_percent": round(disk.used / disk.total * 100, 1),
            }
        except Exception:
            disk_info = {}
        
        return {
            "cpu_load": {"1m": round(load1, 2), "5m": round(load5, 2), "15m": round(load15, 2)},
            "memory_mb": {
                "total": mem_total,
                "available": mem_available,
                "used_percent": round((mem_total - mem_available) / mem_total * 100, 1) if mem_total > 0 else 0,
            },
            "disk": disk_info,
        }

    def health_check(self) -> Dict[str, Any]:
        """Full health check."""
        services = self.get_services_status()
        online_count = sum(1 for s in services.values() if s.get("online"))
        total_count = len(services)
        
        return {
            "healthy": online_count >= total_count * 0.8,  # 80% threshold
            "services_online": online_count,
            "services_total": total_count,
            "services": services,
            "gpu": self.get_gpu_status(),
            "system": self.get_system_resources(),
        }


class InfraHandler(BaseHTTPRequestHandler):
    admin: InfraAdmin = None

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
        if self.path == "/infra/status":
            return self._json(200, self.admin.health_check())
        
        if self.path == "/infra/services":
            return self._json(200, {"services": self.admin.get_services_status()})
        
        if self.path == "/infra/gpu":
            return self._json(200, self.admin.get_gpu_status())
        
        if self.path == "/infra/resources":
            return self._json(200, self.admin.get_system_resources())
        
        if self.path == "/infra/health":
            health = self.admin.health_check()
            return self._json(200, {"healthy": health["healthy"], "online": health["services_online"], "total": health["services_total"]})
        
        self._json(404, {"error": "not found"})

    def log_message(self, format, *args):
        pass  # Suppress logging


def main():
    parser = argparse.ArgumentParser(description="InfraAdmin service monitor")
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--compose-dir", default="/app")
    parser.add_argument("--ollama-url", default="http://10.1.1.7:11434")
    args = parser.parse_args()

    admin = InfraAdmin(compose_dir=args.compose_dir, ollama_url=args.ollama_url)
    InfraHandler.admin = admin
    
    server = HTTPServer(("0.0.0.0", args.port), InfraHandler)
    print(f"[OK] InfraAdmin running on port {args.port}", flush=True)
    print(f"     Monitoring {len(SERVICES)} services", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
