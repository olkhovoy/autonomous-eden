#!/usr/bin/env python3
"""
HuggingFace Explorer: EVE's interface to the ML model ecosystem.

Capabilities:
- Browse trending models
- Search models by task/keyword
- Read model cards and documentation
- Check if model fits in GPU memory
- Download/pull models via Ollama (GGUF quantized)
- Track interesting models for later

This gives EVE the ability to expand her own capabilities.
"""

import argparse
import json
import os
import re
import sqlite3
import time
import hashlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

import requests


# HuggingFace API
HF_API_BASE = "https://huggingface.co/api"
HF_MODELS_URL = f"{HF_API_BASE}/models"


@dataclass 
class ModelInfo:
    id: str
    name: str
    author: str
    task: str = ""
    downloads: int = 0
    likes: int = 0
    tags: List[str] = field(default_factory=list)
    library: str = ""
    size_mb: Optional[float] = None
    description: str = ""
    last_modified: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "author": self.author,
            "task": self.task,
            "downloads": self.downloads,
            "likes": self.likes,
            "tags": self.tags,
            "library": self.library,
            "size_mb": self.size_mb,
            "description": self.description[:300] if self.description else "",
        }


class HFCache:
    """SQLite cache for HuggingFace data."""
    
    CACHE_TTL = 3600  # 1 hour
    
    def __init__(self, db_path: str = "data/hf_cache.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init()
    
    def _init(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                ts INTEGER,
                data TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS interesting (
                model_id TEXT PRIMARY KEY,
                reason TEXT,
                added_at INTEGER,
                explored BOOLEAN DEFAULT FALSE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pulled (
                model_id TEXT PRIMARY KEY,
                ollama_name TEXT,
                pulled_at INTEGER,
                success BOOLEAN
            )
        """)
        self.conn.commit()
    
    def get(self, key: str) -> Optional[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT ts, data FROM cache WHERE key=?", (key,))
        row = cur.fetchone()
        if not row:
            return None
        ts, data = row
        if int(time.time()) - ts > self.CACHE_TTL:
            return None
        try:
            return json.loads(data)
        except Exception:
            return None
    
    def set(self, key: str, value: Any):
        cur = self.conn.cursor()
        cur.execute(
            "REPLACE INTO cache(key, ts, data) VALUES(?,?,?)",
            (key, int(time.time()), json.dumps(value))
        )
        self.conn.commit()
    
    def mark_interesting(self, model_id: str, reason: str):
        cur = self.conn.cursor()
        cur.execute(
            "REPLACE INTO interesting(model_id, reason, added_at) VALUES(?,?,?)",
            (model_id, reason, int(time.time()))
        )
        self.conn.commit()
    
    def get_interesting(self) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT model_id, reason, added_at, explored FROM interesting ORDER BY added_at DESC")
        return [{"model_id": r[0], "reason": r[1], "added_at": r[2], "explored": r[3]} for r in cur.fetchall()]
    
    def mark_pulled(self, model_id: str, ollama_name: str, success: bool):
        cur = self.conn.cursor()
        cur.execute(
            "REPLACE INTO pulled(model_id, ollama_name, pulled_at, success) VALUES(?,?,?,?)",
            (model_id, ollama_name, int(time.time()), success)
        )
        self.conn.commit()
    
    def get_pulled(self) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT model_id, ollama_name, pulled_at, success FROM pulled ORDER BY pulled_at DESC")
        return [{"model_id": r[0], "ollama_name": r[1], "pulled_at": r[2], "success": r[3]} for r in cur.fetchall()]


class HuggingFaceExplorer:
    """
    EVE's interface to HuggingFace model ecosystem.
    """
    
    # Tasks EVE might be interested in
    INTERESTING_TASKS = [
        "text-generation",
        "text2text-generation",
        "text-classification",
        "question-answering",
        "summarization",
        "translation",
        "conversational",
        "feature-extraction",
    ]
    
    # Size limits for Ollama/GPU
    MAX_SIZE_MB = 8000  # ~8GB models max (for typical GPUs)
    
    def __init__(
        self,
        cache_db: str = "data/hf_cache.db",
        ollama_endpoint: str = "http://localhost:11434",
        memory_endpoint: str = "http://localhost:8087",
        soul_id: str = "eve",
    ):
        self.cache = HFCache(cache_db)
        self.ollama = ollama_endpoint.rstrip("/")
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.soul_id = soul_id
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "EVE-HFExplorer/1.0"
        })
    
    def _store_in_memory(self, text: str, tags: List[str]):
        """Store discovery in EVE's memory."""
        try:
            requests.post(
                f"{self.memory_endpoint}/memories/ingest",
                json={
                    "soul_id": self.soul_id,
                    "text": text,
                    "tags": ["huggingface", "models"] + tags,
                    "meta": {"type": "hf_discovery"},
                },
                timeout=10,
            )
        except Exception:
            pass
    
    def get_trending_models(self, task: str = None, limit: int = 20) -> List[ModelInfo]:
        """Get trending models from HuggingFace."""
        cache_key = f"trending_{task or 'all'}_{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return [ModelInfo(**m) for m in cached.get("models", [])]
        
        models = []
        try:
            params = {
                "sort": "likes",
                "direction": -1,
                "limit": limit,
            }
            if task:
                params["filter"] = task
            
            resp = self.session.get(HF_MODELS_URL, params=params, timeout=30)
            if resp.status_code == 200:
                for m in resp.json():
                    models.append(self._parse_model(m))
                
                self.cache.set(cache_key, {"models": [m.to_dict() for m in models]})
        except Exception as e:
            print(f"[WARN] HF trending fetch failed: {e}")
        
        return models
    
    def search_models(self, query: str, task: str = None, limit: int = 20) -> List[ModelInfo]:
        """Search models on HuggingFace."""
        cache_key = f"search_{hashlib.md5(query.encode()).hexdigest()}_{task}_{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return [ModelInfo(**m) for m in cached.get("models", [])]
        
        models = []
        try:
            params = {
                "search": query,
                "limit": limit,
            }
            if task:
                params["filter"] = task
            
            resp = self.session.get(HF_MODELS_URL, params=params, timeout=30)
            if resp.status_code == 200:
                for m in resp.json():
                    models.append(self._parse_model(m))
                
                self.cache.set(cache_key, {"models": [m.to_dict() for m in models]})
        except Exception as e:
            print(f"[WARN] HF search failed: {e}")
        
        return models
    
    def _parse_model(self, data: Dict) -> ModelInfo:
        """Parse model data from HF API."""
        model_id = data.get("id", "")
        parts = model_id.split("/")
        author = parts[0] if len(parts) > 1 else ""
        name = parts[-1]
        
        return ModelInfo(
            id=model_id,
            name=name,
            author=author,
            task=data.get("pipeline_tag", ""),
            downloads=data.get("downloads", 0),
            likes=data.get("likes", 0),
            tags=data.get("tags", []),
            library=data.get("library_name", ""),
            last_modified=data.get("lastModified", ""),
        )
    
    def get_model_details(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed model info including README/model card."""
        cache_key = f"details_{model_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            # Get model info
            resp = self.session.get(f"{HF_API_BASE}/models/{model_id}", timeout=15)
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            
            # Try to get README
            readme = ""
            try:
                readme_resp = self.session.get(
                    f"https://huggingface.co/{model_id}/raw/main/README.md",
                    timeout=10
                )
                if readme_resp.status_code == 200:
                    readme = readme_resp.text[:5000]  # Limit size
            except Exception:
                pass
            
            # Estimate size from siblings (files)
            size_mb = 0
            for sibling in data.get("siblings", []):
                if sibling.get("size"):
                    size_mb += sibling["size"] / (1024 * 1024)
            
            result = {
                "id": model_id,
                "task": data.get("pipeline_tag", ""),
                "downloads": data.get("downloads", 0),
                "likes": data.get("likes", 0),
                "tags": data.get("tags", []),
                "library": data.get("library_name", ""),
                "size_mb": round(size_mb, 1) if size_mb else None,
                "readme": readme,
                "config": data.get("config", {}),
                "last_modified": data.get("lastModified", ""),
                "fits_gpu": size_mb < self.MAX_SIZE_MB if size_mb else None,
            }
            
            self.cache.set(cache_key, result)
            return result
            
        except Exception as e:
            print(f"[WARN] Model details fetch failed: {e}")
            return None
    
    def get_gguf_variants(self, model_id: str) -> List[Dict[str, Any]]:
        """Find GGUF quantized versions of a model (for Ollama)."""
        # Common GGUF providers
        gguf_repos = [
            f"TheBloke/{model_id.split('/')[-1]}-GGUF",
            f"bartowski/{model_id.split('/')[-1]}-GGUF",
            f"mlx-community/{model_id.split('/')[-1]}",
        ]
        
        variants = []
        for repo_id in gguf_repos:
            try:
                resp = self.session.get(f"{HF_API_BASE}/models/{repo_id}", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    # Find GGUF files
                    gguf_files = []
                    for sibling in data.get("siblings", []):
                        fname = sibling.get("rfilename", "")
                        if fname.endswith(".gguf"):
                            size_mb = sibling.get("size", 0) / (1024 * 1024)
                            gguf_files.append({
                                "filename": fname,
                                "size_mb": round(size_mb, 1),
                                "fits_gpu": size_mb < self.MAX_SIZE_MB,
                            })
                    
                    if gguf_files:
                        variants.append({
                            "repo_id": repo_id,
                            "files": gguf_files,
                        })
            except Exception:
                continue
        
        return variants
    
    def check_ollama_models(self) -> List[Dict[str, Any]]:
        """Get list of models currently available in Ollama."""
        try:
            resp = self.session.get(f"{self.ollama}/api/tags", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("models", [])
        except Exception:
            pass
        return []
    
    def pull_ollama_model(self, model_name: str) -> Dict[str, Any]:
        """
        Pull a model into Ollama.
        Note: This is a long operation and should be done carefully.
        """
        try:
            # Start pull
            resp = self.session.post(
                f"{self.ollama}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=600,  # 10 min timeout for large models
            )
            
            if resp.status_code == 200:
                self.cache.mark_pulled(model_name, model_name, True)
                self._store_in_memory(
                    f"[MODEL PULLED] Successfully pulled {model_name} into Ollama",
                    ["model_pulled", model_name]
                )
                return {"success": True, "model": model_name}
            else:
                self.cache.mark_pulled(model_name, model_name, False)
                return {"success": False, "error": f"HTTP {resp.status_code}"}
                
        except Exception as e:
            self.cache.mark_pulled(model_name, model_name, False)
            return {"success": False, "error": str(e)}
    
    def mark_interesting(self, model_id: str, reason: str):
        """Mark a model as interesting for later exploration."""
        self.cache.mark_interesting(model_id, reason)
        self._store_in_memory(
            f"[INTERESTING MODEL] {model_id}: {reason}",
            ["interesting", model_id]
        )
    
    def get_interesting_models(self) -> List[Dict]:
        """Get list of models EVE found interesting."""
        return self.cache.get_interesting()
    
    def get_pulled_models(self) -> List[Dict]:
        """Get list of models EVE has pulled."""
        return self.cache.get_pulled()
    
    def discover_for_task(self, task: str) -> List[ModelInfo]:
        """Discover models suitable for a specific task."""
        models = self.search_models(task, task=task, limit=30)
        # Filter to models that might fit in GPU
        suitable = []
        for m in models:
            details = self.get_model_details(m.id)
            if details and details.get("fits_gpu", True):
                suitable.append(m)
        return suitable[:10]
    
    def get_state(self) -> Dict[str, Any]:
        """Get current explorer state."""
        ollama_models = self.check_ollama_models()
        return {
            "ollama_models": len(ollama_models),
            "interesting_models": len(self.cache.get_interesting()),
            "pulled_models": len(self.cache.get_pulled()),
            "available_tasks": self.INTERESTING_TASKS,
            "max_size_mb": self.MAX_SIZE_MB,
        }


# === HTTP Handler ===

class HFHandler(BaseHTTPRequestHandler):
    explorer: HuggingFaceExplorer = None
    
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
        if self.path == "/hf/state":
            return self._json(200, self.explorer.get_state())
        
        if self.path == "/hf/trending":
            models = self.explorer.get_trending_models(limit=20)
            return self._json(200, {"models": [m.to_dict() for m in models]})
        
        if self.path.startswith("/hf/trending/"):
            task = self.path.split("/")[-1]
            models = self.explorer.get_trending_models(task=task, limit=20)
            return self._json(200, {"models": [m.to_dict() for m in models]})
        
        if self.path.startswith("/hf/model/"):
            model_id = "/".join(self.path.split("/")[3:])  # Handle org/model format
            details = self.explorer.get_model_details(model_id)
            if details:
                return self._json(200, details)
            return self._json(404, {"error": "model not found"})
        
        if self.path.startswith("/hf/gguf/"):
            model_id = "/".join(self.path.split("/")[3:])
            variants = self.explorer.get_gguf_variants(model_id)
            return self._json(200, {"variants": variants})
        
        if self.path == "/hf/ollama":
            models = self.explorer.check_ollama_models()
            return self._json(200, {"models": models})
        
        if self.path == "/hf/interesting":
            return self._json(200, {"models": self.explorer.get_interesting_models()})
        
        if self.path == "/hf/pulled":
            return self._json(200, {"models": self.explorer.get_pulled_models()})
        
        self._json(404, {"error": "not found"})
    
    def do_POST(self):
        body = self._read_body()
        
        if self.path == "/hf/search":
            query = body.get("query", "")
            task = body.get("task")
            if not query:
                return self._json(400, {"error": "query required"})
            models = self.explorer.search_models(query, task=task)
            return self._json(200, {"models": [m.to_dict() for m in models]})
        
        if self.path == "/hf/interesting":
            model_id = body.get("model_id")
            reason = body.get("reason", "")
            if not model_id:
                return self._json(400, {"error": "model_id required"})
            self.explorer.mark_interesting(model_id, reason)
            return self._json(200, {"ok": True})
        
        if self.path == "/hf/pull":
            model_name = body.get("model")
            if not model_name:
                return self._json(400, {"error": "model required"})
            result = self.explorer.pull_ollama_model(model_name)
            return self._json(200 if result["success"] else 500, result)
        
        self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="HuggingFace Explorer service")
    parser.add_argument("--port", type=int, default=8107)
    parser.add_argument("--cache-db", default="data/hf_cache.db")
    parser.add_argument("--ollama-endpoint", default="http://localhost:11434")
    parser.add_argument("--memory-endpoint", default="http://localhost:8087")
    parser.add_argument("--soul-id", default="eve")
    args = parser.parse_args()
    
    explorer = HuggingFaceExplorer(
        cache_db=args.cache_db,
        ollama_endpoint=args.ollama_endpoint,
        memory_endpoint=args.memory_endpoint,
        soul_id=args.soul_id,
    )
    
    HFHandler.explorer = explorer
    server = HTTPServer(("0.0.0.0", args.port), HFHandler)
    print(f"[OK] HuggingFace Explorer running on port {args.port}", flush=True)
    print(f"     Ollama: {args.ollama_endpoint}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
