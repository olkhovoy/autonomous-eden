#!/usr/bin/env python3
"""
SoulMemoryNode: Autobiographical episodic memory service.

Uses an embedding model (via Ollama) to convert conversation logs into
"Experience Vectors". Stores them in Qdrant under a Soul/Identity ID,
with saliency-weighted retention and time decay.
"""

import argparse
import os
import json
import math
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import requests


# -----------------------------
# Data model
# -----------------------------

@dataclass
class MemoryObject:
    """Canonical memory record stored in the vector DB payload."""
    id: str
    soul_id: str
    text: str
    vector: List[float]
    saliency: float
    created_at: float
    last_accessed: float
    strength: float
    tags: List[str]
    meta: Dict[str, Any]


# -----------------------------
# Embedding client (Ollama)
# -----------------------------

class OllamaEmbeddingClient:
    def __init__(self, model: str = "nomic-embed-text:latest", endpoint: str = "http://localhost:11434/api/embeddings"):
        self.model = model
        self.endpoint = endpoint

    def embed(self, text: str) -> List[float]:
        resp = requests.post(self.endpoint, json={"model": self.model, "prompt": text}, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama embeddings error {resp.status_code}: {resp.text}")
        data = resp.json()
        vec = data.get("embedding")
        if not vec:
            raise RuntimeError("No embedding returned from Ollama")
        return vec


# -----------------------------
# Saliency router
# -----------------------------

class EmotionalSaliencyRouter:
    """
    Heuristic saliency: surprise, pain/error, high utility.
    Returns scalar in [0, 1].
    """
    def __init__(self):
        self.surprise_terms = ["unexpected", "suddenly", "shock", "surprise", "astonish", "unbelievable"]
        self.pain_terms = ["error", "fail", "wrong", "crash", "bug", "fix", "correct", "warning"]
        self.utility_terms = ["important", "remember", "note", "key", "critical", "must", "urgent"]

    def score(self, text: str) -> float:
        t = text.lower()
        surprise = sum(1 for w in self.surprise_terms if w in t)
        pain = sum(1 for w in self.pain_terms if w in t)
        utility = sum(1 for w in self.utility_terms if w in t)
        raw = 0.1 + 0.25 * surprise + 0.35 * pain + 0.3 * utility
        return max(0.05, min(1.0, raw))


# -----------------------------
# Qdrant HTTP client (minimal)
# -----------------------------

class QdrantHTTP:
    def __init__(self, base_url: str = "http://localhost:6333"):
        self.base_url = base_url.rstrip("/")

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def ensure_collection(self, name: str, vector_size: int, distance: str = "Cosine"):
        info = requests.get(self._url(f"/collections/{name}"), timeout=10)
        if info.status_code == 200:
            return
        payload = {
            "vectors": {"size": vector_size, "distance": distance},
        }
        resp = requests.put(self._url(f"/collections/{name}"), json=payload, timeout=30)
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Qdrant create collection failed: {resp.status_code} {resp.text}")

    def upsert(self, name: str, point_id: str, vector: List[float], payload: Dict[str, Any]):
        data = {"points": [{"id": point_id, "vector": vector, "payload": payload}]}
        resp = requests.put(self._url(f"/collections/{name}/points"), json=data, timeout=30)
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Qdrant upsert failed: {resp.status_code} {resp.text}")

    def search(self, name: str, vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        data = {"vector": vector, "limit": limit, "with_payload": True}
        resp = requests.post(self._url(f"/collections/{name}/points/search"), json=data, timeout=30)
        if resp.status_code == 404:
            return []
        if resp.status_code != 200:
            raise RuntimeError(f"Qdrant search failed: {resp.status_code} {resp.text}")
        return resp.json().get("result", [])

    def scroll(
        self,
        name: str,
        limit: int = 100,
        offset: Optional[Dict[str, Any]] = None,
        with_vectors: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        data = {"limit": limit, "with_payload": True, "with_vectors": with_vectors}
        if offset:
            data["offset"] = offset
        resp = requests.post(self._url(f"/collections/{name}/points/scroll"), json=data, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Qdrant scroll failed: {resp.status_code} {resp.text}")
        result = resp.json().get("result", {})
        return result.get("points", []), result.get("next_page_offset")

    def update_payload(self, name: str, point_id: str, payload: Dict[str, Any]):
        data = {"payload": payload, "points": [point_id]}
        resp = requests.post(self._url(f"/collections/{name}/points/payload"), json=data, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Qdrant payload update failed: {resp.status_code} {resp.text}")


# -----------------------------
# SoulMemoryNode
# -----------------------------

class SoulMemoryNode:
    def __init__(
        self,
        embedding_model: str,
        qdrant_url: str = "http://localhost:6333",
        decay_rate: float = 1.0 / (7 * 24 * 3600),
        embedding_endpoint: str = "http://localhost:11434/api/embeddings",
    ):
        self.embedder = OllamaEmbeddingClient(model=embedding_model, endpoint=embedding_endpoint)
        self.router = EmotionalSaliencyRouter()
        self.qdrant = QdrantHTTP(qdrant_url)
        self.decay_rate = decay_rate

    def _collection(self, soul_id: str) -> str:
        return f"soul_{soul_id}"

    def ingest(self, soul_id: str, text: str, tags: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None) -> MemoryObject:
        tags = tags or []
        meta = meta or {}
        vec = self.embedder.embed(text)
        saliency = self.router.score(text)
        now = time.time()
        mem_id = str(uuid.uuid4())
        mem = MemoryObject(
            id=mem_id,
            soul_id=soul_id,
            text=text,
            vector=vec,
            saliency=saliency,
            created_at=now,
            last_accessed=now,
            strength=saliency,
            tags=tags,
            meta=meta,
        )
        self.qdrant.ensure_collection(self._collection(soul_id), vector_size=len(vec))
        payload = asdict(mem).copy()
        payload.pop("vector")
        self.qdrant.upsert(self._collection(soul_id), mem_id, vec, payload)
        return mem

    def query(self, soul_id: str, query_text: str, limit: int = 5) -> List[MemoryObject]:
        vec = self.embedder.embed(query_text)
        results = self.qdrant.search(self._collection(soul_id), vec, limit=limit)
        memories = []
        now = time.time()
        for r in results:
            payload = r.get("payload", {})
            mem = MemoryObject(
                id=str(r.get("id")),
                soul_id=soul_id,
                text=payload.get("text", ""),
                vector=r.get("vector", vec),
                saliency=float(payload.get("saliency", 0.1)),
                created_at=float(payload.get("created_at", now)),
                last_accessed=now,
                strength=float(payload.get("strength", 0.1)),
                tags=payload.get("tags", []),
                meta=payload.get("meta", {}),
            )
            # reinforce on access
            mem.strength = min(1.0, mem.strength + 0.05 * mem.saliency)
            payload["last_accessed"] = now
            payload["strength"] = mem.strength
            self.qdrant.update_payload(self._collection(soul_id), mem.id, payload)
            memories.append(mem)
        return memories

    def decay(self, soul_id: str):
        """Decay memory strength over time unless recently accessed."""
        collection = self._collection(soul_id)
        now = time.time()
        offset = None
        while True:
            points, offset = self.qdrant.scroll(collection, limit=100, offset=offset)
            for p in points:
                payload = p.get("payload", {})
                last = float(payload.get("last_accessed", now))
                strength = float(payload.get("strength", 0.1))
                age = max(0.0, now - last)
                decay = math.exp(-self.decay_rate * age)
                new_strength = max(0.01, strength * decay)
                payload["strength"] = new_strength
                self.qdrant.update_payload(collection, p.get("id"), payload)
            if not offset:
                break

    def recent(self, soul_id: str, limit: int = 10) -> List[MemoryObject]:
        """Fetch most recent memories by created_at timestamp."""
        collection = self._collection(soul_id)
        points: List[Dict[str, Any]] = []
        offset = None
        while True:
            batch, offset = self.qdrant.scroll(collection, limit=100, offset=offset, with_vectors=True)
            points.extend(batch)
            if not offset:
                break
        points.sort(key=lambda p: float(p.get("payload", {}).get("created_at", 0.0)), reverse=True)
        now = time.time()
        memories: List[MemoryObject] = []
        for p in points[:limit]:
            payload = p.get("payload", {})
            mem = MemoryObject(
                id=str(p.get("id")),
                soul_id=soul_id,
                text=payload.get("text", ""),
                vector=p.get("vector", []),
                saliency=float(payload.get("saliency", 0.1)),
                created_at=float(payload.get("created_at", now)),
                last_accessed=now,
                strength=float(payload.get("strength", 0.1)),
                tags=payload.get("tags", []),
                meta=payload.get("meta", {}),
            )
            memories.append(mem)
        return memories


# -----------------------------
# Minimal HTTP server (std lib)
# -----------------------------

from http.server import BaseHTTPRequestHandler, HTTPServer


class SoulMemoryHandler(BaseHTTPRequestHandler):
    node: SoulMemoryNode = None  # set externally

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
            if self.path == "/memories/ingest":
                soul_id = data.get("soul_id")
                text = data.get("text", "")
                if not soul_id or not text:
                    return self._json(400, {"error": "soul_id and text required"})
                mem = self.node.ingest(soul_id, text, data.get("tags"), data.get("meta"))
                return self._json(200, {"memory": asdict(mem)})

            if self.path == "/memories/query":
                soul_id = data.get("soul_id")
                query_text = data.get("query", "")
                if not soul_id or not query_text:
                    return self._json(400, {"error": "soul_id and query required"})
                mems = self.node.query(soul_id, query_text, limit=int(data.get("limit", 5)))
                return self._json(200, {"results": [asdict(m) for m in mems]})

            if self.path == "/memories/decay":
                soul_id = data.get("soul_id")
                if not soul_id:
                    return self._json(400, {"error": "soul_id required"})
                self.node.decay(soul_id)
                return self._json(200, {"ok": True})

            if self.path == "/memories/recent":
                soul_id = data.get("soul_id")
                if not soul_id:
                    return self._json(400, {"error": "soul_id required"})
                limit = int(data.get("limit", 10))
                mems = self.node.recent(soul_id, limit=limit)
                return self._json(200, {"results": [asdict(m) for m in mems]})
        except Exception as exc:
            return self._json(500, {"error": str(exc)})

        return self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="SoulMemoryNode service")
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--embedding-model", type=str, default="nomic-embed-text:latest")
    parser.add_argument("--qdrant-url", type=str, default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--embedding-endpoint", type=str, default=os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings"))
    args = parser.parse_args()

    node = SoulMemoryNode(
        embedding_model=args.embedding_model,
        qdrant_url=args.qdrant_url,
        embedding_endpoint=args.embedding_endpoint,
    )
    SoulMemoryHandler.node = node
    server = HTTPServer(("0.0.0.0", args.port), SoulMemoryHandler)
    print(f"SoulMemoryNode listening on :{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
