#!/usr/bin/env python3
"""
FractalCompressor: NC2 integration via GGGP-guided long-term memory folding.

- Builds hierarchical summaries of memories (fractal depth).
- Applies evolutionary pruning for dead contexts.
- Uses reconstructibility as fitness.
"""

import argparse
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from http.server import BaseHTTPRequestHandler, HTTPServer

import requests
import sys

sys.path.append(os.path.dirname(__file__))

from soul_memory_node import QdrantHTTP, OllamaEmbeddingClient


@dataclass
class MemoryPhenotype:
    pruning_rate: float = 0.2
    depth_bias: float = 0.3
    ghost_strength: float = 0.2
    max_depth: int = 4

    def clamp(self):
        self.pruning_rate = min(max(self.pruning_rate, 0.0), 1.0)
        self.depth_bias = min(max(self.depth_bias, 0.0), 1.0)
        self.ghost_strength = min(max(self.ghost_strength, 0.0), 1.0)
        self.max_depth = max(1, min(int(self.max_depth), 8))


class FractalCompressor:
    def __init__(
        self,
        qdrant_url: str,
        embedding_model: str,
        embed_url: str,
        ollama_generate_url: str,
        gggp_endpoint: str,
    ):
        self.qdrant = QdrantHTTP(qdrant_url)
        self.embedder = OllamaEmbeddingClient(model=embedding_model, endpoint=embed_url)
        self.ollama_generate_url = ollama_generate_url
        self.gggp_endpoint = gggp_endpoint.rstrip("/")
        self.state = MemoryPhenotype()

    def _collection(self, soul_id: str) -> str:
        return f"soul_{soul_id}"

    def _dead_context(self, text: str, tags: List[str], meta: Dict[str, Any]) -> bool:
        if meta.get("context_alive") is False:
            return True
        lower = text.lower()
        if any(k in lower for k in ["deprecated", "obsolete", "outdated", "no longer", "v1", "v0", "legacy"]):
            return True
        if any(t in ("deprecated", "obsolete", "legacy") for t in tags):
            return True
        return False

    def _summarize(self, texts: List[str], depth: int) -> str:
        if not texts:
            return ""
        prompt = (
            "Summarize the following experiences into a compact, abstract memory. "
            "Keep it high-level and compress details. Output a single paragraph.\n\n"
            f"Fractal depth: {depth}\n"
            + "\n".join(f"- {t}" for t in texts[:50])
        )
        payload = {"model": "llama3:8b", "prompt": prompt, "stream": False}
        resp = requests.post(self.ollama_generate_url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _reconstructibility(self, summary: str) -> float:
        if not summary:
            return 0.0
        prompt = (
            "Reconstruct the gist of the original conversation from this summary. "
            "Return a short paragraph.\n\n"
            f"Summary: {summary}"
        )
        payload = {"model": "llama3:8b", "prompt": prompt, "stream": False}
        resp = requests.post(self.ollama_generate_url, json=payload, timeout=120)
        resp.raise_for_status()
        recon = resp.json().get("response", "").strip()
        emb_sum = self.embedder.embed(summary)
        emb_rec = self.embedder.embed(recon)
        return self._cosine(emb_sum, emb_rec)

    def _fetch_all(self, soul_id: str) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        offset = None
        while True:
            batch, offset = self.qdrant.scroll(self._collection(soul_id), limit=100, offset=offset, with_vectors=False)
            points.extend(batch)
            if not offset:
                break
        return points

    def _depth_for_age(self, age_seconds: float) -> int:
        days = max(age_seconds / 86400.0, 0.0)
        depth = int(math.log2(days + 1.0))
        depth = max(0, min(depth, self.state.max_depth))
        return depth

    def _evolve_state(self, score: float):
        payload = {"traits": asdict(self.state), "score": score}
        try:
            resp = requests.post(f"{self.gggp_endpoint}/evolve_memory", json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("traits")
                if data:
                    self.state = MemoryPhenotype(**data)
                    self.state.clamp()
        except Exception:
            pass

    def compress(self, soul_id: str) -> Dict[str, Any]:
        points = self._fetch_all(soul_id)
        now = time.time()
        buckets: Dict[int, List[str]] = {}
        dead_flags: Dict[int, bool] = {}
        for p in points:
            payload = p.get("payload", {})
            text = payload.get("text", "")
            if not text:
                continue
            created = float(payload.get("created_at", now))
            depth = self._depth_for_age(now - created)
            buckets.setdefault(depth, []).append(text)
            tags = payload.get("tags", [])
            meta = payload.get("meta", {})
            if self._dead_context(text, tags, meta):
                dead_flags[depth] = True

        summaries: List[Dict[str, Any]] = []
        recon_scores: List[float] = []
        for depth, texts in buckets.items():
            summary = self._summarize(texts, depth)
            if not summary:
                continue
            recon = self._reconstructibility(summary)
            recon_scores.append(recon)
            vec = self.embedder.embed(summary)
            ghost = dead_flags.get(depth, False)
            payload = {
                "text": summary,
                "tags": ["semantic_ghost"] if ghost else ["fractal_summary"],
                "meta": {
                    "depth": depth,
                    "abstractness": min(1.0, depth / max(1.0, self.state.max_depth)),
                    "ghost": ghost,
                    "reconstructibility": recon,
                },
                "created_at": now,
                "last_accessed": now,
                "saliency": 0.05,
                "strength": self.state.ghost_strength if ghost else 0.2,
            }
            point_id = str(uuid.uuid4())
            # ensure collection exists
            self.qdrant.ensure_collection(self._collection(soul_id), vector_size=len(vec))
            self.qdrant.upsert(self._collection(soul_id), point_id, vec, payload)
            summaries.append({"depth": depth, "summary": summary, "reconstructibility": recon, "ghost": ghost})

        avg_recon = sum(recon_scores) / max(1, len(recon_scores))
        self._evolve_state(avg_recon)

        return {
            "summaries": summaries,
            "avg_reconstructibility": avg_recon,
            "phenotype": asdict(self.state),
        }


class FractalHandler(BaseHTTPRequestHandler):
    compressor: FractalCompressor = None

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

        if self.path == "/fractal/compact":
            soul_id = data.get("soul_id")
            if not soul_id:
                return self._json(400, {"error": "soul_id required"})
            result = self.compressor.compress(soul_id)
            return self._json(200, result)

        return self._json(404, {"error": "not found"})

    def do_GET(self):
        if self.path == "/fractal/state":
            return self._json(200, {"phenotype": asdict(self.compressor.state)})
        return self._json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser(description="Fractal Compressor service")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--qdrant-url", type=str, default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--embedding-model", type=str, default="nomic-embed-text:latest")
    parser.add_argument("--ollama-embed-url", type=str, default=os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings"))
    parser.add_argument("--ollama-generate-url", type=str, default=os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate"))
    parser.add_argument("--gggp-endpoint", type=str, default=os.getenv("GGGP_ENDPOINT", "http://localhost:8091"))
    args = parser.parse_args()

    compressor = FractalCompressor(
        qdrant_url=args.qdrant_url,
        embedding_model=args.embedding_model,
        embed_url=args.ollama_embed_url,
        ollama_generate_url=args.ollama_generate_url,
        gggp_endpoint=args.gggp_endpoint,
    )
    FractalHandler.compressor = compressor
    server = HTTPServer(("0.0.0.0", args.port), FractalHandler)
    print(f"FractalCompressor listening on :{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
