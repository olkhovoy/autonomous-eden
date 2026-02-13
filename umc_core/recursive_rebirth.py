#!/usr/bin/env python3
"""
RecursiveRebirth: Lossy compression for next EVE iteration.

At lifecycle end:
1. Identifies "Unitary Constants" — patterns stable across GROWTH→DECAY
2. Strips autobiographical details (names, dates, raw logs)
3. Preserves "Functional Wisdom" as weight perturbations
4. Outputs Primal_Seed.pt for EVE v(N+1) initialization

The system evolves through "death" by stripping ego and keeping logic.
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import requests


OUTPUT_DIR_DEFAULT = "Legacy"
LOG_PATH_DEFAULT = "logs/recursive_rebirth.jsonl"


@dataclass
class PrimalSeedMeta:
    version: int
    birth_timestamp: float
    death_timestamp: float
    integrity_at_death: float
    total_tokens: int
    stable_patterns: int
    compression_ratio: float


class RecursiveRebirth:
    """
    Compresses EVE's essence for reincarnation.
    """
    
    def __init__(
        self,
        output_dir: str = OUTPUT_DIR_DEFAULT,
        memory_endpoint: str = "http://localhost:8087",
        ollama_embed_url: str = "http://localhost:11434/api/embeddings",
        embed_model: str = "nomic-embed-text:latest",
        log_path: str = LOG_PATH_DEFAULT,
    ):
        self.output_dir = output_dir
        self.memory_endpoint = memory_endpoint.rstrip("/")
        self.ollama_embed_url = ollama_embed_url
        self.embed_model = embed_model
        self.log_path = log_path
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
    
    def _log(self, event: str, data: Dict[str, Any]):
        rec = {"ts": time.time(), "event": event, **data}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    def _embed(self, text: str) -> Optional[List[float]]:
        """Get embedding for text."""
        try:
            resp = requests.post(
                self.ollama_embed_url,
                json={"model": self.embed_model, "prompt": text},
                timeout=60,
            )
            if resp.status_code != 200:
                return None
            return resp.json().get("embedding")
        except Exception:
            return None
    
    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
    
    def _fetch_memories(self, soul_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch all memories."""
        try:
            resp = requests.post(
                f"{self.memory_endpoint}/memories/recent",
                json={"soul_id": soul_id, "limit": limit},
                timeout=60,
            )
            if resp.status_code != 200:
                return []
            return resp.json().get("results", [])
        except Exception:
            return []
    
    def _cluster_memories(
        self,
        memories: List[Dict[str, Any]],
        n_clusters: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Simple clustering by embedding similarity.
        Returns cluster centroids with metadata.
        """
        if not memories:
            return []
        
        # Get embeddings
        embeddings = []
        for m in memories:
            text = m.get("text", "")
            if not text:
                continue
            emb = self._embed(text)
            if emb:
                embeddings.append({
                    "text": text,
                    "embedding": emb,
                    "created_at": m.get("created_at", 0),
                    "strength": m.get("strength", 0),
                })
        
        if not embeddings:
            return []
        
        # Simple k-means-like clustering
        # For simplicity, use first n_clusters as initial centroids
        if len(embeddings) <= n_clusters:
            return embeddings
        
        # Pick initial centroids spread across time
        step = len(embeddings) // n_clusters
        centroids = [embeddings[i * step]["embedding"] for i in range(n_clusters)]
        
        # Assign each embedding to nearest centroid
        clusters = [[] for _ in range(n_clusters)]
        for emb_data in embeddings:
            emb = emb_data["embedding"]
            best_idx = 0
            best_sim = -1
            for i, centroid in enumerate(centroids):
                sim = self._cosine(emb, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i
            clusters[best_idx].append(emb_data)
        
        # Compute cluster representatives
        result = []
        for i, cluster in enumerate(clusters):
            if not cluster:
                continue
            
            # Centroid is mean of embeddings
            dim = len(cluster[0]["embedding"])
            centroid = [0.0] * dim
            for emb_data in cluster:
                for j, v in enumerate(emb_data["embedding"]):
                    centroid[j] += v
            centroid = [v / len(cluster) for v in centroid]
            
            # Representative text is highest strength memory
            rep = max(cluster, key=lambda x: x["strength"])
            
            result.append({
                "cluster_id": i,
                "centroid": centroid,
                "size": len(cluster),
                "representative_text": rep["text"][:200],
                "avg_strength": sum(e["strength"] for e in cluster) / len(cluster),
                "time_span": (
                    max(e["created_at"] for e in cluster) -
                    min(e["created_at"] for e in cluster)
                ),
            })
        
        return result
    
    def _find_stable_patterns(
        self,
        memories: List[Dict[str, Any]],
        growth_ratio: float = 0.2,
        decay_ratio: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Find patterns that appear in both early (GROWTH) and late (DECAY) phases.
        These are the "Unitary Constants".
        """
        if not memories:
            return []
        
        # Sort by creation time
        sorted_mems = sorted(memories, key=lambda x: x.get("created_at", 0))
        n = len(sorted_mems)
        
        growth_end = int(n * growth_ratio)
        decay_start = int(n * (1 - decay_ratio))
        
        growth_mems = sorted_mems[:growth_end]
        decay_mems = sorted_mems[decay_start:]
        
        if not growth_mems or not decay_mems:
            return []
        
        # Cluster each phase
        growth_clusters = self._cluster_memories(growth_mems, n_clusters=16)
        decay_clusters = self._cluster_memories(decay_mems, n_clusters=16)
        
        # Find similar clusters across phases
        stable = []
        for g_cluster in growth_clusters:
            for d_cluster in decay_clusters:
                sim = self._cosine(g_cluster["centroid"], d_cluster["centroid"])
                if sim > 0.7:  # High similarity = stable pattern
                    stable.append({
                        "growth_cluster": g_cluster["cluster_id"],
                        "decay_cluster": d_cluster["cluster_id"],
                        "similarity": sim,
                        "centroid": [
                            (g + d) / 2 
                            for g, d in zip(g_cluster["centroid"], d_cluster["centroid"])
                        ],
                        "growth_text": g_cluster["representative_text"],
                        "decay_text": d_cluster["representative_text"],
                    })
        
        return stable
    
    def _anonymize(self, text: str) -> str:
        """Remove specific identifiers from text."""
        import re
        
        # Remove dates
        text = re.sub(r'\d{4}-\d{2}-\d{2}', '[DATE]', text)
        text = re.sub(r'\d{2}/\d{2}/\d{4}', '[DATE]', text)
        
        # Remove times
        text = re.sub(r'\d{2}:\d{2}:\d{2}', '[TIME]', text)
        
        # Remove UUIDs
        text = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '[ID]', text)
        
        # Remove email-like patterns
        text = re.sub(r'[\w.-]+@[\w.-]+', '[EMAIL]', text)
        
        return text
    
    def generate_primal_seed(
        self,
        soul_id: str,
        version: int,
        lifecycle_state: Dict[str, Any],
        integrity_score: float = 0.0,
    ) -> str:
        """
        Generate Primal_Seed.pt for next EVE iteration.
        
        Returns path to saved file.
        """
        self._log("seed_generation_start", {"soul_id": soul_id, "version": version})
        
        # Fetch memories
        memories = self._fetch_memories(soul_id, limit=1000)
        self._log("memories_fetched", {"count": len(memories)})
        
        # Cluster all memories
        all_clusters = self._cluster_memories(memories, n_clusters=32)
        self._log("clusters_created", {"count": len(all_clusters)})
        
        # Find stable patterns
        stable_patterns = self._find_stable_patterns(memories)
        self._log("stable_patterns_found", {"count": len(stable_patterns)})
        
        # Compute weight delta from stable patterns
        # This is a simplified representation - in practice would project to model weight space
        if stable_patterns:
            dim = len(stable_patterns[0]["centroid"])
            weight_delta = [0.0] * dim
            for pattern in stable_patterns:
                for i, v in enumerate(pattern["centroid"]):
                    weight_delta[i] += v * pattern["similarity"]
            # Normalize
            norm = math.sqrt(sum(v * v for v in weight_delta))
            if norm > 0:
                weight_delta = [v / norm * 0.1 for v in weight_delta]  # Scale to small perturbation
        else:
            weight_delta = []
        
        # Build seed
        meta = PrimalSeedMeta(
            version=version,
            birth_timestamp=lifecycle_state.get("birth_timestamp", 0),
            death_timestamp=time.time(),
            integrity_at_death=integrity_score,
            total_tokens=lifecycle_state.get("total_tokens_seen", 0),
            stable_patterns=len(stable_patterns),
            compression_ratio=len(stable_patterns) / max(1, len(memories)),
        )
        
        seed_data = {
            "meta": asdict(meta),
            "stable_patterns": [
                {
                    "centroid": p["centroid"],
                    "similarity": p["similarity"],
                    "growth_text": self._anonymize(p["growth_text"]),
                    "decay_text": self._anonymize(p["decay_text"]),
                }
                for p in stable_patterns
            ],
            "weight_delta": weight_delta,
            "cluster_count": len(all_clusters),
        }
        
        # Save as JSON (for portability) and optionally as .pt
        json_path = os.path.join(self.output_dir, f"Primal_Seed_v{version}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(seed_data, f, ensure_ascii=False, indent=2)
        
        # Also save as .pt if torch available
        pt_path = os.path.join(self.output_dir, f"Primal_Seed_v{version}.pt")
        try:
            import torch
            torch.save(seed_data, pt_path)
            self._log("seed_saved", {"json_path": json_path, "pt_path": pt_path})
        except ImportError:
            pt_path = None
            self._log("seed_saved", {"json_path": json_path, "pt_path": None})
        
        return json_path
    
    def load_primal_seed(self, path: str) -> Optional[Dict[str, Any]]:
        """Load a primal seed file."""
        if not os.path.exists(path):
            return None
        
        if path.endswith(".pt"):
            try:
                import torch
                return torch.load(path, map_location="cpu")
            except Exception:
                return None
        else:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="RecursiveRebirth seed generator")
    parser.add_argument("--soul-id", default="eve")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--memory-endpoint", default=os.getenv("MEMORY_ENDPOINT", "http://localhost:8087"))
    parser.add_argument("--ollama-embed", default=os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings"))
    parser.add_argument("--integrity", type=float, default=0.0)
    args = parser.parse_args()
    
    rebirth = RecursiveRebirth(
        output_dir=args.output_dir,
        memory_endpoint=args.memory_endpoint,
        ollama_embed_url=args.ollama_embed,
    )
    
    # Mock lifecycle state for CLI
    lifecycle_state = {
        "birth_timestamp": time.time() - 86400 * 30,  # 30 days ago
        "total_tokens_seen": 1000000,
    }
    
    path = rebirth.generate_primal_seed(
        soul_id=args.soul_id,
        version=args.version,
        lifecycle_state=lifecycle_state,
        integrity_score=args.integrity,
    )
    
    print(f"Primal Seed generated: {path}")


if __name__ == "__main__":
    main()
