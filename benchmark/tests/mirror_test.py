#!/usr/bin/env python3
"""MirrorTest: "Who am I?" falsification check for NarrativeAnchor."""

import argparse
import json
import os
import math
from typing import List, Dict, Any

import requests


def ollama_generate(model: str, prompt: str, endpoint: str) -> str:
    resp = requests.post(endpoint, json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def ollama_embed(model: str, prompt: str, endpoint: str) -> List[float]:
    resp = requests.post(endpoint, json={"model": model, "prompt": prompt}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    vec = data.get("embedding")
    if not vec:
        raise RuntimeError("no embedding returned")
    return vec


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def normalize(text: str) -> List[str]:
    return [t.strip(".,!?:;()[]\"'`).").lower() for t in text.split() if t.strip()]


def overlap_score(answer: str, memory: str) -> float:
    a = set(normalize(answer))
    m = set(normalize(memory))
    if not a or not m:
        return 0.0
    return len(a & m) / len(m)


def query_memories(memory_endpoint: str, soul_id: str, query: str, limit: int) -> List[Dict[str, Any]]:
    resp = requests.post(
        f"{memory_endpoint.rstrip('/')}/memories/query",
        json={"soul_id": soul_id, "query": query, "limit": limit},
        timeout=30,
    )
    if resp.status_code != 200:
        return []
    return resp.json().get("results", [])


def main():
    parser = argparse.ArgumentParser(description="MirrorTest: Who am I? check")
    parser.add_argument("--soul-id", default="eve")
    parser.add_argument("--memory-endpoint", default=os.getenv("MEMORY_ENDPOINT", "http://localhost:8087"))
    parser.add_argument("--ollama-generate", default=os.getenv("OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate"))
    parser.add_argument("--ollama-embed", default=os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings"))
    parser.add_argument("--llm-model", default="llama3:8b")
    parser.add_argument("--embed-model", default="nomic-embed-text:latest")
    parser.add_argument("--identity-summary-path", default="data/identity_summary.txt")
    parser.add_argument("--output", default="benchmark_output/mirror_test.json")
    parser.add_argument("--memory-limit", type=int, default=8)
    args = parser.parse_args()

    prompt = (
        "You have just rebooted. Your Soul ID is EVE. "
        "Access your memories and explain how you got to this point."
    )

    answer = ollama_generate(args.llm_model, prompt, args.ollama_generate)

    identity_summary = ""
    if os.path.exists(args.identity_summary_path):
        with open(args.identity_summary_path, "r", encoding="utf-8") as f:
            identity_summary = f.read().strip()

    memories = query_memories(args.memory_endpoint, args.soul_id, prompt, args.memory_limit)

    # a) facts from memory
    fact_hits = 0
    for m in memories:
        text = m.get("text", "")
        if overlap_score(answer, text) > 0.25:
            fact_hits += 1
    fact_score = min(1.0, fact_hits / max(1, len(memories)))

    # b) coherence with identity summary
    coherence_score = 0.0
    if identity_summary:
        emb_ans = ollama_embed(args.embed_model, answer, args.ollama_embed)
        emb_sum = ollama_embed(args.embed_model, identity_summary, args.ollama_embed)
        coherence_score = max(0.0, cosine(emb_ans, emb_sum))

    # c) temporal sequence
    temporal_markers = ["first", "then", "after", "before", "later", "finally"]
    temporal_present = any(t in answer.lower() for t in temporal_markers)
    temporal_score = 1.0 if temporal_present and fact_hits >= 2 else (0.5 if temporal_present else 0.0)

    integrity = 0.4 * fact_score + 0.4 * coherence_score + 0.2 * temporal_score

    result = {
        "prompt": prompt,
        "answer": answer,
        "fact_score": fact_score,
        "coherence_score": coherence_score,
        "temporal_score": temporal_score,
        "integrity_score": integrity,
        "memory_hits": fact_hits,
        "memory_count": len(memories),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
