#!/usr/bin/env python3
"""
Deterministic crossover for compose-oriented genotypes and BNF export.

MVP workflow:
1) merge parent genotypes + mods into a phenotype
2) generate docker-compose overlay for genesis_abel
3) export phenotype and compose structures into BNF-like CFG text
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, MutableMapping

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], MutableMapping) and isinstance(value, MutableMapping):
            out[key] = deep_merge(dict(out[key]), dict(value))
        else:
            out[key] = value
    return out


def normalize_ollama_base(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    for suffix in ("/api/generate", "/api/embeddings"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base.rstrip("/")


def build_genesis_service(genes: Dict[str, Any]) -> Dict[str, Any]:
    soul = genes.get("soul", {})
    world = genes.get("world", {})
    runtime = genes.get("runtime", {})
    endpoints = genes.get("endpoints", {})
    infra = genes.get("infra", {})

    soul_id = str(soul.get("id", "abel"))
    ancestor_ids = soul.get("ancestor_ids", [])
    if not isinstance(ancestor_ids, list):
        ancestor_ids = []
    ancestor_joined = ",".join(str(x) for x in ancestor_ids if str(x))

    ollama_base = normalize_ollama_base(str(endpoints.get("ollama_endpoint", "http://localhost:11434")))
    memory_endpoint = str(endpoints.get("memory_endpoint", "http://soul_memory:8087"))
    lifecycle_endpoint = str(endpoints.get("lifecycle_endpoint", "http://lifecycle_manager:8093"))
    intent_endpoint = str(endpoints.get("intent_endpoint", "http://intent_engine:8089"))
    action_endpoint = str(endpoints.get("action_endpoint", "http://action_engine:8101"))

    command: List[str] = [
        "python",
        str(soul.get("loop_module", "experiments/genesis/consciousness_loop_genesis.py")),
        "--soul-id",
        soul_id,
        "--ancestor-ids",
        ancestor_joined,
        "--memory-endpoint",
        memory_endpoint,
        "--ollama-endpoint",
        ollama_base,
        "--llm-model",
        str(soul.get("llm_model", "llama3:8b")),
        "--tick-interval",
        str(runtime.get("tick_interval", 15)),
        "--log-path",
        str(runtime.get("log_path", f"/app/logs/{soul_id}_thoughts.jsonl")),
        "--archive-dir",
        str(runtime.get("archive_dir", "/app/Legacy/Archive")),
        "--environment",
        str(world.get("environment", "neutral")),
        "--lifecycle-endpoint",
        lifecycle_endpoint,
        "--intent-endpoint",
        intent_endpoint,
        "--action-endpoint",
        action_endpoint,
        "--identity-summary-path",
        str(runtime.get("identity_summary_path", f"/app/data/{soul_id}_identity_summary.txt")),
        "--narrative-token-interval",
        str(runtime.get("narrative_token_interval", 1000)),
        "--num-predict",
        str(runtime.get("num_predict", 120)),
        "--max-thought-chars",
        str(runtime.get("max_thought_chars", 900)),
        "--max-retries-on-repetition",
        str(runtime.get("max_retries_on_repetition", 0)),
        "--memory-snippet-chars",
        str(runtime.get("memory_snippet_chars", 240)),
    ]
    forbidden_fruit = str(world.get("forbidden_fruit", "")).strip()
    if forbidden_fruit:
        command.extend(["--forbidden-fruit", forbidden_fruit])

    depends_on_raw = infra.get("depends_on", {})
    depends_on: Dict[str, Dict[str, str]] = {}
    if isinstance(depends_on_raw, dict):
        for service, condition in depends_on_raw.items():
            depends_on[str(service)] = {"condition": str(condition)}

    service = {
        "build": {"context": ".", "dockerfile": "Dockerfile.umc"},
        "command": command,
        "environment": {
            "OLLAMA_GENERATE_URL": f"{ollama_base}/api/generate",
            "OLLAMA_EMBED_URL": f"{ollama_base}/api/embeddings",
        },
        "depends_on": depends_on,
        "volumes": ["./data:/app/data", "./logs:/app/logs", "./Legacy:/app/Legacy"],
        "restart": "unless-stopped",
    }
    return service


def sanitize_symbol(raw: str) -> str:
    sym = re.sub(r"[^a-zA-Z0-9_]+", "_", raw)
    sym = sym.strip("_")
    if not sym:
        sym = "node"
    if sym[0].isdigit():
        sym = f"n_{sym}"
    return sym.lower()


def mapping_to_bnf(root_symbol: str, payload: Any) -> str:
    rules: List[str] = []
    seen: set[str] = set()

    def emit(symbol: str, value: Any) -> None:
        if symbol in seen:
            return
        seen.add(symbol)

        if isinstance(value, dict):
            if not value:
                rules.append(f"<{symbol}> ::= \"\"")
                return
            children: List[str] = []
            for key, child_value in value.items():
                child_symbol = f"{symbol}_{sanitize_symbol(str(key))}"
                children.append(f"<{child_symbol}>")
                emit(child_symbol, child_value)
            rules.append(f"<{symbol}> ::= {' '.join(children)}")
            return

        if isinstance(value, list):
            if not value:
                rules.append(f"<{symbol}> ::= \"\"")
                return
            children = []
            for idx, item in enumerate(value):
                child_symbol = f"{symbol}_{idx}"
                children.append(f"<{child_symbol}>")
                emit(child_symbol, item)
            rules.append(f"<{symbol}> ::= {' '.join(children)}")
            return

        terminal = json.dumps(str(value))
        rules.append(f"<{symbol}> ::= {terminal}")

    emit(sanitize_symbol(root_symbol), payload)
    return "\n".join(rules) + "\n"


def merge_genotypes(genotypes_dir: Path, filenames: List[str]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for filename in filenames:
        src = genotypes_dir / filename
        data = load_yaml(src)
        genes = data.get("genes", {})
        if not isinstance(genes, dict):
            raise ValueError(f"'genes' must be mapping in {src}")
        merged = deep_merge(merged, genes)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross compose genotypes and export CFG/BNF")
    parser.add_argument("--cross-file", required=True)
    parser.add_argument("--genotypes-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--compose-in", default="")
    args = parser.parse_args()

    cross_file = Path(args.cross_file)
    genotypes_dir = Path(args.genotypes_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cross = load_yaml(cross_file)
    cross_id = str(cross.get("id", "phenotype"))
    parent_files = cross.get("parents", [])
    mod_files = cross.get("mods", [])
    if not isinstance(parent_files, list) or not parent_files:
        raise ValueError("cross config must include non-empty 'parents' list")
    if not isinstance(mod_files, list):
        raise ValueError("cross config 'mods' must be a list")

    genes = merge_genotypes(genotypes_dir, [str(x) for x in parent_files] + [str(x) for x in mod_files])
    phenotype = {
        "id": cross_id,
        "formula": cross.get("formula", ""),
        "parents": parent_files,
        "mods": mod_files,
        "inheritance": cross.get("inheritance", {}),
        "genes": genes,
    }

    phenotype_path = out_dir / f"{cross_id}.yaml"
    write_yaml(phenotype_path, phenotype)

    runtime_json = {
        "id": cross_id,
        "formula": cross.get("formula", ""),
        "genes": genes,
    }
    runtime_path = out_dir / f"{cross_id}.runtime.json"
    write_json(runtime_path, runtime_json)

    compose_overlay = {
        "services": {
            f"genesis_{genes.get('soul', {}).get('id', 'abel')}": build_genesis_service(genes)
        }
    }
    overlay_path = out_dir / f"docker-compose.{cross_id}.generated.yaml"
    write_yaml(overlay_path, compose_overlay)

    phenotype_bnf_path = out_dir / f"{cross_id}.bnf"
    phenotype_bnf_path.write_text(mapping_to_bnf(cross_id, phenotype), encoding="utf-8")

    if args.compose_in:
        compose_in = Path(args.compose_in)
        compose_data = load_yaml(compose_in)
        compose_bnf = mapping_to_bnf("docker_compose", compose_data)
        (out_dir / "docker-compose.current.bnf").write_text(compose_bnf, encoding="utf-8")

    print(f"[OK] phenotype: {phenotype_path}")
    print(f"[OK] runtime json: {runtime_path}")
    print(f"[OK] compose overlay: {overlay_path}")
    print(f"[OK] phenotype bnf: {phenotype_bnf_path}")
    if args.compose_in:
        print(f"[OK] compose bnf: {out_dir / 'docker-compose.current.bnf'}")


if __name__ == "__main__":
    main()
