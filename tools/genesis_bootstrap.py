#!/usr/bin/env python3
"""
Bootstrap a descendant generation from an evolutionary runtime config.

Flow:
1) Load generated phenotype runtime JSON (from gggp_bundle evolution step)
2) Ingest ancestor archives into SoulMemory
3) Exec the Genesis consciousness loop with resolved parameters
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{ts}] [genesis-bootstrap] {message}", flush=True)


def wait_for_service(endpoint: str, timeout: int = 180) -> bool:
    deadline = time.time() + timeout
    health_url = f"{endpoint.rstrip('/')}/health"
    log(f"Waiting for memory service: {health_url} (timeout={timeout}s)")
    while time.time() < deadline:
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200:
                log("Memory service is ready.")
                return True
        except Exception:
            pass
        time.sleep(2)
    log("Memory service did not become ready in time.")
    return False


def post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            log(f"HTTP {resp.status_code} from {url}: {resp.text[:240]}")
            return None
        data = resp.json()
        return data if isinstance(data, dict) else None
    except Exception as exc:
        log(f"Request failed {url}: {exc}")
        return None


def ingest_ancestor(memory_endpoint: str, archive_dir: str, ancestor_id: str) -> int:
    ingest_url = f"{memory_endpoint.rstrip('/')}/memories/ingest_archive"
    archive_path = str(Path(archive_dir) / f"{ancestor_id}_archive")
    log(f"Ingesting ancestor '{ancestor_id}' from '{archive_path}'")
    data = post_json(
        ingest_url,
        {"soul_id": ancestor_id, "archive_path": archive_path},
        timeout=900,
    )
    if not data:
        return 0
    count = int(data.get("ingested", 0))
    log(f"Ingested {count} memories for ancestor '{ancestor_id}'")
    return count


def load_runtime_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("runtime config root must be an object")
    genes = data.get("genes", data)
    if not isinstance(genes, dict):
        raise ValueError("runtime config must contain object field 'genes'")
    return genes


def load_ingest_state(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_ingest_state(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def archive_signature(archive_dir: str, ancestor_id: str) -> Dict[str, Any]:
    snapshot = Path(archive_dir) / f"{ancestor_id}_archive" / "memories_snapshot.json"
    if not snapshot.exists():
        return {"path": str(snapshot), "exists": False}
    stat = snapshot.stat()
    return {
        "path": str(snapshot),
        "exists": True,
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
    }


def maybe_refresh_config(args: argparse.Namespace) -> None:
    if not args.refresh_config:
        return
    cmd = [
        sys.executable,
        args.cross_tool,
        "--cross-file",
        args.cross_file,
        "--genotypes-dir",
        args.genotypes_dir,
        "--out-dir",
        args.out_dir,
    ]
    if args.compose_in:
        cmd.extend(["--compose-in", args.compose_in])
    log("Refreshing evolutionary runtime config from genotype cross...")
    log(" ".join(cmd))
    subprocess.run(cmd, check=True)


def build_loop_command(genes: Dict[str, Any], runtime_config_path: str) -> List[str]:
    soul = genes.get("soul", {}) if isinstance(genes.get("soul"), dict) else {}
    world = genes.get("world", {}) if isinstance(genes.get("world"), dict) else {}
    runtime = genes.get("runtime", {}) if isinstance(genes.get("runtime"), dict) else {}
    endpoints = genes.get("endpoints", {}) if isinstance(genes.get("endpoints"), dict) else {}

    soul_id = str(soul.get("id", "abel")).strip() or "abel"
    ancestors = soul.get("ancestor_ids", [])
    if not isinstance(ancestors, list):
        ancestors = []
    ancestor_ids = ",".join(str(x).strip() for x in ancestors if str(x).strip())

    command: List[str] = [
        sys.executable,
        str(soul.get("loop_module", "experiments/genesis/consciousness_loop_genesis.py")),
        "--soul-id",
        soul_id,
        "--ancestor-ids",
        ancestor_ids,
        "--memory-endpoint",
        str(endpoints.get("memory_endpoint", "http://soul_memory:8087")),
        "--ollama-endpoint",
        str(endpoints.get("ollama_endpoint", "http://localhost:11434")),
        "--llm-model",
        str(soul.get("llm_model", os.getenv("LLM_MODEL", "llama3:8b"))),
        "--tick-interval",
        str(runtime.get("tick_interval", 15)),
        "--log-path",
        str(runtime.get("log_path", f"/app/logs/{soul_id}_thoughts.jsonl")),
        "--archive-dir",
        str(runtime.get("archive_dir", "/app/Legacy/Archive")),
        "--environment",
        str(world.get("environment", "neutral")),
        "--lifecycle-endpoint",
        str(endpoints.get("lifecycle_endpoint", "http://lifecycle_manager:8093")),
        "--intent-endpoint",
        str(endpoints.get("intent_endpoint", "http://intent_engine:8089")),
        "--action-endpoint",
        str(endpoints.get("action_endpoint", "http://action_engine:8101")),
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
        "--social-tick-every",
        str(runtime.get("social_tick_every", 3)),
        "--echo-num-predict",
        str(runtime.get("echo_num_predict", 60)),
    ]
    echo_model = str(runtime.get("echo_model", "")).strip()
    if echo_model:
        command.extend(["--echo-model", echo_model])
    forbidden_fruit = str(world.get("forbidden_fruit", "")).strip()
    if forbidden_fruit:
        command.extend(["--forbidden-fruit", forbidden_fruit])

    log(f"Using runtime config: {runtime_config_path}")
    log(f"Soul: {soul_id}; ancestors: {ancestor_ids or '(none)'}")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap descendant generation from evolution config")
    parser.add_argument(
        "--runtime-config",
        default="gggp_bundle/evolution/phenotypes/abel_env.runtime.json",
        help="JSON runtime config generated by cross_compose_cfg.py",
    )
    parser.add_argument("--wait", type=int, default=180, help="Wait time for memory service readiness")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ancestor archive ingestion")
    parser.add_argument("--force-ingest", action="store_true", help="Force ingest even if archive signature is unchanged")
    parser.add_argument(
        "--ingest-state-path",
        default="/app/data/genesis_ancestor_ingest_state.json",
        help="Path to ingestion state marker (used to skip duplicate archive ingest on restart)",
    )
    parser.add_argument(
        "--refresh-config",
        action="store_true",
        help="Regenerate runtime config from cross + genotypes before launch",
    )
    parser.add_argument("--cross-tool", default="gggp_bundle/tools/cross_compose_cfg.py")
    parser.add_argument("--cross-file", default="gggp_bundle/evolution/crosses/abel_env.yaml")
    parser.add_argument("--genotypes-dir", default="gggp_bundle/evolution/genotypes")
    parser.add_argument("--out-dir", default="gggp_bundle/evolution/phenotypes")
    parser.add_argument("--compose-in", default="")
    args = parser.parse_args()

    try:
        maybe_refresh_config(args)
    except Exception as exc:
        log(f"Failed to refresh runtime config: {exc}")
        sys.exit(1)

    try:
        genes = load_runtime_config(args.runtime_config)
    except Exception as exc:
        log(f"Failed to load runtime config '{args.runtime_config}': {exc}")
        sys.exit(1)

    endpoints = genes.get("endpoints", {}) if isinstance(genes.get("endpoints"), dict) else {}
    memory_endpoint = str(endpoints.get("memory_endpoint", "http://soul_memory:8087"))
    runtime = genes.get("runtime", {}) if isinstance(genes.get("runtime"), dict) else {}
    archive_dir = str(runtime.get("archive_dir", "/app/Legacy/Archive"))
    soul = genes.get("soul", {}) if isinstance(genes.get("soul"), dict) else {}
    ancestor_ids = soul.get("ancestor_ids", [])
    if not isinstance(ancestor_ids, list):
        ancestor_ids = []

    if not wait_for_service(memory_endpoint, timeout=max(10, args.wait)):
        sys.exit(1)

    if not args.skip_ingest:
        state = load_ingest_state(args.ingest_state_path)
        state_ancestors = state.get("ancestors", {}) if isinstance(state.get("ancestors"), dict) else {}
        updated_state = {"ancestors": dict(state_ancestors)}
        total = 0
        for ancestor_id in ancestor_ids:
            ancestor = str(ancestor_id).strip()
            if not ancestor:
                continue
            sig = archive_signature(archive_dir, ancestor)
            if not args.force_ingest and state_ancestors.get(ancestor) == sig:
                log(f"Skipping ingest for '{ancestor}' (archive signature unchanged)")
                continue
            total += ingest_ancestor(memory_endpoint, archive_dir, ancestor)
            updated_state["ancestors"][ancestor] = sig
        save_ingest_state(args.ingest_state_path, updated_state)
        log(f"Total ingested ancestor memories: {total}")
    else:
        log("Skipping ancestor ingestion by request.")

    command = build_loop_command(genes, args.runtime_config)
    log("Executing genesis loop:")
    log(" ".join(command))
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
