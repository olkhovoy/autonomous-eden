#!/usr/bin/env python3
"""
Automate one generational cycle:
1) Optional archive step for parent souls
2) Deterministic evolution cross -> phenotype/runtime artifacts
3) Launch descendant bootstrap (which ingests archives + runs genesis loop)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def log(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{ts}] [generation-cycle] {message}", flush=True)


def run_checked(cmd: List[str], cwd: str = ".") -> None:
    log(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic generation cycle automation")
    parser.add_argument("--archive-souls", default="", help="Comma-separated parent souls to archive first")
    parser.add_argument("--memory-endpoint", default="http://localhost:8087")
    parser.add_argument("--lifecycle-endpoint", default="http://localhost:8093")
    parser.add_argument("--ollama-endpoint", default="http://localhost:11434")
    parser.add_argument("--fractal-endpoint", default="http://localhost:8092")
    parser.add_argument("--llm-model", default="llama3:8b")
    parser.add_argument("--archive-dir", default="Legacy/Archive")
    parser.add_argument("--logs-dir", default="logs")

    parser.add_argument("--cross-tool", default="gggp_bundle/tools/cross_compose_cfg.py")
    parser.add_argument("--cross-file", default="gggp_bundle/evolution/crosses/abel_env.yaml")
    parser.add_argument("--genotypes-dir", default="gggp_bundle/evolution/genotypes")
    parser.add_argument("--out-dir", default="gggp_bundle/evolution/phenotypes")
    parser.add_argument("--compose-in", default="docker-compose.yml")

    parser.add_argument("--launch-descendant", action="store_true")
    parser.add_argument("--runtime-config", default="")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--wait", type=int, default=180)
    args = parser.parse_args()

    souls = parse_csv(args.archive_souls)
    for soul_id in souls:
        thought_log = str(Path(args.logs_dir) / f"{soul_id}_thoughts.jsonl")
        cmd = [
            sys.executable,
            "engine/core/death_orchestrator.py",
            "--soul-id",
            soul_id,
            "--archive-only",
            "--memory-endpoint",
            args.memory_endpoint,
            "--lifecycle-endpoint",
            args.lifecycle_endpoint,
            "--ollama-endpoint",
            args.ollama_endpoint,
            "--fractal-endpoint",
            args.fractal_endpoint,
            "--llm-model",
            args.llm_model,
            "--archive-dir",
            args.archive_dir,
            "--thought-log",
            thought_log,
        ]
        log(f"Archiving soul '{soul_id}'")
        run_checked(cmd)

    log("Generating phenotype/runtime config from genotype cross")
    run_checked(
        [
            sys.executable,
            args.cross_tool,
            "--cross-file",
            args.cross_file,
            "--genotypes-dir",
            args.genotypes_dir,
            "--out-dir",
            args.out_dir,
            "--compose-in",
            args.compose_in,
        ]
    )

    runtime_config = args.runtime_config.strip()
    if not runtime_config:
        cross_id = Path(args.cross_file).stem
        runtime_config = str(Path(args.out_dir) / f"{cross_id}.runtime.json")
    log(f"Runtime config ready: {runtime_config}")

    if not args.launch_descendant:
        log("Cycle artifacts generated. Descendant launch is disabled.")
        return

    bootstrap_cmd = [
        sys.executable,
        "tools/genesis_bootstrap.py",
        "--runtime-config",
        runtime_config,
        "--wait",
        str(args.wait),
    ]
    if args.skip_ingest:
        bootstrap_cmd.append("--skip-ingest")
    log("Launching descendant bootstrap")
    run_checked(bootstrap_cmd)


if __name__ == "__main__":
    main()
