#!/usr/bin/env python3
"""Run vLLM throughput benchmark + small quality slice for selected models."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class ModelSpec:
    label: str
    model: str
    trust_remote_code: bool = False
    speculative_config: dict[str, Any] | None = None
    load_format: str | None = None
    generation_config: str | None = None


DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec(
        label="gigachat3_ollama_gguf",
        model="/usr/share/ollama/.ollama/models/blobs/sha256-d5b2c4b50565f503d00c5f1c334d45bdc219e9b2e6293da84018343ceac09e59",
        load_format="gguf",
        generation_config="vllm",
    ),
    ModelSpec(
        label="llama3_8b_ollama_gguf",
        model="/usr/share/ollama/.ollama/models/blobs/sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa",
        load_format="gguf",
        generation_config="vllm",
    ),
    ModelSpec(
        label="mistral_7b_ollama_gguf",
        model="/usr/share/ollama/.ollama/models/blobs/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f",
        load_format="gguf",
        generation_config="vllm",
    ),
]

QUALITY_PROMPTS: list[dict[str, str]] = [
    {
        "id": "ru_networking",
        "prompt": (
            "Объясни разницу между TCP и UDP в 6 пунктах, "
            "и добавь по одному практическому примеру для каждого протокола."
        ),
    },
    {
        "id": "python_code",
        "prompt": (
            "Write a Python function rotate_matrix_clockwise(matrix) that rotates "
            "an N x N matrix in-place. Then provide 3 simple tests."
        ),
    },
    {
        "id": "math_reasoning",
        "prompt": (
            "Solve step by step: A store gives 20% discount and then adds 10% tax. "
            "What is the final price for an item that initially costs $250?"
        ),
    },
    {
        "id": "ru_summary",
        "prompt": (
            "Сделай краткое резюме в 5 пунктах:\n"
            "Large language models are useful for coding assistance, summarization, "
            "and classification. However, their latency and infrastructure cost can "
            "differ dramatically depending on architecture, quantization, and serving stack."
        ),
    },
    {
        "id": "system_design",
        "prompt": (
            "Give a concise design for a rate limiter for an HTTP API handling 50k RPS. "
            "Mention data structures, distributed consistency, and failure modes."
        ),
    },
    {
        "id": "sql",
        "prompt": (
            "Write an SQL query to return top 3 products by revenue in the last 30 days "
            "from orders(order_id, created_at), order_items(order_id, product_id, qty, price)."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", default="data/ShareGPT_V3_unfiltered_cleaned_split.json")
    p.add_argument("--num-prompts", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-num-batched-tokens", type=int, default=1024)
    p.add_argument("--output-dir", default="serving_bench_vllm")
    p.add_argument("--ready-timeout-s", type=int, default=3600)
    p.add_argument("--request-timeout-s", type=int, default=1800)
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument(
        "--models",
        nargs="*",
        choices=[m.label for m in DEFAULT_MODELS],
        help="Subset of predefined labels to run",
    )
    return p.parse_args()


def sh(cmd: list[str], *, env: dict[str, str] | None = None, log_file: Path | None = None) -> int:
    stdout = None
    stderr = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fp = log_file.open("a", encoding="utf-8")
        stdout = fp
        stderr = subprocess.STDOUT
    else:
        fp = None
    try:
        proc = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr, text=True)
        return proc.returncode
    finally:
        if fp is not None:
            fp.close()


def wait_ready(base_url: str, timeout_s: int, proc: subprocess.Popen[str] | None = None) -> None:
    start = time.time()
    last_err: str | None = None
    while time.time() - start < timeout_s:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {proc.returncode}")
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=10)
            if r.status_code == 200:
                return
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        time.sleep(2)
    raise TimeoutError(f"Server not ready within {timeout_s}s. Last error: {last_err}")


def start_server(
    spec: ModelSpec,
    *,
    host: str,
    port: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    out_dir: Path,
) -> tuple[subprocess.Popen[str], str, Path]:
    served_name = spec.label
    log_file = out_dir / "logs" / f"{spec.label}_serve.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "vllm",
        "serve",
        spec.model,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        "auto",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--served-model-name",
        served_name,
    ]
    if spec.trust_remote_code:
        cmd.append("--trust-remote-code")
    if spec.load_format is not None:
        cmd.extend(["--load-format", spec.load_format])
    if spec.generation_config is not None:
        cmd.extend(["--generation-config", spec.generation_config])
    if spec.speculative_config is not None:
        cmd.extend(["--speculative-config", json.dumps(spec.speculative_config, ensure_ascii=False)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_DISABLE_XET"] = "1"

    fp = log_file.open("w", encoding="utf-8")
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=fp,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        preexec_fn=os.setsid,
    )
    return proc, served_name, log_file


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    for _ in range(40):
        if proc.poll() is not None:
            return
        time.sleep(0.5)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def run_benchmark(
    spec: ModelSpec,
    *,
    served_name: str,
    dataset_path: Path,
    num_prompts: int,
    seed: int,
    host: str,
    port: int,
    out_dir: Path,
) -> Path:
    filename = f"{spec.label}_throughput_serving.json"
    bench_log = out_dir / "logs" / f"{spec.label}_bench.log"

    cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        spec.model,
        "--served-model-name",
        served_name,
        "--dataset-name",
        "sharegpt",
        "--dataset-path",
        str(dataset_path),
        "--num-prompts",
        str(num_prompts),
        "--seed",
        str(seed),
        "--max-concurrency",
        "1",
        "--ignore-eos",
        "--save-result",
        "--result-dir",
        str(out_dir),
        "--result-filename",
        filename,
        "--ready-check-timeout-sec",
        "3600",
    ]

    code = sh(cmd, log_file=bench_log)
    if code != 0:
        raise RuntimeError(f"Benchmark failed for {spec.label}. See: {bench_log}")

    return out_dir / filename


def run_quality_slice(
    *,
    served_name: str,
    base_url: str,
    timeout_s: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in QUALITY_PROMPTS:
        payload = {
            "model": served_name,
            "prompt": item["prompt"],
            "max_tokens": 220,
            "temperature": 0.2,
            "top_p": 0.95,
        }
        t0 = time.perf_counter()
        r = requests.post(f"{base_url}/v1/completions", json=payload, timeout=timeout_s)
        dt = time.perf_counter() - t0
        r.raise_for_status()
        obj = r.json()
        text = ""
        if obj.get("choices"):
            text = obj["choices"][0].get("text", "")
        rows.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "response": text.strip(),
                "latency_s": dt,
                "usage": obj.get("usage", {}),
            }
        )
    return rows


def extract_key_metrics(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        # vllm can append list entries; take the last one.
        data = data[-1]

    keys = [
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "completed",
        "total_input_tokens",
        "total_output_tokens",
    ]
    out = {k: data.get(k) for k in keys if k in data}

    # If metrics nested under `metrics`, lift the same keys.
    if "metrics" in data and isinstance(data["metrics"], dict):
        for k in keys:
            if k in data["metrics"]:
                out[k] = data["metrics"][k]

    return out


def main() -> int:
    args = parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = DEFAULT_MODELS
    if args.models:
        selected_labels = set(args.models)
        selected = [m for m in DEFAULT_MODELS if m.label in selected_labels]

    base_url = f"http://{args.host}:{args.port}"

    summary: list[dict[str, Any]] = []

    for spec in selected:
        print(f"\n=== {spec.label} :: {spec.model} ===", flush=True)
        proc: subprocess.Popen[str] | None = None
        serve_log: Path | None = None
        try:
            proc, served_name, serve_log = start_server(
                spec,
                host=args.host,
                port=args.port,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_seqs=args.max_num_seqs,
                max_num_batched_tokens=args.max_num_batched_tokens,
                out_dir=out_dir,
            )
            wait_ready(base_url, args.ready_timeout_s, proc=proc)
            bench_file = run_benchmark(
                spec,
                served_name=served_name,
                dataset_path=dataset_path,
                num_prompts=args.num_prompts,
                seed=args.seed,
                host=args.host,
                port=args.port,
                out_dir=out_dir,
            )
            metrics = extract_key_metrics(bench_file)

            quality_file = out_dir / f"{spec.label}_quality.json"
            if not args.skip_quality:
                quality_rows = run_quality_slice(
                    served_name=served_name,
                    base_url=base_url,
                    timeout_s=args.request_timeout_s,
                )
                quality_file.write_text(
                    json.dumps(quality_rows, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            else:
                quality_rows = []

            summary.append(
                {
                    "label": spec.label,
                    "model": spec.model,
                    "speculative_config": spec.speculative_config,
                    "load_format": spec.load_format,
                    "generation_config": spec.generation_config,
                    "benchmark_file": str(bench_file),
                    "quality_file": str(quality_file) if quality_rows else None,
                    "metrics": metrics,
                    "status": "ok",
                }
            )
            print(f"Done: {spec.label}", flush=True)

        except Exception as e:  # noqa: BLE001
            summary.append(
                {
                    "label": spec.label,
                    "model": spec.model,
                    "speculative_config": spec.speculative_config,
                    "load_format": spec.load_format,
                    "generation_config": spec.generation_config,
                    "status": "failed",
                    "error": str(e),
                    "serve_log": str(serve_log) if serve_log else None,
                }
            )
            print(f"FAILED: {spec.label}: {e}", flush=True)

        finally:
            if proc is not None:
                stop_server(proc)
            time.sleep(3)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
