#!/usr/bin/env python3
"""
Benchmark Ollama models on ShareGPT prompts with vLLM-like throughput metrics.

Outputs JSON per model with:
- request_throughput (req/s)
- output_throughput (generated tok/s)
- total_token_throughput (prompt+generated tok/s)
- mean_ttft_ms

This script is designed to compare models under identical local Ollama settings.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

DATASET_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/"
    "resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Ollama models on ShareGPT")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names in Ollama, e.g. llama3:8b forzer/GigaChat3-10B-A1.8B:latest",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/ShareGPT_V3_unfiltered_cleaned_split.json"),
        help="Path to ShareGPT JSON dataset",
    )
    parser.add_argument(
        "--dataset-url",
        default=DATASET_URL,
        help="Dataset URL used if dataset-path does not exist",
    )
    parser.add_argument("--num-prompts", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="Ollama host base URL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("serving_bench_ollama"),
    )
    parser.add_argument(
        "--min-prompt-chars",
        type=int,
        default=8,
        help="Drop extremely short prompts",
    )
    parser.add_argument(
        "--max-predict",
        type=int,
        default=256,
        help="Upper bound for per-request output token budget",
    )
    parser.add_argument(
        "--min-predict",
        type=int,
        default=16,
        help="Lower bound for per-request output token budget",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=600.0,
        help="Per-request timeout (seconds)",
    )
    parser.add_argument(
        "--keep-alive",
        default="30m",
        help="Ollama keep_alive for each request",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one warmup request before timed benchmark",
    )
    parser.add_argument(
        "--stop-active-models",
        action="store_true",
        help="Stop already loaded Ollama models before each model run",
    )
    parser.add_argument(
        "--sample-cache",
        type=Path,
        default=None,
        help="Optional path to persist sampled prompts for reproducibility",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry count per failed request before aborting",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort benchmark on first request that fails after retries",
    )
    return parser.parse_args()


def ensure_ollama_up(host: str, timeout: float = 10.0) -> None:
    url = f"{host.rstrip('/')}/api/tags"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Ollama is not reachable at {url}: {exc}") from exc


def download_dataset(dataset_url: str, dataset_path: Path) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dataset_path.with_suffix(dataset_path.suffix + ".part")
    print(f"Downloading dataset to {dataset_path} ...", flush=True)
    with requests.get(dataset_url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        written = 0
        chunk_size = 1024 * 1024
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if total > 0 and written % (25 * chunk_size) < chunk_size:
                    pct = (written / total) * 100
                    print(f"  downloaded {written / 1e6:.1f}MB / {total / 1e6:.1f}MB ({pct:.1f}%)")
    tmp.rename(dataset_path)


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    print(f"Loading dataset: {dataset_path}", flush=True)
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Unexpected dataset format: top-level JSON is not a list")
    return data


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _estimate_tokens(text: str, min_predict: int, max_predict: int) -> int:
    # Rough multilingual token estimate used only to set num_predict budget.
    approx = max(1, int(math.ceil(len(text) / 4.0)))
    return max(min_predict, min(max_predict, approx))


def extract_sharegpt_samples(
    raw_data: list[dict[str, Any]],
    *,
    num_prompts: int,
    seed: int,
    min_prompt_chars: int,
    min_predict: int,
    max_predict: int,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []

    for item in raw_data:
        conv = item.get("conversations")
        if not isinstance(conv, list) or not conv:
            continue

        user_text: str | None = None
        assistant_text: str | None = None

        for turn in conv:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", "")).lower()
            value = turn.get("value", "")
            if not isinstance(value, str):
                continue
            value = _clean_text(value)
            if not value:
                continue

            if user_text is None and role in {"human", "user"}:
                user_text = value
                continue

            if user_text is not None and assistant_text is None and role in {"gpt", "assistant"}:
                assistant_text = value
                break

        if user_text is None or assistant_text is None:
            continue
        if len(user_text) < min_prompt_chars:
            continue

        pairs.append(
            {
                "prompt": user_text,
                "target_tokens": _estimate_tokens(
                    assistant_text,
                    min_predict=min_predict,
                    max_predict=max_predict,
                ),
            }
        )

    if len(pairs) < num_prompts:
        raise ValueError(
            f"Not enough valid prompt/response pairs: found {len(pairs)}, need {num_prompts}"
        )

    rng = random.Random(seed)
    return rng.sample(pairs, num_prompts)


def maybe_stop_active_models(except_model: str | None = None) -> None:
    ps = subprocess.run(
        ["ollama", "ps"],
        check=False,
        capture_output=True,
        text=True,
    )
    if ps.returncode != 0:
        return

    lines = [line.strip() for line in ps.stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return

    for line in lines[1:]:
        name = line.split()[0]
        if except_model and name == except_model:
            continue
        subprocess.run(["ollama", "stop", name], check=False, capture_output=True)


def run_single_request(
    *,
    host: str,
    model: str,
    prompt: str,
    target_tokens: int,
    temperature: float,
    timeout_s: float,
    keep_alive: str,
) -> dict[str, Any]:
    url = f"{host.rstrip('/')}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
            "num_predict": target_tokens,
        },
    }

    start = time.perf_counter()
    first_token_ts: float | None = None
    final_chunk: dict[str, Any] | None = None

    with requests.post(url, json=payload, stream=True, timeout=(10.0, timeout_s)) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            chunk = json.loads(raw_line)
            if first_token_ts is None and chunk.get("response"):
                first_token_ts = time.perf_counter()
            if chunk.get("done"):
                final_chunk = chunk
                break

    end = time.perf_counter()

    if final_chunk is None:
        raise RuntimeError("Stream ended without final done chunk")

    if first_token_ts is None:
        ttft_ms = (end - start) * 1000.0
    else:
        ttft_ms = (first_token_ts - start) * 1000.0

    return {
        "wall_time_s": end - start,
        "ttft_ms": ttft_ms,
        "prompt_eval_count": int(final_chunk.get("prompt_eval_count", 0) or 0),
        "eval_count": int(final_chunk.get("eval_count", 0) or 0),
        "prompt_eval_duration_ns": int(final_chunk.get("prompt_eval_duration", 0) or 0),
        "eval_duration_ns": int(final_chunk.get("eval_duration", 0) or 0),
        "total_duration_ns": int(final_chunk.get("total_duration", 0) or 0),
        "load_duration_ns": int(final_chunk.get("load_duration", 0) or 0),
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def benchmark_model(
    *,
    host: str,
    model: str,
    samples: list[dict[str, Any]],
    temperature: float,
    timeout_s: float,
    keep_alive: str,
    warmup: bool,
    max_retries: int,
    fail_fast: bool,
) -> dict[str, Any]:
    print(f"\n=== Benchmarking {model} ({len(samples)} prompts) ===", flush=True)

    if warmup:
        warmup_sample = samples[0]
        print("Running warmup request...", flush=True)
        _ = run_single_request(
            host=host,
            model=model,
            prompt=warmup_sample["prompt"],
            target_tokens=min(32, int(warmup_sample["target_tokens"])),
            temperature=temperature,
            timeout_s=timeout_s,
            keep_alive=keep_alive,
        )

    per_request: list[dict[str, Any]] = []
    failed_requests: list[dict[str, Any]] = []
    started = time.perf_counter()

    for idx, sample in enumerate(samples, start=1):
        attempt = 0
        result: dict[str, Any] | None = None
        while True:
            attempt += 1
            try:
                result = run_single_request(
                    host=host,
                    model=model,
                    prompt=sample["prompt"],
                    target_tokens=int(sample["target_tokens"]),
                    temperature=temperature,
                    timeout_s=timeout_s,
                    keep_alive=keep_alive,
                )
                break
            except Exception as exc:  # noqa: BLE001
                if attempt > max_retries:
                    if fail_fast:
                        raise RuntimeError(
                            f"Request {idx}/{len(samples)} failed after {max_retries} retries: {exc}"
                        ) from exc
                    failed_requests.append(
                        {
                            "index": idx,
                            "error": str(exc),
                            "retries": max_retries,
                        }
                    )
                    print(
                        f"  request {idx}/{len(samples)} failed after retries and will be skipped: {exc}",
                        flush=True,
                    )
                    break
                backoff_s = min(10.0, float(2**attempt))
                print(
                    f"  request {idx}/{len(samples)} failed (attempt {attempt}/{max_retries}): {exc}; "
                    f"retrying in {backoff_s:.1f}s",
                    flush=True,
                )
                time.sleep(backoff_s)
        if result is not None:
            per_request.append(result)

        if idx % 25 == 0 or idx == len(samples):
            done_pct = (idx / len(samples)) * 100
            avg_ttft = statistics.mean(r["ttft_ms"] for r in per_request) if per_request else 0.0
            print(
                f"  progress: {idx}/{len(samples)} ({done_pct:.1f}%)"
                f" | avg_ttft={avg_ttft:.1f}ms"
                f" | failed={len(failed_requests)}",
                flush=True,
            )

    ended = time.perf_counter()

    if not per_request:
        raise RuntimeError("All requests failed; no metrics to report.")

    total_wall_s = ended - started
    total_prompt_tokens = sum(r["prompt_eval_count"] for r in per_request)
    total_output_tokens = sum(r["eval_count"] for r in per_request)
    total_tokens = total_prompt_tokens + total_output_tokens

    ttft_values = [r["ttft_ms"] for r in per_request]
    prompt_eval_duration_s = sum(r["prompt_eval_duration_ns"] for r in per_request) / 1e9
    eval_duration_s = sum(r["eval_duration_ns"] for r in per_request) / 1e9

    result = {
        "model": model,
        "num_prompts": len(samples),
        "completed_prompts": len(per_request),
        "failed_prompts": len(failed_requests),
        "seed": None,
        "metrics": {
            "request_throughput": len(per_request) / total_wall_s if total_wall_s > 0 else 0.0,
            "output_throughput": total_output_tokens / total_wall_s if total_wall_s > 0 else 0.0,
            "total_token_throughput": total_tokens / total_wall_s if total_wall_s > 0 else 0.0,
            "mean_ttft_ms": statistics.mean(ttft_values) if ttft_values else 0.0,
        },
        "extra_metrics": {
            "p50_ttft_ms": percentile(ttft_values, 0.5),
            "p95_ttft_ms": percentile(ttft_values, 0.95),
            "mean_prompt_eval_tps": (
                total_prompt_tokens / prompt_eval_duration_s if prompt_eval_duration_s > 0 else 0.0
            ),
            "mean_eval_tps": total_output_tokens / eval_duration_s if eval_duration_s > 0 else 0.0,
            "total_wall_time_s": total_wall_s,
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "mean_total_duration_ms": (
                statistics.mean(r["total_duration_ns"] for r in per_request) / 1e6 if per_request else 0.0
            ),
            "mean_load_duration_ms": (
                statistics.mean(r["load_duration_ns"] for r in per_request) / 1e6 if per_request else 0.0
            ),
        },
        "failed_requests": failed_requests,
    }

    return result


def model_slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model)


def main() -> int:
    args = parse_args()

    ensure_ollama_up(args.host)

    if not args.dataset_path.exists():
        download_dataset(args.dataset_url, args.dataset_path)

    raw_data = load_dataset(args.dataset_path)
    samples = extract_sharegpt_samples(
        raw_data,
        num_prompts=args.num_prompts,
        seed=args.seed,
        min_prompt_chars=args.min_prompt_chars,
        min_predict=args.min_predict,
        max_predict=args.max_predict,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_cache is not None:
        args.sample_cache.parent.mkdir(parents=True, exist_ok=True)
        with args.sample_cache.open("w", encoding="utf-8") as f:
            for row in samples:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary: list[dict[str, Any]] = []

    for model in args.models:
        if args.stop_active_models:
            maybe_stop_active_models(except_model=None)

        result = benchmark_model(
            host=args.host,
            model=model,
            samples=samples,
            temperature=args.temperature,
            timeout_s=args.request_timeout_s,
            keep_alive=args.keep_alive,
            warmup=args.warmup,
            max_retries=args.max_retries,
            fail_fast=args.fail_fast,
        )
        result["seed"] = args.seed

        out_file = args.output_dir / f"{model_slug(model)}_throughput_serving.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Saved: {out_file}", flush=True)
        summary.append(result)

    summary_file = args.output_dir / "summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved: {summary_file}")

    print("\n=== Summary ===")
    for row in summary:
        m = row["metrics"]
        print(
            f"{row['model']}: "
            f"req/s={m['request_throughput']:.3f}, "
            f"out_tok/s={m['output_throughput']:.3f}, "
            f"total_tok/s={m['total_token_throughput']:.3f}, "
            f"mean_ttft_ms={m['mean_ttft_ms']:.3f}"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
