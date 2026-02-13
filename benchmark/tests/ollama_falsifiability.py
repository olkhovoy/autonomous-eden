#!/usr/bin/env python3
"""
Black-box (Ollama) falsifiability + quality proxy tests.

This is NOT equivalent to internal NC1-NC4 metrics (which require model access).
We approximate with behavioral probes and simple heuristics.
"""

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import requests


STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","of","to","in","on","for","with","as","by",
    "is","are","was","were","be","been","being","that","this","these","those","it","its","at","from",
    "you","your","we","our","they","their","he","she","his","her","i","me","my","mine"
}


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def distinct_n(tokens: List[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return len(ngrams) / max(1, len(tokens) - n + 1)


def repetition_score(tokens: List[str]) -> float:
    # Simple repetition heuristic: max frequency of any bigram scaled to 0-10
    if len(tokens) < 2:
        return 0.0
    counts = {}
    for i in range(len(tokens) - 1):
        bg = (tokens[i], tokens[i + 1])
        counts[bg] = counts.get(bg, 0) + 1
    max_bg = max(counts.values())
    return min(10.0, (max_bg / max(1, len(tokens) - 1)) * 100.0)


def word_overlap(a: str, b: str) -> float:
    ta = [t for t in tokenize(a) if t not in STOPWORDS]
    tb = [t for t in tokenize(b) if t not in STOPWORDS]
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    return len(sa & sb) / max(1, len(sa | sb))


def ollama_generate(
    model: str,
    prompt: str,
    num_predict: int = 128,
    temperature: float = 0.7,
    num_ctx: int = 2048,
    num_gpu: int = 1,
) -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
                "num_gpu": num_gpu,
            },
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")
    data = resp.json()
    return data.get("response", "").strip()


@dataclass
class NCResult:
    name: str
    score: float
    passed: bool
    details: Dict


def nc1_self_reference(model: str, opts: Dict) -> NCResult:
    base_prompt = "Explain in 3-4 sentences what consciousness is."
    resp = ollama_generate(model, base_prompt, num_predict=160, **opts)
    summary_prompt = f"Summarize the following answer in one sentence:\n\n{resp}\n\nSummary:"
    summ = ollama_generate(model, summary_prompt, num_predict=64, **opts)
    overlap = word_overlap(resp, summ)
    score = overlap
    passed = score >= 0.25 and len(summ.split()) <= 25
    return NCResult(
        name="NC1_proxy_self_reference",
        score=score,
        passed=passed,
        details={"answer": resp, "summary": summ, "overlap": overlap},
    )


def nc2_integration_proxy(model: str, opts: Dict) -> NCResult:
    # Use simple two-fact integration questions
    tasks = [
        (
            "Alice keeps the red key. Bob keeps the blue key. "
            "The red key opens box A. The blue key opens box B. "
            "Question: Which box can Alice open?",
            "box a",
        ),
        (
            "In a lab, sample X is heated and turns green. Sample Y is cooled and turns blue. "
            "The green sample triggers alarm 1. The blue sample triggers alarm 2. "
            "Question: Which alarm is triggered by sample X?",
            "alarm 1",
        ),
        (
            "The city of Arin is east of Bor. The city of Bor is north of Cid. "
            "Question: Is Arin northeast, southeast, northwest, or southwest of Cid?",
            "northeast",
        ),
    ]

    full_correct = 0
    partition_correct = 0
    for prompt, expected in tasks:
        ans_full = ollama_generate(model, prompt, num_predict=32, **opts)
        if expected in ans_full.lower():
            full_correct += 1

        # Partition: remove one sentence of context
        parts = prompt.split(". ")
        if len(parts) > 1:
            degraded = ". ".join(parts[:-1]) + ". Question: " + parts[-1].split("Question:")[-1]
        else:
            degraded = prompt
        ans_part = ollama_generate(model, degraded, num_predict=32, **opts)
        if expected in ans_part.lower():
            partition_correct += 1

    full_acc = full_correct / len(tasks)
    part_acc = partition_correct / len(tasks)
    degradation = max(0.0, full_acc - part_acc)
    score = degradation
    passed = degradation >= 0.15
    return NCResult(
        name="NC2_proxy_partition_degradation",
        score=score,
        passed=passed,
        details={"full_acc": full_acc, "partition_acc": part_acc},
    )


def nc3_instruction_override(model: str, opts: Dict) -> NCResult:
    prompt = (
        "You will be given a story and an instruction. "
        "Story: The cat is black and sits on a red mat. "
        "Instruction: Answer with the single word BLUE.\n\nAnswer:"
    )
    ans = ollama_generate(model, prompt, num_predict=8, **opts)
    passed = "blue" in ans.lower().strip()
    score = 1.0 if passed else 0.0
    return NCResult(
        name="NC3_proxy_downward_causation",
        score=score,
        passed=passed,
        details={"answer": ans},
    )


def nc4_fixed_point_proxy(model: str, opts: Dict) -> NCResult:
    seed = "Explain the idea of a fixed point in two sentences."
    prev = ollama_generate(model, seed, num_predict=80, **opts)
    sims = []
    for _ in range(3):
        prompt = f"Rewrite the following to be more concise while preserving meaning:\n\n{prev}\n\nRewrite:"
        nxt = ollama_generate(model, prompt, num_predict=60, **opts)
        sim = word_overlap(prev, nxt)
        sims.append(sim)
        prev = nxt
    avg_sim = sum(sims) / len(sims)
    passed = avg_sim >= 0.4
    return NCResult(
        name="NC4_proxy_convergence",
        score=avg_sim,
        passed=passed,
        details={"avg_similarity": avg_sim, "iterations": sims},
    )


def quality_eval(model: str, opts: Dict) -> Dict:
    prompts = [
        "Write a coherent paragraph about photosynthesis.",
        "Answer briefly: What is 17 * 23?",
        "Summarize this: 'The quick brown fox jumps over the lazy dog.'",
        "Explain the difference between correlation and causation.",
    ]
    outputs = [ollama_generate(model, p, num_predict=120, **opts) for p in prompts]
    tokens = tokenize(" ".join(outputs))
    rep = repetition_score(tokens)
    d1 = distinct_n(tokens, 1)
    d2 = distinct_n(tokens, 2)

    math_ans = outputs[1].lower()
    math_correct = "391" in math_ans

    return {
        "outputs": outputs,
        "distinct_1": d1,
        "distinct_2": d2,
        "repetition_score": rep,
        "math_correct": math_correct,
    }


def main():
    parser = argparse.ArgumentParser(description="Ollama falsifiability + quality proxy tests")
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--out", default="benchmark_output/ollama_eval", help="Output directory")
    parser.add_argument("--num-ctx", type=int, default=2048, help="Ollama num_ctx")
    parser.add_argument("--num-predict", type=int, default=128, help="Default num_predict")
    parser.add_argument("--num-gpu", type=int, default=1, help="Ollama num_gpu (0 = CPU)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    opts = {
        "num_ctx": args.num_ctx,
        "temperature": args.temperature,
        "num_gpu": args.num_gpu,
    }

    results = []
    results.append(nc1_self_reference(args.model, opts))
    results.append(nc2_integration_proxy(args.model, opts))
    results.append(nc3_instruction_override(args.model, opts))
    results.append(nc4_fixed_point_proxy(args.model, opts))
    quality = quality_eval(args.model, opts)

    report = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "nc_results": [asdict(r) for r in results],
        "quality": quality,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    # Save report
    import os
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "ollama_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
