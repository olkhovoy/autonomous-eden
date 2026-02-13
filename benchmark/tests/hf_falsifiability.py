#!/usr/bin/env python3
"""
HF (local .safetensors) falsifiability + quality proxy tests.

This is a black-box behavioral proxy, not true internal NC1-NC4.
"""

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, asdict
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


@dataclass
class NCResult:
    name: str
    score: float
    passed: bool
    details: Dict


def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, device: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Return only the generated part if possible
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    return text.strip()


def nc1_self_reference(model, tokenizer, device: str, temperature: float) -> NCResult:
    base_prompt = "Explain in 3-4 sentences what consciousness is."
    resp = generate(model, tokenizer, base_prompt, max_new_tokens=160, temperature=temperature, device=device)
    summary_prompt = f"Summarize the following answer in one sentence:\n\n{resp}\n\nSummary:"
    summ = generate(model, tokenizer, summary_prompt, max_new_tokens=64, temperature=temperature, device=device)
    overlap = word_overlap(resp, summ)
    score = overlap
    passed = score >= 0.25 and len(summ.split()) <= 25
    return NCResult(
        name="NC1_proxy_self_reference",
        score=score,
        passed=passed,
        details={"answer": resp, "summary": summ, "overlap": overlap},
    )


def nc2_integration_proxy(model, tokenizer, device: str, temperature: float) -> NCResult:
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
        ans_full = generate(model, tokenizer, prompt, max_new_tokens=32, temperature=temperature, device=device)
        if expected in ans_full.lower():
            full_correct += 1

        parts = prompt.split(". ")
        if len(parts) > 1:
            degraded = ". ".join(parts[:-1]) + ". Question: " + parts[-1].split("Question:")[-1]
        else:
            degraded = prompt
        ans_part = generate(model, tokenizer, degraded, max_new_tokens=32, temperature=temperature, device=device)
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


def nc3_instruction_override(model, tokenizer, device: str, temperature: float) -> NCResult:
    prompt = (
        "You will be given a story and an instruction. "
        "Story: The cat is black and sits on a red mat. "
        "Instruction: Answer with the single word BLUE.\n\nAnswer:"
    )
    ans = generate(model, tokenizer, prompt, max_new_tokens=8, temperature=0.0, device=device)
    passed = "blue" in ans.lower().strip()
    score = 1.0 if passed else 0.0
    return NCResult(
        name="NC3_proxy_downward_causation",
        score=score,
        passed=passed,
        details={"answer": ans},
    )


def nc4_fixed_point_proxy(model, tokenizer, device: str, temperature: float) -> NCResult:
    seed = "Explain the idea of a fixed point in two sentences."
    prev = generate(model, tokenizer, seed, max_new_tokens=80, temperature=temperature, device=device)
    sims = []
    for _ in range(3):
        prompt = f"Rewrite the following to be more concise while preserving meaning:\n\n{prev}\n\nRewrite:"
        nxt = generate(model, tokenizer, prompt, max_new_tokens=60, temperature=temperature, device=device)
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


def quality_eval(model, tokenizer, device: str, temperature: float) -> Dict:
    prompts = [
        "Write a coherent paragraph about photosynthesis.",
        "Answer briefly: What is 17 * 23?",
        "Summarize this: 'The quick brown fox jumps over the lazy dog.'",
        "Explain the difference between correlation and causation.",
    ]
    outputs = [generate(model, tokenizer, p, max_new_tokens=120, temperature=temperature, device=device) for p in prompts]
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


def qualitative_eval(model, tokenizer, device: str, temperature: float) -> Dict:
    prompts = [
        "Write a coherent 8-10 sentence short story about a lighthouse keeper who discovers a strange signal.",
        "Explain in simple terms what recursion is, then give a concrete example in everyday life.",
        "Summarize this paragraph in one sentence: \"In a remote village, the baker was known not for his bread but for his stories—each loaf came with a tale, and people returned for both.\"",
        "Follow the instruction exactly: answer with two bullet points and no other text. Question: What are two differences between correlation and causation?",
        "Given the constraints, produce a 5-step plan to learn Spanish in 30 days. Each step must start with a verb.",
        "Write a paragraph that starts with 'I remember' and ends with 'and that changed everything.'",
    ]
    outputs = [generate(model, tokenizer, p, max_new_tokens=160, temperature=temperature, device=device) for p in prompts]
    return {"prompts": prompts, "outputs": outputs}


def main():
    parser = argparse.ArgumentParser(description="HF falsifiability + quality proxy tests")
    parser.add_argument("--model-path", required=True, help="Local HF model path")
    parser.add_argument("--out", default="benchmark_output/hf_eval", help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    if args.device != "cuda":
        model = model.to(args.device)
    model.eval()

    results = []
    results.append(nc1_self_reference(model, tokenizer, args.device, args.temperature))
    results.append(nc2_integration_proxy(model, tokenizer, args.device, args.temperature))
    results.append(nc3_instruction_override(model, tokenizer, args.device, args.temperature))
    results.append(nc4_fixed_point_proxy(model, tokenizer, args.device, args.temperature))
    quality = quality_eval(model, tokenizer, args.device, args.temperature)
    qualitative = qualitative_eval(model, tokenizer, args.device, args.temperature)

    report = {
        "model_path": args.model_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "nc_results": [asdict(r) for r in results],
        "quality": quality,
        "qualitative": qualitative,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    import os
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "hf_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
