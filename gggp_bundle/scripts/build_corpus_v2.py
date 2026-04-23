"""
gggp_bundle/scripts/build_corpus_v2.py

MEDP A2 / S0a -- paraphrase corpus generator (gamma.c LLMOnly), v2 edition.

Delta vs build_corpus_v1.py:
  * PARAPHRASES_PER_SEED = 32  (was 16)   -> corpus size 256 (was 128)
  * Output file: corpus_v2.jsonl          (v1 untouched for reproducibility)

Everything else -- seed list, prompt, parser, provider config, retry
logic, row schema -- is deliberately identical to v1 so that the 256
rows of v2 are a strict superset of the same 8 classes at a higher
paraphrase density. Rows of v1 are NOT reused verbatim: regenerating
from scratch preserves one-corpus-one-timestamp invariants.

Input:   8 hard-coded seed concepts from v1 (domain-diverse).
Output:  demos/semiotic_hypercube/corpus_v2.jsonl
         256 rows = 8 classes x 32 paraphrases.
         Schema (same as v1):
           {id, class_id, class_name, text, method, provider,
            model, seed, temperature, ts_utc}

Run:
    cd <repo-root>
    gggp_bundle/.venv/bin/python gggp_bundle/scripts/build_corpus_v2.py

Idempotent at filesystem level (overwrites output). NOT idempotent at
content level -- LLM outputs drift between runs. Intended flow:
generate once, commit the snapshot, downstream consumes the snapshot.

MEDP ref: A2 plan.md §Unit table (S0a) and §Locked Decisions Q1=256.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "gggp_bundle" / "scripts"))

from providers import load_provider, Provider, ProviderError  # noqa: E402

CONFIG_PATH = REPO_ROOT / "gggp_bundle" / "config" / "providers.toml"
OUT_PATH = (
    REPO_ROOT
    / "gggp_bundle"
    / "demos"
    / "semiotic_hypercube"
    / "corpus_v2.jsonl"
)

# ---------------------------------------------------------------------
# Seed concepts. IDENTICAL to v1 by design: v2 is "v1 at higher density",
# not a different domain sampling. Class count stays 8; class_id indexes
# this list.
# ---------------------------------------------------------------------
SEEDS: list[str] = [
    "sort integers",
    "extract dates from text",
    "classify support tickets",
    "translate Russian to English",
    "detect anomalies in time series",
    "summarize an article",
    "route a payment",
    "parse a configuration file",
]

# v2 raises per-class density 16 -> 32. Keeps 8 classes for a clean
# ground-truth ARI, but gives CMA-ES / EA substantially more within-class
# samples for NC1/NC2 metrics that rely on pair statistics.
PARAPHRASES_PER_SEED = 32
MAX_RETRIES = 2


PROMPT_TEMPLATE = """\
You are writing a dataset for a paraphrase-clustering benchmark.

Task description (the single underlying task): "{seed}"

Produce exactly {n} DIFFERENT English paraphrases that all describe
THE SAME underlying task. Vary:
  - syntax (active/passive, imperative/descriptive, command/question)
  - register (formal / casual / technical)
  - synonyms and domain vocabulary
  - length (from terse 3-word commands to longer instructions)

Hard rules:
  1. Every paraphrase must be a plausible description a programmer or
     operator would use to request THIS task. Do not drift to a
     different task.
  2. Exactly {n} paraphrases, one per line.
  3. NO numbering, NO bullets, NO markdown, NO explanations, NO
     blank lines between items. One paraphrase per line, nothing else.
  4. Each line must be one complete sentence or phrase, not two.

Output now, {n} lines, plain text:
/no_think
"""

NUMBER_PREFIX_RE = re.compile(
    r"^\s*(?:[-*\u2022]|\d+[.)\]]|[(\[]?\d+[)\]])\s+"
)


@dataclass
class CorpusRow:
    id: int
    class_id: int
    class_name: str
    text: str
    method: str
    provider: str
    model: str
    seed: int
    temperature: float
    ts_utc: str


def clean_line(raw: str) -> str:
    """Strip numbering/bullets/markdown emphasis, collapse whitespace."""
    s = raw.strip()
    s = NUMBER_PREFIX_RE.sub("", s)
    s = s.strip().strip("`").strip('"').strip("'")
    s = re.sub(r"\s+", " ", s)
    return s


def parse_paraphrases(response: str, target: int) -> list[str]:
    """Extract clean paraphrase lines from a chat response."""
    lines = [clean_line(ln) for ln in response.splitlines()]
    lines = [ln for ln in lines if ln and len(ln) >= 3]

    # De-dupe by case-insensitive, punctuation-stripped form.
    seen: set[str] = set()
    uniq: list[str] = []
    for ln in lines:
        key = re.sub(r"[^\w\s]", "", ln.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            uniq.append(ln)
        if len(uniq) == target:
            break
    return uniq


def generate_class(
    provider: Provider, seed: str, class_id: int
) -> list[str]:
    """Generate PARAPHRASES_PER_SEED unique paraphrases for a seed concept."""
    for attempt in range(MAX_RETRIES + 1):
        prompt = PROMPT_TEMPLATE.format(
            seed=seed, n=PARAPHRASES_PER_SEED
        )
        if attempt > 0:
            prompt = (
                f"(retry {attempt}: previous output had fewer than "
                f"{PARAPHRASES_PER_SEED} unique paraphrases. Try again, "
                f"with MORE variation.)\n\n" + prompt
            )
        t0 = time.time()
        try:
            raw = provider.chat(prompt)
        except ProviderError as exc:
            if attempt == MAX_RETRIES:
                raise
            print(
                f"[S0a] class={class_id} attempt={attempt} ProviderError: "
                f"{exc}; retrying...",
                file=sys.stderr,
            )
            continue
        elapsed = time.time() - t0
        uniq = parse_paraphrases(raw, PARAPHRASES_PER_SEED)
        print(
            f"[S0a] class={class_id} seed='{seed}' attempt={attempt} "
            f"got {len(uniq)}/{PARAPHRASES_PER_SEED} unique in {elapsed:.1f}s",
            file=sys.stderr,
        )
        if len(uniq) == PARAPHRASES_PER_SEED:
            return uniq

    raise RuntimeError(
        f"[S0a] class={class_id} seed='{seed}': failed to get "
        f"{PARAPHRASES_PER_SEED} unique paraphrases after "
        f"{MAX_RETRIES + 1} attempts. "
        f"Hint: increase MAX_RETRIES, try a larger chat_model, or relax "
        f"the dedup rule in parse_paraphrases()."
    )


def build_corpus(provider: Provider) -> list[CorpusRow]:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[CorpusRow] = []
    global_id = 0

    for class_id, seed in enumerate(SEEDS):
        paraphrases = generate_class(provider, seed, class_id)
        for p in paraphrases:
            rows.append(
                CorpusRow(
                    id=global_id,
                    class_id=class_id,
                    class_name=seed,
                    text=p,
                    method="gamma.c-LLMOnly",
                    provider=provider.name,
                    model=provider.chat_model,
                    seed=provider.chat_seed,
                    temperature=provider.chat_temperature,
                    ts_utc=ts,
                )
            )
            global_id += 1

    return rows


def main() -> None:
    provider = load_provider(CONFIG_PATH)
    print(
        f"[S0a] provider={provider.name} chat_model={provider.chat_model} "
        f"seed={provider.chat_seed} temp={provider.chat_temperature}",
        file=sys.stderr,
    )

    rows = build_corpus(provider)

    expected = len(SEEDS) * PARAPHRASES_PER_SEED
    assert len(rows) == expected, (
        f"corpus size mismatch: got {len(rows)}, expected {expected}"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    print(f"[S0a] wrote {len(rows)} rows to {OUT_PATH}")
    print(
        f"[S0a] classes: {len(SEEDS)}  "
        f"paraphrases/class: {PARAPHRASES_PER_SEED}",
    )


if __name__ == "__main__":
    main()
