"""
gggp_bundle/scripts/build_corpus_v1.py

MEDP A1 / T1 -- paraphrase corpus generator (gamma.c LLMOnly).

Input:   8 hard-coded seed concepts (domain-diverse).
Output:  demos/semiotic_hypercube/corpus_v1.jsonl
         128 rows = 8 classes x 16 paraphrases.
         Each row:  {id, class_id, class_name, text, method, provider,
                     model, seed, temperature, ts_utc}

Strategy:
  * One chat call per class, asking for 16 paraphrases in a single
    response. This amortizes the Qwen3 CoT cost (~200 tokens of
    thinking per call, stripped by Ollama) across all 16 paraphrases.
  * Resilient parser: strips numbering / bullets / markdown; de-dupes
    on lowercase-trimmed form; accepts 16..24 candidates and picks the
    first 16 unique. If fewer than 16 survive, retry up to
    MAX_RETRIES with a corrective prompt.
  * Deterministic ordering: seeds are sorted by class_id, paraphrases
    by arrival order within each class. Result is stable enough for
    `git diff corpus_v1.jsonl` to be meaningful across regenerations
    (beyond provider-level non-determinism).

Run:
    cd <repo-root>
    gggp_bundle/.venv/bin/python gggp_bundle/scripts/build_corpus_v1.py

The script is idempotent at the filesystem level: rewrites the file
each run. It is NOT idempotent at content level -- LLM outputs drift
between runs. Intended use: run once, commit the snapshot, downstream
consumes the snapshot.
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
    / "corpus_v1.jsonl"
)

# ---------------------------------------------------------------------
# Seed concepts. Ordered; class_id = index in this list.
# Hand-picked to be domain-diverse (numeric, text, routing, control).
# Changing this list = new corpus version (commit a corpus_v2.jsonl).
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

PARAPHRASES_PER_SEED = 16
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
                f"[T1] class={class_id} attempt={attempt} ProviderError: "
                f"{exc}; retrying...",
                file=sys.stderr,
            )
            continue
        elapsed = time.time() - t0
        uniq = parse_paraphrases(raw, PARAPHRASES_PER_SEED)
        print(
            f"[T1] class={class_id} seed='{seed}' attempt={attempt} "
            f"got {len(uniq)}/{PARAPHRASES_PER_SEED} unique in {elapsed:.1f}s",
            file=sys.stderr,
        )
        if len(uniq) == PARAPHRASES_PER_SEED:
            return uniq

    raise RuntimeError(
        f"[T1] class={class_id} seed='{seed}': failed to get "
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
        f"[T1] provider={provider.name} chat_model={provider.chat_model} "
        f"seed={provider.chat_seed} temp={provider.chat_temperature}",
        file=sys.stderr,
    )

    rows = build_corpus(provider)

    assert len(rows) == len(SEEDS) * PARAPHRASES_PER_SEED, (
        f"corpus size mismatch: got {len(rows)}, expected "
        f"{len(SEEDS) * PARAPHRASES_PER_SEED}"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    print(f"[T1] wrote {len(rows)} rows to {OUT_PATH}")
    print(
        f"[T1] classes: {len(SEEDS)}  "
        f"paraphrases/class: {PARAPHRASES_PER_SEED}",
    )


if __name__ == "__main__":
    main()
