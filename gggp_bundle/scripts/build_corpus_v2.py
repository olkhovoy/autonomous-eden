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

Idempotent at the class level: if OUT_PATH already exists, any class
that already has exactly PARAPHRASES_PER_SEED rows is reused verbatim
(original ts_utc preserved); only missing or partial classes are
regenerated. A completed run is bit-identical to the previous one for
completed classes. To force a full regeneration, delete OUT_PATH first.

Write is atomic: rows are buffered, written to OUT_PATH.tmp, then
renamed over OUT_PATH. A crash mid-generation leaves the previous
snapshot intact.

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


def load_existing_rows_by_class(path: Path) -> dict[int, list[dict]]:
    """Read an existing corpus JSONL (if present) and group rows by class_id.

    Malformed lines are skipped with a stderr warning (rather than crashing
    resume), but a malformed FILE header / unreadable file is fatal: better
    to fail loudly than to silently discard a half-finished run.

    Returns empty dict if the file does not exist. Note that rows inside a
    class keep their on-disk order, which matters: generate_class returns
    paraphrases in arrival order, so preserving that order keeps row ids
    stable across resumes of the same class.
    """
    if not path.is_file():
        return {}
    grouped: dict[int, list[dict]] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[S0a] WARN {path.name}:{ln_no} malformed JSON, "
                    f"dropping: {exc}",
                    file=sys.stderr,
                )
                continue
            cid = row.get("class_id")
            if not isinstance(cid, int):
                print(
                    f"[S0a] WARN {path.name}:{ln_no} row lacks int class_id, "
                    f"dropping: {row!r}",
                    file=sys.stderr,
                )
                continue
            grouped.setdefault(cid, []).append(row)
    return grouped


def build_corpus_with_resume(
    provider: Provider,
    existing: dict[int, list[dict]],
    checkpoint_path: Path,
) -> list[dict]:
    """Assemble the full corpus, reusing any complete classes from `existing`.

    A class is "complete" iff it has exactly PARAPHRASES_PER_SEED rows in
    `existing`. Complete classes are copied verbatim (original ts_utc kept).
    Partial or missing classes are regenerated via generate_class.

    Incremental durability: after every class whose rows were freshly
    generated, the full-so-far corpus is flushed atomically to
    `checkpoint_path`. A crash (timeout, OOM, Ctrl-C) during a later
    class therefore leaves earlier classes durably on disk, and the next
    invocation will find them and SKIP their regeneration. This is the
    whole point of the resume protocol -- the alternative (single flush
    at the very end) turns one bad timeout into hours of redundant LLM
    calls.

    Ids are reassigned in the final order to match v1's schema (id ==
    position in the output file). Ids of rows inside complete classes are
    stable across resumes because complete classes occupy the same
    contiguous block on every run.
    """
    ts_new = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    all_rows: list[dict] = []

    for class_id, seed in enumerate(SEEDS):
        existing_rows = existing.get(class_id, [])
        if len(existing_rows) == PARAPHRASES_PER_SEED:
            print(
                f"[S0a] class={class_id} seed='{seed}' SKIP "
                f"({PARAPHRASES_PER_SEED} rows already present)",
                file=sys.stderr,
            )
            all_rows.extend(existing_rows)
            continue

        if existing_rows:
            print(
                f"[S0a] class={class_id} seed='{seed}' PARTIAL "
                f"({len(existing_rows)}/{PARAPHRASES_PER_SEED} rows), "
                f"regenerating whole class",
                file=sys.stderr,
            )

        paraphrases = generate_class(provider, seed, class_id)
        for p in paraphrases:
            all_rows.append(
                asdict(
                    CorpusRow(
                        id=-1,  # reassigned below
                        class_id=class_id,
                        class_name=seed,
                        text=p,
                        method="gamma.c-LLMOnly",
                        provider=provider.name,
                        model=provider.chat_model,
                        seed=provider.chat_seed,
                        temperature=provider.chat_temperature,
                        ts_utc=ts_new,
                    )
                )
            )

        # Checkpoint after every freshly-generated class. Reassign ids
        # first so the on-disk snapshot is always in a consistent,
        # resumable state.
        for i, row in enumerate(all_rows):
            row["id"] = i
        write_jsonl_atomic(checkpoint_path, all_rows)
        print(
            f"[S0a] class={class_id} checkpoint: flushed {len(all_rows)} "
            f"rows to {checkpoint_path.name}",
            file=sys.stderr,
        )

    # Final id reassignment is a no-op if a checkpoint just ran; still
    # cheap and keeps the contract explicit for the all-classes-skipped path.
    for i, row in enumerate(all_rows):
        row["id"] = i

    return all_rows


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    """Write rows to path via a tmp file + os.replace to prevent torn snapshots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    existing = load_existing_rows_by_class(OUT_PATH)
    complete = sum(
        1 for rows in existing.values() if len(rows) == PARAPHRASES_PER_SEED
    )
    if existing:
        print(
            f"[S0a] resume: {OUT_PATH.name} has "
            f"{sum(len(r) for r in existing.values())} rows across "
            f"{len(existing)} classes; {complete} classes are complete.",
            file=sys.stderr,
        )

    provider = load_provider(CONFIG_PATH)
    print(
        f"[S0a] provider={provider.name} chat_model={provider.chat_model} "
        f"seed={provider.chat_seed} temp={provider.chat_temperature} "
        f"timeout_s={provider.request_timeout_s}",
        file=sys.stderr,
    )

    rows = build_corpus_with_resume(provider, existing, OUT_PATH)

    expected = len(SEEDS) * PARAPHRASES_PER_SEED
    assert len(rows) == expected, (
        f"corpus size mismatch: got {len(rows)}, expected {expected}"
    )

    write_jsonl_atomic(OUT_PATH, rows)

    print(f"[S0a] wrote {len(rows)} rows to {OUT_PATH}")
    print(
        f"[S0a] classes: {len(SEEDS)}  "
        f"paraphrases/class: {PARAPHRASES_PER_SEED}",
    )


if __name__ == "__main__":
    main()
