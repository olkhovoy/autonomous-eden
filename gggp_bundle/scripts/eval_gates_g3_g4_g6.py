"""
gggp_bundle/scripts/eval_gates_g3_g4_g6.py

MEDP A1 / T8 -- evaluate gates G3 (F > F_0 + 0.10), G4 (ARI > 0.30),
and G6 (combined chromosome length < 12) against the best individual
from runA1_seed0_best.json.

G3 is re-evaluated on the FULL corpus (128 rows) for an honest F number,
not just the train split. G4 runs k-means(k=8) on the encoder's codes
{c_i} and compares to ground-truth classes via ARI. G6 reads chromosome
lengths directly from the best-individual json.

Verdicts go to:
  * checkpoints.md    (human-readable table, one row per gate)
  * log.jsonl         (one gate_eval JSON line per gate)
  * stdout            (summary table)

Run:
    gggp_bundle/.venv/bin/python gggp_bundle/scripts/eval_gates_g3_g4_g6.py \
        [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "gggp_bundle" / "scripts"))

DEMO_DIR = REPO_ROOT / "gggp_bundle" / "demos" / "semiotic_hypercube"
T_PATH = DEMO_DIR / "T.npy"
CLASSES_PATH = DEMO_DIR / "classes.npy"
GRAMMAR_ENCODER = DEMO_DIR / "grammar_encoder.cfg"
GRAMMAR_DECODER = DEMO_DIR / "grammar_decoder.cfg"
CHECKPOINTS_MD = (
    REPO_ROOT / "gggp_bundle" / "docs" / "medp" / "branches" / "A1" / "checkpoints.md"
)
LOG_JSONL = REPO_ROOT / "gggp_bundle" / "docs" / "medp" / "log.jsonl"

G3_THRESHOLD_EXTRA = 0.10       # F must beat F_0 + 0.10
G4_THRESHOLD_ARI = 0.30
G6_THRESHOLD_LEN = 12
F_0 = 0.6132                     # G2 baseline (see checkpoints.md)

CODE_DIM = 16
TARGET_DIM = 1024


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True, cwd=REPO_ROOT,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def evaluate(seed: int) -> dict:
    best_path = DEMO_DIR / f"runA1_seed{seed}_best.json"
    if not best_path.is_file():
        raise SystemExit(
            f"{best_path} missing. Run scripts/run_A1.py --seed {seed} first."
        )
    best = json.loads(best_path.read_text())
    chromo_g = best["chromo_g"]
    chromo_d = best["chromo_d"]

    T = np.load(T_PATH).astype(np.float64)
    classes = np.load(CLASSES_PATH)
    assert T.shape == (128, TARGET_DIM)

    from semiotic_hypercube import SemioticHypercube
    sh = SemioticHypercube(str(GRAMMAR_ENCODER))
    sh.attach_decoder_grammar(str(GRAMMAR_DECODER))

    res = sh.batch_render_dual(chromo_g, chromo_d, T, CODE_DIM, TARGET_DIM)
    F_full = float(res["F"])
    c_matrix = np.asarray(res["c"])

    # --- G3 ---
    g3_threshold = F_0 + G3_THRESHOLD_EXTRA
    g3_pass = F_full > g3_threshold

    # --- G4 (k-means on encoder codes, ARI vs ground truth) ---
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
    except ImportError as e:
        raise SystemExit(f"sklearn required for G4 eval: {e}")

    n_classes = int(classes.max()) + 1
    km = KMeans(n_clusters=n_classes, random_state=seed, n_init=10).fit(c_matrix)
    ari = float(adjusted_rand_score(classes, km.labels_))
    g4_pass = ari > G4_THRESHOLD_ARI

    # --- G6 ---
    combined_len = len(chromo_g) + len(chromo_d)
    g6_pass = combined_len < G6_THRESHOLD_LEN

    return {
        "seed": seed,
        "F_full": F_full,
        "F_train_raw": best.get("F_train_raw"),
        "F_test_raw": best.get("F_test_raw"),
        "g3": {
            "threshold": g3_threshold,
            "observed": F_full,
            "passed": g3_pass,
            "margin": F_full - g3_threshold,
        },
        "g4": {
            "threshold": G4_THRESHOLD_ARI,
            "observed": ari,
            "passed": g4_pass,
            "margin": ari - G4_THRESHOLD_ARI,
        },
        "g6": {
            "threshold": G6_THRESHOLD_LEN,
            "observed": combined_len,
            "len_g": len(chromo_g),
            "len_d": len(chromo_d),
            "passed": g6_pass,
            "margin": G6_THRESHOLD_LEN - combined_len,
        },
    }


def status_marker(passed: bool) -> str:
    return "[PASS]" if passed else "[FAIL]"


def update_checkpoints_md(e: dict, head: str, ts: str) -> None:
    md = CHECKPOINTS_MD.read_text(encoding="utf-8")
    g3 = e["g3"]
    g4 = e["g4"]
    g6 = e["g6"]
    g3_row = (
        f"| G3 | {status_marker(g3['passed'])} | 2026-04-22T20:30Z | "
        f"F={g3['observed']:.4f} | > F_0 + 0.10 ({g3['threshold']:.4f}) | "
        f"{ts} | {head} |"
    )
    g4_row = (
        f"| G4 | {status_marker(g4['passed'])} | 2026-04-23T00:30Z | "
        f"ARI={g4['observed']:.4f} | > {g4['threshold']:.2f} | {ts} | {head} |"
    )
    g6_row = (
        f"| G6 | {status_marker(g6['passed'])} | 2026-04-23T16:30Z | "
        f"len={g6['observed']} (G={g6['len_g']}, D={g6['len_d']}) | "
        f"< {g6['threshold']} | {ts} | {head} |"
    )

    lines = md.splitlines()
    out_lines = []
    for line in lines:
        if line.startswith("| G3 |"):
            out_lines.append(g3_row)
        elif line.startswith("| G4 |"):
            out_lines.append(g4_row)
        elif line.startswith("| G6 |"):
            out_lines.append(g6_row)
        else:
            out_lines.append(line)
    CHECKPOINTS_MD.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def append_log(e: dict, head: str, ts: str) -> None:
    events = [
        {
            "ts": ts,
            "event": "gate_eval",
            "branch": "A1",
            "gate": "G3",
            "status": "pass" if e["g3"]["passed"] else "fail",
            "criterion": f"F > F_0 + 0.10 ({e['g3']['threshold']:.4f})",
            "observed": e["g3"]["observed"],
            "margin": e["g3"]["margin"],
            "source_seed": e["seed"],
            "F_full_corpus": e["F_full"],
            "F_train_raw": e["F_train_raw"],
            "F_test_raw": e["F_test_raw"],
            "commit": head,
        },
        {
            "ts": ts,
            "event": "gate_eval",
            "branch": "A1",
            "gate": "G4",
            "status": "pass" if e["g4"]["passed"] else "fail",
            "criterion": f"ARI > {e['g4']['threshold']:.2f}",
            "observed": e["g4"]["observed"],
            "margin": e["g4"]["margin"],
            "k": 8,
            "source_seed": e["seed"],
            "commit": head,
        },
        {
            "ts": ts,
            "event": "gate_eval",
            "branch": "A1",
            "gate": "G6",
            "status": "pass" if e["g6"]["passed"] else "fail",
            "criterion": f"combined_len < {e['g6']['threshold']}",
            "observed": e["g6"]["observed"],
            "len_g": e["g6"]["len_g"],
            "len_d": e["g6"]["len_d"],
            "margin": e["g6"]["margin"],
            "source_seed": e["seed"],
            "commit": head,
        },
    ]
    with LOG_JSONL.open("a", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    e = evaluate(args.seed)
    head = git_head()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("=" * 72)
    print(f"A1 gate evaluation (seed={e['seed']}, F_0={F_0:.4f})")
    print("-" * 72)
    for gate_id in ("g3", "g4", "g6"):
        g = e[gate_id]
        status = status_marker(g["passed"])
        print(
            f"  {gate_id.upper()}  {status}   "
            f"observed={g['observed']}   "
            f"threshold={g['threshold']}   "
            f"margin={g['margin']:+.4f}"
        )
    print("=" * 72)

    update_checkpoints_md(e, head, ts)
    append_log(e, head, ts)
    print(f"[T8] checkpoints.md updated")
    print(f"[T8] log.jsonl appended with 3 gate_eval events")


if __name__ == "__main__":
    main()
