"""
gggp_bundle/scripts/eval_gates_g1_g2.py

MEDP A1 / T3 + T4 -- evaluate gates G1 (corpus structural) and G2
(baseline fitness anchor F_0). Reads T.npy + classes.npy produced by
T2, writes verdicts to checkpoints.md and log.jsonl.

G1 criterion (from ROOT.md schedule):
  |M| = 128  AND  dim(T_i) = 1024    -> pass
  else -> fail -> A1 must backtrack (pre-execution refinement already
  used; another threshold_adjusted is not allowed within A1).

G2 criterion:
  F_0 = mean_i cos(mean_T, T_i)   (anchor, no pass/fail; recorded)

Diagnostics (informational, not gated):
  * intra_vs_inter cos-sim (how separable are the 8 classes in T_i)
  * spectral ratio of T^T T (how low-dim is the embedding manifold)
  * raw k-means(k=8) ARI against ground truth (upper bound for what G4
    can achieve with the trivial "pass-through" decoder; G3 must beat
    F_0, G4 must hit ARI > 0.30)

Run:
    cd <repo-root>
    gggp_bundle/.venv/bin/python gggp_bundle/scripts/eval_gates_g1_g2.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "gggp_bundle" / "demos" / "semiotic_hypercube"
T_PATH = DEMO_DIR / "T.npy"
CLASSES_PATH = DEMO_DIR / "classes.npy"
META_PATH = DEMO_DIR / "embed_meta.json"
CHECKPOINTS_MD = (
    REPO_ROOT / "gggp_bundle" / "docs" / "medp" / "branches" / "A1" / "checkpoints.md"
)
LOG_JSONL = REPO_ROOT / "gggp_bundle" / "docs" / "medp" / "log.jsonl"

EXPECTED_N = 128
EXPECTED_DIM = 1024
EXPECTED_CLASSES = 8


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def load_arrays() -> tuple[np.ndarray, np.ndarray, dict]:
    T = np.load(T_PATH)
    classes = np.load(CLASSES_PATH)
    meta = json.loads(META_PATH.read_text())
    return T, classes, meta


def evaluate_g1(T: np.ndarray, classes: np.ndarray) -> dict:
    n_ok = T.shape[0] == EXPECTED_N
    d_ok = T.shape[1] == EXPECTED_DIM
    c_ok = int(classes.max()) + 1 == EXPECTED_CLASSES
    passed = n_ok and d_ok and c_ok
    return {
        "gate": "G1",
        "criterion": f"|M|={EXPECTED_N} AND dim(T_i)={EXPECTED_DIM} AND n_classes={EXPECTED_CLASSES}",
        "observed": {
            "n": int(T.shape[0]),
            "dim": int(T.shape[1]),
            "n_classes": int(classes.max()) + 1,
        },
        "passed": bool(passed),
    }


def evaluate_g2(T: np.ndarray) -> dict:
    """F_0 = mean_i cos(mean_T, T_i).

    T is already L2-normalized (from embed_corpus.py). mean_T itself is
    NOT unit-norm in general; we normalize it for the cosine definition
    to be meaningful.
    """
    mean_T = T.mean(axis=0)
    mean_T_unit = mean_T / (np.linalg.norm(mean_T) + 1e-12)
    f0_per_i = T @ mean_T_unit  # since rows of T are unit-norm
    F_0 = float(f0_per_i.mean())
    return {
        "gate": "G2",
        "metric": "F_0 = mean_i cos(mean_T, T_i)  (unit-mean used for cos)",
        "F_0": F_0,
        "F_0_std": float(f0_per_i.std()),
        "F_0_min": float(f0_per_i.min()),
        "F_0_max": float(f0_per_i.max()),
        "passed": True,
    }


def diagnose_separability(T: np.ndarray, classes: np.ndarray) -> dict:
    """Intra-class vs inter-class cosine similarity on raw embeddings."""
    n_classes = int(classes.max()) + 1
    intra: list[float] = []
    inter: list[float] = []
    # T rows are unit-norm so cosine == dot product.
    sim = T @ T.T
    n = T.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if classes[i] == classes[j]:
                intra.append(float(sim[i, j]))
            else:
                inter.append(float(sim[i, j]))
    return {
        "intra_mean": float(np.mean(intra)),
        "intra_std": float(np.std(intra)),
        "inter_mean": float(np.mean(inter)),
        "inter_std": float(np.std(inter)),
        "gap": float(np.mean(intra) - np.mean(inter)),
        "n_classes": n_classes,
    }


def spectral_ratio(T: np.ndarray, ks=(8, 16, 32, 64)) -> dict:
    """Cumulative explained variance at top-k singular values."""
    _, s, _ = np.linalg.svd(T, full_matrices=False)
    total = float((s ** 2).sum())
    out = {}
    for k in ks:
        if k <= len(s):
            out[f"top_{k}"] = float((s[:k] ** 2).sum() / total)
    return out


def kmeans_ari(T: np.ndarray, classes: np.ndarray, k: int = 8, seed: int = 0) -> float:
    """ARI of k-means(k) directly on T vs ground-truth classes.

    Upper bound on what G4 can hit with a trivial pass-through decoder:
    if even raw embeddings cluster poorly, A1 should backtrack fast.
    """
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.metrics import adjusted_rand_score  # type: ignore

    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(T)
    return float(adjusted_rand_score(classes, km.labels_))


def update_checkpoints_md(g1: dict, g2: dict, head: str, ts: str) -> None:
    """Replace the G1/G2 rows in checkpoints.md."""
    md = CHECKPOINTS_MD.read_text(encoding="utf-8")
    status_g1 = "[PASS]" if g1["passed"] else "[FAIL]"
    status_g2 = "[RECORDED]"

    g1_observed = (
        f"n={g1['observed']['n']}, dim={g1['observed']['dim']}, "
        f"k={g1['observed']['n_classes']}"
    )
    g1_row = (
        f"| G1 | {status_g1} | 2026-04-22T17:00Z | {g1_observed} | "
        f"100% match (M=128, dim=1024) | {ts} | {head} |"
    )
    g2_row = (
        f"| G2 | {status_g2} | 2026-04-22T18:30Z | F_0={g2['F_0']:.4f} | "
        f"(record) | {ts} | {head} |"
    )

    lines = md.splitlines()
    out_lines = []
    for line in lines:
        if line.startswith("| G1 |"):
            out_lines.append(g1_row)
        elif line.startswith("| G2 |"):
            out_lines.append(g2_row)
        else:
            out_lines.append(line)
    CHECKPOINTS_MD.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def append_log_entries(
    g1: dict,
    g2: dict,
    sep: dict,
    spec: dict,
    kmeans_ari_val: float,
    meta: dict,
    head: str,
    ts: str,
) -> None:
    events = [
        {
            "ts": ts,
            "event": "gate_eval",
            "branch": "A1",
            "gate": "G1",
            "status": "pass" if g1["passed"] else "fail",
            "criterion": g1["criterion"],
            "observed": g1["observed"],
            "commit": head,
            "inputs": {
                "T_path": "gggp_bundle/demos/semiotic_hypercube/T.npy",
                "classes_path": "gggp_bundle/demos/semiotic_hypercube/classes.npy",
                "corpus_sha256": meta.get("corpus_sha256"),
            },
        },
        {
            "ts": ts,
            "event": "gate_eval",
            "branch": "A1",
            "gate": "G2",
            "status": "recorded",
            "F_0": g2["F_0"],
            "F_0_std": g2["F_0_std"],
            "F_0_min": g2["F_0_min"],
            "F_0_max": g2["F_0_max"],
            "commit": head,
            "note": (
                "Baseline anchor for G3 which requires F > F_0 + 0.10."
            ),
        },
        {
            "ts": ts,
            "event": "diagnostic",
            "branch": "A1",
            "kind": "separability",
            "intra_mean": sep["intra_mean"],
            "inter_mean": sep["inter_mean"],
            "gap": sep["gap"],
            "spectral": spec,
            "kmeans_on_raw_T_ARI_k8": kmeans_ari_val,
            "note": (
                "kmeans-on-raw-T ARI is an upper bound for what G4 can hit "
                "with a trivial pass-through decoder. G4 threshold is 0.30."
            ),
        },
    ]
    with LOG_JSONL.open("a", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def main() -> None:
    T, classes, meta = load_arrays()

    g1 = evaluate_g1(T, classes)
    g2 = evaluate_g2(T)
    sep = diagnose_separability(T, classes)
    spec = spectral_ratio(T)

    try:
        kmeans_ari_val = kmeans_ari(T, classes)
    except ImportError:
        print(
            "[T3/T4] sklearn not installed; kmeans ARI diagnostic skipped",
            file=sys.stderr,
        )
        kmeans_ari_val = float("nan")

    head = git_head()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("=" * 72)
    print(f"G1: {g1['criterion']}")
    print(f"    observed: {g1['observed']}  passed={g1['passed']}")
    print()
    print(f"G2: {g2['metric']}")
    print(
        f"    F_0 = {g2['F_0']:.4f}  std={g2['F_0_std']:.4f}  "
        f"min={g2['F_0_min']:.4f}  max={g2['F_0_max']:.4f}"
    )
    print()
    print("Diagnostics (informational):")
    print(
        f"  intra-class cos: {sep['intra_mean']:.4f} +/- {sep['intra_std']:.4f}"
    )
    print(
        f"  inter-class cos: {sep['inter_mean']:.4f} +/- {sep['inter_std']:.4f}"
    )
    print(f"  gap:             {sep['gap']:.4f}")
    print(f"  spectral top-k:  {spec}")
    print(f"  k-means(k=8) ARI on raw T: {kmeans_ari_val:.4f}")
    print("=" * 72)

    update_checkpoints_md(g1, g2, head, ts)
    append_log_entries(
        g1, g2, sep, spec, kmeans_ari_val, meta, head, ts
    )
    print(f"[T3/T4] checkpoints.md updated")
    print(f"[T3/T4] log.jsonl appended with 3 events")


if __name__ == "__main__":
    main()
