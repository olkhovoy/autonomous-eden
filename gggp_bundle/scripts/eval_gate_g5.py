"""
gggp_bundle/scripts/eval_gate_g5.py

MEDP / evaluate G5 (seed-stability: sigma(F) < 0.05) across N seeds.

Works for both raw (A1) and pca (A1.1) modes, selected via --mode.
Reads runA1[_mode]_seed{k}_best.json for k in range(--n-seeds),
computes the F_test_raw sample mean/std, writes a gate_eval event
and updates the matching checkpoints.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "gggp_bundle" / "demos" / "semiotic_hypercube"
LOG_JSONL = REPO_ROOT / "gggp_bundle" / "docs" / "medp" / "log.jsonl"

G5_SIGMA_THRESHOLD = 0.05

MODE_PROFILES = {
    "raw": {
        "branch": "A1",
        "gate": "G5",
        "best_glob": "runA1_seed{seed}_best.json",
        "checkpoints_md": REPO_ROOT / "gggp_bundle" / "docs" / "medp"
            / "branches" / "A1" / "checkpoints.md",
        "uses_deadline_col": True,
    },
    "pca": {
        "branch": "A1.1",
        "gate": "G5.1",
        "best_glob": "runA1_pca_seed{seed}_best.json",
        "checkpoints_md": REPO_ROOT / "gggp_bundle" / "docs" / "medp"
            / "branches" / "A1.1" / "checkpoints.md",
        "uses_deadline_col": False,
    },
}


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True, cwd=REPO_ROOT,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def collect_f(profile: dict, n_seeds: int) -> tuple[np.ndarray, np.ndarray]:
    f_train, f_test = [], []
    for seed in range(n_seeds):
        p = DEMO_DIR / profile["best_glob"].format(seed=seed)
        if not p.is_file():
            raise SystemExit(
                f"{p} missing. Run run_A1.py --seed {seed} --mode "
                f"{'pca' if profile['branch']=='A1.1' else 'raw'} first."
            )
        b = json.loads(p.read_text())
        f_train.append(float(b["F_train_raw"]))
        f_test.append(float(b["F_test_raw"]))
    return np.asarray(f_train), np.asarray(f_test)


def update_md(profile: dict, mean_test: float, std_test: float, passed: bool,
              head: str, ts: str) -> None:
    md_path = profile["checkpoints_md"]
    md = md_path.read_text(encoding="utf-8")
    status = "[PASS]" if passed else "[FAIL]"
    if profile["uses_deadline_col"]:
        row = (
            f"| {profile['gate']} | {status} | 2026-04-23T12:30Z | "
            f"sigma(F_test)={std_test:.4f} "
            f"(mean={mean_test:.4f}) | < {G5_SIGMA_THRESHOLD} | "
            f"{ts} | {head} |"
        )
    else:
        row = (
            f"| {profile['gate']} | {status} | "
            f"sigma(F_test)={std_test:.4f} (mean={mean_test:.4f}) | "
            f"< {G5_SIGMA_THRESHOLD} | {ts} | {head} |"
        )
    out = []
    for line in md.splitlines():
        if line.startswith(f"| {profile['gate']} |"):
            out.append(row)
        else:
            out.append(line)
    md_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def append_log(profile: dict, n_seeds: int, f_train: np.ndarray, f_test: np.ndarray,
               passed: bool, head: str, ts: str) -> None:
    ev = {
        "ts": ts,
        "event": "gate_eval",
        "branch": profile["branch"],
        "gate": profile["gate"],
        "status": "pass" if passed else "fail",
        "criterion": f"sigma(F_test_raw) < {G5_SIGMA_THRESHOLD}",
        "observed": {
            "n_seeds": n_seeds,
            "F_train_mean": float(f_train.mean()),
            "F_train_std":  float(f_train.std(ddof=1)),
            "F_test_mean":  float(f_test.mean()),
            "F_test_std":   float(f_test.std(ddof=1)),
            "F_test_min":   float(f_test.min()),
            "F_test_max":   float(f_test.max()),
        },
        "margin": G5_SIGMA_THRESHOLD - float(f_test.std(ddof=1)),
        "commit": head,
    }
    with LOG_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODE_PROFILES), default="pca")
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()
    profile = MODE_PROFILES[args.mode]

    f_train, f_test = collect_f(profile, args.n_seeds)
    std_test = float(f_test.std(ddof=1))
    passed = std_test < G5_SIGMA_THRESHOLD

    head = git_head()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("=" * 72)
    print(f"{profile['branch']} {profile['gate']} eval  mode={args.mode}  "
          f"n_seeds={args.n_seeds}")
    print("-" * 72)
    print(f"  F_train_raw: mean={f_train.mean():.4f}  std={f_train.std(ddof=1):.4f}  "
          f"range=[{f_train.min():.4f}, {f_train.max():.4f}]")
    print(f"  F_test_raw:  mean={f_test.mean():.4f}  std={std_test:.4f}  "
          f"range=[{f_test.min():.4f}, {f_test.max():.4f}]")
    print(f"  sigma(F_test) = {std_test:.4f}  threshold = {G5_SIGMA_THRESHOLD}  "
          f"=> {'PASS' if passed else 'FAIL'}")
    print("=" * 72)

    update_md(profile, float(f_test.mean()), std_test, passed, head, ts)
    append_log(profile, args.n_seeds, f_train, f_test, passed, head, ts)
    print(f"[{profile['gate']}] {profile['checkpoints_md']} updated")
    print(f"[{profile['gate']}] log.jsonl appended")


if __name__ == "__main__":
    main()
