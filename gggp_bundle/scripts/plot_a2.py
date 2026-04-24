"""
gggp_bundle/scripts/plot_a2.py

MEDP A2 / S3c -- figures for the A2 postmortem.

Figures produced (all PNG, 150 DPI, monochrome-friendly):
  1. a2_fig_trajectories.png
     4 subplots (NC1, NC2, NC3, NC4) x (best-per-seed line + pop-mean
     shaded band). X = generation. Lets the reader see when each
     constraint saturates.
  2. a2_fig_c_scatter.png
     2D PCA of the best seed's c_i matrix, colored by ground-truth
     class. Also shows T_v2_pca PCA2 as a reference panel.
  3. a2_fig_gates.png
     Horizontal bar chart of each gate's value vs threshold.

Inputs: runA2_seed{0,1,2}.jsonl, runA2_seed{0,1,2}_best.json,
        T_v2_pca.npy, classes_v2.npy, a2_gate_report.json.

Run:
    gggp_bundle/.venv/bin/python gggp_bundle/scripts/plot_a2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "gggp_bundle" / "demos" / "semiotic_hypercube"
SEEDS = (0, 1, 2)

# Locked visual defaults -- keep figures reproducible.
FIG_DPI = 150
CMAP = "tab10"   # 10-color qualitative map; we have 8 classes.


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_trajectories(out_png: Path) -> None:
    import matplotlib.pyplot as plt
    per_seed = []
    for s in SEEDS:
        jp = DEMO_DIR / f"runA2_seed{s}.jsonl"
        if not jp.is_file():
            print(f"[S3c] WARN missing {jp.name}, skipping seed {s}")
            continue
        per_seed.append((s, load_jsonl(jp)))
    if not per_seed:
        raise SystemExit("[S3c] no runA2_seed*.jsonl files found.")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=FIG_DPI)
    panels = (
        ("F_best_nc1", "NC1 — dual fixed point", 0, 1.0),
        ("F_best_nc2", "NC2 — midpoint composition", 0, 1.0),
        ("F_best_nc3", "NC3 — code-gated op fraction", 0, 1.05),
        ("F_best_nc4", "NC4 — reconstruction (A1 carry-over)", 0, 1.0),
    )
    for (key, title, ymin, ymax), ax in zip(panels, axes.ravel()):
        for s, rows in per_seed:
            gens = [r["gen"] for r in rows]
            vals = [r[key] for r in rows]
            ax.plot(gens, vals, marker=".", linewidth=1.2,
                    label=f"seed {s}")
        ax.set_title(title)
        ax.set_xlabel("generation")
        ax.set_ylabel(key.replace("F_best_", "F_"))
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.3)
        # Gate line if applicable (threshold from plan.md).
        gate_threshold = {
            "F_best_nc1": 0.50,
            "F_best_nc2": 0.55,
            "F_best_nc3": 0.20,
            "F_best_nc4": 0.50,
        }.get(key)
        if gate_threshold is not None:
            ax.axhline(
                gate_threshold, color="red", linewidth=0.8,
                linestyle="--", alpha=0.7,
            )
            ax.text(
                0.02, gate_threshold + 0.02,
                f"gate = {gate_threshold}",
                transform=ax.get_yaxis_transform(),
                color="red", fontsize=8,
            )
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("A2 UMC — NC1..NC4 trajectories (best individual per gen)")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[S3c] wrote {out_png}")


def plot_c_scatter(out_png: Path) -> None:
    """2D PCA scatter of c_i colored by class, for the best seed.

    Side-by-side with T_v2_pca 2D PCA for visual comparison. Uses the
    seed whose F_shaped is highest on the train split.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from semiotic_hypercube import SemioticHypercube

    T = np.load(DEMO_DIR / "T_v2_pca.npy").astype(np.float64)
    classes = np.load(DEMO_DIR / "classes_v2.npy")

    # Pick the seed with the highest train F_shaped.
    best_seed = None
    best_F = -np.inf
    for s in SEEDS:
        p = DEMO_DIR / f"runA2_seed{s}_best.json"
        if not p.is_file():
            continue
        b = json.loads(p.read_text())
        if b["train"]["F_shaped"] > best_F:
            best_F = b["train"]["F_shaped"]
            best_seed = (s, b)
    if best_seed is None:
        raise SystemExit("[S3c] no runA2_seed*_best.json files found.")
    s_id, best = best_seed

    sh = SemioticHypercube(str(DEMO_DIR / "grammar_encoder_a2.cfg"))
    sh.attach_decoder_grammar(str(DEMO_DIR / "grammar_decoder_a2_nc3.cfg"))
    res = sh.batch_render_dual(
        best["chromo_g"], best["chromo_d"], T,
        int(best["ea_config"]["code_dim"]),
        int(best["ea_config"]["target_dim"]),
    )
    c = np.asarray(res["c"], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=FIG_DPI)
    for ax, X, title in (
        (axes[0], T, f"T_v2_pca (target embeddings, dim={T.shape[1]})"),
        (axes[1], c, f"c_i from seed={s_id} G/D (dim={c.shape[1]})"),
    ):
        # PCA-2 for visualization.
        pca2 = PCA(n_components=2, random_state=0)
        proj = pca2.fit_transform(X)
        scatter = ax.scatter(
            proj[:, 0], proj[:, 1], c=classes, cmap=CMAP,
            s=28, alpha=0.85, edgecolor="black", linewidth=0.3,
        )
        ev = pca2.explained_variance_ratio_
        ax.set_title(
            f"{title}\n"
            f"PCA2 explains {ev.sum():.2%} of variance"
        )
        ax.set_xlabel(f"PC1 ({ev[0]:.2%})")
        ax.set_ylabel(f"PC2 ({ev[1]:.2%})")
        ax.grid(alpha=0.3)
    # Single legend across both panels.
    handles, _ = scatter.legend_elements(prop="colors", num=len(set(classes.tolist())))
    labels = [f"class {c}" for c in sorted(set(classes.tolist()))]
    fig.legend(
        handles, labels, loc="lower center", ncol=8,
        bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8,
    )
    fig.suptitle(f"A2 — code vs target structure (winning seed: {s_id})")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[S3c] wrote {out_png}")


def plot_gates(out_png: Path) -> None:
    import matplotlib.pyplot as plt
    report_path = DEMO_DIR / "a2_gate_report.json"
    if not report_path.is_file():
        print(
            f"[S3c] {report_path.name} missing; run eval_gates_a2.py first. "
            f"Skipping gate plot."
        )
        return
    report = json.loads(report_path.read_text())
    gates = report["gates"]

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=FIG_DPI)
    names = [g["gate"] for g in gates]
    values = [g["value"] for g in gates]
    passed = [g["pass"] for g in gates]
    # Threshold numeric extraction from the rule string.
    thresholds = []
    op_signs = []
    for g in gates:
        op, thr = g["rule"].split()
        thresholds.append(float(thr))
        op_signs.append(op)
    colors = ["#2a9d8f" if p else "#e76f51" for p in passed]

    y = np.arange(len(names))
    ax.barh(y, values, color=colors, alpha=0.85, edgecolor="black",
            linewidth=0.5)
    for i, (v, thr, op, p) in enumerate(
        zip(values, thresholds, op_signs, passed)
    ):
        ax.plot([thr, thr], [i - 0.4, i + 0.4], "k-", linewidth=1.5)
        ax.text(
            max(v, thr) * 1.02, i,
            f" {v:+.3f}  ({op} {thr:g}) — {'PASS' if p else 'FAIL'}",
            va="center", fontsize=9,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("value (bar) vs threshold (black tick)")
    ax.set_title(
        f"A2 gates — verdict: {report['verdict']} "
        f"({sum(passed)}/{len(passed)} pass)"
    )
    ax.grid(axis="x", alpha=0.3)
    # Give the text some room on the right.
    xmax = max(1.05, max(values) * 1.25, max(thresholds) * 1.25)
    ax.set_xlim(0, xmax)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[S3c] wrote {out_png}")


def main() -> None:
    sys.path.insert(0, str(REPO_ROOT / "gggp_bundle" / "scripts"))
    # Use a non-interactive backend so the script runs headless.
    import matplotlib
    matplotlib.use("Agg")
    plot_trajectories(DEMO_DIR / "a2_fig_trajectories.png")
    plot_c_scatter(DEMO_DIR / "a2_fig_c_scatter.png")
    plot_gates(DEMO_DIR / "a2_fig_gates.png")


if __name__ == "__main__":
    main()
