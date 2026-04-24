"""
gggp_bundle/scripts/eval_ari_a2.py

MEDP A2 / S3a -- clustering quality of the evolved code matrix c_i.

For each seed's winning (G, D), reconstruct c via batch_render_dual on the
full 256-row corpus, KMeans-cluster into K = n_classes groups, and score
against ground-truth labels via:
  * ARI (Adjusted Rand Index)      -- chance-corrected, range [-1, 1]
  * AMI (Adjusted Mutual Info)     -- chance-corrected, range [0, 1]
  * silhouette (cosine)            -- intrinsic cluster tightness
  * V-measure (hom + completeness) -- balanced purity

Baseline rows (for comparison):
  * T_v2_pca                       -- raw PCA-16 target embeddings
  * random (seed-shuffled labels)  -- ARI should be ~0

Outputs:
  demos/semiotic_hypercube/a2_ari_report.json
  demos/semiotic_hypercube/a2_ari_report.md     (human-readable table)

Run:
    gggp_bundle/.venv/bin/python gggp_bundle/scripts/eval_ari_a2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "gggp_bundle" / "demos" / "semiotic_hypercube"
SEEDS = (0, 1, 2)
KMEANS_RANDOM_STATE = 0  # locked so reruns are bit-identical


def load_sh():
    """Instantiate the Rust SemioticHypercube bound to the A2 grammars."""
    from semiotic_hypercube import SemioticHypercube
    sh = SemioticHypercube(str(DEMO_DIR / "grammar_encoder_a2.cfg"))
    sh.attach_decoder_grammar(str(DEMO_DIR / "grammar_decoder_a2_nc3.cfg"))
    return sh


def cluster_scores(X: np.ndarray, y: np.ndarray, k: int) -> dict:
    """KMeans-cluster X, compare to labels y, return a score dict."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        silhouette_score,
        v_measure_score,
    )
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"cluster_scores: X rows {X.shape[0]} != y len {y.shape[0]}. "
            f"The cluster matrix and labels must be row-aligned."
        )
    km = KMeans(n_clusters=k, random_state=KMEANS_RANDOM_STATE, n_init=10)
    pred = km.fit_predict(X)
    return {
        "ari":        float(adjusted_rand_score(y, pred)),
        "ami":        float(adjusted_mutual_info_score(y, pred)),
        "v_measure":  float(v_measure_score(y, pred)),
        "silhouette": float(silhouette_score(X, y, metric="cosine")),
        "inertia":    float(km.inertia_),
    }


def reconstruct_c_for_seed(
    sh, best: dict, T: np.ndarray, code_dim: int, target_dim: int
) -> np.ndarray:
    """Run batch_render_dual to recover the c matrix the EA settled on.

    We don't persist c during the EA (too large); reconstructing is cheap
    (<0.1 s per seed on 256 rows) and perfectly reproducible because the
    Rust rendering is deterministic in (chromosome, input).
    """
    res = sh.batch_render_dual(
        best["chromo_g"], best["chromo_d"], T, code_dim, target_dim
    )
    return np.ascontiguousarray(np.asarray(res["c"], dtype=np.float64))


def main() -> None:
    # Make semiotic_hypercube importable from the bundle venv path.
    sys.path.insert(0, str(REPO_ROOT / "gggp_bundle" / "scripts"))

    T = np.load(DEMO_DIR / "T_v2_pca.npy").astype(np.float64)
    classes = np.load(DEMO_DIR / "classes_v2.npy")
    n_classes = int(len(set(classes.tolist())))
    print(
        f"[S3a] corpus: N={T.shape[0]} target_dim={T.shape[1]} "
        f"K={n_classes}"
    )

    sh = load_sh()

    per_seed: list[dict] = []
    for seed in SEEDS:
        best_path = DEMO_DIR / f"runA2_seed{seed}_best.json"
        if not best_path.is_file():
            print(f"[S3a] WARN missing {best_path.name}, skipping seed {seed}")
            continue
        best = json.loads(best_path.read_text())
        code_dim = int(best["ea_config"]["code_dim"])
        target_dim = int(best["ea_config"]["target_dim"])
        c = reconstruct_c_for_seed(sh, best, T, code_dim, target_dim)
        scores = cluster_scores(c, classes, n_classes)
        scores["seed"] = seed
        scores["len_g"] = len(best["chromo_g"])
        scores["len_d"] = len(best["chromo_d"])
        per_seed.append(scores)
        print(
            f"[S3a] seed={seed} "
            f"ARI={scores['ari']:+.4f} AMI={scores['ami']:+.4f} "
            f"V={scores['v_measure']:+.4f} sil={scores['silhouette']:+.4f}"
        )

    # Baselines.
    t_scores = cluster_scores(T, classes, n_classes)
    print(
        f"[S3a] T_v2_pca baseline   "
        f"ARI={t_scores['ari']:+.4f} AMI={t_scores['ami']:+.4f} "
        f"V={t_scores['v_measure']:+.4f} sil={t_scores['silhouette']:+.4f}"
    )
    # Random-label baseline. ARI/AMI should be near zero. We don't do
    # KMeans on a random matrix (meaningless); instead we shuffle the
    # labels once and measure ARI against that. This is the canonical
    # "chance level" anchor.
    rng = np.random.default_rng(KMEANS_RANDOM_STATE)
    y_shuf = classes.copy()
    rng.shuffle(y_shuf)
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
    )
    random_scores = {
        "ari": float(adjusted_rand_score(classes, y_shuf)),
        "ami": float(adjusted_mutual_info_score(classes, y_shuf)),
    }
    print(
        f"[S3a] random-shuffle baseline "
        f"ARI={random_scores['ari']:+.4f} AMI={random_scores['ami']:+.4f}"
    )

    # Aggregate mean/std.
    def agg(key: str) -> dict[str, float]:
        vals = np.array([s[key] for s in per_seed], dtype=np.float64)
        return {
            "mean": float(vals.mean()),
            "std":  float(vals.std()),
            "min":  float(vals.min()),
            "max":  float(vals.max()),
        }

    report = {
        "n_seeds": len(per_seed),
        "n_rows": int(T.shape[0]),
        "n_classes": n_classes,
        "kmeans_random_state": KMEANS_RANDOM_STATE,
        "per_seed": per_seed,
        "aggregate": {
            "ari":        agg("ari"),
            "ami":        agg("ami"),
            "v_measure":  agg("v_measure"),
            "silhouette": agg("silhouette"),
        },
        "baseline_t_pca": t_scores,
        "baseline_random": random_scores,
    }
    out_json = DEMO_DIR / "a2_ari_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[S3a] wrote {out_json}")

    # Markdown summary.
    md_lines = [
        "# A2 — ARI / AMI clustering report",
        "",
        f"Rows: {T.shape[0]}, classes: {n_classes}, seeds: {len(per_seed)}.",
        "",
        "## Per-seed (cluster c_i with KMeans, compare to ground truth)",
        "",
        "| seed | len(G) | len(D) | ARI | AMI | V-measure | silhouette (cos) |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in per_seed:
        md_lines.append(
            f"| {s['seed']} | {s['len_g']} | {s['len_d']} | "
            f"{s['ari']:+.4f} | {s['ami']:+.4f} | "
            f"{s['v_measure']:+.4f} | {s['silhouette']:+.4f} |"
        )
    md_lines += [
        "",
        "## Aggregate across seeds",
        "",
        "| metric | mean | std | min | max |",
        "|---|---|---|---|---|",
    ]
    for k in ("ari", "ami", "v_measure", "silhouette"):
        a = report["aggregate"][k]
        md_lines.append(
            f"| {k} | {a['mean']:+.4f} | {a['std']:+.4f} | "
            f"{a['min']:+.4f} | {a['max']:+.4f} |"
        )
    md_lines += [
        "",
        "## Baselines",
        "",
        "| source | ARI | AMI | V-measure | silhouette (cos) |",
        "|---|---|---|---|---|",
        f"| T_v2_pca (cluster raw embeddings) "
        f"| {t_scores['ari']:+.4f} | {t_scores['ami']:+.4f} "
        f"| {t_scores['v_measure']:+.4f} | {t_scores['silhouette']:+.4f} |",
        f"| shuffled-labels (chance floor) "
        f"| {random_scores['ari']:+.4f} | {random_scores['ami']:+.4f} "
        f"| — | — |",
        "",
    ]
    out_md = DEMO_DIR / "a2_ari_report.md"
    out_md.write_text("\n".join(md_lines))
    print(f"[S3a] wrote {out_md}")


if __name__ == "__main__":
    main()
