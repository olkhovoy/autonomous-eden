"""
gggp_bundle/scripts/pca_reduce.py

MEDP A1.1 / R1 and A2 / S0c -- reduce T<suffix>.npy to T<suffix>_pca.npy
via sklearn PCA + L2 row-normalization.

Version selector (--version v1|v2):
  v1 (default, A1): T.npy (128, 1024) -> T_pca.npy (128, 16)
  v2 (A2)        : T_v2.npy (256, 1024) -> T_v2_pca.npy (256, 16)

Why PCA: A1 diagnostic showed that axis-index grammar operations
cannot climb above F=0.17 because embedding energy lives in top
singular-value directions, not in the first 16 raw coordinates.
PCA rotates T into a basis where the first 16 coordinates ARE the
high-energy directions, giving the grammar a fair chance.

Outputs are gitignored -- regenerate from T<suffix>.npy deterministically
(sklearn PCA is not seeded-stable across versions, but the components
hash in the meta file lets you detect a basis-change if one happens).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "gggp_bundle" / "demos" / "semiotic_hypercube"

_VERSION_PATHS = {
    "v1": {
        "T":      DEMO_DIR / "T.npy",
        "T_pca":  DEMO_DIR / "T_pca.npy",
        "meta":   DEMO_DIR / "pca_meta.json",
    },
    "v2": {
        "T":      DEMO_DIR / "T_v2.npy",
        "T_pca":  DEMO_DIR / "T_v2_pca.npy",
        "meta":   DEMO_DIR / "pca_meta_v2.json",
    },
}

N_COMPONENTS = 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--version", choices=["v1", "v2"], default="v1",
        help="corpus version (v1=A1 T.npy, v2=A2 T_v2.npy)",
    )
    ap.add_argument(
        "--n-components", type=int, default=N_COMPONENTS,
        help=f"number of PCA components (default {N_COMPONENTS})",
    )
    args = ap.parse_args()
    paths = _VERSION_PATHS[args.version]
    t_path = paths["T"]
    t_pca_path = paths["T_pca"]
    meta_path = paths["meta"]
    n_components = int(args.n_components)
    tag = f"pca_reduce/{args.version}"

    from sklearn.decomposition import PCA
    if not t_path.is_file():
        raise SystemExit(
            f"{t_path} missing. Run scripts/embed_corpus.py "
            f"--version {args.version} first."
        )
    T = np.load(t_path).astype(np.float64)
    print(f"[{tag}] T shape={T.shape} dtype={T.dtype}")

    pca = PCA(n_components=n_components, random_state=0, whiten=False)
    T_pca_raw = pca.fit_transform(T)
    print(
        f"[{tag}] fit PCA(n_components={n_components}); "
        f"explained_variance cumulative = "
        f"{pca.explained_variance_ratio_.cumsum()[-1]:.4f}"
    )

    # L2-row-normalize so downstream cosine similarity == dot product,
    # matching T.npy's normalization convention (set in embed_corpus.py).
    norms = np.linalg.norm(T_pca_raw, axis=1, keepdims=True)
    assert (norms > 1e-10).all(), (
        f"{int((norms <= 1e-10).sum())} zero-norm PCA rows; "
        f"PCA collapsed some rows to origin. Inspect explained_variance."
    )
    T_pca = (T_pca_raw / norms).astype(np.float32)

    np.save(t_pca_path, T_pca)

    # Meta: everything that lets us re-derive T_pca independently if needed.
    components_hash = hashlib.sha256(
        pca.components_.astype(np.float64).tobytes()
    ).hexdigest()[:16]
    meta = {
        "corpus_version": args.version,
        "source_path": str(t_path.relative_to(REPO_ROOT)),
        "output_path": str(t_pca_path.relative_to(REPO_ROOT)),
        "n_components": n_components,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_cumsum": pca.explained_variance_ratio_.cumsum().tolist(),
        "total_explained": float(pca.explained_variance_ratio_.sum()),
        "pca_components_sha256_16": components_hash,
        "l2_row_normalized": True,
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[{tag}] wrote {t_pca_path} shape={T_pca.shape}")
    print(f"[{tag}] wrote {meta_path}")
    print(
        f"[{tag}] top-{n_components} captures "
        f"{meta['total_explained']:.4f} of total variance"
    )


if __name__ == "__main__":
    main()
