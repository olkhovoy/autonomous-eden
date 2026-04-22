"""
gggp_bundle/scripts/pca_reduce.py

MEDP A1.1 / R1 -- reduce T.npy (128 x 1024) to T_pca.npy (128 x 16)
via sklearn PCA and normalize rows to unit L2.

Why PCA: A1 diagnostic showed that axis-index grammar operations
cannot climb above F=0.17 because embedding energy lives in top
singular-value directions, not in the first 16 raw coordinates.
PCA rotates T into a basis where the first 16 coordinates ARE the
high-energy directions, giving the grammar a fair chance.

Output:
  T_pca.npy               (128, 16) float32, L2-row-normalized
  pca_meta.json           explained_variance_ratio, mean, components hash

Commit T_pca.npy? No -- gitignored. Regenerate from T.npy deterministically.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "gggp_bundle" / "demos" / "semiotic_hypercube"
T_PATH = DEMO_DIR / "T.npy"
T_PCA_PATH = DEMO_DIR / "T_pca.npy"
PCA_META_PATH = DEMO_DIR / "pca_meta.json"

N_COMPONENTS = 16


def main() -> None:
    from sklearn.decomposition import PCA
    if not T_PATH.is_file():
        raise SystemExit(
            f"{T_PATH} missing. Run scripts/embed_corpus.py first (T2)."
        )
    T = np.load(T_PATH).astype(np.float64)
    print(f"[pca_reduce] T shape={T.shape} dtype={T.dtype}")

    pca = PCA(n_components=N_COMPONENTS, random_state=0, whiten=False)
    T_pca_raw = pca.fit_transform(T)
    print(
        f"[pca_reduce] fit PCA(n_components={N_COMPONENTS}); "
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

    np.save(T_PCA_PATH, T_pca)

    # Meta: everything that lets us re-derive T_pca independently if needed.
    components_hash = hashlib.sha256(
        pca.components_.astype(np.float64).tobytes()
    ).hexdigest()[:16]
    meta = {
        "source_path": str(T_PATH.relative_to(REPO_ROOT)),
        "output_path": str(T_PCA_PATH.relative_to(REPO_ROOT)),
        "n_components": N_COMPONENTS,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_cumsum": pca.explained_variance_ratio_.cumsum().tolist(),
        "total_explained": float(pca.explained_variance_ratio_.sum()),
        "pca_components_sha256_16": components_hash,
        "l2_row_normalized": True,
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    PCA_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"[pca_reduce] wrote {T_PCA_PATH} shape={T_pca.shape}")
    print(f"[pca_reduce] wrote {PCA_META_PATH}")
    print(
        f"[pca_reduce] top-{N_COMPONENTS} captures "
        f"{meta['total_explained']:.4f} of total variance"
    )


if __name__ == "__main__":
    main()
