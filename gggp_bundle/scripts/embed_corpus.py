"""
gggp_bundle/scripts/embed_corpus.py

MEDP A1 / T2 -- batch-embed corpus_v1.jsonl via provider-router.

Input:   demos/semiotic_hypercube/corpus_v1.jsonl  (128 rows)
Output:  demos/semiotic_hypercube/T.npy            (128, 1024) float32
         demos/semiotic_hypercube/classes.npy      (128,) int32
         demos/semiotic_hypercube/embed_meta.json  {provider, model, dim, ts, sha256(corpus)}

The embeddings are normalized to unit L2 length so downstream cosine
similarity reduces to dot product. Class labels are copied from
corpus_v1.jsonl so they stay aligned with T.npy by row index.

Run:
    cd <repo-root>
    gggp_bundle/.venv/bin/python gggp_bundle/scripts/embed_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "gggp_bundle" / "scripts"))

from providers import load_provider  # noqa: E402

CORPUS_PATH = (
    REPO_ROOT / "gggp_bundle" / "demos" / "semiotic_hypercube" / "corpus_v1.jsonl"
)
OUT_DIR = REPO_ROOT / "gggp_bundle" / "demos" / "semiotic_hypercube"
T_PATH = OUT_DIR / "T.npy"
CLASSES_PATH = OUT_DIR / "classes.npy"
META_PATH = OUT_DIR / "embed_meta.json"
CONFIG_PATH = REPO_ROOT / "gggp_bundle" / "config" / "providers.toml"

# Batch size: Ollama /api/embed handles arbitrary lists but larger batches
# hold VRAM. 32 balances throughput and latency on RTX3090 for 1024-dim.
BATCH_SIZE = 32


def load_corpus(path: Path) -> tuple[list[str], np.ndarray]:
    """Return (texts, class_ids) preserving file order."""
    texts: list[str] = []
    class_ids: list[int] = []
    expected_id = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            assert row["id"] == expected_id, (
                f"corpus_v1.jsonl row order broken: row['id']={row['id']} "
                f"but expected {expected_id}. Regenerate the corpus."
            )
            texts.append(row["text"])
            class_ids.append(int(row["class_id"]))
            expected_id += 1
    return texts, np.asarray(class_ids, dtype=np.int32)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not CORPUS_PATH.is_file():
        raise SystemExit(
            f"corpus_v1.jsonl not found at {CORPUS_PATH}. "
            f"Run scripts/build_corpus_v1.py first (T1)."
        )

    texts, classes = load_corpus(CORPUS_PATH)
    print(f"[T2] loaded {len(texts)} rows, {len(set(classes.tolist()))} classes")

    provider = load_provider(CONFIG_PATH)
    print(
        f"[T2] provider={provider.name} embed_model={provider.embed_model} "
        f"dim={provider.embed_dim}"
    )

    embeddings: list[np.ndarray] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        emb = provider.embed(batch)
        embeddings.append(emb)

    T = np.concatenate(embeddings, axis=0).astype(np.float32)
    assert T.shape == (len(texts), provider.embed_dim), (
        f"[T2] T shape mismatch: got {T.shape}, "
        f"expected ({len(texts)}, {provider.embed_dim})"
    )

    # L2-normalize so downstream cos_sim reduces to dot product.
    norms = np.linalg.norm(T, axis=1, keepdims=True)
    assert (norms > 1e-8).all(), (
        f"[T2] {int((norms <= 1e-8).sum())} zero-norm embeddings; "
        f"encoder returned degenerate vectors. Check provider."
    )
    T = T / norms

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(T_PATH, T)
    np.save(CLASSES_PATH, classes)

    meta = {
        "provider": provider.name,
        "embed_model": provider.embed_model,
        "embed_dim": provider.embed_dim,
        "n_rows": int(T.shape[0]),
        "n_classes": int(len(set(classes.tolist()))),
        "corpus_path": str(CORPUS_PATH.relative_to(REPO_ROOT)),
        "corpus_sha256": sha256_file(CORPUS_PATH),
        "t_path": str(T_PATH.relative_to(REPO_ROOT)),
        "classes_path": str(CLASSES_PATH.relative_to(REPO_ROOT)),
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "l2_normalized": True,
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"[T2] wrote T: {T_PATH} shape={T.shape} dtype={T.dtype}")
    print(f"[T2] wrote classes: {CLASSES_PATH} shape={classes.shape}")
    print(f"[T2] wrote meta: {META_PATH}")


if __name__ == "__main__":
    main()
