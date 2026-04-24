"""
gggp_bundle/scripts/embed_corpus.py

MEDP A1 / T2 and A2 / S0b -- batch-embed the corpus via provider-router.

Version selector (--version v1|v2):
  v1 (A1, default): corpus_v1.jsonl (128 rows) -> T.npy + classes.npy
  v2 (A2)        : corpus_v2.jsonl (256 rows) -> T_v2.npy + classes_v2.npy

Shared output schema per version:
  T<suffix>.npy           (N, embed_dim) float32, L2-row-normalized
  classes<suffix>.npy     (N,) int32
  embed_meta<suffix>.json {provider, model, dim, ts, sha256(corpus)}

The embeddings are normalized to unit L2 length so downstream cosine
similarity reduces to dot product. Class labels are copied from the
corpus JSONL so they stay aligned with T by row index.

Run:
    cd <repo-root>
    gggp_bundle/.venv/bin/python gggp_bundle/scripts/embed_corpus.py [--version v1|v2]
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
sys.path.insert(0, str(REPO_ROOT / "gggp_bundle" / "scripts"))

from providers import load_provider  # noqa: E402

OUT_DIR = REPO_ROOT / "gggp_bundle" / "demos" / "semiotic_hypercube"
CONFIG_PATH = REPO_ROOT / "gggp_bundle" / "config" / "providers.toml"

# Per-version asset paths. v2 appends a '_v2' suffix to every artefact
# so v1 and v2 runs coexist without any cross-wiring risk.
_VERSION_PATHS = {
    "v1": {
        "corpus":  OUT_DIR / "corpus_v1.jsonl",
        "T":       OUT_DIR / "T.npy",
        "classes": OUT_DIR / "classes.npy",
        "meta":    OUT_DIR / "embed_meta.json",
    },
    "v2": {
        "corpus":  OUT_DIR / "corpus_v2.jsonl",
        "T":       OUT_DIR / "T_v2.npy",
        "classes": OUT_DIR / "classes_v2.npy",
        "meta":    OUT_DIR / "embed_meta_v2.json",
    },
}

# Batch size: Ollama /api/embed handles arbitrary lists but larger batches
# hold VRAM. 32 balances throughput and latency on RTX3090 for 1024-dim.
BATCH_SIZE = 32


def load_corpus(path: Path) -> tuple[list[str], np.ndarray]:
    """Return (texts, class_ids) preserving file order.

    Row 'id' field must match line index exactly; otherwise the
    T / classes arrays would be silently mis-aligned with the corpus.
    """
    texts: list[str] = []
    class_ids: list[int] = []
    expected_id = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            assert row["id"] == expected_id, (
                f"{path.name} row order broken: row['id']={row['id']} "
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
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--version", choices=["v1", "v2"], default="v1",
        help="corpus version to embed (v1=A1, v2=A2 with 32 paraphrases/class)",
    )
    args = ap.parse_args()
    paths = _VERSION_PATHS[args.version]
    corpus_path = paths["corpus"]
    t_path = paths["T"]
    classes_path = paths["classes"]
    meta_path = paths["meta"]
    tag = f"T2/{args.version}"

    if not corpus_path.is_file():
        raise SystemExit(
            f"{corpus_path.name} not found at {corpus_path}. "
            f"Run scripts/build_corpus_{args.version}.py first."
        )

    texts, classes = load_corpus(corpus_path)
    print(f"[{tag}] loaded {len(texts)} rows, {len(set(classes.tolist()))} classes")

    provider = load_provider(CONFIG_PATH)
    print(
        f"[{tag}] provider={provider.name} embed_model={provider.embed_model} "
        f"dim={provider.embed_dim}"
    )

    embeddings: list[np.ndarray] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        emb = provider.embed(batch)
        embeddings.append(emb)

    T = np.concatenate(embeddings, axis=0).astype(np.float32)
    assert T.shape == (len(texts), provider.embed_dim), (
        f"[{tag}] T shape mismatch: got {T.shape}, "
        f"expected ({len(texts)}, {provider.embed_dim})"
    )

    # L2-normalize so downstream cos_sim reduces to dot product.
    norms = np.linalg.norm(T, axis=1, keepdims=True)
    assert (norms > 1e-8).all(), (
        f"[{tag}] {int((norms <= 1e-8).sum())} zero-norm embeddings; "
        f"encoder returned degenerate vectors. Check provider."
    )
    T = T / norms

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(t_path, T)
    np.save(classes_path, classes)

    meta = {
        "corpus_version": args.version,
        "provider": provider.name,
        "embed_model": provider.embed_model,
        "embed_dim": provider.embed_dim,
        "n_rows": int(T.shape[0]),
        "n_classes": int(len(set(classes.tolist()))),
        "corpus_path": str(corpus_path.relative_to(REPO_ROOT)),
        "corpus_sha256": sha256_file(corpus_path),
        "t_path": str(t_path.relative_to(REPO_ROOT)),
        "classes_path": str(classes_path.relative_to(REPO_ROOT)),
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "l2_normalized": True,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print(f"[{tag}] wrote T: {t_path} shape={T.shape} dtype={T.dtype}")
    print(f"[{tag}] wrote classes: {classes_path} shape={classes.shape}")
    print(f"[{tag}] wrote meta: {meta_path}")


if __name__ == "__main__":
    main()
