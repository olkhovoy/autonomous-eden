#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGETS_FILE="$ROOT_DIR/demo_targets.txt"
OUT_DIR="$ROOT_DIR/out"
CARGO_MANIFEST="$ROOT_DIR/../../rust/Cargo.toml"
mkdir -p "$OUT_DIR"

MODEL="all-minilm:33m"
URL="http://localhost:11434/api/embeddings"
SEED=42
GENS=200
POP=120
ELITE=8

idx=0
while IFS= read -r target; do
  if [ -z "$target" ]; then
    continue
  fi
  idx=$((idx+1))
  prefix="$OUT_DIR/target_${idx}"
  echo "=== TARGET ${idx}: $target"
  cargo run --manifest-path "$CARGO_MANIFEST" --bin embedding_gggp -- \
    --target "$target" \
    --model "$MODEL" \
    --url "$URL" \
    --seed "$SEED" \
    --gens "$GENS" \
    --pop "$POP" \
    --elite "$ELITE" \
    --save-best "${prefix}_best.jsonl" \
    --plot-2d "${prefix}_best_2d.svg" \
    --plot-3d "${prefix}_best_3d.csv"
  echo ""
done < "$TARGETS_FILE"

echo "Outputs in: $OUT_DIR"
