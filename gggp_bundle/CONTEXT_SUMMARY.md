# Compressed context (GGGP + embedding demo)

## Core idea
- A GGGP engine (Rust) is used to evolve programs that construct high-dimensional vectors.
- Fitness is cosine similarity between the evolved vector and an embedding of a target text (from Ollama).
- Grammar-based programs output a sequence of vector ops (AX, SCALE, NORM, MIX, ROT, FRAC, ZERO).

## Key binaries
- `rust/src/bin/embedding_gggp.rs`
  - Main demo: GGGP -> vector phenotype.
  - CLI flags: `--model`, `--url`, `--seed`, `--gens`, `--pop`, `--elite`, `--max-ops`, `--axis-step`, `--value-step`, `--target`, `--cfg`, `--dump-cfg`, `--save-best`, `--plot-2d`, `--plot-3d`, `--crossover`, `--mutation`.
  - If `--cfg` includes an `EmbeddingGGGP` node, run parameters are read from it (Target/Model/Seed/etc). If it also contains `Grammar`, that grammar is used.
  - Saves best records to JSONL and optional SVG/CSV projections.
- `rust/src/bin/embedding_poc.rs`
  - Simple GA directly mutating vectors (baseline).
- `rust/src/bin/embedding_gggp_gemini.rs`
  - Alternate implementation using clap/anyhow/nalgebra (kept for comparison).

## GGGP details
- Grammar auto-generated if `--cfg` is not supplied; placeholders are resolved using target dimension.
- Vector operations:
  - `AX axis value` adds to a coordinate.
  - `SCALE value` scales vector.
  - `NORM` normalizes.
  - `MIX` mixes against another random vector.
  - `ROT` rotates (pairwise swap/rotate in subspaces).
  - `FRAC` fractional transform (non-linear dampening).
  - `ZERO` resets vector.

## Demo scripts
- `demos/semiotic_hypercube/run_demo.sh`
  - Runs `embedding_gggp` on 3 targets in `demo_targets.txt` and stores results in `out/`.
- `demos/prompt_breeding/c1`
  - C1 dataset: `c1_prompt_seed.txt`, `c1_schema.json`, `generate_cases.py`, `eval_ollama.py`, `c1_cases.jsonl`.

## Docs
- `docs/EMBEDDING_GGGP.md` (how the demo works)
- `docs/SEMIOTIC-HYPERCUBE.md` (concept + PoC plan)
- `docs/TECH_SPEC_RUST_PORT.md`, `docs/UMC_HYPOTHESES.md`
- `docs/DEMO_CANDIDATES.md`, `docs/EVOLUTIONARY_ENGINE_ROADMAP.md`, `docs/RUST_PORT_STATUS.md`

## Recent runs
- `bge-large:latest`, target "extract dates and amounts from invoices",
  - `gens=2000`, `pop=300`, `elite=12`, best fitness ~0.3701.
- `all-minilm:33m` was faster but lower quality; best ~0.69 on earlier run for a different config.

## Next work ideas (not done yet)
- More targets + larger runs, model comparison.
- Selection tweaks, adaptive mutation, bloat control.
- Stronger visualization and reports.
- Full .cfg integration for grammar + run config.

## Goals alignment (user-defined)
- 1) Evolutionary Engine as standalone SDK product.
- 2) EVO applications and demos outside trading.
- 2.2) UMC research track.
