# Genesis Experiment

`consciousness_loop_genesis.py` launches a descendant soul loop (for example, `abel`) that:
- loads archived ancestor context from `Legacy/Archive/{ancestor}_archive/`
- reads `testament.txt`, `identity_summary.txt`, and `primal_seed.json`
- queries own memory and lineage memory (`/memories/query_lineage`) every tick
- generates autonomous thoughts through Ollama
- stores new thoughts in memory and appends JSONL logs

## Run

```bash
python experiments/genesis/consciousness_loop_genesis.py \
  --soul-id abel \
  --ancestor-ids eve,adam \
  --memory-endpoint http://localhost:8087 \
  --ollama-endpoint http://localhost:11434 \
  --llm-model llama3:8b \
  --tick-interval 15 \
  --environment neutral
```

## Run With Evolution Runtime Config

```bash
python tools/genesis_bootstrap.py \
  --runtime-config gggp_bundle/evolution/phenotypes/abel_env.runtime.json \
  --wait 180
```

This flow ingests ancestor archives automatically and then starts the loop with
parameters produced by `(eve + env) x (adam + eden) + mods`.

## One-Command Generation Cycle

```bash
python tools/generational_cycle.py \
  --archive-souls eve,adam \
  --launch-descendant
```

## Notes

- `OLLAMA_GENERATE_URL` overrides `--ollama-endpoint` when set.
- Default log file is `logs/{soul_id}_thoughts.jsonl`.
- The loop is resilient by design: network and parsing errors are logged and the process continues.
