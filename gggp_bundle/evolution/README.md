# GGGP Evolution MVP

This folder contains the first deterministic step from manual compose engineering
to evolutionary configuration synthesis.

## Formula

Current experiment is encoded as:

```text
(eve + env) x (adam + eden) + (mods) = (abel + env)
```

## Contents

- `genotypes/` - predefined parent/modifier gene sets
- `crosses/` - crossover recipe
- `phenotypes/` - generated outputs:
  - merged phenotype YAML
  - runtime JSON consumed by `tools/genesis_bootstrap.py`
  - generated compose overlay
  - generated BNF/CFG representation
- `../tools/cross_compose_cfg.py` - generator tool

## Run

```bash
python gggp_bundle/tools/cross_compose_cfg.py \
  --cross-file gggp_bundle/evolution/crosses/abel_env.yaml \
  --genotypes-dir gggp_bundle/evolution/genotypes \
  --out-dir gggp_bundle/evolution/phenotypes \
  --compose-in docker-compose.yml

python tools/genesis_bootstrap.py \
  --runtime-config gggp_bundle/evolution/phenotypes/abel_env.runtime.json

python tools/generational_cycle.py \
  --archive-souls eve,adam \
  --launch-descendant
```

## Notes

- This is deterministic synthesis with preset options.
- It is intentionally simple and reproducible.
- Next step is moving rules into Rust GGGP runtime and replacing static merge with grammar-driven operators.
