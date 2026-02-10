Semiotic Hypercube: embedding GGGP demo

This demo supports grammar loading from .cfg (binary). You can generate a template
cfg with the built-in vector grammar using --dump-cfg, then reuse it with --cfg.

Example:
  cargo run --bin embedding_gggp -- \
    --target "sort a list of integers" \
    --dump-cfg /tmp/vector_grammar.cfg

Then run with that grammar:
  cargo run --bin embedding_gggp -- \
    --target "sort a list of integers" \
    --cfg /tmp/vector_grammar.cfg

Placeholders supported in Text fields:
  {DIM}        -> replaced with (dimension - 1)
  {AXIS_STEP}  -> replaced with --axis-step value
  {VALUE_STEP} -> replaced with --value-step value

Visualization options:
  --save-best /tmp/best_vectors.jsonl
  --plot-2d /tmp/best_2d.svg   (SVG polyline)
  --plot-3d /tmp/best_3d.csv   (CSV)

Extended ops in grammar:
  AX, SCALE, NORM, MIX, ROT, FRAC

Full demo run (3 targets):
  ./run_demo.sh

Optional GA tuning:
  --crossover 0.7
  --mutation 0.3
