# Python bridge smoke tests

These scripts live outside the normal `cargo test` tree because they exercise
the PyO3 extension module built by `maturin`, which requires a Python runtime
and the `python` cargo feature.

## Prerequisites

From `gggp_bundle/rust/`:

```bash
cargo run --bin gen_grammar                 # writes ../test_grammar.cfg
python -m pip install -U pip maturin numpy
maturin develop --release --features python  # installs semiotic_hypercube
```

## Run

From `gggp_bundle/` (so `test_grammar.cfg` resolves):

```bash
python rust/tests/python_bridge/test_bridge.py
```

Expected output ends with `[OK] CMA-ES converged ...`. Low-fitness warnings
indicate the grammar fixture has drifted or the CMA-ES budget is too small —
regenerate the grammar and retry before filing an issue.
