# GGGP Bundle Split Plan: Rust Library & SDK

This document describes a practical extraction path for moving `gggp_bundle/`
into a standalone repository as a Rust library + SDK, including Python bindings.

## Why split

- decouple GGGP evolution engine from runtime orchestration changes
- make Rust core versioned and reusable across projects
- reduce coupling to `docker-compose.yml` complexity
- enable separate release cadence for SDK artifacts

## Target repository layout

```text
gggp-sdk/
  Cargo.toml                 # workspace
  crates/
    gggp-core/               # pure Rust library: grammar, crossover, mutation
    gggp-cli/                # CLI binaries (cfgdump, gggp, demos)
    gggp-py/                 # pyo3 bindings layer
  python/
    pyproject.toml           # maturin build config
    src/gggp_sdk/__init__.py
  docs/
  examples/
  .github/workflows/
```

## Extraction commands

Run from current mono-repo root:

```bash
# 1) Create split branch from folder history
git subtree split --prefix gggp_bundle -b split/gggp-bundle

# 2) Bootstrap new repo
mkdir -p ../gggp-sdk
cd ../gggp-sdk
git init
git pull ../mcs split/gggp-bundle

# 3) Set remote and push
git remote add origin <new-repo-url>
git push -u origin main
```

## Rust crate strategy

- keep existing `rust/` code as initial `gggp-core`
- move binaries (`cfgdump`, `gggp`, demos) into `gggp-cli`
- keep serde-free core where possible, expose stable API:
  - grammar parsing
  - chromosome handling
  - tree generation
  - crossover/mutation operators

## Python bindings strategy

Use `pyo3 + maturin`:

- `gggp-py` wraps stable functions from `gggp-core`
- expose deterministic APIs first:
  - `parse_grammar(text) -> Grammar`
  - `generate(grammar, chromosome, seed) -> str`
  - `crossover(chromosome_a, chromosome_b, seed) -> (a, b)`
  - `mutate(chromosome, seed) -> chromosome`
- publish wheels for Linux/macOS in CI

## Versioning

- semantic versioning for SDK (`v0.x` initially)
- tag Rust + Python artifacts from same commit
- changelog per release

## First milestone

- extract repo with history
- keep CLI behavior unchanged
- add minimal Python wheel (`generate`, `mutate`, `crossover`)
- integrate back in mono-repo via dependency pin (git tag / package version)

