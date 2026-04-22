# GGGP Embedding Demo Bundle (private)

This bundle is a self-contained snapshot of the GGGP + embedding demo and related docs/scripts.

## Requirements
- Rust toolchain (stable)
- Ollama running at http://localhost:11434
- Embedding model pulled (e.g. `all-minilm:33m` or `bge-large:latest`)

## Quick start
```
cd rust
cargo run --bin embedding_gggp -- --target "extract dates and amounts from invoices"
```

## Run the demo set
```
cd demos/semiotic_hypercube
./run_demo.sh
```
Outputs are written to `demos/semiotic_hypercube/out`.

## Run config via .cfg
If `--cfg` is supplied, the file may contain an `EmbeddingGGGP` node with run
parameters, and an optional `Grammar` node. Attributes can be written with or
without `@` prefix. Example (as text via `cfgdump`):
```
EmbeddingGGGP
  @Target (Str) = \"extract dates and amounts from invoices\"
  @Model (Str) = \"bge-large:latest\"
  @Seed (Int) = 42
  @Gens (Int) = 2000
  @Pop (Int) = 300
  @Elite (Int) = 12
  @MaxOps (Int) = 24
  @AxisStep (Real) = 1.0
  @ValueStep (Real) = 0.1
  @Crossover (Real) = 0.7
  @Mutation (Real) = 0.3
  @SaveBest (Str) = \"out/best.jsonl\"
  @Plot2D (Str) = \"out/best_2d.svg\"
  @Plot3D (Str) = \"out/best_3d.csv\"
Grammar
  ... grammar rules ...
```

## Integration (library)
This bundle includes a Rust crate in `rust/` with:
- CFG storage and node tree (`storage` module)
- GGGP engine (`gggp` module)
- Binaries: `embedding_gggp`, `embedding_poc`, `gggp`, `cfgdump`

Use as a path dependency in another Rust project:
```
# Cargo.toml
[dependencies]
semiotic_hypercube = { path = "../gggp_bundle/rust" }
```

Minimal example (load cfg and parse a GGGP output):
```rust
use semiotic_hypercube::{Node};
use semiotic_hypercube::gggp::{parse_text, Gggp};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Node::from_file("./some.cfg")?;
    let mut gggp = Gggp::new();
    // configure gggp as needed
    let _tree = parse_text(&cfg, "AX 0 1.0");
    Ok(())
}
```

## Files
- `rust/`: source code for GGGP + CFG storage and demo binaries.
- `docs/`: technical docs and roadmap.
- `demos/`: demo scripts, targets, and prompt-breeding C1 seed data.
- `CONTEXT_SUMMARY.md`: compressed chat context for continuing elsewhere.

## Notes
This is a private repo snapshot. Keep outputs and model settings consistent by fixing `--seed` and `--model`.
