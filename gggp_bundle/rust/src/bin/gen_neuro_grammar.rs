// ============================================================================
// gen_neuro_grammar -- emit VECGRAM grammars for the Semiotic Hypercube A1
// PoC. Two roles, two grammars, one binary.
//
//   cargo run --release --bin gen_neuro_grammar -- encoder <out_path>
//   cargo run --release --bin gen_neuro_grammar -- decoder <out_path>
//   cargo run --release --bin gen_neuro_grammar -- custom <dim> <out_path>
//
// Roles:
//   encoder: dim=16   (CODE_DIM;  matches spectral top-16 of T.npy)
//   decoder: dim=1024 (TARGET_DIM; matches Ollama Qwen3-Embedding-0.6B)
//   custom:  any dim
//
// Grammar shape (same for all roles, parameterized only by `dim`):
//   START -> <SEQ>
//   SEQ   -> <OP> <SEQ>        (choice 0)
//         -> <OP> <SEQ>        (choice 1, duplicated to bias toward longer
//                               programs: with 3 choices, 2/3 continuation =
//                               expected length ~3 instead of ~2 with 1/2)
//         -> <OP>              (choice 2)
//   OP    -> AX   axis val     (axis: 0..dim-1; val: -1..1 step 0.5)
//         -> SCALE val         (val:  0.5..1.5 step 0.5)
//         -> NORM
//         -> MIX  a b w        (a, b: 0..dim-1; w: 0..1 step 0.25)
//         -> ROT  a b theta    (a, b: 0..dim-1; theta: -1..1 step 0.5)
//         -> FRAC exp          (exp:  0.5..2.0 step 0.5)
//
// Differences from the previous grammar (doc-for-record):
//   - dropped ZERO (collapsed signal, killed random-sweep F)
//   - added MIX, ROT, FRAC (already supported by vector.rs ops dispatch)
//   - added SEQ length bias (mean program length ~3 vs ~2 with uniform weights)
//   - axis range now scales with role-specific dim (16 or 1024)
//
// MEDP ref: A1 T6 -- grammar regeneration prerequisite to T7 EA runner.
// ============================================================================

use semiotic_hypercube::Node;
use semiotic_hypercube::gggp::parse_text;
use std::env;
use std::process;

fn build_neuro_grammar(dim: i32) -> Node {
    if dim < 2 {
        eprintln!("gen_neuro_grammar: dim must be >= 2, got {}", dim);
        process::exit(2);
    }

    let mut root = Node::new("VECGRAM");
    root.set_int("MaxDepth", 24);
    root.set_int("MaxCrossoverNodes", 4);
    root.set_int("MaxMutationNodes", 4);

    let rules = root.get_or_create_child("RULES");

    // START -> <SEQ>
    let mut start = Node::new("START");
    let choices = start.get_or_create_child("CHOICES");
    let mut c0 = Node::new("0");
    c0.set_str("Text", b"<SEQ>".to_vec());
    choices.add_child(c0);
    rules.add_child(start);

    // SEQ -> <OP> <SEQ> | <OP> <SEQ> | <OP>
    let mut seq = Node::new("SEQ");
    let choices = seq.get_or_create_child("CHOICES");
    for name in ["0", "1"].iter() {
        let mut c = Node::new(*name);
        c.set_str("Text", b"<OP> <SEQ>".to_vec());
        choices.add_child(c);
    }
    let mut c_term = Node::new("2");
    c_term.set_str("Text", b"<OP>".to_vec());
    choices.add_child(c_term);
    rules.add_child(seq);

    // OP -> AX | SCALE | NORM | MIX | ROT | FRAC
    let mut op = Node::new("OP");
    let choices = op.get_or_create_child("CHOICES");

    let mut c = Node::new("0");
    c.set_str(
        "Text",
        format!(
            "AX <axis from=0 to={} inc=1> <val from=-1 to=1 inc=0.5>",
            dim - 1
        )
        .as_bytes()
        .to_vec(),
    );
    choices.add_child(c);

    let mut c = Node::new("1");
    c.set_str("Text", b"SCALE <val from=0.5 to=1.5 inc=0.5>".to_vec());
    choices.add_child(c);

    let mut c = Node::new("2");
    c.set_str("Text", b"NORM".to_vec());
    choices.add_child(c);

    let mut c = Node::new("3");
    c.set_str(
        "Text",
        format!(
            "MIX <a from=0 to={d} inc=1> <b from=0 to={d} inc=1> <w from=0 to=1 inc=0.25>",
            d = dim - 1
        )
        .as_bytes()
        .to_vec(),
    );
    choices.add_child(c);

    let mut c = Node::new("4");
    c.set_str(
        "Text",
        format!(
            "ROT <a from=0 to={d} inc=1> <b from=0 to={d} inc=1> <theta from=-1 to=1 inc=0.5>",
            d = dim - 1
        )
        .as_bytes()
        .to_vec(),
    );
    choices.add_child(c);

    let mut c = Node::new("5");
    c.set_str("Text", b"FRAC <exp from=0.5 to=2.0 inc=0.5>".to_vec());
    choices.add_child(c);

    rules.add_child(op);

    root
}

fn usage_and_exit() -> ! {
    eprintln!(
        "usage:\n  \
         gen_neuro_grammar encoder <out_path>       (dim=16)\n  \
         gen_neuro_grammar decoder <out_path>       (dim=1024)\n  \
         gen_neuro_grammar custom <dim> <out_path>"
    );
    process::exit(2);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        usage_and_exit();
    }

    let (dim, out_path) = match args[1].as_str() {
        "encoder" => (16_i32, args[2].clone()),
        "decoder" => (1024_i32, args[2].clone()),
        "custom" => {
            if args.len() < 4 {
                usage_and_exit();
            }
            let d = args[2].parse::<i32>().unwrap_or_else(|e| {
                eprintln!("invalid dim '{}': {}", args[2], e);
                process::exit(2);
            });
            (d, args[3].clone())
        }
        _ => usage_and_exit(),
    };

    // Parse all text templates so get_tree_from_chromosome sees choice nodes
    // with their PARAMS already expanded. Without this the runtime rebuild
    // only sees raw Text and fails to expand <axis> / <val> / etc.
    let mut grammar = build_neuro_grammar(dim);
    if let Some(rules) = grammar.child_mut("RULES") {
        for symbol in rules.children_mut() {
            if let Some(choices) = symbol.child_mut("CHOICES") {
                for choice in choices.children_mut() {
                    parse_text(choice);
                }
            }
        }
    }

    if let Err(e) = grammar.to_file(&out_path) {
        eprintln!("Failed to write {}: {}", out_path, e);
        process::exit(1);
    }
    println!(
        "gen_neuro_grammar: wrote {} (role-dim={}, 6 ops, SEQ length bias x3/2, no ZERO)",
        out_path, dim
    );
}
