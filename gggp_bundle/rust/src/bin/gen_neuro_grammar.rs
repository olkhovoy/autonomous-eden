// ============================================================================
// gen_neuro_grammar -- emit VECGRAM grammars for the Semiotic Hypercube
// A1/A2 PoCs. Multiple roles, one binary.
//
//   cargo run --release --bin gen_neuro_grammar -- encoder <out_path>
//   cargo run --release --bin gen_neuro_grammar -- decoder <out_path>
//   cargo run --release --bin gen_neuro_grammar -- custom <dim> <out_path>
//   cargo run --release --bin gen_neuro_grammar -- decoder-nc3 <out_path>
//   cargo run --release --bin gen_neuro_grammar -- decoder-nc3-custom \
//                                                  <target_dim> <code_dim> <out_path>
//
// Roles:
//   encoder:       dim=16   (CODE_DIM;  matches spectral top-16 of T.npy).
//   decoder:       dim=1024 (TARGET_DIM; matches Ollama Qwen3-Embedding-0.6B).
//   custom:        any dim.
//   decoder-nc3:   target_dim=16, code_dim=8 (A2 PCA defaults).
//   decoder-nc3-custom: explicit (target_dim, code_dim).
//
// Grammar shape for encoder/decoder/custom (parameterized only by `dim`):
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
// decoder-nc3 extends OP with three NC3 (downward-causation) ops that read
// the runtime `input` buffer (== c_i when D is rendered with the code):
//   OP    -> ... 6 baseline ops ...
//         -> CTRL axis cidx     (axis: 0..target_dim-1; cidx: 0..code_dim-1)
//         -> SBC  cidx          (scalar broadcast by input[cidx])
//         -> ADDC axis cidx     (multiplicative gating: s[a] += input[c]*s[a])
// Semantics are executed in gggp::vector::execute_ops (see S1a).
//
// Differences from the previous grammar (doc-for-record):
//   - dropped ZERO (collapsed signal, killed random-sweep F).
//   - added MIX, ROT, FRAC (already supported by vector.rs ops dispatch).
//   - added SEQ length bias (mean program length ~3 vs ~2 with uniform weights).
//   - axis range scales with role-specific dim.
//   - A2 S1b: added decoder-nc3 role (6 + 3 ops, two-dim parameterization).
//
// MEDP ref: A1 T6 / A2 S1b -- grammar regeneration prerequisite to EA runner.
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

/// Build a VECGRAM with the 6 baseline ops + the 3 NC3 downward-causation
/// ops (CTRL / SBC / ADDC). Two independent dims:
///   target_dim -- axis bound for AX / MIX / ROT / CTRL / ADDC (== dim of
///                 the decoder's output state, e.g. 16 in A2 PCA mode).
///   code_dim   -- cidx bound for CTRL / SBC / ADDC (== dim of the code
///                 buffer `c_i` handed to the decoder, e.g. 8 in A2).
/// See module docstring for the full grammar shape.
fn build_neuro_grammar_nc3(target_dim: i32, code_dim: i32) -> Node {
    if target_dim < 2 {
        eprintln!(
            "gen_neuro_grammar: target_dim must be >= 2, got {}",
            target_dim
        );
        process::exit(2);
    }
    if code_dim < 1 {
        eprintln!(
            "gen_neuro_grammar: code_dim must be >= 1, got {}",
            code_dim
        );
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

    // SEQ -> <OP> <SEQ> | <OP> <SEQ> | <OP>     (length bias ×3/2)
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

    // OP -> 6 baseline ops + 3 NC3 ops (9 total)
    let mut op = Node::new("OP");
    let choices = op.get_or_create_child("CHOICES");
    let t_hi = target_dim - 1;
    let c_hi = code_dim - 1;

    // --- Baseline 6 (identical to build_neuro_grammar) --------------------
    let mut c = Node::new("0");
    c.set_str(
        "Text",
        format!(
            "AX <axis from=0 to={t} inc=1> <val from=-1 to=1 inc=0.5>",
            t = t_hi
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
            "MIX <a from=0 to={t} inc=1> <b from=0 to={t} inc=1> <w from=0 to=1 inc=0.25>",
            t = t_hi
        )
        .as_bytes()
        .to_vec(),
    );
    choices.add_child(c);

    let mut c = Node::new("4");
    c.set_str(
        "Text",
        format!(
            "ROT <a from=0 to={t} inc=1> <b from=0 to={t} inc=1> <theta from=-1 to=1 inc=0.5>",
            t = t_hi
        )
        .as_bytes()
        .to_vec(),
    );
    choices.add_child(c);

    let mut c = Node::new("5");
    c.set_str("Text", b"FRAC <exp from=0.5 to=2.0 inc=0.5>".to_vec());
    choices.add_child(c);

    // --- NC3 downward-causation ops (A2 S1b) ------------------------------
    let mut c = Node::new("6");
    c.set_str(
        "Text",
        format!(
            "CTRL <axis from=0 to={t} inc=1> <cidx from=0 to={c} inc=1>",
            t = t_hi,
            c = c_hi
        )
        .as_bytes()
        .to_vec(),
    );
    choices.add_child(c);

    let mut c = Node::new("7");
    c.set_str(
        "Text",
        format!(
            "SBC <cidx from=0 to={c} inc=1>",
            c = c_hi
        )
        .as_bytes()
        .to_vec(),
    );
    choices.add_child(c);

    let mut c = Node::new("8");
    c.set_str(
        "Text",
        format!(
            "ADDC <axis from=0 to={t} inc=1> <cidx from=0 to={c} inc=1>",
            t = t_hi,
            c = c_hi
        )
        .as_bytes()
        .to_vec(),
    );
    choices.add_child(c);

    rules.add_child(op);

    root
}

fn usage_and_exit() -> ! {
    eprintln!(
        "usage:\n  \
         gen_neuro_grammar encoder <out_path>              (dim=16)\n  \
         gen_neuro_grammar decoder <out_path>              (dim=1024)\n  \
         gen_neuro_grammar custom <dim> <out_path>\n  \
         gen_neuro_grammar decoder-nc3 <out_path>          (target_dim=16, code_dim=8)\n  \
         gen_neuro_grammar decoder-nc3-custom <target_dim> <code_dim> <out_path>"
    );
    process::exit(2);
}

/// Parsed CLI spec for a single grammar generation request.
enum GrammarSpec {
    /// Flat axis-grammar with 6 ops and a single `dim` bound.
    Flat { dim: i32, out_path: String },
    /// Axis-grammar with 6 baseline ops + 3 NC3 ops; two independent bounds.
    Nc3 {
        target_dim: i32,
        code_dim: i32,
        out_path: String,
    },
}

fn parse_i32_or_die(arg: &str, what: &str) -> i32 {
    arg.parse::<i32>().unwrap_or_else(|e| {
        eprintln!("invalid {} '{}': {}", what, arg, e);
        process::exit(2);
    })
}

fn parse_cli(args: &[String]) -> GrammarSpec {
    if args.len() < 3 {
        usage_and_exit();
    }
    match args[1].as_str() {
        "encoder" => GrammarSpec::Flat {
            dim: 16,
            out_path: args[2].clone(),
        },
        "decoder" => GrammarSpec::Flat {
            dim: 1024,
            out_path: args[2].clone(),
        },
        "custom" => {
            if args.len() < 4 {
                usage_and_exit();
            }
            GrammarSpec::Flat {
                dim: parse_i32_or_die(&args[2], "dim"),
                out_path: args[3].clone(),
            }
        }
        "decoder-nc3" => GrammarSpec::Nc3 {
            target_dim: 16,
            code_dim: 8,
            out_path: args[2].clone(),
        },
        "decoder-nc3-custom" => {
            if args.len() < 5 {
                usage_and_exit();
            }
            GrammarSpec::Nc3 {
                target_dim: parse_i32_or_die(&args[2], "target_dim"),
                code_dim: parse_i32_or_die(&args[3], "code_dim"),
                out_path: args[4].clone(),
            }
        }
        _ => usage_and_exit(),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let spec = parse_cli(&args);

    let (mut grammar, out_path, summary) = match spec {
        GrammarSpec::Flat { dim, out_path } => {
            let g = build_neuro_grammar(dim);
            let s = format!(
                "role-dim={}, 6 ops, SEQ length bias x3/2, no ZERO",
                dim
            );
            (g, out_path, s)
        }
        GrammarSpec::Nc3 {
            target_dim,
            code_dim,
            out_path,
        } => {
            let g = build_neuro_grammar_nc3(target_dim, code_dim);
            let s = format!(
                "role=decoder-nc3, target_dim={}, code_dim={}, 9 ops (6 baseline + CTRL/SBC/ADDC), SEQ length bias x3/2",
                target_dim, code_dim
            );
            (g, out_path, s)
        }
    };

    // Parse all text templates so get_tree_from_chromosome sees choice nodes
    // with their PARAMS already expanded. Without this the runtime rebuild
    // only sees raw Text and fails to expand <axis> / <val> / etc.
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
    println!("gen_neuro_grammar: wrote {} ({})", out_path, summary);
}
