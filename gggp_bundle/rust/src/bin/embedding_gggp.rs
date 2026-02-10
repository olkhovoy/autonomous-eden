use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde_json::json;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;

use gor_data_storage::gggp::{calc_lengths, parse_text, Gggp, GpIndividual};
use gor_data_storage::Node;

struct Config {
    model: String,
    url: String,
    seed: u64,
    gens: usize,
    pop: usize,
    elite: usize,
    max_ops: usize,
    axis_step: f64,
    value_step: f64,
    target_text: String,
    crossover_rate: f64,
    mutation_rate: f64,
    cfg_path: Option<String>,
    dump_cfg: Option<String>,
    save_best: Option<String>,
    plot_2d: Option<String>,
    plot_3d: Option<String>,
}

fn main() {
    let mut cfg = parse_args();
    let mut cfg_node: Option<Node> = None;

    if let Some(path) = cfg.cfg_path.as_deref() {
        match Node::from_file(path) {
            Ok(node) => {
                apply_run_cfg(&mut cfg, &node);
                cfg_node = Some(node);
            }
            Err(err) => {
                eprintln!("error reading cfg {}: {}", path, err);
                std::process::exit(1);
            }
        }
    }

    if cfg.target_text.is_empty() {
        eprintln!("missing target: pass --target or provide Target in cfg");
        std::process::exit(1);
    }

    let target_vec = match fetch_embedding(&cfg.url, &cfg.model, &cfg.target_text) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("embedding error: {err}");
            std::process::exit(1);
        }
    };

    let dim = target_vec.len();
    println!("target: {}", cfg.target_text);
    println!("dim: {}", dim);
    println!("seed: {}", cfg.seed);

    let mut grammar = if let Some(node) = cfg_node.as_ref() {
        if let Some(grammar_node) = find_grammar_node(node) {
            grammar_node.clone()
        } else {
            build_vector_grammar(dim as i32, cfg.max_ops as i32, cfg.axis_step, cfg.value_step)
        }
    } else {
        build_vector_grammar(dim as i32, cfg.max_ops as i32, cfg.axis_step, cfg.value_step)
    };
    apply_placeholders(&mut grammar, dim, cfg.axis_step, cfg.value_step);
    finalize_grammar(&mut grammar);

    if let Some(path) = cfg.dump_cfg.as_deref() {
        if let Err(err) = grammar.to_file(path) {
            eprintln!("error writing cfg {}: {}", path, err);
            std::process::exit(1);
        }
        println!("saved grammar to {}", path);
    }

    let cfgs = vec![grammar.clone()];
    let mut gggp = Gggp::new();
    gggp.set_on_get_fitness({
        let target_vec = target_vec.clone();
        move |ind| Some(score_individual(ind, &target_vec))
    });
    gggp
        .init_from_nodes(
            &cfgs,
            cfg.pop,
            cfg.elite,
            cfg.crossover_rate,
            cfg.mutation_rate,
        )
        .expect("init gggp failed");

    let mut best_score = f64::NEG_INFINITY;
    let mut best_ind: Option<GpIndividual> = None;
    let mut best_so_far = f64::NEG_INFINITY;
    let proj_2d = cfg
        .plot_2d
        .as_deref()
        .map(|_| Projection::new(dim, 2, cfg.seed ^ 0xA5A5_A5A5));
    let proj_3d = cfg
        .plot_3d
        .as_deref()
        .map(|_| Projection::new(dim, 3, cfg.seed ^ 0xC3C3_C3C3));
    let mut points_2d: Vec<(usize, f64, f64, f64)> = Vec::new();
    let mut points_3d: Vec<(usize, f64, f64, f64, f64)> = Vec::new();

    for gen in 0..cfg.gens {
        gggp.step();
        for ind in gggp.individuals() {
            let score = score_individual(ind, &target_vec);
            if score > best_score {
                best_score = score;
                best_ind = Some(ind.clone());
            }
        }
        if gen % 10 == 0 || gen + 1 == cfg.gens {
            println!("gen {:03} best {:.5}", gen + 1, best_score);
        }
        if best_score > best_so_far + 1e-12 {
            best_so_far = best_score;
            if let Some(best_ind) = &best_ind {
                let text = best_ind.trees()[0].text();
                let vec = vector_from_text(&text, target_vec.len());
                if let Some(path) = cfg.save_best.as_deref() {
                    if let Err(err) = save_best_record(
                        path,
                        gen + 1,
                        best_score,
                        &vec,
                        &text,
                        &cfg.target_text,
                        &cfg.model,
                        cfg.seed,
                    ) {
                        eprintln!("warning: failed to save best record: {err}");
                    }
                }
                if let Some(proj) = &proj_2d {
                    let p = proj.project(&vec);
                    if p.len() >= 2 {
                        points_2d.push((gen + 1, best_score, p[0], p[1]));
                        if let Some(path) = cfg.plot_2d.as_deref() {
                            if path.ends_with(".svg") {
                                let svg_path = numbered_svg_path(path, gen + 1);
                                if let Err(err) = write_plot_2d(&svg_path, &points_2d) {
                                    eprintln!("warning: failed to write 2d plot: {err}");
                                } else {
                                    println!("saved 2d plot: {}", svg_path);
                                }
                            }
                        }
                    }
                }
                if let Some(proj) = &proj_3d {
                    let p = proj.project(&vec);
                    if p.len() >= 3 {
                        points_3d.push((gen + 1, best_score, p[0], p[1], p[2]));
                    }
                }
            }
        }
    }

    let best_ind = best_ind.expect("no best individual");
    let text = best_ind.trees()[0].text();
    let best_vec = vector_from_text(&text, target_vec.len());
    let best_sim = cosine_similarity(&best_vec, &target_vec);
    println!("best fitness: {:.6}", best_sim);
    println!("best program: {}", text.trim());

    if let Some(path) = cfg.plot_2d.as_deref() {
        if !path.ends_with(".svg") {
            if let Err(err) = write_plot_2d(path, &points_2d) {
                eprintln!("warning: failed to write 2d plot: {err}");
            } else {
                println!("saved 2d plot: {}", path);
            }
        }
    }
    if let Some(path) = cfg.plot_3d.as_deref() {
        if let Err(err) = write_plot_3d(path, &points_3d) {
            eprintln!("warning: failed to write 3d plot: {err}");
        } else {
            println!("saved 3d plot: {}", path);
        }
    }
}

fn parse_args() -> Config {
    let mut cfg = Config {
        model: "all-minilm:33m".to_string(),
        url: "http://localhost:11434/api/embeddings".to_string(),
        seed: 42,
        gens: 200,
        pop: 120,
        elite: 8,
        max_ops: 24,
        axis_step: 1.0,
        value_step: 0.1,
        target_text: String::new(),
        crossover_rate: 0.7,
        mutation_rate: 0.3,
        cfg_path: None,
        dump_cfg: None,
        save_best: None,
        plot_2d: None,
        plot_3d: None,
    };

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => cfg.model = read_string(&mut args, "--model"),
            "--url" => cfg.url = read_string(&mut args, "--url"),
            "--seed" => cfg.seed = read_u64(&mut args, "--seed"),
            "--gens" => cfg.gens = read_usize(&mut args, "--gens"),
            "--pop" => cfg.pop = read_usize(&mut args, "--pop"),
            "--elite" => cfg.elite = read_usize(&mut args, "--elite"),
            "--max-ops" => cfg.max_ops = read_usize(&mut args, "--max-ops"),
            "--axis-step" => cfg.axis_step = read_f64(&mut args, "--axis-step"),
            "--value-step" => cfg.value_step = read_f64(&mut args, "--value-step"),
            "--crossover" => cfg.crossover_rate = read_f64(&mut args, "--crossover"),
            "--mutation" => cfg.mutation_rate = read_f64(&mut args, "--mutation"),
            "--target" => cfg.target_text = read_string(&mut args, "--target"),
            "--cfg" => cfg.cfg_path = Some(read_string(&mut args, "--cfg")),
            "--dump-cfg" => cfg.dump_cfg = Some(read_string(&mut args, "--dump-cfg")),
            "--save-best" => cfg.save_best = Some(read_string(&mut args, "--save-best")),
            "--plot-2d" => cfg.plot_2d = Some(read_string(&mut args, "--plot-2d")),
            "--plot-3d" => cfg.plot_3d = Some(read_string(&mut args, "--plot-3d")),
            _ if arg.starts_with("--model=") => cfg.model = arg[8..].to_string(),
            _ if arg.starts_with("--url=") => cfg.url = arg[6..].to_string(),
            _ if arg.starts_with("--seed=") => cfg.seed = parse_u64(&arg[7..], "--seed"),
            _ if arg.starts_with("--gens=") => cfg.gens = parse_usize(&arg[7..], "--gens"),
            _ if arg.starts_with("--pop=") => cfg.pop = parse_usize(&arg[6..], "--pop"),
            _ if arg.starts_with("--elite=") => cfg.elite = parse_usize(&arg[8..], "--elite"),
            _ if arg.starts_with("--max-ops=") => cfg.max_ops = parse_usize(&arg[10..], "--max-ops"),
            _ if arg.starts_with("--axis-step=") => cfg.axis_step = parse_f64(&arg[12..], "--axis-step"),
            _ if arg.starts_with("--value-step=") => cfg.value_step = parse_f64(&arg[13..], "--value-step"),
            _ if arg.starts_with("--crossover=") => cfg.crossover_rate = parse_f64(&arg[12..], "--crossover"),
            _ if arg.starts_with("--mutation=") => cfg.mutation_rate = parse_f64(&arg[11..], "--mutation"),
            _ if arg.starts_with("--target=") => cfg.target_text = arg[9..].to_string(),
            _ if arg.starts_with("--cfg=") => cfg.cfg_path = Some(arg[6..].to_string()),
            _ if arg.starts_with("--dump-cfg=") => cfg.dump_cfg = Some(arg[11..].to_string()),
            _ if arg.starts_with("--save-best=") => cfg.save_best = Some(arg[12..].to_string()),
            _ if arg.starts_with("--plot-2d=") => cfg.plot_2d = Some(arg[10..].to_string()),
            _ if arg.starts_with("--plot-3d=") => cfg.plot_3d = Some(arg[10..].to_string()),
            _ => {
                eprintln!("unknown arg: {arg}");
                std::process::exit(2);
            }
        }
    }

    cfg
}

fn apply_run_cfg(cfg: &mut Config, root: &Node) {
    let run_node = find_run_cfg_node(root).unwrap_or(root);
    if let Some(value) = read_str_attr(run_node, "Target") {
        if cfg.target_text.is_empty() {
            cfg.target_text = value;
        }
    }
    if let Some(value) = read_str_attr(run_node, "Model") {
        if cfg.model == "all-minilm:33m" {
            cfg.model = value;
        }
    }
    if let Some(value) = read_str_attr(run_node, "Url") {
        if cfg.url == "http://localhost:11434/api/embeddings" {
            cfg.url = value;
        }
    }
    if let Some(value) = read_u64_attr(run_node, "Seed") {
        if cfg.seed == 42 {
            cfg.seed = value;
        }
    }
    if let Some(value) = read_usize_attr(run_node, "Gens") {
        if cfg.gens == 200 {
            cfg.gens = value;
        }
    }
    if let Some(value) = read_usize_attr(run_node, "Pop") {
        if cfg.pop == 120 {
            cfg.pop = value;
        }
    }
    if let Some(value) = read_usize_attr(run_node, "Elite") {
        if cfg.elite == 8 {
            cfg.elite = value;
        }
    }
    if let Some(value) = read_usize_attr(run_node, "MaxOps") {
        if cfg.max_ops == 24 {
            cfg.max_ops = value;
        }
    }
    if let Some(value) = read_f64_attr(run_node, "AxisStep") {
        if (cfg.axis_step - 1.0).abs() < f64::EPSILON {
            cfg.axis_step = value;
        }
    }
    if let Some(value) = read_f64_attr(run_node, "ValueStep") {
        if (cfg.value_step - 0.1).abs() < f64::EPSILON {
            cfg.value_step = value;
        }
    }
    if let Some(value) = read_f64_attr(run_node, "Crossover") {
        if (cfg.crossover_rate - 0.7).abs() < f64::EPSILON {
            cfg.crossover_rate = value;
        }
    }
    if let Some(value) = read_f64_attr(run_node, "Mutation") {
        if (cfg.mutation_rate - 0.3).abs() < f64::EPSILON {
            cfg.mutation_rate = value;
        }
    }
    if let Some(value) = read_str_attr(run_node, "SaveBest") {
        if cfg.save_best.is_none() {
            cfg.save_best = Some(value);
        }
    }
    if let Some(value) = read_str_attr(run_node, "Plot2D") {
        if cfg.plot_2d.is_none() {
            cfg.plot_2d = Some(value);
        }
    }
    if let Some(value) = read_str_attr(run_node, "Plot3D") {
        if cfg.plot_3d.is_none() {
            cfg.plot_3d = Some(value);
        }
    }
}

fn find_run_cfg_node<'a>(root: &'a Node) -> Option<&'a Node> {
    for name in ["EmbeddingGGGP", "EMBEDDING_GGGP", "Embedding", "Run"] {
        if let Some(node) = find_node_by_name(root, name) {
            return Some(node);
        }
    }
    None
}

fn find_grammar_node<'a>(root: &'a Node) -> Option<&'a Node> {
    find_node_by_name(root, "Grammar")
}

fn find_node_by_name<'a>(root: &'a Node, name: &str) -> Option<&'a Node> {
    if name_eq(root, name) {
        return Some(root);
    }
    for child in root.children() {
        if let Some(found) = find_node_by_name(child, name) {
            return Some(found);
        }
    }
    None
}

fn name_eq(node: &Node, name: &str) -> bool {
    let bytes = node.name();
    let name_bytes = name.as_bytes();
    if bytes.len() != name_bytes.len() {
        return false;
    }
    bytes
        .iter()
        .zip(name_bytes.iter())
        .all(|(a, b)| a.to_ascii_lowercase() == b.to_ascii_lowercase())
}

fn read_str_attr(node: &Node, base: &str) -> Option<String> {
    if let Some(value) = read_str_attr_raw(node, base) {
        return Some(value);
    }
    let prefixed = format!("@{base}");
    read_str_attr_raw(node, &prefixed)
}

fn read_str_attr_raw(node: &Node, name: &str) -> Option<String> {
    if !node.attr_exists(name.as_bytes()) {
        return None;
    }
    let raw = node.get_str(name.as_bytes());
    if raw.is_empty() {
        return None;
    }
    Some(String::from_utf8_lossy(&raw).to_string())
}

fn read_u64_attr(node: &Node, base: &str) -> Option<u64> {
    read_i64_attr(node, base).and_then(|v| if v >= 0 { Some(v as u64) } else { None })
}

fn read_usize_attr(node: &Node, base: &str) -> Option<usize> {
    read_i64_attr(node, base).and_then(|v| if v >= 0 { Some(v as usize) } else { None })
}

fn read_i64_attr(node: &Node, base: &str) -> Option<i64> {
    if node.attr_exists(base.as_bytes()) {
        return Some(node.get_int(base.as_bytes()) as i64);
    }
    let prefixed = format!("@{base}");
    if node.attr_exists(prefixed.as_bytes()) {
        return Some(node.get_int(prefixed.as_bytes()) as i64);
    }
    None
}

fn read_f64_attr(node: &Node, base: &str) -> Option<f64> {
    if node.attr_exists(base.as_bytes()) {
        return Some(node.get_real(base.as_bytes()));
    }
    let prefixed = format!("@{base}");
    if node.attr_exists(prefixed.as_bytes()) {
        return Some(node.get_real(prefixed.as_bytes()));
    }
    None
}

fn read_string(args: &mut impl Iterator<Item = String>, flag: &str) -> String {
    args.next().unwrap_or_else(|| panic!("missing value for {flag}"))
}

fn read_usize(args: &mut impl Iterator<Item = String>, flag: &str) -> usize {
    parse_usize(&read_string(args, flag), flag)
}

fn read_u64(args: &mut impl Iterator<Item = String>, flag: &str) -> u64 {
    parse_u64(&read_string(args, flag), flag)
}

fn read_f64(args: &mut impl Iterator<Item = String>, flag: &str) -> f64 {
    parse_f64(&read_string(args, flag), flag)
}

fn parse_usize(value: &str, flag: &str) -> usize {
    value
        .trim()
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("invalid {flag}"))
}

fn parse_u64(value: &str, flag: &str) -> u64 {
    value
        .trim()
        .parse::<u64>()
        .unwrap_or_else(|_| panic!("invalid {flag}"))
}

fn parse_f64(value: &str, flag: &str) -> f64 {
    value
        .trim()
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("invalid {flag}"))
}

fn fetch_embedding(url: &str, model: &str, prompt: &str) -> Result<Vec<f64>, String> {
    let client = reqwest::blocking::Client::new();
    let body = json!({ "model": model, "prompt": prompt });
    let resp = client.post(url).json(&body).send().map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("http {}", resp.status()));
    }
    let value: serde_json::Value = resp.json().map_err(|e| e.to_string())?;
    if let Some(arr) = value.get("embedding").and_then(|v| v.as_array()) {
        let mut vec = Vec::with_capacity(arr.len());
        for item in arr {
            let v = item.as_f64().ok_or_else(|| "embedding not numeric".to_string())?;
            vec.push(v);
        }
        if vec.is_empty() {
            return Err("empty embedding".to_string());
        }
        return Ok(vec);
    }
    Err("unexpected embedding response".to_string())
}

fn build_vector_grammar(dim: i32, max_ops: i32, axis_step: f64, value_step: f64) -> Node {
    let mut root = Node::new("VECGRAM");
    root.set_int("MaxDepth", max_ops);
    root.set_int("MaxCrossoverNodes", 4);
    root.set_int("MaxMutationNodes", 4);

    let rules = root.get_or_create_child("RULES");

    let mut start = Node::new("START");
    let choices = start.get_or_create_child("CHOICES");
    add_choice(choices, 0, "<SEQ>");
    rules.add_child(start);

    let mut seq = Node::new("SEQ");
    let choices = seq.get_or_create_child("CHOICES");
    add_choice(choices, 0, "<OP> <SEQ>");
    add_choice(choices, 1, "<OP>");
    rules.add_child(seq);

    let mut op = Node::new("OP");
    let choices = op.get_or_create_child("CHOICES");
    add_choice(
        choices,
        0,
        "AX <axis from=0 to={axis_max} inc={axis_step}> <val from=-1 to=1 inc={val_step}>",
    );
    add_choice(choices, 1, "SCALE <val from=0.5 to=1.5 inc=0.1>");
    add_choice(choices, 2, "NORM");
    add_choice(
        choices,
        3,
        "MIX <axis from=0 to={DIM} inc={AXIS_STEP}> <axis from=0 to={DIM} inc={AXIS_STEP}> <val from=0 to=1 inc={VALUE_STEP}>",
    );
    add_choice(
        choices,
        4,
        "ROT <axis from=0 to={DIM} inc={AXIS_STEP}> <axis from=0 to={DIM} inc={AXIS_STEP}> <ang from=-180 to=180 inc=15>",
    );
    add_choice(choices, 5, "FRAC <val from=0.5 to=1.5 inc=0.1>");
    rules.add_child(op);

    root.set_str(
        "Description",
        format!(
            "Vector grammar: dim={}, max_ops={}, axis_step={}, val_step={}",
            dim, max_ops, axis_step, value_step
        )
        .into_bytes(),
    );

    let axis_max = (dim - 1).max(0);
    // Replace template variables in OP choice 0
    if let Some(rules) = root.child_mut("RULES") {
        if let Some(op_node) = rules.child_mut("OP") {
            if let Some(choices) = op_node.child_mut("CHOICES") {
                if let Some(choice) = choices.child_mut("0") {
                    let text = format!(
                        "AX <axis from=0 to={} inc={}> <val from=-1 to=1 inc={}>",
                        axis_max, axis_step, value_step
                    );
                    choice.set_str("Text", text.into_bytes());
                }
            }
        }
    }

    root
}

fn add_choice(choices: &mut Node, index: i32, text: &str) {
    let mut choice = Node::new(index.to_string());
    choice.set_str("Text", text.as_bytes().to_vec());
    choices.add_child(choice);
}

fn finalize_grammar(cfg: &mut Node) {
    if let Some(rules) = cfg.child_mut("RULES") {
        for symbol in rules.children_mut() {
            if let Some(choices) = symbol.child_mut("CHOICES") {
                for choice in choices.children_mut() {
                    parse_text(choice);
                }
            }
        }
    }
    calc_lengths(cfg).expect("calc_lengths failed");
}

fn apply_placeholders(cfg: &mut Node, dim: usize, axis_step: f64, value_step: f64) {
    let dim_max = dim.saturating_sub(1);
    if let Some(rules) = cfg.child_mut("RULES") {
        for symbol in rules.children_mut() {
            if let Some(choices) = symbol.child_mut("CHOICES") {
                for choice in choices.children_mut() {
                    let raw = choice.get_str("Text");
                    if raw.is_empty() {
                        continue;
                    }
                    let text = String::from_utf8_lossy(&raw);
                    if text.contains("{DIM}") || text.contains("{AXIS_STEP}") || text.contains("{VALUE_STEP}") {
                        let replaced = text
                            .replace("{DIM}", &dim_max.to_string())
                            .replace("{AXIS_STEP}", &axis_step.to_string())
                            .replace("{VALUE_STEP}", &value_step.to_string());
                        choice.set_str("Text", replaced.into_bytes());
                    }
                }
            }
        }
    }
}

fn score_individual(ind: &GpIndividual, target: &[f64]) -> f64 {
    if ind.trees().is_empty() {
        return 0.0;
    }
    let text = ind.trees()[0].text();
    let vec = vector_from_text(&text, target.len());
    cosine_similarity(&vec, target)
}

fn vector_from_text(text: &str, dim: usize) -> Vec<f64> {
    let mut out = vec![0.0; dim];
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let mut i = 0usize;
    while i < tokens.len() {
        match tokens[i] {
            "AX" | "ADD" => {
                if i + 2 < tokens.len() {
                    if let (Some(axis), Some(val)) =
                        (parse_number(tokens[i + 1]), parse_number(tokens[i + 2]))
                    {
                        let idx = axis.round() as isize;
                        if idx >= 0 && (idx as usize) < dim {
                            out[idx as usize] += val;
                        }
                    }
                    i += 3;
                } else {
                    i += 1;
                }
            }
            "SCALE" => {
                if i + 1 < tokens.len() {
                    if let Some(scale) = parse_number(tokens[i + 1]) {
                        for v in out.iter_mut() {
                            *v *= scale;
                        }
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "NORM" => {
                normalize(&mut out);
                i += 1;
            }
            "MIX" => {
                if i + 3 < tokens.len() {
                    if let (Some(a), Some(b), Some(w)) = (
                        parse_number(tokens[i + 1]),
                        parse_number(tokens[i + 2]),
                        parse_number(tokens[i + 3]),
                    ) {
                        let ia = a.round() as isize;
                        let ib = b.round() as isize;
                        if ia >= 0 && ib >= 0 && (ia as usize) < dim && (ib as usize) < dim {
                            let w = w.clamp(0.0, 1.0);
                            let a_idx = ia as usize;
                            let b_idx = ib as usize;
                            if a_idx != b_idx {
                                let va = out[a_idx];
                                let vb = out[b_idx];
                                out[a_idx] = va * (1.0 - w) + vb * w;
                                out[b_idx] = vb * (1.0 - w) + va * w;
                            }
                        }
                    }
                    i += 4;
                } else {
                    i += 1;
                }
            }
            "ROT" => {
                if i + 3 < tokens.len() {
                    if let (Some(a), Some(b), Some(ang)) = (
                        parse_number(tokens[i + 1]),
                        parse_number(tokens[i + 2]),
                        parse_number(tokens[i + 3]),
                    ) {
                        let ia = a.round() as isize;
                        let ib = b.round() as isize;
                        if ia >= 0 && ib >= 0 && (ia as usize) < dim && (ib as usize) < dim {
                            let a_idx = ia as usize;
                            let b_idx = ib as usize;
                            if a_idx != b_idx {
                                let rad = ang.to_radians();
                                let (sin, cos) = rad.sin_cos();
                                let x = out[a_idx];
                                let y = out[b_idx];
                                out[a_idx] = x * cos - y * sin;
                                out[b_idx] = x * sin + y * cos;
                            }
                        }
                    }
                    i += 4;
                } else {
                    i += 1;
                }
            }
            "FRAC" => {
                if i + 1 < tokens.len() {
                    if let Some(exp) = parse_number(tokens[i + 1]) {
                        if exp.is_finite() {
                            for v in out.iter_mut() {
                                let sign = if *v >= 0.0 { 1.0 } else { -1.0 };
                                let mag = v.abs().powf(exp);
                                *v = sign * mag;
                            }
                        }
                    }
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "ZERO" => {
                for v in out.iter_mut() {
                    *v = 0.0;
                }
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    out
}

fn parse_number(token: &str) -> Option<f64> {
    let mut end = 0usize;
    for (idx, ch) in token.char_indices() {
        if ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == '+' || ch == 'e' || ch == 'E' {
            end = idx + ch.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    token[..end].parse::<f64>().ok()
}

fn normalize(vec: &mut [f64]) {
    let n = norm(vec);
    if n > 1e-12 {
        for v in vec.iter_mut() {
            *v /= n;
        }
    }
}

fn norm(vec: &[f64]) -> f64 {
    vec.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
    dot / denom
}

struct Projection {
    rows: Vec<Vec<f64>>,
}

impl Projection {
    fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut rows = Vec::with_capacity(output_dim);
        for _ in 0..output_dim {
            let mut row = Vec::with_capacity(input_dim);
            for _ in 0..input_dim {
                row.push(rand_normal(&mut rng));
            }
            rows.push(row);
        }
        Projection { rows }
    }

    fn project(&self, vec: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            let mut acc = 0.0;
            for i in 0..row.len() {
                acc += row[i] * vec[i];
            }
            out.push(acc);
        }
        out
    }
}

fn rand_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn save_best_record(
    path: &str,
    generation: usize,
    fitness: f64,
    vector: &[f64],
    program: &str,
    target_text: &str,
    model: &str,
    seed: u64,
) -> Result<(), String> {
    let record = json!({
        "generation": generation,
        "fitness": fitness,
        "dim": vector.len(),
        "vector": vector,
        "program": program,
        "target_text": target_text,
        "model": model,
        "seed": seed,
    });
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    let line = serde_json::to_string(&record).map_err(|e| e.to_string())?;
    file.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
    file.write_all(b"\n").map_err(|e| e.to_string())?;
    Ok(())
}

fn write_plot_2d(path: &str, points: &[(usize, f64, f64, f64)]) -> Result<(), String> {
    if points.is_empty() {
        return Ok(());
    }
    if path.ends_with(".svg") {
        let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
        for (_, _, x, y) in points {
            min_x = min_x.min(*x);
            max_x = max_x.max(*x);
            min_y = min_y.min(*y);
            max_y = max_y.max(*y);
        }
        let width = 800.0;
        let height = 600.0;
        let pad = 40.0;
        let scale_x = if (max_x - min_x).abs() < 1e-12 {
            1.0
        } else {
            (width - 2.0 * pad) / (max_x - min_x)
        };
        let scale_y = if (max_y - min_y).abs() < 1e-12 {
            1.0
        } else {
            (height - 2.0 * pad) / (max_y - min_y)
        };
        let mut svg = String::new();
        svg.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\">\n",
            width as i32, height as i32
        ));
        svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n");
        svg.push_str("<path d=\"");
        for (i, (_, _, x, y)) in points.iter().enumerate() {
            let px = pad + (x - min_x) * scale_x;
            let py = height - (pad + (y - min_y) * scale_y);
            if i == 0 {
                svg.push_str(&format!("M {:.2} {:.2} ", px, py));
            } else {
                svg.push_str(&format!("L {:.2} {:.2} ", px, py));
            }
        }
        svg.push_str("\" fill=\"none\" stroke=\"#111\" stroke-width=\"2\"/>\n");
        svg.push_str("</svg>\n");
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        file.write_all(svg.as_bytes()).map_err(|e| e.to_string())?;
        Ok(())
    } else {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        file.write_all(b"generation,fitness,x,y\n").map_err(|e| e.to_string())?;
        for (gen, fit, x, y) in points {
            let line = format!("{},{},{},{}\n", gen, fit, x, y);
            file.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

fn write_plot_3d(path: &str, points: &[(usize, f64, f64, f64, f64)]) -> Result<(), String> {
    if points.is_empty() {
        return Ok(());
    }
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    file.write_all(b"generation,fitness,x,y,z\n").map_err(|e| e.to_string())?;
    for (gen, fit, x, y, z) in points {
        let line = format!("{},{},{},{},{}\n", gen, fit, x, y, z);
        file.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn numbered_svg_path(path: &str, generation: usize) -> String {
    if let Some(stripped) = path.strip_suffix(".svg") {
        format!("{}.{}.svg", stripped, format!("{:05}", generation))
    } else {
        format!("{}.{}", path, format!("{:05}", generation))
    }
}
