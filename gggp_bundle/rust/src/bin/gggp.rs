use std::env;
use std::fs::File;
use std::io::{self, Write};

use gor_data_storage::gggp::{
    crossover_individuals,
    gp_data_from_config,
    grammar_config_replace_group_tags,
    grammar_config_replace_group_tags_with_options,
    mutate_individual,
    text_replace_groups_tags,
    GggpError,
    GpConfig,
    GpIndividual,
};
use gor_data_storage::Node;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommandKind {
    Text,
    Crossover,
    Mutate,
}

struct Config {
    command: CommandKind,
    input: String,
    chromosome: Option<String>,
    chromosome_b: Option<String>,
    grammar_path: Option<String>,
    global_cfg: Option<String>,
    output: Option<String>,
    seed: Option<u64>,
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print!("{}", usage());
        return;
    }

    let config = match parse_args(args.into_iter()) {
        Ok(config) => config,
        Err(msg) => {
            eprintln!("{msg}");
            eprint!("{}", usage());
            std::process::exit(2);
        }
    };

    let root = match load_node(&config.input) {
        Ok(node) => node,
        Err(err) => {
            eprintln!("error reading {}: {err}", config.input);
            std::process::exit(1);
        }
    };

    let global_cfg = match config.global_cfg.as_deref() {
        Some(path) => match load_node(path) {
            Ok(node) => Some(node),
            Err(err) => {
                eprintln!("error reading {}: {err}", path);
                std::process::exit(1);
            }
        },
        None => None,
    };

    let global_cfg_ref = global_cfg.as_ref();

    let (grammar_node, grammar_path) =
        match select_grammar_node(&root, config.grammar_path.as_deref()) {
            Ok(result) => result,
            Err(err) => {
                eprintln!("{err}");
                std::process::exit(2);
            }
        };
    let method_name = if config.command == CommandKind::Text {
        derive_method_name(&grammar_path, &root)
    } else {
        None
    };

    let mut grammar_cfg = grammar_node.clone();
    let mut include_inactive = false;
    if let Err(err) =
        grammar_config_replace_group_tags(&root, &mut grammar_cfg, global_cfg_ref, false)
    {
        eprintln!(
            "warning: group tag resolution failed ({}); retrying with inactive series",
            format_gggp_error(err)
        );
        grammar_cfg = grammar_node.clone();
        include_inactive = true;
        if let Err(err) = grammar_config_replace_group_tags_with_options(
            &root,
            &mut grammar_cfg,
            global_cfg_ref,
            false,
            true,
        ) {
            eprintln!(
                "warning: group tag resolution still failed ({}); using cleanup mode",
                format_gggp_error(err)
            );
            grammar_cfg = grammar_node.clone();
            if let Err(err) =
                grammar_config_replace_group_tags(&root, &mut grammar_cfg, global_cfg_ref, true)
            {
                eprintln!("error preparing grammar: {}", format_gggp_error(err));
                std::process::exit(1);
            }
        }
    }

    let gp_cfg = match GpConfig::from_node(&grammar_cfg) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("error loading grammar: {}", format_gggp_error(err));
            std::process::exit(1);
        }
    };

    let output = match config.command {
        CommandKind::Text => {
            let chromosome = match resolve_chromosome(grammar_node, config.chromosome.as_deref()) {
                Ok(chromosome) => chromosome,
                Err(err) => exit_with_error(err, 2),
            };

            let tree = match gp_cfg.tree_from_chromosome(&chromosome) {
                Ok(tree) => tree,
                Err(err) => exit_with_error(
                    format!("error building tree: {}", format_gggp_error(err)),
                    1,
                ),
            };

            let mut text = tree.indented_text(0);
            if text.contains("%group") || text.contains("%ind") {
                match gp_data_from_config(&root, global_cfg_ref, include_inactive) {
                    Ok(mut data) => match text_replace_groups_tags(&text, &mut data) {
                        Ok(replaced) => text = replaced,
                        Err(err) => eprintln!(
                            "warning: failed to replace group tags: {}",
                            format_gggp_error(err)
                        ),
                    },
                    Err(err) => eprintln!(
                        "warning: failed to build inputs list: {}",
                        format_gggp_error(err)
                    ),
                }
            }

            if let Some(name) = method_name.as_deref() {
                text = replace_name_placeholder(&text, name);
            }
            text = replace_id_placeholders(&text);
            text = replace_escaped_operators(&text);
            text = normalize_crlf(&text);
            if !text.ends_with("\r\n") {
                text.push_str("\r\n");
            }
            text
        }
        CommandKind::Crossover => {
            let chromosome_a = match config.chromosome.as_deref() {
                Some(value) => match normalize_chromosome_arg(value, "chromosome") {
                    Ok(value) => value,
                    Err(err) => exit_with_error(err, 2),
                },
                None => exit_with_error("crossover requires -c/--chromosome".to_string(), 2),
            };
            let chromosome_b = match config.chromosome_b.as_deref() {
                Some(value) => match normalize_chromosome_arg(value, "chromosome2") {
                    Ok(value) => value,
                    Err(err) => exit_with_error(err, 2),
                },
                None => exit_with_error("crossover requires -d/--chromosome2".to_string(), 2),
            };

            let mut ind1 = GpIndividual::new();
            let tree1 = match gp_cfg.tree_from_chromosome(&chromosome_a) {
                Ok(tree) => tree,
                Err(err) => exit_with_error(
                    format!("error building tree: {}", format_gggp_error(err)),
                    1,
                ),
            };
            ind1.trees_mut().push(tree1);

            let mut ind2 = GpIndividual::new();
            let tree2 = match gp_cfg.tree_from_chromosome(&chromosome_b) {
                Ok(tree) => tree,
                Err(err) => exit_with_error(
                    format!("error building tree: {}", format_gggp_error(err)),
                    1,
                ),
            };
            ind2.trees_mut().push(tree2);

            let mut rng = make_rng(config.seed);
            let changed = crossover_individuals(&mut rng, &mut ind1, &mut ind2);
            if !changed {
                eprintln!("warning: crossover produced no changes");
            }

            let mut out = String::new();
            let child1 = ind1.trees()[0].chromosome();
            let child2 = ind2.trees()[0].chromosome();
            out.push_str(&child1);
            out.push_str("\r\n");
            out.push_str(&child2);
            out.push_str("\r\n");
            out
        }
        CommandKind::Mutate => {
            let chromosome = match resolve_chromosome(grammar_node, config.chromosome.as_deref()) {
                Ok(chromosome) => chromosome,
                Err(err) => exit_with_error(err, 2),
            };

            let mut ind = GpIndividual::new();
            let tree = match gp_cfg.tree_from_chromosome(&chromosome) {
                Ok(tree) => tree,
                Err(err) => exit_with_error(
                    format!("error building tree: {}", format_gggp_error(err)),
                    1,
                ),
            };
            ind.trees_mut().push(tree);

            let mut rng = make_rng(config.seed);
            let changed = mutate_individual(&mut rng, &mut ind);
            if !changed {
                eprintln!("warning: mutation produced no changes");
            }

            let mut out = ind.trees()[0].chromosome();
            out.push_str("\r\n");
            out
        }
    };

    if let Err(err) = write_output(config.output.as_deref(), output.as_bytes()) {
        eprintln!("error writing output: {err}");
        std::process::exit(1);
    }
}

fn usage() -> &'static str {
    "Usage:\n\
  gggp [text] [-c <chromosome>] [-g <grammar_path>] [--global-cfg <path>] [-o OUTPUT] <input.cfg|->\n\
  gggp crossover -c <chromosome> -d <chromosome> [-g <grammar_path>] [--global-cfg <path>] [-o OUTPUT] <input.cfg|->\n\
  gggp mutate [-c <chromosome>] [-g <grammar_path>] [--global-cfg <path>] [-o OUTPUT] <input.cfg|->\n\
\n\
Options:\n\
  -c, --chromosome   Chromosome string (e.g. \"0-1-2\"). When omitted in text/mutate,\n\
                     uses @Chromosome from the grammar node if @OPTIMIZE is False.\n\
  -d, --chromosome2  Second chromosome for crossover.\n\
  -g, --grammar      Path to grammar node containing RULES (e.g. \"TREND/Grammar/Grammar\")\n\
  --global-cfg       Path to global cfg (PluginHost) for group/index resolution\n\
  -o, --output       Output file (default: stdout)\n\
  -s, --seed         RNG seed (u64) for crossover/mutate reproducibility\n\
  -h, --help         Show this help message\n\
  crossover outputs two chromosomes (child1 then child2), one per line.\n\
"
}

fn parse_args<I>(args: I) -> Result<Config, String>
where
    I: Iterator<Item = String>,
{
    let mut args = args.peekable();
    let mut command = CommandKind::Text;
    if let Some(arg) = args.peek() {
        if !arg.starts_with('-') {
            match arg.as_str() {
                "text" => {
                    command = CommandKind::Text;
                    args.next();
                }
                "crossover" => {
                    command = CommandKind::Crossover;
                    args.next();
                }
                "mutate" => {
                    command = CommandKind::Mutate;
                    args.next();
                }
                _ => {}
            }
        }
    }

    let mut input = None;
    let mut chromosome = None;
    let mut chromosome_b = None;
    let mut grammar_path = None;
    let mut global_cfg = None;
    let mut output = None;
    let mut seed = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-c" | "--chromosome" => {
                let value = args.next().ok_or("missing value for -c")?;
                chromosome = Some(value);
            }
            "-d" | "--chromosome2" | "--chromosome-b" => {
                let value = args.next().ok_or("missing value for -d")?;
                chromosome_b = Some(value);
            }
            "-g" | "--grammar" | "--grammar-path" => {
                let value = args.next().ok_or("missing value for -g")?;
                grammar_path = Some(value);
            }
            "-o" | "--output" => {
                let value = args.next().ok_or("missing value for -o")?;
                output = Some(value);
            }
            "-s" | "--seed" => {
                let value = args.next().ok_or("missing value for -s")?;
                seed = Some(parse_seed(&value)?);
            }
            "--global-cfg" => {
                let value = args.next().ok_or("missing value for --global-cfg")?;
                global_cfg = Some(value);
            }
            _ if arg.starts_with("-c=") => {
                chromosome = Some(arg[3..].to_string());
            }
            _ if arg.starts_with("--chromosome=") => {
                chromosome = Some(arg["--chromosome=".len()..].to_string());
            }
            _ if arg.starts_with("-d=") => {
                chromosome_b = Some(arg[3..].to_string());
            }
            _ if arg.starts_with("--chromosome2=") => {
                chromosome_b = Some(arg["--chromosome2=".len()..].to_string());
            }
            _ if arg.starts_with("--chromosome-b=") => {
                chromosome_b = Some(arg["--chromosome-b=".len()..].to_string());
            }
            _ if arg.starts_with("-g=") => {
                grammar_path = Some(arg[3..].to_string());
            }
            _ if arg.starts_with("--grammar=") => {
                grammar_path = Some(arg["--grammar=".len()..].to_string());
            }
            _ if arg.starts_with("--grammar-path=") => {
                grammar_path = Some(arg["--grammar-path=".len()..].to_string());
            }
            _ if arg.starts_with("-o=") => {
                output = Some(arg[3..].to_string());
            }
            _ if arg.starts_with("--output=") => {
                output = Some(arg["--output=".len()..].to_string());
            }
            _ if arg.starts_with("-s=") => {
                seed = Some(parse_seed(&arg[3..])?);
            }
            _ if arg.starts_with("--seed=") => {
                seed = Some(parse_seed(&arg["--seed=".len()..])?);
            }
            _ if arg.starts_with("--global-cfg=") => {
                global_cfg = Some(arg["--global-cfg=".len()..].to_string());
            }
            _ if arg.starts_with('-') => {
                return Err(format!("unknown option: {arg}"));
            }
            _ => {
                if input.is_some() {
                    return Err("multiple input files provided".to_string());
                }
                input = Some(arg);
            }
        }
    }

    let input = input.ok_or("missing input file".to_string())?;
    match command {
        CommandKind::Text | CommandKind::Mutate => {
            if chromosome_b.is_some() {
                return Err("option -d/--chromosome2 only valid for crossover".to_string());
            }
        }
        CommandKind::Crossover => {
            if chromosome_b.is_none() {
                return Err("crossover requires -d/--chromosome2".to_string());
            }
            if chromosome.is_none() {
                return Err("crossover requires -c/--chromosome".to_string());
            }
        }
    }

    Ok(Config {
        command,
        input,
        chromosome,
        chromosome_b,
        grammar_path,
        global_cfg,
        output,
        seed,
    })
}

fn load_node(input: &str) -> Result<Node, String> {
    if input == "-" {
        let mut stdin = io::stdin().lock();
        Node::from_reader(&mut stdin).map_err(|err| err.to_string())
    } else {
        Node::from_file(input).map_err(|err| err.to_string())
    }
}

fn write_output(output: Option<&str>, data: &[u8]) -> io::Result<()> {
    match output {
        Some(path) => {
            let mut file = File::create(path)?;
            file.write_all(data)
        }
        None => {
            let mut stdout = io::stdout().lock();
            stdout.write_all(data)
        }
    }
}

fn exit_with_error(msg: String, code: i32) -> ! {
    eprintln!("{msg}");
    std::process::exit(code);
}

fn select_grammar_node<'a>(
    root: &'a Node,
    path: Option<&str>,
) -> Result<(&'a Node, String), String> {
    if let Some(path) = path {
        let parts = normalize_path_parts(root, path);
        let node = find_node_by_path(root, &parts)
            .ok_or_else(|| format!("grammar path not found: {path}"))?;
        if node.child("RULES").is_none() {
            return Err(format!(
                "node at '{path}' does not contain RULES"
            ));
        }
        return Ok((node, path.to_string()));
    }

    let mut paths = Vec::new();
    collect_grammar_paths(root, &mut Vec::new(), &mut paths);

    match paths.len() {
        0 => Err("no grammar nodes with RULES found in config".to_string()),
        1 => {
            let path = paths[0].clone();
            let parts = normalize_path_parts(root, &path);
            find_node_by_path(root, &parts)
                .map(|node| (node, path))
                .ok_or_else(|| "grammar path not found".to_string())
        }
        _ => {
            let mut msg = String::from("multiple grammar nodes found; use -g to choose:\n");
            for path in paths {
                msg.push_str("  ");
                msg.push_str(&path);
                msg.push('\n');
            }
            Err(msg)
        }
    }
}

fn resolve_chromosome(grammar: &Node, cli: Option<&str>) -> Result<String, String> {
    if let Some(cli) = cli {
        let trimmed = cli.trim();
        if trimmed.is_empty() {
            return Err("chromosome is empty".to_string());
        }
        return Ok(trimmed.to_string());
    }

    let optimize_present = grammar.attr_exists("OPTIMIZE");
    let optimize = grammar.get_bool("OPTIMIZE");
    if optimize_present && optimize {
        return Err("OPTIMIZE=true: provide chromosome with -c".to_string());
    }

    let chromosome = String::from_utf8_lossy(&grammar.get_str("Chromosome"))
        .trim()
        .to_string();
    if !chromosome.is_empty() {
        return Ok(chromosome);
    }

    if optimize_present && !optimize {
        return Err("OPTIMIZE=false but @Chromosome is empty; provide -c".to_string());
    }

    Err("chromosome not provided and @Chromosome not found".to_string())
}

fn normalize_chromosome_arg(value: &str, label: &str) -> Result<String, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(format!("{label} is empty"));
    }
    Ok(trimmed.to_string())
}

fn parse_seed(value: &str) -> Result<u64, String> {
    value
        .trim()
        .parse::<u64>()
        .map_err(|_| format!("invalid seed: {value}"))
}

fn make_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    }
}

fn normalize_path_parts<'a>(root: &Node, path: &'a str) -> Vec<&'a str> {
    let mut parts: Vec<&str> = path
        .split(|c| c == '/' || c == '\\')
        .filter(|s| !s.is_empty())
        .collect();
    if let Some(first) = parts.first() {
        if name_eq(first, root.name()) {
            parts.remove(0);
        }
    }
    parts
}

fn find_node_by_path<'a>(root: &'a Node, parts: &[&str]) -> Option<&'a Node> {
    let mut current = root;
    for part in parts {
        current = current.child(part.as_bytes())?;
    }
    Some(current)
}

fn collect_grammar_paths(node: &Node, path: &mut Vec<String>, out: &mut Vec<String>) {
    if node.child("RULES").is_some() && node.attr_exists("MaxDepth") {
        if path.is_empty() {
            out.push(node_name(node));
        } else {
            out.push(path.join("/"));
        }
    }
    for child in node.children() {
        path.push(node_name(child));
        collect_grammar_paths(child, path, out);
        path.pop();
    }
}

fn node_name(node: &Node) -> String {
    String::from_utf8_lossy(node.name()).into_owned()
}

fn name_eq(part: &str, name: &[u8]) -> bool {
    let name_str = String::from_utf8_lossy(name);
    part.eq_ignore_ascii_case(&name_str)
}

fn derive_method_name(path: &str, root: &Node) -> Option<String> {
    let mut parts: Vec<String> = path
        .split(|c| c == '/' || c == '\\')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    if let Some(first) = parts.first() {
        if name_eq(first, root.name()) {
            parts.remove(0);
        }
    }
    parts.retain(|part| !part.eq_ignore_ascii_case("Grammar"));
    if let Some(first) = parts.first() {
        if first.eq_ignore_ascii_case("STRATEGIES") {
            parts.remove(0);
        }
    }
    if parts.is_empty() {
        return None;
    }
    Some(parts.join("_").to_uppercase())
}

fn replace_name_placeholder(text: &str, name: &str) -> String {
    text.replace("%Name%", name)
}

fn replace_id_placeholders(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut last = 0usize;
    let mut next_id = 0usize;
    let mut i = 0usize;
    while i + 4 <= bytes.len() {
        if bytes[i] == b'%'
            && bytes[i + 1].to_ascii_lowercase() == b'i'
            && bytes[i + 2].to_ascii_lowercase() == b'd'
            && bytes[i + 3] == b'%'
        {
            out.push_str(&text[last..i]);
            out.push_str(&next_id.to_string());
            next_id += 1;
            i += 4;
            last = i;
        } else {
            i += 1;
        }
    }
    out.push_str(&text[last..]);
    out
}

fn replace_escaped_operators(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'<' && i + 1 < bytes.len() && bytes[i + 1] == b'<' {
            if i + 3 < bytes.len() && bytes[i + 2] == b'>' && bytes[i + 3] == b'>' {
                out.push_str("<>");
                i += 4;
                continue;
            }
            if i + 2 < bytes.len() && bytes[i + 2] == b'=' {
                out.push_str("<=");
                i += 3;
                continue;
            }
            out.push('<');
            i += 2;
            continue;
        }
        if bytes[i] == b'>' && i + 1 < bytes.len() && bytes[i + 1] == b'>' {
            if i + 2 < bytes.len() && bytes[i + 2] == b'=' {
                out.push_str(">=");
                i += 3;
                continue;
            }
            out.push('>');
            i += 2;
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

fn normalize_crlf(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\r' {
            if let Some('\n') = chars.peek().copied() {
                chars.next();
            }
            out.push_str("\r\n");
        } else if ch == '\n' {
            out.push_str("\r\n");
        } else {
            out.push(ch);
        }
    }
    out
}

fn format_gggp_error(err: GggpError) -> String {
    match err {
        GggpError::InvalidConfig(msg) => format!("invalid config: {msg}"),
        GggpError::InvalidData(msg) => format!("invalid data: {msg}"),
        GggpError::Io(err) => format!("io error: {err}"),
        GggpError::MissingStart => "missing START symbol".to_string(),
        GggpError::MissingRules => "missing RULES".to_string(),
        GggpError::MissingSymbol(name) => format!("missing symbol: {name}"),
        GggpError::InvalidChoice(msg) => format!("invalid choice: {msg}"),
    }
}
