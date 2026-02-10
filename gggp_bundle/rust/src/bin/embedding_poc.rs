use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde_json::json;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;

#[derive(Clone)]
struct Individual {
    vec: Vec<f64>,
    fitness: f64,
}

struct Config {
    model: String,
    url: String,
    seed: u64,
    gens: usize,
    pop: usize,
    elite: usize,
    mutation_rate: f64,
    mutation_sigma: f64,
    target_texts: Vec<String>,
    save_best: Option<String>,
}

fn main() {
    let config = parse_args();

    if config.target_texts.is_empty() {
        eprintln!("no target text provided; use --target 'text'");
        std::process::exit(2);
    }

    for (idx, target) in config.target_texts.iter().enumerate() {
        println!("target: {}", target);
        let target_vec = match fetch_embedding(&config.url, &config.model, target) {
            Ok(v) => v,
            Err(err) => {
                eprintln!("embedding error: {err}");
                std::process::exit(1);
            }
        };
        run_evolution(&config, idx, target, &target_vec);
        println!();
    }
}

fn parse_args() -> Config {
    let mut cfg = Config {
        model: "all-minilm:33m".to_string(),
        url: "http://localhost:11434/api/embeddings".to_string(),
        seed: 42,
        gens: 80,
        pop: 80,
        elite: 6,
        mutation_rate: 0.15,
        mutation_sigma: 0.08,
        target_texts: Vec::new(),
        save_best: None,
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
            "--mutation" => cfg.mutation_rate = read_f64(&mut args, "--mutation"),
            "--sigma" => cfg.mutation_sigma = read_f64(&mut args, "--sigma"),
            "--target" => cfg.target_texts.push(read_string(&mut args, "--target")),
            "--save-best" => cfg.save_best = Some(read_string(&mut args, "--save-best")),
            _ if arg.starts_with("--model=") => cfg.model = arg[8..].to_string(),
            _ if arg.starts_with("--url=") => cfg.url = arg[6..].to_string(),
            _ if arg.starts_with("--seed=") => cfg.seed = parse_u64(&arg[7..], "--seed"),
            _ if arg.starts_with("--gens=") => cfg.gens = parse_usize(&arg[7..], "--gens"),
            _ if arg.starts_with("--pop=") => cfg.pop = parse_usize(&arg[6..], "--pop"),
            _ if arg.starts_with("--elite=") => cfg.elite = parse_usize(&arg[8..], "--elite"),
            _ if arg.starts_with("--mutation=") => cfg.mutation_rate = parse_f64(&arg[11..], "--mutation"),
            _ if arg.starts_with("--sigma=") => cfg.mutation_sigma = parse_f64(&arg[8..], "--sigma"),
            _ if arg.starts_with("--target=") => cfg.target_texts.push(arg[9..].to_string()),
            _ if arg.starts_with("--save-best=") => cfg.save_best = Some(arg[12..].to_string()),
            _ => {
                eprintln!("unknown arg: {arg}");
                std::process::exit(2);
            }
        }
    }

    cfg
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
    let body = json!({"model": model, "prompt": prompt});
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

fn run_evolution(cfg: &Config, target_index: usize, target_text: &str, target: &[f64]) {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let target_norm = norm(target);
    let dim = target.len();

    let mut population = Vec::with_capacity(cfg.pop);
    for _ in 0..cfg.pop {
        let mut vec = Vec::with_capacity(dim);
        for _ in 0..dim {
            vec.push(rand_normal(&mut rng));
        }
        if target_norm > 0.0 {
            scale_to_norm(&mut vec, target_norm);
        }
        let fitness = cosine_similarity(&vec, target);
        population.push(Individual { vec, fitness });
    }

    let mut best_so_far = f64::NEG_INFINITY;
    for gen in 0..cfg.gens {
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        let best = population[0].fitness;
        let avg = population.iter().map(|i| i.fitness).sum::<f64>() / population.len() as f64;
        if gen % 10 == 0 || gen + 1 == cfg.gens {
            println!("gen {:03} best {:.5} avg {:.5}", gen + 1, best, avg);
        }
        if best > best_so_far + 1e-12 {
            best_so_far = best;
            if let Some(path) = cfg.save_best.as_deref() {
                if let Err(err) = save_best_record(
                    path,
                    target_index,
                    target_text,
                    &cfg.model,
                    cfg.seed,
                    gen + 1,
                    best,
                    &population[0].vec,
                ) {
                    eprintln!("warning: failed to save best record: {err}");
                }
            }
        }

        let mut next = Vec::with_capacity(cfg.pop);
        for i in 0..cfg.elite.min(population.len()) {
            next.push(population[i].clone());
        }

        while next.len() < cfg.pop {
            let a = tournament_select(&population, &mut rng, 3);
            let b = tournament_select(&population, &mut rng, 3);
            let mut child = crossover(&a.vec, &b.vec, &mut rng);
            mutate(&mut child, &mut rng, cfg.mutation_rate, cfg.mutation_sigma);
            if target_norm > 0.0 {
                scale_to_norm(&mut child, target_norm);
            }
            let fitness = cosine_similarity(&child, target);
            next.push(Individual { vec: child, fitness });
        }

        population = next;
    }

    population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    println!("best fitness: {:.6}", population[0].fitness);
}

fn save_best_record(
    path: &str,
    target_index: usize,
    target_text: &str,
    model: &str,
    seed: u64,
    generation: usize,
    fitness: f64,
    vector: &[f64],
) -> Result<(), String> {
    let record = json!({
        "target_index": target_index,
        "target_text": target_text,
        "model": model,
        "seed": seed,
        "generation": generation,
        "fitness": fitness,
        "dim": vector.len(),
        "vector": vector,
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

fn tournament_select<'a>(pop: &'a [Individual], rng: &mut impl Rng, k: usize) -> &'a Individual {
    let mut best: Option<&'a Individual> = None;
    for _ in 0..k {
        let idx = rng.gen_range(0..pop.len());
        let cand = &pop[idx];
        match best {
            Some(b) => {
                if cand.fitness > b.fitness {
                    best = Some(cand);
                }
            }
            None => best = Some(cand),
        }
    }
    best.unwrap()
}

fn crossover(a: &[f64], b: &[f64], rng: &mut impl Rng) -> Vec<f64> {
    let mut out = Vec::with_capacity(a.len());
    let alpha: f64 = rng.gen_range(0.0..1.0);
    for i in 0..a.len() {
        out.push(alpha * a[i] + (1.0 - alpha) * b[i]);
    }
    out
}

fn mutate(vec: &mut [f64], rng: &mut impl Rng, rate: f64, sigma: f64) {
    for v in vec.iter_mut() {
        if rng.gen::<f64>() < rate {
            *v += rand_normal(rng) * sigma;
        }
    }
}

fn rand_normal(rng: &mut impl Rng) -> f64 {
    // Box-Muller transform
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn norm(vec: &[f64]) -> f64 {
    vec.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn scale_to_norm(vec: &mut [f64], target_norm: f64) {
    let n = norm(vec);
    if n > 0.0 {
        let scale = target_norm / n;
        for v in vec.iter_mut() {
            *v *= scale;
        }
    }
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
