use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::PathBuf;

struct Config {
    memory_endpoint: String,
    soul_id: String,
    ollama_url: String,
    model: String,
    limit: usize,
    summary_path: Option<PathBuf>,
    store_identity: bool,
}

fn parse_args() -> Config {
    let mut cfg = Config {
        memory_endpoint: "http://localhost:8087".to_string(),
        soul_id: "eve".to_string(),
        ollama_url: "http://localhost:11434/api/generate".to_string(),
        model: "llama3:8b".to_string(),
        limit: 10,
        summary_path: None,
        store_identity: false,
    };

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--memory-endpoint" => cfg.memory_endpoint = args.next().unwrap_or_default(),
            "--soul-id" => cfg.soul_id = args.next().unwrap_or_default(),
            "--ollama-url" => cfg.ollama_url = args.next().unwrap_or_default(),
            "--model" => cfg.model = args.next().unwrap_or_default(),
            "--limit" => cfg.limit = args.next().unwrap_or("10".to_string()).parse().unwrap_or(10),
            "--summary-path" => cfg.summary_path = args.next().map(PathBuf::from),
            "--store-identity" => cfg.store_identity = true,
            _ => {}
        }
    }

    cfg
}

fn load_previous_summary(path: &Option<PathBuf>) -> String {
    if let Some(p) = path {
        if let Ok(s) = fs::read_to_string(p) {
            return s;
        }
    }
    "".to_string()
}

fn fetch_recent_memories(client: &Client, endpoint: &str, soul_id: &str, limit: usize) -> Result<Vec<Value>, String> {
    let url = format!("{}/memories/recent", endpoint.trim_end_matches('/'));
    let payload = json!({"soul_id": soul_id, "limit": limit});
    let resp = client
        .post(url)
        .json(&payload)
        .send()
        .map_err(|e| format!("memory request failed: {e}"))?;
    let status = resp.status();
    let data: Value = resp.json().map_err(|e| format!("memory response parse failed: {e}"))?;
    if !status.is_success() {
        return Err(format!("memory error: {status} {data}"));
    }
    Ok(data.get("results").and_then(|v| v.as_array()).cloned().unwrap_or_default())
}

fn build_prompt(prev_summary: &str, memories: &[Value]) -> String {
    let mut lines: Vec<String> = Vec::new();
    if !prev_summary.trim().is_empty() {
        lines.push(format!("Previous Identity Summary: {}", prev_summary.trim()));
    } else {
        lines.push("Previous Identity Summary: (none)".to_string());
    }
    lines.push("Recent Experiences:".to_string());
    for (i, m) in memories.iter().enumerate() {
        let text = m.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let created = m.get("created_at").and_then(|v| v.as_f64()).unwrap_or(0.0);
        lines.push(format!("{}. {} (t={})", i + 1, text, created));
    }
    lines.push("".to_string());
    lines.push("Task: Based on previous state X and new inputs Y, generate a concise self-model update in this exact template:".to_string());
    lines.push("\"Based on previous state X and new inputs Y, my current self-model is now Z.\"".to_string());
    lines.push("Return a single paragraph. Do not add bullet points.".to_string());
    lines.join("\n")
}

fn call_ollama(client: &Client, url: &str, model: &str, prompt: &str) -> Result<String, String> {
    let payload = json!({"model": model, "prompt": prompt, "stream": false});
    let resp = client
        .post(url)
        .json(&payload)
        .send()
        .map_err(|e| format!("ollama request failed: {e}"))?;
    let status = resp.status();
    let data: Value = resp.json().map_err(|e| format!("ollama response parse failed: {e}"))?;
    if !status.is_success() {
        return Err(format!("ollama error: {status} {data}"));
    }
    Ok(data
        .get("response")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string())
}

fn store_identity_summary(client: &Client, endpoint: &str, soul_id: &str, summary: &str) -> Result<(), String> {
    let url = format!("{}/memories/ingest", endpoint.trim_end_matches('/'));
    let payload = json!({"soul_id": soul_id, "text": summary, "tags": ["identity_summary"], "meta": {"type": "identity_summary"}});
    let resp = client
        .post(url)
        .json(&payload)
        .send()
        .map_err(|e| format!("identity ingest failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("identity ingest error: {}", resp.status()));
    }
    Ok(())
}

fn main() {
    let cfg = parse_args();
    let client = Client::new();
    let prev_summary = load_previous_summary(&cfg.summary_path);

    let memories = match fetch_recent_memories(&client, &cfg.memory_endpoint, &cfg.soul_id, cfg.limit) {
        Ok(m) => m,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    let prompt = build_prompt(&prev_summary, &memories);
    let summary = match call_ollama(&client, &cfg.ollama_url, &cfg.model, &prompt) {
        Ok(s) => s,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    if let Some(path) = cfg.summary_path.as_ref() {
        if let Err(err) = fs::write(path, &summary) {
            eprintln!("failed to write summary: {err}");
        }
    }

    if cfg.store_identity {
        if let Err(err) = store_identity_summary(&client, &cfg.memory_endpoint, &cfg.soul_id, &summary) {
            eprintln!("{err}");
        }
    }

    println!("{summary}");
}
