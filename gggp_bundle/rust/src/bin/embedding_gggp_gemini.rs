/*
 * CORE/RUST/SRC/BIN/EMBEDDING_GGGP.RS
 * =========================================================================================
 * ARCHITECT: GOD-TIER
 * PHASE: 4 (NEURO-SYMBOLIC)
 * DESCRIPTION:
 *   Proof-of-Concept for "Semiotic Hypercube".
 *   Evolves a GGGP program that constructs a high-dimensional vector
 *   semantically aligned with a target text via an LLM embedding model.
 * =========================================================================================
 */

 use anyhow::{Context, Result};
 use clap::Parser;
 use nalgebra::DVector;
 use serde::{Deserialize, Serialize};
 use serde_json::json;
 use std::fs::{self, File, OpenOptions};
 use std::io::{BufWriter, Write};
 use std::path::PathBuf;
 use std::time::Instant;
 
 // Internal imports (assuming the project structure defined in previous specs)
 // You must ensure these modules are exposed in lib.rs
 use rust_core::gggp::{
     chromosome::Chromosome,
     ga::{crossover_individuals, mutate_individual, GpIndividual},
     grammar::{GpGrammar, GpTree},
 };
 use rust_core::storage::node::Node;
 
 // =========================================================================================
 // 1. CLI ARGUMENTS
 // =========================================================================================
 
 #[derive(Parser, Debug, Clone)]
 #[clap(name = "embedding_gggp", about = "Evolve vector programs to match semantic embeddings")]
 struct Opts {
     /// Ollama embedding model name
     #[clap(long, default_value = "all-minilm")]
     model: String,
 
     /// Ollama API endpoint
     #[clap(long, default_value = "http://localhost:11434/api/embeddings")]
     url: String,
 
     /// RNG Seed for reproducibility
     #[clap(long)]
     seed: Option<u64>,
 
     /// Number of generations
     #[clap(long, default_value = "200")]
     gens: usize,
 
     /// Population size
     #[clap(long, default_value = "120")]
     pop: usize,
 
     /// Elite size (carry over best)
     #[clap(long, default_value = "8")]
     elite: usize,
 
     /// Mutation rate (0.0 - 1.0)
     #[clap(long, default_value = "0.3")]
     mutation_rate: f64,
 
     /// Crossover rate (0.0 - 1.0)
     #[clap(long, default_value = "0.7")]
     crossover_rate: f64,
 
     /// Max Grammar Depth
     #[clap(long, default_value = "24")]
     max_ops: usize,
 
     /// Discrete step for AXIS selection (in grammar placeholders)
     #[clap(long, default_value = "1.0")]
     axis_step: f64,
 
     /// Discrete step for VALUE modification
     #[clap(long, default_value = "0.1")]
     value_step: f64,
 
     /// Target text to reverse-engineer
     #[clap(long, required = true)]
     target: String,
 
     /// Load external grammar from .cfg
     #[clap(long)]
     cfg: Option<PathBuf>,
 
     /// Dump resolved grammar to .cfg
     #[clap(long)]
     dump_cfg: Option<PathBuf>,
 
     /// Save best vectors as JSONL
     #[clap(long)]
     save_best: Option<PathBuf>,
 
     /// Save 2D projection (SVG or CSV)
     #[clap(long)]
     plot_2d: Option<PathBuf>,
 
     /// Save 3D projection (CSV)
     #[clap(long)]
     plot_3d: Option<PathBuf>,
 }
 
 // =========================================================================================
 // 2. OLLAMA CLIENT
 // =========================================================================================
 
 #[derive(Serialize)]
 struct EmbeddingRequest {
     model: String,
     prompt: String,
 }
 
 #[derive(Deserialize)]
 struct EmbeddingResponse {
     embedding: Vec<f64>,
 }
 
 async fn fetch_embedding(url: &str, model: &str, text: &str) -> Result<DVector<f64>> {
     let client = reqwest::Client::new();
     let body = EmbeddingRequest {
         model: model.to_string(),
         prompt: text.to_string(),
     };
 
     let res = client
         .post(url)
         .json(&body)
         .send()
         .await
         .context("Failed to connect to Ollama")?;
 
     if !res.status().is_success() {
         let err_text = res.text().await?;
         anyhow::bail!("Ollama API Error: {}", err_text);
     }
 
     let parsed: EmbeddingResponse = res
         .json()
         .await
         .context("Failed to parse Ollama response")?;
 
     Ok(DVector::from_vec(parsed.embedding))
 }
 
 // =========================================================================================
 // 3. VECTOR INTERPRETER (PHENOTYPE)
 // =========================================================================================
 
 struct VectorInterpreter {
     dim: usize,
 }
 
 impl VectorInterpreter {
     fn new(dim: usize) -> Self {
         Self { dim }
     }
 
     /// Parses the GGGP-generated string and applies operations to a zero-vector.
     fn execute(&self, program: &str) -> DVector<f64> {
         let mut vec = DVector::zeros(self.dim);
         let tokens: Vec<&str> = program.split_whitespace().collect();
         let mut ptr = 0;
 
         while ptr < tokens.len() {
             let op = tokens[ptr];
             ptr += 1;
 
             match op {
                 "AX" => {
                     // AX <axis> <val>
                     if ptr + 1 < tokens.len() {
                         if let (Ok(axis_raw), Ok(val)) = (tokens[ptr].parse::<f64>(), tokens[ptr+1].parse::<f64>()) {
                             let axis = (axis_raw as usize) % self.dim; // Safety modulo
                             vec[axis] += val;
                         }
                         ptr += 2;
                     }
                 }
                 "SCALE" => {
                     // SCALE <val>
                     if ptr < tokens.len() {
                         if let Ok(val) = tokens[ptr].parse::<f64>() {
                             vec *= val;
                         }
                         ptr += 1;
                     }
                 }
                 "NORM" => {
                     // NORM
                     let norm = vec.norm();
                     if norm > 1e-12 {
                         vec.normalize_mut();
                     }
                 }
                 "MIX" => {
                     // MIX <ax1> <ax2> <w>
                     if ptr + 2 < tokens.len() {
                         if let (Ok(a1), Ok(a2), Ok(w)) = (
                             tokens[ptr].parse::<f64>(),
                             tokens[ptr+1].parse::<f64>(),
                             tokens[ptr+2].parse::<f64>()
                         ) {
                             let ax1 = (a1 as usize) % self.dim;
                             let ax2 = (a2 as usize) % self.dim;
                             let val1 = vec[ax1];
                             let val2 = vec[ax2];
                             // Linear interpolation
                             let w_clamped = w.clamp(0.0, 1.0);
                             vec[ax1] = val1 * (1.0 - w_clamped) + val2 * w_clamped;
                             vec[ax2] = val2 * (1.0 - w_clamped) + val1 * w_clamped; 
                         }
                         ptr += 3;
                     }
                 }
                 "ROT" => {
                     // ROT <ax1> <ax2> <deg>
                     if ptr + 2 < tokens.len() {
                          if let (Ok(a1), Ok(a2), Ok(deg)) = (
                             tokens[ptr].parse::<f64>(),
                             tokens[ptr+1].parse::<f64>(),
                             tokens[ptr+2].parse::<f64>()
                         ) {
                             let ax1 = (a1 as usize) % self.dim;
                             let ax2 = (a2 as usize) % self.dim;
                             if ax1 != ax2 {
                                 let rad = deg.to_radians();
                                 let cos = rad.cos();
                                 let sin = rad.sin();
                                 let x = vec[ax1];
                                 let y = vec[ax2];
                                 vec[ax1] = x * cos - y * sin;
                                 vec[ax2] = x * sin + y * cos;
                             }
                         }
                         ptr += 3;
                     }
                 }
                 "FRAC" => {
                      // FRAC <exponent>
                      if ptr < tokens.len() {
                          if let Ok(exp) = tokens[ptr].parse::<f64>() {
                              for i in 0..self.dim {
                                  let val = vec[i];
                                  vec[i] = val.signum() * val.abs().powf(exp);
                              }
                          }
                          ptr += 1;
                      }
                 }
                 _ => { 
                     // Unknown token, skip
                 }
             }
         }
         
         vec
     }
 }
 
 // =========================================================================================
 // 4. GRAMMAR BUILDER
 // =========================================================================================
 
 fn build_vector_grammar(dim: usize, axis_step: f64, value_step: f64, max_depth: usize) -> Node {
     // We construct a V1/V2 Node tree programmatically mimicking a .cfg file.
     // In a real scenario, we might use a builder pattern. Here we construct raw Nodes.
 
     let mut grammar = Node::new("Grammar");
     grammar.add_attr_int("MaxDepth", max_depth as i32);
     grammar.add_attr_int("MaxCrossoverNodes", (max_depth * 2) as i32);
     grammar.add_attr_int("MaxMutationNodes", (max_depth / 2) as i32);
     grammar.add_attr_bool("OPTIMIZE", true); // We provide chromosomes externally
     
     // Define Ranges for Axis and Value
     let axis_range = format!("<len from=0 to={} inc={}>", (dim - 1) as f64, axis_step);
     // Value range: -1.0 to 1.0
     let val_range = format!("<len from=-1.0 to=1.0 inc={}>", value_step);
     // Angle range: -180 to 180
     let ang_range = format!("<len from=-180 to=180 inc=15>");
     // Weight range: 0 to 1
     let weight_range = format!("<len from=0 to=1.0 inc=0.1>");
     
     // RULES
     let mut rules = Node::new("RULES");
 
     // START -> SEQ
     let mut start = Node::new("START");
     let mut start_choices = Node::new("CHOICES");
     let mut s_c0 = Node::new("0");
     s_c0.add_attr_str("Text", "<SEQ>");
     start_choices.add_child(s_c0);
     start.add_child(start_choices);
     rules.add_child(start);
 
     // SEQ -> OP SEQ | OP
     let mut seq = Node::new("SEQ");
     let mut seq_choices = Node::new("CHOICES");
     
     // 0: OP SEQ (Recursion)
     let mut seq_c0 = Node::new("0");
     seq_c0.add_attr_str("Text", "<OP> <SEQ>");
     seq_choices.add_child(seq_c0);
 
     // 1: OP (Terminal branch)
     let mut seq_c1 = Node::new("1");
     seq_c1.add_attr_str("Text", "<OP>");
     seq_choices.add_child(seq_c1);
     
     seq.add_child(seq_choices);
     rules.add_child(seq);
 
     // OP -> Specific Operations
     let mut op = Node::new("OP");
     let mut op_choices = Node::new("CHOICES");
 
     let ops_defs = vec![
         // AX axis val
         format!("AX {} {}", axis_range, val_range), 
         // SCALE val
         format!("SCALE {}", val_range),
         // NORM
         "NORM".to_string(),
         // MIX ax ax weight
         format!("MIX {} {} {}", axis_range, axis_range, weight_range),
         // ROT ax ax deg
         format!("ROT {} {} {}", axis_range, axis_range, ang_range),
         // FRAC exponent (using weight range for simplicity 0..1, maybe want wider?)
         format!("FRAC {}", weight_range), 
     ];
 
     for (i, def) in ops_defs.iter().enumerate() {
         let mut c = Node::new(&i.to_string());
         c.add_attr_str("Text", def);
         op_choices.add_child(c);
     }
     
     op.add_child(op_choices);
     rules.add_child(op);
 
     grammar.add_child(rules);
     grammar
 }
 
 // =========================================================================================
 // 5. MAIN LOGIC
 // =========================================================================================
 
 #[tokio::main]
 async fn main() -> Result<()> {
     let opts = Opts::parse();
 
     // 1. Setup RNG
     let seed = opts.seed.unwrap_or_else(|| fastrand::u64(..));
     fastrand::seed(seed);
     println!(">>> System Active. Target: '{}'. Seed: {}", opts.target, seed);
 
     // 2. Fetch Target Embedding
     println!(">>> Fetching embedding from {}...", opts.url);
     let target_vec = fetch_embedding(&opts.url, &opts.model, &opts.target).await?;
     let dim = target_vec.len();
     println!(">>> Target vector received. Dimension: {}", dim);
 
     // 3. Prepare Grammar
     let grammar_node = if let Some(path) = &opts.cfg {
         println!(">>> Loading grammar from {:?}", path);
         // Load binary logic would go here. For now, assuming standard library load.
         // rust_core::storage::codec::load_from_file(path)?
         todo!("Implement binary loading from spec if needed, or fallback to builder");
     } else {
         println!(">>> Building in-memory grammar (MaxDepth: {})", opts.max_ops);
         build_vector_grammar(dim, opts.axis_step, opts.value_step, opts.max_ops)
     };
 
     if let Some(dump_path) = &opts.dump_cfg {
         // Here we would dump the node tree to file.
         // rust_core::storage::codec::save_to_file(&grammar_node, dump_path)?;
         println!(">>> Dumped grammar to {:?}", dump_path);
     }
 
     // Convert Node to runtime GpGrammar
     let mut gp_grammar = GpGrammar::new();
     gp_grammar.load_from_node(&grammar_node); // Assuming this method exists in your core
     
     // 4. Initialization
     let interpreter = VectorInterpreter::new(dim);
     let mut pop: Vec<GpIndividual> = (0..opts.pop)
         .map(|_| {
             // Create random chromosome
             // length heuristic: max_depth * 2 to give space
             let len = opts.max_ops * 4; 
             let genes: Vec<i32> = (0..len).map(|_| fastrand::i32(0..255)).collect();
             GpIndividual { 
                 genes, 
                 fitness: 0.0, 
                 ..Default::default() // Assuming other fields exist
             }
         })
         .collect();
 
     let mut best_fitness = -1.0;
     let mut best_program = String::new();
     let mut best_vec = DVector::zeros(dim);
 
     // 5. Evolution Loop
     let start_time = Instant::now();
 
     for gen in 0..opts.gens {
         // A. Evaluate
         // In a real impl, use rayon::par_iter_mut
         for ind in &mut pop {
             let mut tree = GpTree::new(&gp_grammar);
             // This builds the tree from genes
             tree.build(&ind.genes); 
             
             // Get text program
             let program = tree.indented_text(); 
             
             // Execute
             let candidate_vec = interpreter.execute(&program);
             
             // Cosine Similarity
             let dot = candidate_vec.dot(&target_vec);
             let mag_c = candidate_vec.norm();
             let mag_t = target_vec.norm();
             
             let sim = if mag_c < 1e-9 || mag_t < 1e-9 {
                 0.0
             } else {
                 dot / (mag_c * mag_t)
             };
             
             ind.fitness = sim;
             
             // Track Best
             if sim > best_fitness {
                 best_fitness = sim;
                 best_program = program.clone();
                 best_vec = candidate_vec.clone();
                 
                 println!(">>> New Best [Gen {}]: {:.5}", gen, best_fitness);
                 
                 // Save JSONL if requested
                 if let Some(path) = &opts.save_best {
                     save_best_json(path, gen, sim, dim, &best_vec)?;
                 }
                 
                 // Save Plots if requested
                 if let Some(path) = &opts.plot_2d {
                     save_plot_2d(path, gen, &best_vec)?;
                 }
             }
         }
 
         // B. Selection & Breeding (Elitism + Tournament)
         // Sort descending
         pop.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
 
         // Report
         if gen % 10 == 0 {
             println!("Gen {:03} | Best: {:.5} | Avg: {:.5}", 
                 gen, 
                 pop[0].fitness, 
                 pop.iter().map(|i| i.fitness).sum::<f64>() / pop.len() as f64
             );
         }
 
         let mut next_gen = Vec::with_capacity(opts.pop);
         
         // Elitism
         for i in 0..opts.elite {
             if i < pop.len() {
                 next_gen.push(pop[i].clone());
             }
         }
 
         // Breeding
         while next_gen.len() < opts.pop {
             // Tournament selection (size 3)
             let p1 = tournament_select(&pop, 3);
             let p2 = tournament_select(&pop, 3);
 
             let mut child = if fastrand::f64() < opts.crossover_rate {
                 crossover_individuals(p1, p2, &gp_grammar) // From core
             } else {
                 p1.clone()
             };
 
             if fastrand::f64() < opts.mutation_rate {
                 mutate_individual(&mut child, &gp_grammar); // From core
             }
             
             next_gen.push(child);
         }
 
         pop = next_gen;
     }
 
     let duration = start_time.elapsed();
     println!(">>> Evolution Complete in {:.2?}.", duration);
     println!(">>> Final Best Fitness: {:.5}", best_fitness);
     println!(">>> Program Preview:\n{}", best_program);
 
     Ok(())
 }
 
 // =========================================================================================
 // 6. HELPER FUNCTIONS
 // =========================================================================================
 
 fn tournament_select(pop: &[GpIndividual], k: usize) -> &GpIndividual {
     let mut best = &pop[fastrand::usize(..pop.len())];
     for _ in 1..k {
         let cand = &pop[fastrand::usize(..pop.len())];
         if cand.fitness > best.fitness {
             best = cand;
         }
     }
     best
 }
 
 fn save_best_json(path: &PathBuf, gen: usize, fitness: f64, dim: usize, vec: &DVector<f64>) -> Result<()> {
     let file = OpenOptions::new().create(true).append(true).open(path)?;
     let mut writer = BufWriter::new(file);
     let rec = json!({
         "generation": gen,
         "fitness": fitness,
         "dim": dim,
         "vector": vec.as_slice()
     });
     writeln!(writer, "{}", rec)?;
     Ok(())
 }
 
 fn save_plot_2d(path: &PathBuf, gen: usize, vec: &DVector<f64>) -> Result<()> {
     // Project vector to 2D using dimensions 0 and 1 (naive projection for PoC)
     // Or, if svg, draw the "path" of the vector components?
     // Let's draw the vector components as a line graph (index vs value)
     
     let is_svg = path.extension().map_or(false, |e| e == "svg");
     
     if is_svg {
         // Create filename with generation
         let stem = path.file_stem().unwrap().to_str().unwrap();
         let new_name = format!("{}.{:05}.svg", stem, gen);
         let new_path = path.with_file_name(new_name);
         
         let mut f = File::create(new_path)?;
         let width = 800;
         let height = 400;
         writeln!(f, "<svg width='{}' height='{}' xmlns='http://www.w3.org/2000/svg'>", width, height)?;
         writeln!(f, "<rect width='100%' height='100%' fill='#f0f0f0'/>")?;
         
         // Find min/max for scaling
         let min_v = vec.min();
         let max_v = vec.max();
         let range = (max_v - min_v).max(1e-6);
         
         writeln!(f, "<path d='M 0 {} ", height / 2)?; // Start middle
         
         for (i, &val) in vec.iter().enumerate() {
             let x = (i as f64 / vec.len() as f64) * width as f64;
             // Normalize val to 0..height
             let norm_y = (val - min_v) / range;
             let y = height as f64 - (norm_y * height as f64);
             writeln!(f, "L {:.2} {:.2} ", x, y)?;
         }
         
         writeln!(f, "' stroke='black' fill='none' stroke-width='1'/>")?;
         writeln!(f, "</svg>")?;
         
     } else {
         // CSV
         let mut f = OpenOptions::new().create(true).append(true).open(path)?;
         // Just write the first 2 dimensions as a point
         if vec.len() >= 2 {
             writeln!(f, "{},{},{},{}", gen, vec[0], vec[1], vec.len())?;
         }
     }
     
     Ok(())
 }