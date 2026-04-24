use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use numpy::{PyArray1, ToPyArray};
use nalgebra::DVector;
use rand::{SeedableRng, rngs::StdRng};
use std::rc::Rc;
use std::cell::RefCell;

use crate::storage::Node;
use crate::gggp::{GpConfig, Gggp, GpIndividual, calc_lengths, parse_text};
use crate::gggp::vector::{
    compile_tree_to_vector, compile_tree_to_vector_with_input,
    FractalDecoderConfig, cosine_similarity,
};
use crate::gggp::hybrid::{HybridEvolutionConfig, run_cmaes_optimization};

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
    let _ = calc_lengths(cfg);
}

#[pyclass(unsendable)]
pub struct SemioticHypercube {
    /// Primary grammar config. In the A1 SCL PoC this is the encoder-side
    /// grammar (G, dim=CODE_DIM). `evolve_target` and `fractal_expand` use
    /// it unconditionally.
    cfg: Rc<GpConfig>,
    /// Optional decoder-side grammar config. When set via
    /// `attach_decoder_grammar`, `batch_render_dual` parses `chromosome_d`
    /// against this instead of `cfg`. Unset = old single-grammar behavior
    /// (both G and D parsed against `cfg`).
    decoder_cfg: RefCell<Option<Rc<GpConfig>>>,
    engine: RefCell<Gggp>,
    target_vec: Rc<RefCell<DVector<f64>>>,
}

#[pymethods]
impl SemioticHypercube {
    #[new]
    fn new(grammar_cfg_path: &str) -> PyResult<Self> {
        let mut grammar_node = Node::from_file(grammar_cfg_path).map_err(|e| {
            PyValueError::new_err(format!("Failed to load grammar cfg: {}", e))
        })?;

        // Process grammar text nodes
        finalize_grammar(&mut grammar_node);

        let cfg = GpConfig::from_node(&grammar_node).map_err(|e| {
            PyValueError::new_err(format!("Failed to initialize GpConfig: {:?}", e))
        })?;

        let mut engine = Gggp::new();
        let target_vec = Rc::new(RefCell::new(DVector::zeros(3)));
        
        let target_clone = target_vec.clone();
        engine.set_on_get_fitness(move |ind| {
            let dim = target_clone.borrow().len();
            if dim == 0 { return Some(-100.0); }
            if ind.trees().is_empty() { return Some(-100.0); }
            
            let tree = &ind.trees()[0];
            let vec = compile_tree_to_vector(tree, dim, None);
            
            // Allow hybrid continuous weights to adjust the vector linearly
            let mut final_vec = vec.clone();
            if ind.continuous_weights.len() == dim {
                let cont_vec = DVector::from_vec(ind.continuous_weights.clone());
                final_vec += &cont_vec;
            }
            
            let sim = cosine_similarity(&final_vec, &target_clone.borrow());
            Some(sim - 1.0 * (final_vec.norm() - 1.0).powi(2))
        });

        engine.init_from_nodes(&[grammar_node], 50, 5, 0.7, 0.2).map_err(|e| {
            PyValueError::new_err(format!("Failed to init Gggp: {:?}", e))
        })?;

        Ok(SemioticHypercube {
            cfg,
            decoder_cfg: RefCell::new(None),
            engine: RefCell::new(engine),
            target_vec,
        })
    }

    /// Attach a second grammar used exclusively for parsing the decoder
    /// chromosome in `batch_render_dual`. A1 uses this to give G and D
    /// role-specific search spaces:
    ///   encoder grammar dim=16   (CODE_DIM;   AX axis range 0..15)
    ///   decoder grammar dim=1024 (TARGET_DIM; AX axis range 0..1023)
    ///
    /// Idempotent: calling it twice replaces the previously attached
    /// grammar. Passing an empty string resets to the single-grammar
    /// fallback (decoder_cfg = None).
    fn attach_decoder_grammar(&self, grammar_cfg_path: &str) -> PyResult<()> {
        if grammar_cfg_path.is_empty() {
            *self.decoder_cfg.borrow_mut() = None;
            return Ok(());
        }
        let mut grammar_node = Node::from_file(grammar_cfg_path).map_err(|e| {
            PyValueError::new_err(format!(
                "attach_decoder_grammar: failed to load {}: {}. \
                 Hint: run gen_neuro_grammar decoder <path> first.",
                grammar_cfg_path, e
            ))
        })?;
        finalize_grammar(&mut grammar_node);
        let dec = GpConfig::from_node(&grammar_node).map_err(|e| {
            PyValueError::new_err(format!(
                "attach_decoder_grammar: GpConfig init failed: {:?}", e
            ))
        })?;
        *self.decoder_cfg.borrow_mut() = Some(dec);
        Ok(())
    }

    fn fractal_expand<'py>(
        &self,
        py: Python<'py>,
        chromosome: Vec<i32>,
        depth: i32,
        dim: usize,
    ) -> PyResult<&'py PyArray1<f64>> {
        let chromo_str = chromosome
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join("-");

        let tree = self.cfg.tree_from_chromosome(&chromo_str).map_err(|e| {
            PyValueError::new_err(format!("Failed to parse chromosome: {:?}", e))
        })?;

        let fractal_config = FractalDecoderConfig {
            max_expansion_depth: depth,
            hash_seed: 42,
        };

        let result_vector = compile_tree_to_vector(&tree, dim, Some(&fractal_config));

        let vec_data = result_vector.as_slice().to_vec();
        Ok(vec_data.to_pyarray(py))
    }

    /// Draw a grammar-valid random chromosome using a seeded RNG.
    ///
    /// `role` selects which grammar to sample against:
    ///   "encoder" (default): self.cfg
    ///   "decoder": self.decoder_cfg (must be attached first)
    ///
    /// Used by A1 T7 runner to initialize the EA population. The
    /// chromosome is guaranteed to parse back into a tree (as long as
    /// the grammar and seed don't produce pathological deep recursions
    /// that hit MaxDepth). Output is a Vec<i32>.
    #[pyo3(signature = (seed, role="encoder"))]
    fn random_chromosome(&self, seed: u64, role: &str) -> PyResult<Vec<i32>> {
        let cfg = self.resolve_role_cfg(role, "random_chromosome")?;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut ind = GpIndividual::new();
        ind.random_trees(&[cfg], &mut rng);
        if ind.trees().is_empty() {
            return Err(PyRuntimeError::new_err(
                "random_chromosome: no trees generated. Hint: grammar may \
                 be empty or malformed.",
            ));
        }
        let chromo_str = ind.trees()[0].chromosome();
        let parts: Result<Vec<i32>, _> = chromo_str
            .split('-')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<i32>())
            .collect();
        parts.map_err(|e| {
            PyRuntimeError::new_err(format!(
                "random_chromosome: non-integer token in chromosome \
                 '{}': {}. This is a bug in the grammar or encoding.",
                chromo_str, e
            ))
        })
    }

    /// Render a chromosome tree over a seeded starting state.
    ///
    /// Used by MEDP A1 SCL PoC:
    ///   c_i = render_tree_with_input(G_chromosome, code_dim=16,  input=T_i)
    ///   r_i = render_tree_with_input(D_chromosome, state_dim=1024, input=c_i,
    ///                                role="decoder")
    ///
    /// Arguments:
    ///   chromosome -- genotype integer-array (e.g. [0, 1, 3, 2, ...]).
    ///   dim        -- size of the working state (output length).
    ///   input      -- optional seed vector. If provided, the first
    ///                 min(len(input), dim) components of the state are
    ///                 initialized from input; rest stay zero.
    ///                 If None, state starts at zero (classic path).
    ///   role       -- "encoder" (default) parses `chromosome` against
    ///                 self.cfg. "decoder" parses against the grammar
    ///                 attached via attach_decoder_grammar. A2/S1c added
    ///                 this so NC2 can call D(c_mix) -> T_mix without
    ///                 round-tripping through batch_render_dual.
    ///
    /// Returns: 1-D numpy array of length `dim` (float64).
    ///
    /// Errors: PyValueError if the chromosome is ill-formed for the
    /// selected grammar; PyRuntimeError if role="decoder" but no decoder
    /// grammar is attached.
    #[pyo3(signature = (chromosome, dim, input=None, role="encoder"))]
    fn render_tree_with_input<'py>(
        &self,
        py: Python<'py>,
        chromosome: Vec<i32>,
        dim: usize,
        input: Option<&'py PyArray1<f64>>,
        role: &str,
    ) -> PyResult<&'py PyArray1<f64>> {
        let chromo_str = chromosome
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join("-");

        let cfg = self.resolve_role_cfg(role, "render_tree_with_input")?;
        let tree = cfg.tree_from_chromosome(&chromo_str).map_err(|e| {
            PyValueError::new_err(format!(
                "render_tree_with_input(role='{}'): failed to parse \
                 chromosome '{}': {:?}. Hint: verify the chromosome \
                 matches the {} grammar loaded by this \
                 SemioticHypercube instance -- genome integers must be \
                 valid choice-indices at each grammar rule.",
                role, chromo_str, e, role
            ))
        })?;

        let seed_opt: Option<DVector<f64>> = match input {
            None => None,
            Some(arr) => {
                let ro = arr.readonly();
                let sl = ro.as_slice().map_err(|_| {
                    PyValueError::new_err(
                        "input numpy array must be contiguous 1-D float64",
                    )
                })?;
                Some(DVector::from_vec(sl.to_vec()))
            }
        };

        let result_vector = compile_tree_to_vector_with_input(
            &tree,
            dim,
            seed_opt.as_ref(),
            None,
        );

        Ok(result_vector.as_slice().to_vec().to_pyarray(py))
    }

    /// Batch-render the dual (G, D) tree pair over a T matrix.
    ///
    /// This is the core A1 fitness primitive: one call produces the
    /// full (c_matrix, reconstruction_matrix) for the whole 128-row
    /// corpus, amortizing Python-Rust FFI overhead and GIL acquisition.
    ///
    /// Arguments:
    ///   chromosome_g -- genotype for encoder-grammar G.
    ///   chromosome_d -- genotype for decoder-interpreter D.
    ///   t_matrix     -- numpy (N, target_dim) float64 input embeddings.
    ///   code_dim     -- output dim for c_i = G(T_i).
    ///   target_dim   -- output dim for r_i = D(c_i); typically the
    ///                   same as t_matrix.shape[1].
    ///
    /// Returns: dict with keys
    ///   "c"             -- numpy (N, code_dim)    G(T_i) for each i.
    ///   "reconstruction"-- numpy (N, target_dim)  D(G(T_i)) for each i.
    ///   "per_i_cos"     -- numpy (N,)             cos(r_i, T_i).
    ///   "F"             -- float                  mean_i cos(r_i, T_i).
    ///
    /// Penalties (length / entropy) are NOT applied here -- caller
    /// computes them in Python from `c` and from `len(chromosome_g)`,
    /// `len(chromosome_d)`. This keeps fitness shaping flexible.
    fn batch_render_dual<'py>(
        &self,
        py: Python<'py>,
        chromosome_g: Vec<i32>,
        chromosome_d: Vec<i32>,
        t_matrix: &'py numpy::PyArray2<f64>,
        code_dim: usize,
        target_dim: usize,
    ) -> PyResult<&'py pyo3::types::PyDict> {
        use numpy::{PyArray2, ToPyArray};
        use pyo3::types::PyDict;

        let to_str = |c: &[i32]| {
            c.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("-")
        };
        let g_str = to_str(&chromosome_g);
        let d_str = to_str(&chromosome_d);

        let g_tree = self.cfg.tree_from_chromosome(&g_str).map_err(|e| {
            PyValueError::new_err(format!(
                "encoder chromosome parse failed against encoder grammar: \
                 {:?}", e
            ))
        })?;
        // If a decoder grammar is attached, parse chromosome_d against it.
        // Otherwise fall back to self.cfg (single-grammar path).
        let dec_borrow = self.decoder_cfg.borrow();
        let d_cfg = dec_borrow.as_ref().unwrap_or(&self.cfg);
        let d_tree = d_cfg.tree_from_chromosome(&d_str).map_err(|e| {
            PyValueError::new_err(format!(
                "decoder chromosome parse failed against {} grammar: {:?}",
                if dec_borrow.is_some() { "decoder" } else { "encoder (single-grammar fallback)" },
                e
            ))
        })?;

        let t_ro = t_matrix.readonly();
        let t_shape = t_ro.shape().to_vec();
        if t_shape.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "t_matrix must be 2-D, got shape {:?}",
                t_shape
            )));
        }
        if t_shape[1] != target_dim {
            return Err(PyValueError::new_err(format!(
                "t_matrix.shape[1] = {} but target_dim = {}. \
                 Hint: pass target_dim = T.shape[1].",
                t_shape[1], target_dim
            )));
        }
        let n = t_shape[0];
        let t_slice = t_ro.as_slice().map_err(|_| {
            PyValueError::new_err("t_matrix must be contiguous float64")
        })?;

        let mut c_buf: Vec<f64> = vec![0.0; n * code_dim];
        let mut r_buf: Vec<f64> = vec![0.0; n * target_dim];
        let mut per_i: Vec<f64> = vec![0.0; n];

        for i in 0..n {
            let t_i = DVector::from_row_slice(
                &t_slice[i * target_dim..(i + 1) * target_dim],
            );
            let c_i = compile_tree_to_vector_with_input(
                &g_tree, code_dim, Some(&t_i), None,
            );
            let r_i = compile_tree_to_vector_with_input(
                &d_tree, target_dim, Some(&c_i), None,
            );
            per_i[i] = cosine_similarity(&r_i, &t_i);
            for j in 0..code_dim {
                c_buf[i * code_dim + j] = c_i[j];
            }
            for j in 0..target_dim {
                r_buf[i * target_dim + j] = r_i[j];
            }
        }

        let f_mean: f64 = if n > 0 {
            per_i.iter().sum::<f64>() / n as f64
        } else {
            0.0
        };

        let c_np = PyArray2::from_vec2(
            py,
            &(0..n).map(|i| c_buf[i * code_dim..(i + 1) * code_dim].to_vec()).collect::<Vec<_>>(),
        ).map_err(|e| PyRuntimeError::new_err(format!("c np build: {:?}", e)))?;
        let r_np = PyArray2::from_vec2(
            py,
            &(0..n).map(|i| r_buf[i * target_dim..(i + 1) * target_dim].to_vec()).collect::<Vec<_>>(),
        ).map_err(|e| PyRuntimeError::new_err(format!("r np build: {:?}", e)))?;
        let per_i_np = per_i.to_pyarray(py);

        let out = PyDict::new(py);
        out.set_item("c", c_np)?;
        out.set_item("reconstruction", r_np)?;
        out.set_item("per_i_cos", per_i_np)?;
        out.set_item("F", f_mean)?;
        Ok(out)
    }

    /// Render a chromosome to its string form (the program text).
    ///
    /// Used by MEDP A2 / S1c to let Python-side code analyze the
    /// structural properties of a decoder program, e.g. NC3 requires
    /// counting the fraction of opcode tokens that are code-gated
    /// (CTRL / SBC / ADDC). Without this binding we would either have
    /// to re-implement tree_from_chromosome in Python (duplicating
    /// Rust logic against the binary .cfg format) or add a
    /// task-specific counter in Rust.
    ///
    /// Arguments:
    ///   chromosome -- genotype integer-array (same format as all other
    ///                 APIs on this class).
    ///   role       -- "encoder" (default) parses against self.cfg.
    ///                 "decoder" parses against the attached decoder
    ///                 grammar; PyRuntimeError if none is attached.
    ///
    /// Returns: the GpTree::text() of the parsed chromosome, i.e. the
    /// rendered program with all grammar parameters filled in (e.g.
    /// "AX 2 0.5 CTRL 3 1 SCALE 1.0 ...").
    ///
    /// Errors: PyValueError if the chromosome does not parse against
    /// the selected grammar.
    #[pyo3(signature = (chromosome, role="encoder"))]
    fn chromosome_text(&self, chromosome: Vec<i32>, role: &str) -> PyResult<String> {
        let chromo_str = chromosome
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join("-");
        let cfg = self.resolve_role_cfg(role, "chromosome_text")?;
        let tree = cfg.tree_from_chromosome(&chromo_str).map_err(|e| {
            PyValueError::new_err(format!(
                "chromosome_text(role='{}'): failed to parse chromosome \
                 '{}': {:?}. Hint: verify the chromosome matches the {} \
                 grammar.",
                role, chromo_str, e, role
            ))
        })?;
        Ok(tree.text())
    }

    fn evolve_target<'py>(
        &self,
        py: Python<'py>,
        target_vector: &'py PyArray1<f64>,
        generations: usize,
    ) -> PyResult<(Vec<f64>, &'py PyArray1<f64>, f64)> {
        // Read input target tensor safely (zero-copy when possible)
        let target_slice = target_vector.readonly();
        let target_slice = target_slice.as_slice().map_err(|_| {
            PyValueError::new_err("target_vector must be contiguous")
        })?;
        
        let target_dim = target_slice.len();
        let target = DVector::from_vec(target_slice.to_vec());

        let hybrid_cfg = HybridEvolutionConfig {
            continuous_mutation_sigma: 1.0,
            cmaes_enabled: true,
            cmaes_population_ratio: 0.5,
            cmaes_generations_per_gggp_step: generations,
            cmaes_population_size: None,
        };
        
        // Wrap the evaluate function
        let eval_fn = |weights: &[f64]| -> f64 {
            let candidate = DVector::from_vec(weights.to_vec());
            if candidate.len() == target.len() {
                let sim = cosine_similarity(&candidate, &target);
                sim - 10.0 * (candidate.norm() - 1.0).powi(2)
            } else {
                -100.0
            }
        };

        let initial_weights = vec![0.1; target_dim]; // Match target dimension for pure vector optimization

        let optimized_weights = run_cmaes_optimization(initial_weights, &hybrid_cfg, eval_fn);
        
        let final_candidate = DVector::from_vec(optimized_weights.clone());
        let final_fitness = cosine_similarity(&final_candidate, &target);

        let out_array = optimized_weights.clone().to_pyarray(py);
        
        Ok((optimized_weights, out_array, final_fitness))
    }

    fn evolve_with_fitness<'py>(
        &self,
        py: Python<'py>,
        dim: usize,
        generations: usize,
        population_size: usize,
        fitness_callback: PyObject,
    ) -> PyResult<(Vec<f64>, &'py PyArray1<f64>, f64)> {
        let hybrid_cfg = HybridEvolutionConfig {
            continuous_mutation_sigma: 1.0,
            cmaes_enabled: true,
            cmaes_population_ratio: 0.5,
            cmaes_generations_per_gggp_step: generations,
            cmaes_population_size: Some(population_size),
        };
        
        let eval_fn = |weights: &[f64]| -> f64 {
            Python::with_gil(|py| {
                let array = weights.to_vec().to_pyarray(py);
                match fitness_callback.call1(py, (array,)) {
                    Ok(res) => match res.extract::<f64>(py) {
                        Ok(val) => val,
                        Err(_) => -1e9,
                    },
                    Err(_) => -1e9,
                }
            })
        };

        let initial_weights = vec![0.1; dim];

        let optimized_weights = run_cmaes_optimization(initial_weights, &hybrid_cfg, eval_fn);
        
        let final_fitness = Python::with_gil(|py| {
            let array = optimized_weights.clone().to_pyarray(py);
            match fitness_callback.call1(py, (array,)) {
                Ok(res) => res.extract::<f64>(py).unwrap_or(-1e9),
                Err(_) => -1e9,
            }
        });

        let out_array = optimized_weights.clone().to_pyarray(py);
        Ok((optimized_weights, out_array, final_fitness))
    }

    fn step_evolution<'py>(
        &self,
        py: Python<'py>,
        target_vector: &'py PyArray1<f64>,
    ) -> PyResult<(Vec<i32>, &'py PyArray1<f64>, f64)> {
        let target_slice = target_vector.readonly();
        let target_slice = target_slice.as_slice().map_err(|_| {
            PyValueError::new_err("target_vector must be contiguous")
        })?;
        
        let target_dim = target_slice.len();
        let target = DVector::from_vec(target_slice.to_vec());
        
        *self.target_vec.borrow_mut() = target.clone();

        let mut engine = self.engine.borrow_mut();
        
        // 1. One generation of GGGP
        engine.step();
        
        let mut individuals = engine.individuals_mut();
        if individuals.is_empty() {
            return Err(PyRuntimeError::new_err("Population is empty"));
        }
        
        // Ensure best individual has continuous weights
        if individuals[0].continuous_weights.is_empty() {
            individuals[0].continuous_weights = vec![0.1; target_dim];
        }

        let hybrid_cfg = HybridEvolutionConfig {
            continuous_mutation_sigma: 0.1,
            cmaes_enabled: true,
            cmaes_population_ratio: 0.5,
            cmaes_generations_per_gggp_step: 1, // micro-step
            cmaes_population_size: None,
        };
        
        let target_clone = target.clone();
        
        // Since we are applying CMA-ES on the best discrete topology, we need to capture the current discrete tree's output.
        let base_vec = {
            let best_tree = &individuals[0].trees()[0];
            compile_tree_to_vector(best_tree, target_dim, None)
        };
        
        let eval_fn = |weights: &[f64]| -> f64 {
            let mut final_vec = base_vec.clone();
            if weights.len() == target_dim {
                let cont_vec = DVector::from_vec(weights.to_vec());
                final_vec += &cont_vec;
            }
            if final_vec.len() == target_clone.len() {
                let sim = cosine_similarity(&final_vec, &target_clone);
                sim - 1.0 * (final_vec.norm() - 1.0).powi(2)
            } else {
                -100.0
            }
        };

        let optimized_weights = run_cmaes_optimization(
            individuals[0].continuous_weights.clone(),
            &hybrid_cfg,
            eval_fn
        );
        individuals[0].continuous_weights = optimized_weights.clone();

        let mut final_vec = base_vec.clone();
        if optimized_weights.len() == target_dim {
            let cont_vec = DVector::from_vec(optimized_weights.clone());
            final_vec += &cont_vec;
        }

        let final_fitness = cosine_similarity(&final_vec, &target);
        
        let chromo_str = individuals[0].trees()[0].chromosome();
        let chromo_ints: Vec<i32> = chromo_str.split('-').filter_map(|s| s.parse().ok()).collect();

        Ok((chromo_ints, final_vec.as_slice().to_vec().to_pyarray(py), final_fitness))
    }
}

impl SemioticHypercube {
    /// Resolve a role string to the matching `Rc<GpConfig>`.
    ///
    /// Shared by random_chromosome / render_tree_with_input / chromosome_text.
    /// `api_name` is injected into error messages so the caller always sees
    /// which binding failed, and therefore which grammar they actually need
    /// to attach. Private (not a pymethod) on purpose: role resolution is
    /// an internal concern of the Python bindings, not a public API.
    fn resolve_role_cfg(&self, role: &str, api_name: &str) -> PyResult<Rc<GpConfig>> {
        match role {
            "encoder" => Ok(self.cfg.clone()),
            "decoder" => {
                let dec = self.decoder_cfg.borrow();
                match dec.as_ref() {
                    Some(c) => Ok(c.clone()),
                    None => Err(PyRuntimeError::new_err(format!(
                        "{}(role='decoder'): no decoder grammar attached. \
                         Call attach_decoder_grammar(path) first.",
                        api_name
                    ))),
                }
            }
            other => Err(PyValueError::new_err(format!(
                "{}: unknown role '{}'. Valid: 'encoder' | 'decoder'.",
                api_name, other
            ))),
        }
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn semiotic_hypercube(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SemioticHypercube>()?;
    Ok(())
}
