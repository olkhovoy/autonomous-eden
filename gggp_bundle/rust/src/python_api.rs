use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use numpy::{PyArray1, ToPyArray};
use nalgebra::DVector;
use std::rc::Rc;
use std::cell::RefCell;

use crate::storage::Node;
use crate::gggp::{GpConfig, Gggp, GpIndividual, calc_lengths, parse_text};
use crate::gggp::vector::{compile_tree_to_vector, FractalDecoderConfig, cosine_similarity};
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
    cfg: Rc<GpConfig>,
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
            engine: RefCell::new(engine),
            target_vec,
        })
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

/// A Python module implemented in Rust.
#[pymodule]
fn semiotic_hypercube(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SemioticHypercube>()?;
    Ok(())
}
