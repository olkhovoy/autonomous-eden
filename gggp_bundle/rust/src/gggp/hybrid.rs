use nalgebra::DVector;
use rand::Rng;
use cmaes::{CMAESOptions, DVector as CmaesDVector};
use crate::gggp::{Gggp, GpIndividual};

pub struct HybridEvolutionConfig {
    pub continuous_mutation_sigma: f64,
    pub cmaes_enabled: bool,
    pub cmaes_population_ratio: f64,
    pub cmaes_generations_per_gggp_step: usize,
    pub cmaes_population_size: Option<usize>,
}

pub fn extract_continuous_genome(ind: &GpIndividual) -> Vec<f64> {
    ind.continuous_weights.clone()
}

pub fn inject_continuous_genome(ind: &mut GpIndividual, weights: &[f64]) {
    ind.continuous_weights = weights.to_vec();
}

pub fn continuous_mutate_individual(ind: &mut GpIndividual, sigma: f64, rng: &mut impl Rng) {
    if sigma <= 0.0 || ind.continuous_weights.is_empty() {
        return;
    }
    for w in &mut ind.continuous_weights {
        *w += rand_normal(rng) * sigma;
    }
}

fn rand_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Helper to run CMA-ES optimization on the continuous weights for a given evaluation function.
pub fn run_cmaes_optimization<F>(
    initial_weights: Vec<f64>,
    config: &HybridEvolutionConfig,
    mut eval_fn: F,
) -> Vec<f64>
where
    F: FnMut(&[f64]) -> f64,
{
    if !config.cmaes_enabled || initial_weights.is_empty() {
        return initial_weights;
    }

    // Wrap the maximization function into a minimization function for cmaes crate
    let mut min_fn = |x: &cmaes::DVector<f64>| -> f64 {
        -eval_fn(x.as_slice())
    };

    let mut cmaes_opts = CMAESOptions::new(initial_weights.clone(), config.continuous_mutation_sigma);
    if let Some(pop_size) = config.cmaes_population_size {
        cmaes_opts = cmaes_opts.population_size(pop_size);
    }
    
    let mut cmaes_state = cmaes_opts
        .build(min_fn)
        .unwrap();

    let mut best_result = initial_weights.clone();
    for _ in 0..config.cmaes_generations_per_gggp_step {
        if cmaes_state.next().is_some() {
            break; // Terminated
        }
    }

    if let Some(best) = cmaes_state.overall_best_individual() {
        best_result = best.point.as_slice().to_vec();
    }

    best_result
}
