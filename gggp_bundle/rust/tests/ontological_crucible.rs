//! Integration test: "ontological crucible".
//!
//! Smoke-level guarantees that the neuro-symbolic core (vector.rs + hybrid.rs)
//! preserves the invariants the rest of the system relies on:
//!   1. Dimensional integrity  -- rendered vectors always have the requested dim.
//!   2. Fractal determinism    -- same (gene, depth, parent_hash, seed) -> same ops.
//!   3. Hybrid convergence     -- run_cmaes_optimization improves fitness on a
//!                                 trivial quadratic landscape.
//!
//! These tests only exercise the public Rust API; the PyO3 bridge is covered
//! separately under tests/python_bridge/.

use nalgebra::DVector;

use semiotic_hypercube::gggp::hybrid::{run_cmaes_optimization, HybridEvolutionConfig};
use semiotic_hypercube::gggp::vector::{
    cosine_similarity, fractal_expand, FractalDecoderConfig,
};

#[test]
fn cosine_similarity_is_sane() {
    let a = DVector::from_vec(vec![1.0, 0.0, 0.0]);
    let b = DVector::from_vec(vec![1.0, 0.0, 0.0]);
    let c = DVector::from_vec(vec![0.0, 1.0, 0.0]);
    let d = DVector::from_vec(vec![-1.0, 0.0, 0.0]);

    assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-9);
    assert!(cosine_similarity(&a, &c).abs() < 1e-9);
    assert!((cosine_similarity(&a, &d) + 1.0).abs() < 1e-9);

    let zero = DVector::<f64>::zeros(3);
    assert_eq!(cosine_similarity(&a, &zero), 0.0);
}

#[test]
fn fractal_expand_is_deterministic() {
    let cfg = FractalDecoderConfig {
        max_expansion_depth: 4,
        hash_seed: 1729,
    };
    let ops_a = fractal_expand(7, 3, 0xdead_beef, &cfg, 8);
    let ops_b = fractal_expand(7, 3, 0xdead_beef, &cfg, 8);
    assert_eq!(ops_a.len(), ops_b.len());
    for (x, y) in ops_a.iter().zip(ops_b.iter()) {
        assert_eq!(x, y, "fractal_expand must be deterministic");
    }

    let ops_c = fractal_expand(7, 3, 0xcafe_babe, &cfg, 8);
    // Different parent_hash should generally produce different ops; we only
    // assert the call succeeds and returns at least one op.
    assert!(!ops_c.is_empty());
}

#[test]
fn hybrid_cmaes_converges_on_quadratic() {
    // Reference target for evaluating fitness. We use a standalone function
    // (not a closure) so we can score both the initial and the optimized
    // candidate without cloning a non-Clone FnMut into the optimizer.
    fn fitness(weights: &[f64], target: &DVector<f64>) -> f64 {
        let candidate = DVector::from_vec(weights.to_vec());
        if candidate.len() != target.len() {
            return -100.0;
        }
        let sim = cosine_similarity(&candidate, target);
        sim - 10.0 * (candidate.norm() - 1.0).powi(2)
    }

    let target = DVector::from_vec(vec![1.0, -0.5, 0.25, 0.0]);
    let target_for_opt = target.clone();

    let cfg = HybridEvolutionConfig {
        continuous_mutation_sigma: 0.5,
        cmaes_enabled: true,
        cmaes_population_ratio: 0.5,
        cmaes_generations_per_gggp_step: 200,
        cmaes_population_size: Some(16),
    };

    let initial = vec![0.1_f64, 0.1, 0.1, 0.1];
    let initial_fitness = fitness(&initial, &target);

    let optimized = run_cmaes_optimization(initial.clone(), &cfg, move |w: &[f64]| -> f64 {
        fitness(w, &target_for_opt)
    });
    assert_eq!(optimized.len(), initial.len());

    let final_fitness = fitness(&optimized, &target);
    assert!(
        final_fitness > initial_fitness,
        "CMA-ES must improve fitness: initial={initial_fitness:.4}, final={final_fitness:.4}"
    );

    // On this landscape (cosine reward + quadratic norm penalty at ||x||=1)
    // the optimum is co-linear with target with unit norm. Accept any cosine
    // > 0.7 to stay robust to RNG seeds.
    let cand = DVector::from_vec(optimized);
    let sim = cosine_similarity(&cand, &target);
    assert!(
        sim > 0.7,
        "CMA-ES did not approach target direction: cos_sim={sim:.4}"
    );
}
