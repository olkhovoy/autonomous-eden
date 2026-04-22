use nalgebra::DVector;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use super::{GpTree, TreeType};
use crate::gggp::phenotype::{VectorOp, VectorPhenotype, VectorSymbol};

pub struct FractalDecoderConfig {
    pub max_expansion_depth: i32,
    pub hash_seed: u64,
}

pub fn fractal_expand(
    gene_index: usize,
    depth: i32,
    parent_hash: u64,
    config: &FractalDecoderConfig,
    dim: usize,
) -> Vec<VectorOp> {
    let mut hasher = DefaultHasher::new();
    config.hash_seed.hash(&mut hasher);
    gene_index.hash(&mut hasher);
    depth.hash(&mut hasher);
    parent_hash.hash(&mut hasher);
    let seed = hasher.finish();

    let num_ops = 6;
    let op_type = seed % num_ops;
    let param_seed = seed >> 8;
    
    let mut ops = Vec::new();
    let dim_f64 = dim as f64;
    
    match op_type {
        0 => {
            let axis = (param_seed % dim as u64) as usize;
            let val = hash_to_f64(param_seed >> 10, -1.0, 1.0);
            ops.push(VectorOp::AxisAdd(axis, val));
        }
        1 => {
            let val = hash_to_f64(param_seed, 0.5, 1.5);
            ops.push(VectorOp::Scale(val));
        }
        2 => {
            ops.push(VectorOp::Norm);
        }
        3 => {
            let axis_a = (param_seed % dim as u64) as usize;
            let axis_b = ((param_seed >> 8) % dim as u64) as usize;
            let w = hash_to_f64(param_seed >> 16, 0.0, 1.0);
            ops.push(VectorOp::Mix(axis_a, axis_b, w));
        }
        4 => {
            let axis_a = (param_seed % dim as u64) as usize;
            let axis_b = ((param_seed >> 8) % dim as u64) as usize;
            let ang = hash_to_f64(param_seed >> 16, -180.0, 180.0);
            ops.push(VectorOp::Rotate(axis_a, axis_b, ang));
        }
        5 => {
            let exp = hash_to_f64(param_seed, 0.5, 1.5);
            ops.push(VectorOp::Fractal(exp));
        }
        _ => {}
    }
    ops
}

fn hash_to_f64(h: u64, from: f64, to: f64) -> f64 {
    let t = (h as f64) / (u64::MAX as f64);
    from + (to - from) * t
}

pub fn cosine_similarity(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
    if a.len() != b.len() || a.len() == 0 {
        return 0.0;
    }
    let dot = a.dot(b);
    let norm_a = a.norm();
    let norm_b = b.norm();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

pub fn compile_tree_to_vector(
    tree: &GpTree,
    dim: usize,
    fractal_config: Option<&FractalDecoderConfig>,
) -> DVector<f64> {
    compile_tree_to_vector_with_input(tree, dim, None, fractal_config)
}

/// Like `compile_tree_to_vector` but seeds the working buffer with
/// an optional input vector instead of zeros. Used by A1 SCL PoC:
/// G(T_i) seeds with T_i (truncated/padded to `dim`), applies grammar
/// ops, returns the resulting code c_i; D(c_i) seeds with c_i and
/// renders a reconstruction.
///
/// Truncation / zero-pad semantics:
///   - If `input.len() >= dim`: copy first `dim` components into state.
///   - If `input.len() <  dim`: copy all `input.len()` components,
///     remaining state stays zero.
///   - If `input = None`: state stays zero (classic behavior).
///
/// This is intentionally naive (no PCA / no learned projection) so the
/// A1 PoC stays readable. Higher-fidelity projections live in A3/A4.
pub fn compile_tree_to_vector_with_input(
    tree: &GpTree,
    dim: usize,
    input: Option<&DVector<f64>>,
    fractal_config: Option<&FractalDecoderConfig>,
) -> DVector<f64> {
    let mut out = DVector::zeros(dim);
    if let Some(seed) = input {
        let take = seed.len().min(dim);
        for i in 0..take {
            out[i] = seed[i];
        }
    }
    let mut ops = Vec::new();
    collect_ops(tree, dim, 0, 0, fractal_config, &mut ops);
    
    for op in ops {
        match op {
            VectorOp::AxisAdd(axis, val) => {
                if axis < dim {
                    out[axis] += val;
                }
            }
            VectorOp::Scale(val) => {
                out *= val;
            }
            VectorOp::Norm => {
                let n = out.norm();
                if n > 1e-12 {
                    out /= n;
                }
            }
            VectorOp::Mix(a, b, w) => {
                if a < dim && b < dim && a != b {
                    let w = w.clamp(0.0, 1.0);
                    let va = out[a];
                    let vb = out[b];
                    out[a] = va * (1.0 - w) + vb * w;
                    out[b] = vb * (1.0 - w) + va * w;
                }
            }
            VectorOp::Rotate(a, b, ang) => {
                if a < dim && b < dim && a != b {
                    let rad = ang.to_radians();
                    let (sin, cos) = rad.sin_cos();
                    let x = out[a];
                    let y = out[b];
                    out[a] = x * cos - y * sin;
                    out[b] = x * sin + y * cos;
                }
            }
            VectorOp::Fractal(exp) => {
                if exp.is_finite() {
                    for i in 0..dim {
                        let v = out[i];
                        let sign = if v >= 0.0 { 1.0 } else { -1.0 };
                        out[i] = sign * v.abs().powf(exp);
                    }
                }
            }
            VectorOp::Zero => {
                out.fill(0.0);
            }
            _ => {}
        }
    }
    out
}

fn collect_ops(
    tree: &GpTree,
    dim: usize,
    gene_index: usize,
    parent_hash: u64,
    fractal_config: Option<&FractalDecoderConfig>,
    ops: &mut Vec<VectorOp>
) {
    if let Some(config) = fractal_config {
        if tree.depth() >= config.max_expansion_depth {
            let mut fractal_ops = fractal_expand(gene_index, tree.depth(), parent_hash, config, dim);
            ops.append(&mut fractal_ops);
            return;
        }
    }

    match tree.tree_type() {
        TreeType::Empty => {}
        TreeType::Choice => {
            let choice = match tree.choice() {
                Some(c) => c,
                None => return,
            };
            
            let text = choice.text();
            let tokens: Vec<&str> = text.split_whitespace().collect();
            if let Some(&first) = tokens.first() {
                let is_op = match first {
                    "AX" | "ADD" | "SCALE" | "NORM" | "MIX" | "ROT" | "FRAC" | "ZERO" => true,
                    _ => false,
                };
                
                if is_op {
                    // This is an operation node. Render its full text (including parameters) and parse it.
                    let rendered = tree.text();
                    let r_tokens: Vec<&str> = rendered.split_whitespace().collect();
                    let mut i = 0;
                    while i < r_tokens.len() {
                        match r_tokens[i] {
                            "AX" | "ADD" => {
                                if i + 2 < r_tokens.len() {
                                    if let (Some(axis), Some(val)) = (parse_number(r_tokens[i+1]), parse_number(r_tokens[i+2])) {
                                        ops.push(VectorOp::AxisAdd(axis.round() as usize, val));
                                    }
                                    i += 3;
                                } else { i += 1; }
                            }
                            "SCALE" => {
                                if i + 1 < r_tokens.len() {
                                    if let Some(val) = parse_number(r_tokens[i+1]) {
                                        ops.push(VectorOp::Scale(val));
                                    }
                                    i += 2;
                                } else { i += 1; }
                            }
                            "NORM" => {
                                ops.push(VectorOp::Norm);
                                i += 1;
                            }
                            "MIX" => {
                                if i + 3 < r_tokens.len() {
                                    if let (Some(a), Some(b), Some(w)) = (parse_number(r_tokens[i+1]), parse_number(r_tokens[i+2]), parse_number(r_tokens[i+3])) {
                                        ops.push(VectorOp::Mix(a.round() as usize, b.round() as usize, w));
                                    }
                                    i += 4;
                                } else { i += 1; }
                            }
                            "ROT" => {
                                if i + 3 < r_tokens.len() {
                                    if let (Some(a), Some(b), Some(ang)) = (parse_number(r_tokens[i+1]), parse_number(r_tokens[i+2]), parse_number(r_tokens[i+3])) {
                                        ops.push(VectorOp::Rotate(a.round() as usize, b.round() as usize, ang));
                                    }
                                    i += 4;
                                } else { i += 1; }
                            }
                            "FRAC" => {
                                if i + 1 < r_tokens.len() {
                                    if let Some(exp) = parse_number(r_tokens[i+1]) {
                                        ops.push(VectorOp::Fractal(exp));
                                    }
                                    i += 2;
                                } else { i += 1; }
                            }
                            "ZERO" => {
                                ops.push(VectorOp::Zero);
                                i += 1;
                            }
                            _ => { i += 1; }
                        }
                    }
                    return; // We parsed the operation, no need to recurse into its parameter children
                }
            }

            // Calculate parent hash for children
            let mut hasher = DefaultHasher::new();
            parent_hash.hash(&mut hasher);
            choice.number().hash(&mut hasher);
            let new_hash = hasher.finish();

            for (i, child) in tree.children().iter().enumerate() {
                collect_ops(child, dim, gene_index + i, new_hash, fractal_config, ops);
            }
        }
    }
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
