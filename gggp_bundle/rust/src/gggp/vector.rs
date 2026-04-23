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
    execute_ops(&ops, &mut out, dim, input);
    out
}

/// Apply a flat sequence of `VectorOp`s to `state` in-place.
///
/// Separated from `compile_tree_to_vector_with_input` for two reasons:
///   1. Keeps tree-walk (collect_ops) and op-semantics (this fn) decoupled.
///   2. Lets unit tests exercise op semantics without building GpTree mocks.
///
/// Ops with out-of-range axis / code_idx, non-finite params, or a missing
/// `input` buffer (for NC3 code-gated ops) degrade to no-ops: EA runs are
/// expected to produce malformed programs, and silent no-op keeps fitness
/// well-defined. Grammar bounds must still constrain axis/code_idx to
/// valid ranges to avoid wasting evaluation budget.
pub(crate) fn execute_ops(
    ops: &[VectorOp],
    state: &mut DVector<f64>,
    dim: usize,
    input: Option<&DVector<f64>>,
) {
    for op in ops {
        match *op {
            VectorOp::AxisAdd(axis, val) => {
                if axis < dim {
                    state[axis] += val;
                }
            }
            VectorOp::Scale(val) => {
                *state *= val;
            }
            VectorOp::Norm => {
                let n = state.norm();
                if n > 1e-12 {
                    *state /= n;
                }
            }
            VectorOp::Mix(a, b, w) => {
                if a < dim && b < dim && a != b {
                    let w = w.clamp(0.0, 1.0);
                    let va = state[a];
                    let vb = state[b];
                    state[a] = va * (1.0 - w) + vb * w;
                    state[b] = vb * (1.0 - w) + va * w;
                }
            }
            VectorOp::Rotate(a, b, ang) => {
                if a < dim && b < dim && a != b {
                    let rad = ang.to_radians();
                    let (sin, cos) = rad.sin_cos();
                    let x = state[a];
                    let y = state[b];
                    state[a] = x * cos - y * sin;
                    state[b] = x * sin + y * cos;
                }
            }
            VectorOp::Fractal(exp) => {
                if exp.is_finite() {
                    for i in 0..dim {
                        let v = state[i];
                        let sign = if v >= 0.0 { 1.0 } else { -1.0 };
                        state[i] = sign * v.abs().powf(exp);
                    }
                }
            }
            VectorOp::Zero => {
                state.fill(0.0);
            }
            VectorOp::Ctrl(axis, cidx) => {
                if axis < dim {
                    if let Some(c) = input {
                        if cidx < c.len() && c[cidx].is_finite() {
                            state[axis] += c[cidx];
                        }
                    }
                }
            }
            VectorOp::ScaleByCode(cidx) => {
                if let Some(c) = input {
                    if cidx < c.len() {
                        let s = c[cidx];
                        if s.is_finite() {
                            *state *= s;
                        }
                    }
                }
            }
            VectorOp::AddCode(axis, cidx) => {
                if axis < dim {
                    if let Some(c) = input {
                        if cidx < c.len() {
                            let k = c[cidx];
                            if k.is_finite() {
                                state[axis] += k * state[axis];
                            }
                        }
                    }
                }
            }
            VectorOp::Add | VectorOp::Subtract | VectorOp::Cross | VectorOp::Seq => {
                // Not implemented in the scalar-axis dispatch path.
                // Intentional no-op: these enum variants exist for the
                // higher-level vector-symbolic compositor, not the flat
                // axis-grammar used by A1/A2.
            }
        }
    }
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
                    "AX" | "ADD" | "SCALE" | "NORM" | "MIX" | "ROT" | "FRAC" | "ZERO"
                    | "CTRL" | "SBC" | "ADDC" => true,
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
                            // --- NC3 downward-causation tokens (A2 S1a) ---
                            // Token forms:
                            //   CTRL <axis> <code_idx>
                            //   SBC  <code_idx>
                            //   ADDC <axis> <code_idx>
                            "CTRL" => {
                                if i + 2 < r_tokens.len() {
                                    if let (Some(ax), Some(ci)) = (parse_number(r_tokens[i+1]), parse_number(r_tokens[i+2])) {
                                        ops.push(VectorOp::Ctrl(ax.round() as usize, ci.round() as usize));
                                    }
                                    i += 3;
                                } else { i += 1; }
                            }
                            "SBC" => {
                                if i + 1 < r_tokens.len() {
                                    if let Some(ci) = parse_number(r_tokens[i+1]) {
                                        ops.push(VectorOp::ScaleByCode(ci.round() as usize));
                                    }
                                    i += 2;
                                } else { i += 1; }
                            }
                            "ADDC" => {
                                if i + 2 < r_tokens.len() {
                                    if let (Some(ax), Some(ci)) = (parse_number(r_tokens[i+1]), parse_number(r_tokens[i+2])) {
                                        ops.push(VectorOp::AddCode(ax.round() as usize, ci.round() as usize));
                                    }
                                    i += 3;
                                } else { i += 1; }
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

// ============================================================================
// Unit tests for S1a (NC3 downward-causation ops) and op-execution semantics.
// Exercises execute_ops directly so the parser/tree-walk layer is not on the
// critical test path.
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn v(slice: &[f64]) -> DVector<f64> {
        DVector::from_row_slice(slice)
    }

    // --- NC4 carry-over ops (sanity regression) -----------------------------

    #[test]
    fn axis_add_shifts_single_component() {
        let mut s = DVector::<f64>::zeros(4);
        execute_ops(&[VectorOp::AxisAdd(2, 0.75)], &mut s, 4, None);
        assert_eq!(s, v(&[0.0, 0.0, 0.75, 0.0]));
    }

    #[test]
    fn axis_add_out_of_range_is_noop() {
        let mut s = v(&[1.0, 2.0]);
        execute_ops(&[VectorOp::AxisAdd(5, 9.9)], &mut s, 2, None);
        assert_eq!(s, v(&[1.0, 2.0]));
    }

    #[test]
    fn scale_multiplies_all_components() {
        let mut s = v(&[1.0, -2.0, 3.0]);
        execute_ops(&[VectorOp::Scale(2.0)], &mut s, 3, None);
        assert_eq!(s, v(&[2.0, -4.0, 6.0]));
    }

    // --- NC3 CTRL ------------------------------------------------------------

    #[test]
    fn ctrl_adds_code_component_to_target_axis() {
        let mut s = v(&[0.0, 0.0, 0.0, 0.0]);
        let code = v(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]); // code_dim=8
        execute_ops(&[VectorOp::Ctrl(2, 5)], &mut s, 4, Some(&code));
        //           state[2] += code[5] == 0.6
        assert!((s[2] - 0.6).abs() < 1e-12);
        assert_eq!(s[0], 0.0);
        assert_eq!(s[1], 0.0);
        assert_eq!(s[3], 0.0);
    }

    #[test]
    fn ctrl_accumulates_across_invocations() {
        let mut s = v(&[1.0, 0.0]);
        let code = v(&[0.5, -0.25]);
        execute_ops(
            &[VectorOp::Ctrl(0, 0), VectorOp::Ctrl(0, 1)],
            &mut s,
            2,
            Some(&code),
        );
        // 1.0 + 0.5 + (-0.25) == 1.25
        assert!((s[0] - 1.25).abs() < 1e-12);
    }

    #[test]
    fn ctrl_is_noop_without_input() {
        let mut s = v(&[2.0, 3.0]);
        execute_ops(&[VectorOp::Ctrl(0, 0)], &mut s, 2, None);
        assert_eq!(s, v(&[2.0, 3.0]));
    }

    #[test]
    fn ctrl_is_noop_when_code_idx_out_of_range() {
        let mut s = v(&[2.0]);
        let code = v(&[9.9, 8.8]);
        execute_ops(&[VectorOp::Ctrl(0, 42)], &mut s, 1, Some(&code));
        assert_eq!(s, v(&[2.0]));
    }

    #[test]
    fn ctrl_is_noop_when_axis_out_of_range() {
        let mut s = v(&[2.0]);
        let code = v(&[9.9]);
        execute_ops(&[VectorOp::Ctrl(5, 0)], &mut s, 1, Some(&code));
        assert_eq!(s, v(&[2.0]));
    }

    // --- NC3 SCALE_BY_CODE ---------------------------------------------------

    #[test]
    fn sbc_broadcasts_scalar_from_code() {
        let mut s = v(&[1.0, -2.0, 0.5]);
        let code = v(&[0.0, 3.0, 0.0]);
        execute_ops(&[VectorOp::ScaleByCode(1)], &mut s, 3, Some(&code));
        // all scaled by code[1] == 3
        assert_eq!(s, v(&[3.0, -6.0, 1.5]));
    }

    #[test]
    fn sbc_is_noop_without_input() {
        let mut s = v(&[1.0, 2.0]);
        execute_ops(&[VectorOp::ScaleByCode(0)], &mut s, 2, None);
        assert_eq!(s, v(&[1.0, 2.0]));
    }

    #[test]
    fn sbc_skips_non_finite_code() {
        let mut s = v(&[1.0, 2.0]);
        let code = v(&[f64::NAN, 7.0]);
        execute_ops(&[VectorOp::ScaleByCode(0)], &mut s, 2, Some(&code));
        assert_eq!(s, v(&[1.0, 2.0]));
    }

    // --- NC3 ADD_CODE (multiplicative gating) --------------------------------

    #[test]
    fn add_code_multiplicatively_gates_axis() {
        // state[axis] += code[k] * state[axis]  ==>  state[axis] *= (1 + code[k])
        let mut s = v(&[4.0, 0.0]);
        let code = v(&[0.25, 0.0]);
        execute_ops(&[VectorOp::AddCode(0, 0)], &mut s, 2, Some(&code));
        // 4 + 0.25*4 == 5
        assert!((s[0] - 5.0).abs() < 1e-12);
        assert_eq!(s[1], 0.0);
    }

    #[test]
    fn add_code_is_noop_on_zero_axis() {
        // state[axis]=0 => 0 + k*0 = 0 regardless of k
        let mut s = v(&[0.0, 0.0]);
        let code = v(&[999.0]);
        execute_ops(&[VectorOp::AddCode(0, 0)], &mut s, 2, Some(&code));
        assert_eq!(s, v(&[0.0, 0.0]));
    }

    #[test]
    fn add_code_is_noop_without_input() {
        let mut s = v(&[5.0]);
        execute_ops(&[VectorOp::AddCode(0, 0)], &mut s, 1, None);
        assert_eq!(s, v(&[5.0]));
    }

    // --- NC3 structural signal: functional dependence on `input` -------------

    #[test]
    fn nc3_programs_are_functionally_code_dependent() {
        // Sanity for the F_nc3 metric rationale: running the SAME ops with
        // two different `input` buffers must produce two different states.
        // If this ever fails, NC3 ops would be a simulacrum.
        let ops = vec![
            VectorOp::Ctrl(0, 0),
            VectorOp::ScaleByCode(1),
            VectorOp::AddCode(1, 2),
        ];
        let c1 = v(&[1.0, 2.0, 0.5]);
        let c2 = v(&[-1.0, 0.5, -0.25]);

        let mut s1 = v(&[0.1, 0.1]);
        let mut s2 = v(&[0.1, 0.1]);
        execute_ops(&ops, &mut s1, 2, Some(&c1));
        execute_ops(&ops, &mut s2, 2, Some(&c2));
        assert_ne!(s1, s2, "NC3 ops must make state functionally depend on code");
    }

    // --- parser smoke: token forms map to the right VectorOp variants --------
    // Full tree-walk tests live at integration level; here we only verify
    // parse_number covers the numeric tails our tokens rely on.

    #[test]
    fn parse_number_accepts_common_numeric_forms() {
        assert_eq!(parse_number("0"), Some(0.0));
        assert_eq!(parse_number("7"), Some(7.0));
        assert_eq!(parse_number("-3.5"), Some(-3.5));
        assert_eq!(parse_number("1e2"), Some(100.0));
        assert_eq!(parse_number("abc"), None);
    }
}
