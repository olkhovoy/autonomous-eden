use nalgebra::DVector;

/// The physical manifestation of the Chromosome.
/// In Legacy Mode: This is Source Code (String).
/// In God Mode: This is a Latent Vector (Tensor).
pub trait Phenotype {
    fn render(&self) -> OutputType;
    fn dimension(&self) -> usize;
}

#[derive(Debug, Clone)]
pub enum OutputType {
    /// Classical text generation (current implementation)
    SourceCode(String),
    /// The multidimensional embedding
    Embedding(DVector<f64>),
    /// A fractal visual glyph (for your visual demo extension)
    GlyphMap(Vec<Vec<u8>>),
}

pub trait VectorPhenotype {
    fn render_vector(&self, dim: usize) -> DVector<f64>;
    fn to_vector_symbol(&self, gene_index: usize, depth: i32, parent_hash: u64) -> Option<VectorSymbol>;
}

/// A node in the vector space.
/// Instead of a string literal, a grammar node holds a semantic "direction".
#[derive(Debug, Clone)]
pub struct VectorSymbol {
    pub axis_weights: DVector<f64>,
    pub operation: VectorOp,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VectorOp {
    Add,                       // Compose concepts (v1 + v2)
    Subtract,                  // Remove concepts (v1 - v2)
    Cross,                     // Orthogonalize
    Fractal(f64),              // Recursive scaling: sign(v) * |v|^exp
    Scale(f64),                // v * scalar
    Norm,                      // v / ||v||
    Mix(usize, usize, f64),    // (1-w)*v1 + w*v2 on two axes
    Rotate(usize, usize, f64), // 2D rotation in axis plane
    AxisAdd(usize, f64),       // Add scalar to axis: v[axis] += val
    Seq,                       // Sequence of ops
    Zero,                      // Reset
    // --- NC3 downward-causation ops (A2: code-gated, index `input` buffer) ---
    // CTRL(axis, code_idx):     state[axis] += input[code_idx]
    // ScaleByCode(code_idx):    state *= input[code_idx]                    (scalar broadcast)
    // AddCode(axis, code_idx):  state[axis] += input[code_idx] * state[axis] (multiplicative gating)
    // All three are no-ops if `input` is None or indices are out of range.
    Ctrl(usize, usize),
    ScaleByCode(usize),
    AddCode(usize, usize),
}
