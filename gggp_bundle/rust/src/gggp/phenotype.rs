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

/// A node in the vector space.
/// Instead of a string literal, a grammar node holds a semantic "direction".
#[derive(Debug, Clone)]
pub struct VectorSymbol {
    pub axis_weights: DVector<f64>,
    pub operation: VectorOp,
}

#[derive(Debug, Clone)]
pub enum VectorOp {
    Add,       // Compose concepts (King + Man)
    Subtract,  // Remove concepts (King - Man)
    Cross,     // Orthogonalize (Create new dimension)
    Fractal,   // Recursive scaling (Zoom in)
}
