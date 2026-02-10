mod codec;
mod error;
mod node;
mod types;

pub use codec::{read_tree, write_tree};
pub use error::Error;
pub use node::Node;
pub use types::{Attr, DataType, Value};

pub(crate) fn normalize_name(name: &[u8]) -> Vec<u8> {
    name.iter().map(|b| b.to_ascii_lowercase()).collect()
}
