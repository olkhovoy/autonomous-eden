use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    InvalidHeader,
    InvalidFormat(&'static str),
    InvalidNameId(i32),
    NameTooLarge(u32),
    LengthOverflow,
    DataLengthMismatch { expected: usize, actual: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(err) => write!(f, "io error: {err}"),
            Error::InvalidHeader => write!(f, "invalid file header"),
            Error::InvalidFormat(msg) => write!(f, "invalid format: {msg}"),
            Error::InvalidNameId(id) => write!(f, "invalid name id: {id}"),
            Error::NameTooLarge(len) => write!(f, "name too large: {len}"),
            Error::LengthOverflow => write!(f, "length overflow"),
            Error::DataLengthMismatch { expected, actual } => {
                write!(f, "data length mismatch (expected {expected}, got {actual})")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}
