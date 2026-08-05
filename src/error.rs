/// Error type for fallible, production-facing PIR APIs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PirError {
    /// A database layout shape is invalid.
    InvalidLayout { reason: &'static str },
    /// A cryptosystem/application configuration is invalid.
    InvalidConfig { reason: &'static str },
    /// A requested payload index is outside the configured database capacity.
    PayloadOutOfBounds { index: usize, capacity: usize },
    /// A shard length overflowed `usize` while checking capacity.
    ShardLengthOverflow,
    /// A shard write would exceed the configured database capacity.
    ShardOutOfBounds { end: usize, capacity: usize },
    /// A digit run would exceed the selected logical record height.
    DigitRunOutOfBounds {
        offset: usize,
        len: usize,
        column_height: usize,
    },
    /// A query variant does not match the server/config collapse.
    WrongQueryVariant {
        expected: &'static str,
        actual: &'static str,
    },
    /// A response variant does not match the client/config collapse.
    WrongResponseVariant {
        expected: &'static str,
        actual: &'static str,
    },
}

pub type Result<T> = std::result::Result<T, PirError>;

impl std::fmt::Display for PirError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLayout { reason } => write!(f, "invalid database layout: {reason}"),
            Self::InvalidConfig { reason } => write!(f, "invalid PIR config: {reason}"),
            Self::PayloadOutOfBounds { index, capacity } => {
                write!(f, "payload {index} out of bounds (capacity {capacity})")
            }
            Self::ShardLengthOverflow => write!(f, "shard length overflow"),
            Self::ShardOutOfBounds { end, capacity } => {
                write!(f, "shard ends at {end}, past capacity {capacity}")
            }
            Self::DigitRunOutOfBounds {
                offset,
                len,
                column_height,
            } => write!(
                f,
                "digit run offset {offset} + length {len} exceeds column height {column_height}"
            ),
            Self::WrongQueryVariant { expected, actual } => {
                write!(f, "query variant {actual} does not match server {expected}")
            }
            Self::WrongResponseVariant { expected, actual } => write!(
                f,
                "response variant {actual} does not match client {expected}"
            ),
        }
    }
}

impl std::error::Error for PirError {}
