//! Error types for M4A parsing and synthesis.

use core::fmt;

/// Errors returned while parsing or synthesizing Nintendo M4A data.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Error {
    /// Input buffer ended before the expected structure was complete.
    UnexpectedEof,
    /// Header or table field contained an invalid or out-of-range value.
    InvalidHeader(&'static str),
    /// Invalid or unsupported `WaveData` sample structure.
    InvalidSample(&'static str),
    /// Invalid track pointer or bytecode stream.
    InvalidTrack(&'static str),
    /// Unrecognized M4A track bytecode opcode.
    InvalidOpcode(u8),
    /// Corrupted M4A stream or inconsistent parameters.
    CorruptedData(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "Unexpected end of data while parsing M4A"),
            Self::InvalidHeader(msg) => write!(f, "Invalid M4A header: {msg}"),
            Self::InvalidSample(msg) => write!(f, "Invalid M4A WaveData sample: {msg}"),
            Self::InvalidTrack(msg) => write!(f, "Invalid M4A track: {msg}"),
            Self::InvalidOpcode(op) => write!(f, "Invalid M4A bytecode opcode: 0x{op:02X}"),
            Self::CorruptedData(msg) => write!(f, "Corrupted M4A data: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Result type alias using [`Error`].
pub type Result<T> = core::result::Result<T, Error>;
