use crate::payload::Payload;

/// Payload layout policy for the plaintext database before scheme-specific
/// preprocessing. `column_height` is `n` for interpolation and `gamma0` for
/// InsPIRe².
#[derive(Copy, Clone, Debug)]
pub struct DatabasePreprocessingConfig {
    column_height: usize,
}

impl DatabasePreprocessingConfig {
    pub fn new<P: Payload<[u8; 32]>>(column_height: usize) -> Self {
        assert!(column_height > 0, "column height must be non-zero");
        assert!(
            P::EXPONENT <= column_height,
            "a payload ({} digits) must fit within one column (height = {column_height})",
            P::EXPONENT
        );
        Self { column_height }
    }

    pub fn column_height(&self) -> usize {
        self.column_height
    }

    pub fn payloads_per_column<P: Payload<[u8; 32]>>(&self) -> usize {
        self.column_height / P::EXPONENT
    }
}
