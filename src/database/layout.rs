use std::marker::PhantomData;

use poulpy_core::layouts::ModuleCoreAlloc;
use poulpy_hal::layouts::{Backend, Module};

use crate::payload::Payload;

use super::{address::Address, storage::Database};

/// Shape and addressing math of a coefficient database, shared by both
/// constructions. The layout stores only the coefficient matrix dimensions; the
/// scheme decides how tall one logical column/record is.
pub struct DatabaseLayout<P> {
    rows: usize,
    cols: usize,
    _payload: PhantomData<P>,
}

impl<P> Copy for DatabaseLayout<P> {}
impl<P> Clone for DatabaseLayout<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: Payload<[u8; 32]>> DatabaseLayout<P> {
    /// Raw coefficient matrix dimensions. `rows` and `cols` are coefficient counts,
    /// not scheme-specific batch/block counts.
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "dimensions must be non-zero");
        Self {
            rows,
            cols,
            _payload: PhantomData,
        }
    }

    /// Coefficient rows in the raw database matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Coefficient columns in the raw database matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Block-rows — the second (interpolation) dimension. *(InsPIRe name.)*
    pub fn block_rows(&self, n: usize) -> usize {
        assert_eq!(self.rows % n, 0, "rows must be a multiple of n");
        self.rows / n
    }

    /// Block-cols `columns / n` — the first dimension in `n`-wide blocks. *(InsPIRe.)*
    pub fn block_cols(&self, n: usize) -> usize {
        self.cols.div_ceil(n)
    }

    /// Scheme-derived second dimension for a given logical column height.
    pub fn grid_rows_for(&self, column_height: usize) -> usize {
        assert!(column_height > 0, "column height must be non-zero");
        assert_eq!(
            self.rows % column_height,
            0,
            "rows must be a multiple of the scheme-derived column height"
        );
        self.rows / column_height
    }

    /// Number of `n`-wide chunks needed to cover the first dimension.
    pub fn column_blocks(&self, n: usize) -> usize {
        self.cols.div_ceil(n)
    }

    /// Width of the `block`-th first-dimension chunk.
    pub fn column_block_width(&self, n: usize, block: usize) -> usize {
        assert!(
            block < self.column_blocks(n),
            "column block {block} out of bounds"
        );
        let start = block * n;
        (self.cols - start).min(n)
    }

    /// Plaintext modulus (`P::BASIS`).
    pub fn p(&self) -> u32 {
        P::BASIS
    }

    /// Coefficients spanned by one payload (`P::EXPONENT`).
    pub fn payload_digits(&self) -> usize {
        P::EXPONENT
    }

    /// Payloads stacked down one logical column for the scheme-derived
    /// `column_height`.
    pub fn payloads_per_column(&self, column_height: usize) -> usize {
        assert!(
            P::EXPONENT <= column_height,
            "a payload ({} digits) must fit within one column (height = {column_height})",
            P::EXPONENT
        );
        column_height / P::EXPONENT
    }

    /// Records in the scheme-derived grid.
    pub fn num_records(&self, column_height: usize) -> usize {
        self.grid_rows_for(column_height) * self.cols
    }

    /// Total payloads the database holds for the scheme-derived column height.
    pub fn num_payloads(&self, column_height: usize) -> usize {
        self.num_records(column_height) * self.payloads_per_column(column_height)
    }

    /// Interpolation degree (`next_pow2(block_rows)`), the second-dimension
    /// reduction size. *(InsPIRe.)*
    pub fn interpolation_t(&self, n: usize) -> usize {
        let interpolation_t = self.block_rows(n).next_power_of_two();
        assert!(
            interpolation_t <= 2 * n,
            "second dimension (interpolation degree {interpolation_t}) exceeds 2n = {}",
            2 * n
        );
        interpolation_t
    }

    /// Payloads held by one second-dimension band (`payloads_per_column · cols`).
    pub fn payloads_per_block_row(&self, column_height: usize) -> usize {
        self.payloads_per_column(column_height) * self.cols
    }

    /// Total `i16` coefficient slots of the InsPIRe `n × n` tiling.
    pub fn total_i16_slots(&self) -> usize {
        self.rows * self.cols
    }

    /// Total payload capacity in bytes.
    pub fn total_payload_bytes(&self, column_height: usize) -> usize {
        self.num_payloads(column_height) * size_of::<[u8; 32]>()
    }

    /// The `(rows_bytes, cols_bytes)` byte-matrix shape (`cols_bytes = columns`).
    pub fn byte_matrix_shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Resolve payload index `i` to the three logical coordinates shared by both
    /// constructions. `matrix` is the second-dimension index (interpolation block
    /// row / InsPIRe² batch), `column` is the first-dimension index, and
    /// `row_offset` is the payload's coefficient offset within the returned
    /// `column_height`-digit record.
    pub fn address_for(&self, i: usize, column_height: usize) -> Address {
        assert!(
            i < self.num_payloads(column_height),
            "payload {i} out of bounds (num_payloads {})",
            self.num_payloads(column_height)
        );
        let ppc = self.payloads_per_column(column_height);
        let per_row = ppc * self.cols;
        let row = i / per_row;
        let e_local = i % per_row;
        let column = e_local % self.cols;
        let payload_in_column = e_local / self.cols;
        Address {
            matrix: row,
            column,
            row_offset: payload_in_column * P::EXPONENT,
        }
    }

    /// Compatibility wrapper for older call sites that passed `n`; address
    /// derivation itself only needs the scheme-derived `column_height`.
    pub fn address(&self, i: usize, _n: usize, column_height: usize) -> Address {
        self.address_for(i, column_height)
    }

    /// Allocate an empty [`Database`] matching this layout. `base2k` (from
    /// the cryptosystem `Parameters`) sizes the coefficient storage only.
    pub fn instantiate<BE: Backend<OwnedBuf = Vec<u8>>>(
        &self,
        module: &Module<BE>,
        base2k: usize,
        column_height: usize,
    ) -> Database<BE, P>
    where
        Module<BE>: ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
    {
        Database::from_layout(module, *self, base2k, column_height)
    }
}
