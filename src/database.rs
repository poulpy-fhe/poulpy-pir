//! The database side of the toy PIR.
//!
//! The database is an `a × b` coefficient matrix (`rows × cols`, `cols =
//! block_cols · n`) split into `n × n` blocks, giving a `block_rows × block_cols`
//! grid. The **block-rows** axis (`a / n`) is the PIR second dimension, reduced
//! by interpolation; the **block-cols** axis (`b / n`) is the first dimension.
//!
//! Payloads are generic via [`Payload`]: a payload (here `[u8; 32]`) packs into
//! `P::EXPONENT` consecutive coefficient rows of one column, each a centred-`i16`
//! base-`P::BASIS` digit (see [`Database::encode_shard`]). For the working
//! example `P = U256P65535`: 256-bit payloads, `BASIS = 65535`, `EXPONENT = 17`
//! (the full `2^256` range, unlike a 16-digit bound).
//!
//! [`DatabaseInfos`] is the shared shape / addressing math (implemented by
//! [`DatabaseLayout`]); [`Database`] holds the `i16` matrices. `base2k` is a
//! coefficient-storage detail supplied at [`DatabaseLayout::instantiate`].

use std::marker::PhantomData;

use poulpy_core::layouts::{Base2K, CoeffMatrix, ModuleCoreAlloc, TorusPrecision};
use poulpy_hal::layouts::{Backend, Module, ZnxView, ZnxViewMut};

use crate::payload::Payload;

/// Shape and addressing math of a tiled database. The implementor supplies the
/// primitives (`n`, `block_rows`, `block_cols`, `p`, `payload_digits`,
/// `total_payload_bytes`); everything else derives.
pub trait DatabaseInfos {
    /// Ring degree / `n × n` block size.
    fn n(&self) -> usize;

    /// Block-rows `a / n` — the second (interpolation) dimension.
    fn block_rows(&self) -> usize;

    /// Block-cols `b / n` — the first dimension.
    fn block_cols(&self) -> usize;

    /// Plaintext modulus (`P::BASIS`).
    fn p(&self) -> u16;

    /// Coefficients spanned by one payload (`P::EXPONENT`).
    fn payload_digits(&self) -> usize;

    /// Total payload capacity in bytes.
    fn total_payload_bytes(&self) -> usize;

    /// Coefficient rows `a` (`block_rows · n`).
    fn rows(&self) -> usize {
        self.n() * self.block_rows()
    }

    /// Coefficient columns `b` (`block_cols · n`).
    fn cols(&self) -> usize {
        self.n() * self.block_cols()
    }

    /// Payloads stacked down one column (`n / payload_digits`).
    fn payloads_per_column(&self) -> usize {
        self.n() / self.payload_digits()
    }

    /// Payloads held by one block-row band (`payloads_per_column · cols`).
    fn payloads_per_block_row(&self) -> usize {
        self.payloads_per_column() * self.cols()
    }

    /// Total payloads the database holds (capacity).
    fn num_payloads(&self) -> usize {
        self.block_rows() * self.payloads_per_block_row()
    }

    /// Interpolation degree (`next_pow2(block_rows)`), the second-dimension reduction size.
    fn interpolation_t(&self) -> usize {
        let interpolation_t = self.block_rows().next_power_of_two();
        assert!(
            interpolation_t <= 2 * self.n(),
            "second dimension (interpolation degree {interpolation_t}) exceeds 2n = {}",
            2 * self.n()
        );
        interpolation_t
    }

    /// Total `i16` coefficient slots (`rows · cols`).
    fn total_i16_slots(&self) -> usize {
        self.rows() * self.cols()
    }

    /// The `(rows_bytes, cols_bytes)` byte-matrix shape (`cols_bytes = cols`).
    fn byte_matrix_shape(&self) -> (usize, usize) {
        (self.total_payload_bytes() / self.cols(), self.cols())
    }

    /// Resolve payload index `i` to the coordinate that retrieves it.
    fn address(&self, i: usize) -> PayloadAddress {
        assert!(
            i < self.num_payloads(),
            "payload {i} out of bounds (num_payloads {})",
            self.num_payloads()
        );
        let payloads_per_block_row = self.payloads_per_block_row();
        let e_local = i % payloads_per_block_row;
        let c = e_local % self.cols();
        PayloadAddress {
            matrix: i / payloads_per_block_row,
            block_col: c / self.n(),
            col_in_block: c % self.n(),
            row_offset: (e_local / self.cols()) * self.payload_digits(),
            block_cols: self.block_cols(),
            interpolation_t: self.interpolation_t(),
            digits: self.payload_digits(),
        }
    }
}

/// Pure shape of the database, parameterized by the payload encoding `P`. The
/// block grid (`block_rows × block_cols` of `n × n` blocks) is the input; the
/// coefficient dims and capacity derive (via [`DatabaseInfos`]).
pub struct DatabaseLayout<P> {
    n: usize,
    block_rows: usize,
    block_cols: usize,
    _payload: PhantomData<P>,
}

impl<P> Copy for DatabaseLayout<P> {}
impl<P> Clone for DatabaseLayout<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: Payload<[u8; 32]>> DatabaseInfos for DatabaseLayout<P> {
    fn n(&self) -> usize {
        self.n
    }

    fn block_rows(&self) -> usize {
        self.block_rows
    }

    fn block_cols(&self) -> usize {
        self.block_cols
    }

    fn p(&self) -> u16 {
        P::BASIS
    }

    fn payload_digits(&self) -> usize {
        P::EXPONENT
    }

    fn total_payload_bytes(&self) -> usize {
        self.num_payloads() * size_of::<[u8; 32]>()
    }
}

impl<P: Payload<[u8; 32]>> DatabaseLayout<P> {
    /// Layout over a `block_rows × block_cols` grid of `n × n` blocks.
    pub fn new(n: usize, block_rows: usize, block_cols: usize) -> Self {
        assert!(
            n > 0 && block_rows > 0 && block_cols > 0,
            "dimensions must be non-zero"
        );
        assert!(
            P::EXPONENT <= n,
            "a payload ({} digits) must fit within one column (n = {n})",
            P::EXPONENT
        );
        Self {
            n,
            block_rows,
            block_cols,
            _payload: PhantomData,
        }
    }

    /// Layout whose block-rows are sized to hold at least `min_payloads` payloads
    /// over a `block_cols · n`-wide matrix.
    pub fn with_capacity(n: usize, block_cols: usize, min_payloads: usize) -> Self {
        let per_block_row = (n / P::EXPONENT) * (block_cols * n);
        let block_rows = min_payloads.div_ceil(per_block_row.max(1)).max(1);
        Self::new(n, block_rows, block_cols)
    }

    /// Layout sized to hold at least `total_bytes` worth of payloads.
    pub fn from_total_bytes(n: usize, block_cols: usize, total_bytes: usize) -> Self {
        Self::with_capacity(n, block_cols, total_bytes / size_of::<[u8; 32]>())
    }

    /// Allocate an empty [`Database`] matching this layout. `base2k` (from the
    /// cryptosystem `Parameters`) sizes the coefficient storage only.
    pub fn instantiate<BE: Backend<OwnedBuf = Vec<u8>>>(
        &self,
        module: &Module<BE>,
        base2k: usize,
    ) -> Database<BE, P>
    where
        Module<BE>: ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
    {
        Database::new(module, self.total_i16_slots(), base2k, self.cols())
    }
}

/// The coordinate of one payload in the tiled database: which matrix (second
/// dimension), which column (first dimension, as `block_col`/`col_in_block`),
/// and the `digits`-long run of coefficient rows starting at `row_offset`.
#[derive(Copy, Clone, Debug)]
pub struct PayloadAddress {
    pub matrix: usize,
    pub block_col: usize,
    pub col_in_block: usize,
    pub row_offset: usize,
    pub block_cols: usize,
    pub interpolation_t: usize,
    pub digits: usize,
}

/// The raw database: `block_rows · block_cols` `n x n` `i16` coefficient
/// sub-matrices, ordered `matrices[matrix · block_cols + block]`.
pub struct Database<BE: Backend, P> {
    matrices: Vec<CoeffMatrix<BE::OwnedBuf, i16>>,
    n: usize,
    base2k: usize,
    cols: usize,
    _payload: PhantomData<P>,
}

impl<BE: Backend, P: Payload<[u8; 32]>> Database<BE, P> {
    /// The flat list of `n x n` sub-matrices (`matrix · block_cols + block`).
    pub fn matrices(&self) -> &[CoeffMatrix<BE::OwnedBuf, i16>] {
        &self.matrices
    }

    /// Mutable view of the sub-matrices (used by the in-place interpolation).
    pub fn matrices_mut(&mut self) -> &mut [CoeffMatrix<BE::OwnedBuf, i16>] {
        &mut self.matrices
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn base2k(&self) -> usize {
        self.base2k
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Block-cols `cols / n` — the first dimension.
    pub fn block_cols(&self) -> usize {
        self.cols / self.n
    }

    /// Block-rows — the second (interpolation) dimension.
    pub fn block_rows(&self) -> usize {
        self.matrices.len() / self.block_cols()
    }

    /// Number of payloads the database can hold
    /// (`block_rows · (n / P::EXPONENT) · cols`).
    pub fn payload_capacity(&self) -> usize {
        self.block_rows() * (self.n / P::EXPONENT) * self.cols
    }
}

impl<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload<[u8; 32]>> Database<BE, P>
where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
{
    /// Allocate a zeroed database holding `db_entries = block_rows · n · cols`
    /// coefficient slots, tiled into `n x n` sub-matrices at `base2k`.
    pub fn new(module: &Module<BE>, db_entries: usize, base2k: usize, cols: usize) -> Self {
        let n = module.n();
        assert!(cols % n == 0, "cols must be a multiple of n");
        let per_matrix = n * cols;
        assert!(
            db_entries % per_matrix == 0,
            "db_entries must be a multiple of n·cols"
        );
        let blocks = cols / n;
        let matrices = (0..(db_entries / per_matrix) * blocks)
            .map(|_| {
                module.coeff_matrix_alloc::<i16>(
                    n,
                    n,
                    Base2K(base2k as u32),
                    TorusPrecision(base2k as u32),
                )
            })
            .collect();
        Self {
            matrices,
            n,
            base2k,
            cols,
            _payload: PhantomData,
        }
    }

    /// Encode `payloads` values starting at payload index `start`, each as
    /// `P::EXPONENT` base-`P::BASIS` digits down consecutive rows of one column.
    pub fn encode_shard(&mut self, start: usize, payloads: &[[u8; 32]]) {
        let capacity = self.payload_capacity();
        assert!(
            start + payloads.len() <= capacity,
            "shard writes past the configured capacity ({capacity})"
        );
        let digits_per = P::EXPONENT;
        let payloads_per_matrix = (self.n / digits_per) * self.cols;
        let blocks = self.block_cols();
        let mut digits = vec![0i16; digits_per];
        for (i, &payload) in payloads.iter().enumerate() {
            let e = start + i;
            let e_local = e % payloads_per_matrix;
            let c = e_local % self.cols;
            let row_out_start = (e_local / self.cols) * digits_per;
            let sub = &mut self.matrices[(e / payloads_per_matrix) * blocks + c / self.n];
            let row_in = c % self.n;
            P::encode(&mut digits, payload);
            for (k, &d) in digits.iter().enumerate() {
                sub.data_mut().at_mut(row_out_start + k, 0)[row_in] = d as i64;
            }
        }
    }
}

impl<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload<[u8; 32]>> Database<BE, P> {
    /// Read back the plaintext payload stored at index `i` (the decode inverse of
    /// [`encode_shard`](Self::encode_shard)). The server owns the plaintext DB, so
    /// this is the ground-truth oracle for the value a PIR query should return.
    pub fn payload(&self, i: usize) -> [u8; 32] {
        let digits_per = P::EXPONENT;
        let payloads_per_matrix = (self.n / digits_per) * self.cols;
        let blocks = self.block_cols();
        let e_local = i % payloads_per_matrix;
        let c = e_local % self.cols;
        let row_out_start = (e_local / self.cols) * digits_per;
        let sub = &self.matrices[(i / payloads_per_matrix) * blocks + c / self.n];
        let row_in = c % self.n;
        let digits: Vec<i16> = (0..digits_per)
            .map(|k| sub.data().at(row_out_start + k, 0)[row_in] as i16)
            .collect();
        let mut out = [0u8; 32];
        P::decode(&mut out, &digits);
        out
    }
}
