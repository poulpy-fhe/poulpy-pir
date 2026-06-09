use std::marker::PhantomData;

use poulpy_core::layouts::ModuleCoreAlloc;
use poulpy_hal::layouts::{Backend, Module};

use crate::payload::Payload;

use super::{
    CoeffMatrix, address::Address, layout::DatabaseLayout,
    preprocessing::DatabasePreprocessingConfig,
};

/// The raw PIR database: `physical_rows · block_cols` `n x n` `i16`
/// coefficient sub-matrices, ordered `matrices[row_group · block_cols + block]`.
/// InsPIRe² keeps `gamma0` logical records inside each physical `n`-row group,
/// so the storage layout stays shared with interpolation.
pub struct Database<BE: Backend, P> {
    matrices: Vec<CoeffMatrix>,
    n: usize,
    base2k: usize,
    cols: usize,
    grid_rows: usize,
    physical_rows: usize,
    preprocessing: DatabasePreprocessingConfig,
    _marker: PhantomData<(BE, P)>,
}

impl<BE: Backend, P: Payload<[u8; 32]>> Database<BE, P> {
    /// The flat list of `n x n` sub-matrices (`matrix · block_cols + block`).
    pub fn matrices(&self) -> &[CoeffMatrix] {
        &self.matrices
    }

    /// Mutable view of the sub-matrices (used by the in-place interpolation).
    pub fn matrices_mut(&mut self) -> &mut [CoeffMatrix] {
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

    /// Number of `n`-wide chunks needed to cover the first dimension.
    pub fn column_blocks(&self) -> usize {
        self.cols.div_ceil(self.n)
    }

    /// Width of the `block`-th first-dimension chunk.
    pub fn column_block_width(&self, block: usize) -> usize {
        assert!(
            block < self.column_blocks(),
            "column block {block} out of bounds"
        );
        let start = block * self.n;
        (self.cols - start).min(self.n)
    }

    /// Coefficients used by each logical column (`n` for interpolation,
    /// `γ0` for InsPIRe²).
    pub fn column_height(&self) -> usize {
        self.preprocessing.column_height()
    }

    /// Payload layout policy used by [`encode_shard`](Self::encode_shard) and
    /// [`payload`](Self::payload).
    pub fn preprocessing(&self) -> DatabasePreprocessingConfig {
        self.preprocessing
    }

    /// Block-cols `cols / n` — the first dimension.
    pub fn block_cols(&self) -> usize {
        assert_eq!(
            self.column_height(),
            self.n,
            "block_cols is interpolation-only"
        );
        self.cols / self.n
    }

    /// Block-rows — the second (interpolation) dimension.
    pub fn block_rows(&self) -> usize {
        assert_eq!(
            self.column_height(),
            self.n,
            "block_rows is interpolation-only"
        );
        self.grid_rows
    }

    /// InsPIRe² batches — the shared second-dimension row count.
    pub fn t(&self) -> usize {
        self.grid_rows
    }

    /// InsPIRe² record size (`γ0`) in `Z_p` digits.
    pub fn gamma0(&self) -> usize {
        self.column_height()
    }

    /// Number of logical grid rows packed into one physical `n x n` row group.
    pub fn rows_per_physical_group(&self) -> usize {
        self.n / self.column_height()
    }

    /// Number of physical `n`-row groups in the stored database.
    pub fn physical_rows(&self) -> usize {
        self.physical_rows
    }

    /// The raw coefficient blocks.
    pub fn blocks(&self) -> &[CoeffMatrix] {
        &self.matrices
    }

    /// The physical coefficient block for `grid_row` and first-dimension
    /// `column_block`.
    pub fn block(&self, grid_row: usize, column_block: usize) -> &CoeffMatrix {
        assert!(
            grid_row < self.grid_rows,
            "grid row {grid_row} out of bounds"
        );
        assert!(
            column_block < self.column_blocks(),
            "column block {column_block} out of bounds"
        );
        let row_group = grid_row / self.rows_per_physical_group();
        &self.matrices[row_group * self.column_blocks() + column_block]
    }

    /// The physical `n x n` block for one row group and column block.
    pub fn physical_block(&self, row_group: usize, column_block: usize) -> &CoeffMatrix {
        assert!(
            row_group < self.physical_rows,
            "physical row group {row_group} out of bounds"
        );
        assert!(
            column_block < self.column_blocks(),
            "column block {column_block} out of bounds"
        );
        &self.matrices[row_group * self.column_blocks() + column_block]
    }

    /// Total logical records (`grid_rows · cols`).
    pub fn num_records(&self) -> usize {
        self.grid_rows * self.cols
    }

    /// Number of payloads the database can hold
    /// (`grid_rows · (column_height / P::EXPONENT) · cols`).
    pub fn payload_capacity(&self) -> usize {
        self.grid_rows * self.payloads_per_grid_row()
    }

    fn payloads_per_grid_row(&self) -> usize {
        self.preprocessing.payloads_per_column::<P>() * self.cols
    }

    /// Resolve payload index `i` using this database's preprocessing layout.
    pub fn payload_address(&self, i: usize) -> Address {
        let capacity = self.payload_capacity();
        assert!(
            i < capacity,
            "payload {i} out of bounds (payload_capacity {capacity})"
        );
        let payloads_per_grid_row = self.payloads_per_grid_row();
        let grid_row = i / payloads_per_grid_row;
        let e_local = i % payloads_per_grid_row;
        let column = e_local % self.cols;
        let payload_in_column = e_local / self.cols;
        Address {
            matrix: grid_row,
            column,
            row_offset: payload_in_column * P::EXPONENT,
        }
    }
}

impl<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload<[u8; 32]>> Database<BE, P>
where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
{
    /// Allocate a zeroed database from the shared layout. Both constructions use
    /// physical `n × n` blocks; `column_height` only controls logical payload
    /// addressing and how many logical records are packed in one physical row
    /// group.
    pub fn from_layout(
        module: &Module<BE>,
        layout: DatabaseLayout<P>,
        base2k: usize,
        column_height: usize,
    ) -> Self {
        let n = module.n();
        let preprocessing = DatabasePreprocessingConfig::new::<P>(column_height);
        let grid_rows = layout.grid_rows_for(column_height);
        assert_eq!(
            n % column_height,
            0,
            "column height must divide the ring degree"
        );
        let rows_per_group = n / column_height;
        let physical_rows = grid_rows.div_ceil(rows_per_group);
        let matrices = (0..physical_rows * layout.column_blocks(n))
            .map(|_| CoeffMatrix::zeros(n, n))
            .collect();
        Self {
            matrices,
            n,
            base2k,
            cols: layout.cols(),
            grid_rows,
            physical_rows,
            preprocessing,
            _marker: PhantomData,
        }
    }

    /// Allocate a zeroed database holding `db_entries = block_rows · n · cols`
    /// coefficient slots, tiled into `n x n` sub-matrices at `base2k`.
    pub fn new(module: &Module<BE>, db_entries: usize, base2k: usize, cols: usize) -> Self {
        let n = module.n();
        assert!(cols.is_multiple_of(n), "cols must be a multiple of n");
        let per_matrix = n * cols;
        assert!(
            db_entries.is_multiple_of(per_matrix),
            "db_entries must be a multiple of n·cols"
        );
        let blocks = cols / n;
        let matrices = (0..(db_entries / per_matrix) * blocks)
            .map(|_| CoeffMatrix::zeros(n, n))
            .collect();
        Self {
            matrices,
            n,
            base2k,
            cols,
            grid_rows: db_entries / per_matrix,
            physical_rows: db_entries / per_matrix,
            preprocessing: DatabasePreprocessingConfig::new::<P>(n),
            _marker: PhantomData,
        }
    }

    /// Encode `payloads` values starting at payload index `start`, each as
    /// `P::EXPONENT` base-`P::BASIS` digits down consecutive rows of one column.
    pub fn encode_shard(&mut self, start: usize, payloads: &[[u8; 32]]) {
        let capacity = self.payload_capacity();
        let end = start
            .checked_add(payloads.len())
            .expect("shard length overflow");
        assert!(
            end <= capacity,
            "shard writes past the configured capacity ({capacity})"
        );
        let digits_per = P::EXPONENT;
        let mut digits = vec![0i16; digits_per];
        for (i, &payload) in payloads.iter().enumerate() {
            let addr = self.payload_address(start + i);
            let (matrix_idx, row_out_base, col_in_block) =
                self.matrix_index_and_column(addr.matrix, addr.column);
            let sub = &mut self.matrices[matrix_idx];
            P::encode(&mut digits, payload);
            for (k, &d) in digits.iter().enumerate() {
                sub.row_mut(row_out_base + addr.row_offset + k)[col_in_block] = d;
            }
        }
    }
}

impl<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload<[u8; 32]>> Database<BE, P> {
    fn matrix_index_and_column(&self, grid_row: usize, column: usize) -> (usize, usize, usize) {
        assert!(
            grid_row < self.grid_rows,
            "grid row {grid_row} out of bounds"
        );
        assert!(column < self.cols, "column {column} out of bounds");
        let row_group = grid_row / self.rows_per_physical_group();
        let row_offset = (grid_row % self.rows_per_physical_group()) * self.column_height();
        let column_blocks = self.column_blocks();
        (
            row_group * column_blocks + column / self.n,
            row_offset,
            column % self.n,
        )
    }

    /// Write raw `Z_p` digits into a logical record `(column, grid_row)`.
    pub fn write_digits(
        &mut self,
        column: usize,
        grid_row: usize,
        row_offset: usize,
        digits: &[i64],
    ) {
        assert!(
            row_offset + digits.len() <= self.column_height(),
            "digit run overflows the record"
        );
        let (matrix_idx, row_out_base, col_in_block) =
            self.matrix_index_and_column(grid_row, column);
        let block = &mut self.matrices[matrix_idx];
        for (j, &v) in digits.iter().enumerate() {
            block.row_mut(row_out_base + row_offset + j)[col_in_block] = v as i16;
        }
    }

    /// Read raw `Z_p` digits from a logical record `(column, grid_row)`.
    pub fn read_digits(
        &self,
        column: usize,
        grid_row: usize,
        row_offset: usize,
        len: usize,
    ) -> Vec<i64> {
        assert!(
            row_offset + len <= self.column_height(),
            "digit run overflows the record"
        );
        let (matrix_idx, row_out_base, col_in_block) =
            self.matrix_index_and_column(grid_row, column);
        let block = &self.matrices[matrix_idx];
        // Digits are stored as the centered `i16` representative of a `Z_p` value.
        // Return it signed: `as u16` would reduce mod `2^16`, but `p = 2^16 - 1`,
        // so a negative digit `-k` would come back as `2^16 - k ≡ (1 - k) mod p`
        // — a +1 error on every negative coefficient. The encoder recenters mod
        // `p`, so the signed representative is exact.
        (0..len)
            .map(|j| block.row(row_out_base + row_offset + j)[col_in_block] as i64)
            .collect()
    }

    /// Write one complete record `(column, grid_row)` = `column_height` values.
    pub fn encode_record(&mut self, column: usize, grid_row: usize, record: &[i64]) {
        assert_eq!(
            record.len(),
            self.column_height(),
            "record must hold column_height elements"
        );
        self.write_digits(column, grid_row, 0, record);
    }

    /// Bulk-write all `grid_rows · cols` records in row-major order:
    /// `records[grid_row·cols + column]`.
    pub fn encode(&mut self, records: &[Vec<i64>]) {
        assert_eq!(
            records.len(),
            self.num_records(),
            "expected grid_rows·cols records"
        );
        for grid_row in 0..self.grid_rows {
            for column in 0..self.cols {
                self.encode_record(column, grid_row, &records[grid_row * self.cols + column]);
            }
        }
    }

    /// Read back one complete record `(column, grid_row)`.
    pub fn record(&self, column: usize, grid_row: usize) -> Vec<i64> {
        self.read_digits(column, grid_row, 0, self.column_height())
    }

    /// Read back the plaintext payload stored at index `i` (the decode inverse of
    /// [`encode_shard`](Self::encode_shard)). The server owns the plaintext DB, so
    /// this is the ground-truth oracle for the value a PIR query should return.
    pub fn payload(&self, i: usize) -> [u8; 32] {
        let digits_per = P::EXPONENT;
        let addr = self.payload_address(i);
        let digits: Vec<i16> = self
            .read_digits(addr.column, addr.matrix, addr.row_offset, digits_per)
            .into_iter()
            .map(|v| v as i16)
            .collect();
        let mut out = [0u8; 32];
        P::decode(&mut out, &digits);
        out
    }
}
