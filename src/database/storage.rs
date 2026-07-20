use std::marker::PhantomData;

use poulpy_core::layouts::ModuleCoreAlloc;
use poulpy_hal::layouts::{Backend, Module};

use crate::payload::Payload;
#[cfg(feature = "numa-db-interleave")]
use crate::{numa, parallel::num_threads_setup};

use super::{
    CoeffMatrix, address::Address, layout::DatabaseLayout,
    preprocessing::DatabasePreprocessingConfig,
};

// NOTE (NUMA): DB placement is a compile-time choice via the
// `numa-db-interleave` feature, because the two serving modes want opposite
// policies:
// - Feature OFF (default, batch throughput): no explicit placement — batched
//   serving measures ~12% faster when automatic NUMA balancing is left free
//   to migrate the hot blocks adaptively than under any static policy tried
//   (blanket interleave, per-row-group node binding with pinned workers, or
//   a parallel first-touch spread). The cost is single-query body-product
//   latency on multi-socket hosts: the whole DB is first-touched onto the
//   sequential fill writer's node, so a lone query streams at one node's
//   bandwidth.
// - Feature ON (single-query latency): the pass below spreads the DB pages
//   across the nodes at allocation, letting the memory-bound l1 body GEMV
//   run at every node's bandwidth (~1.5 s -> ~0.2 s online for a 32 GiB DB).

/// Spread the zeroed matrices' physical pages across the NUMA nodes and commit
/// them from parallel workers. `vec![0; …]` maps its pages lazily, so without
/// this every page is first-touched by the (sequential) `encode_shard` caller
/// and the whole DB lands on one node, throttling the memory-bound online body
/// GEMV to one node's bandwidth (measured ~0.9–1.5 s instead of ~140 ms for a
/// 32 GiB DB at one query).
///
/// Placement is pinned with an explicit `mbind(MPOL_INTERLEAVE)` per block
/// where available (Linux/x86-64, multi-node): plain first-touch spreading is
/// silently undone by automatic NUMA balancing, which watches the sequential
/// `encode_shard` writer stream the whole DB and migrates the pages back to
/// its node — an explicit VMA policy is exempt from that. Where `mbind` is
/// unavailable the round-robin parallel touch still spreads pages at block
/// granularity (and the pre-fault alone takes ~12 s off the 32 GiB fill).
#[cfg(feature = "numa-db-interleave")]
fn first_touch_matrices(matrices: &mut [CoeffMatrix]) {
    // Two passes, because `mbind` and the page fault scale differently:
    //
    // 1. `numa::interleave` (`mbind`) takes the process `mmap_lock` in *write*
    //    mode, so issuing it from many threads only forms a lock convoy — it is
    //    no faster in parallel and the contention dominates (a single-threaded
    //    ~4 s SETUP on a 32 GiB DB). Set the interleave policy serially.
    // 2. The actual first-touch faults each take only a per-page-table lock, so
    //    they parallelize across cores — do those at full setup width.
    for m in matrices.iter() {
        numa::interleave(m.flat().as_ptr().cast(), size_of_val(m.flat()));
    }
    let nthreads = num_threads_setup(matrices.len());
    if nthreads <= 1 {
        for m in matrices.iter_mut() {
            m.first_touch();
        }
        return;
    }
    let per = matrices.len().div_ceil(nthreads);
    std::thread::scope(|scope| {
        for chunk in matrices.chunks_mut(per) {
            scope.spawn(move || {
                for m in chunk {
                    m.first_touch();
                }
            });
        }
    });
}

/// Batch-throughput mode: leave placement to the kernel (see NOTE above).
#[cfg(not(feature = "numa-db-interleave"))]
fn first_touch_matrices(_matrices: &mut [CoeffMatrix]) {}

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
        let mut matrices: Vec<CoeffMatrix> = (0..physical_rows * layout.column_blocks(n))
            .map(|_| CoeffMatrix::zeros(n, n))
            .collect();
        first_touch_matrices(&mut matrices);
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
        let mut matrices: Vec<CoeffMatrix> = (0..(db_entries / per_matrix) * blocks)
            .map(|_| CoeffMatrix::zeros(n, n))
            .collect();
        first_touch_matrices(&mut matrices);
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
    ///
    /// The scatter is parallelized over the physical `n x n` matrices the shard
    /// touches: every payload maps to exactly one matrix, so partitioning by
    /// matrix hands each worker a disjoint set of matrices whose coefficient
    /// writes never alias. The result is identical to a sequential scatter.
    pub fn encode_shard(&mut self, start: usize, payloads: &[[u8; 32]]) {
        let capacity = self.payload_capacity();
        let end = start
            .checked_add(payloads.len())
            .expect("shard length overflow");
        assert!(
            end <= capacity,
            "shard writes past the configured capacity ({capacity})"
        );
        if payloads.is_empty() {
            return;
        }

        let n = self.n;
        let cols = self.cols;
        let grid_rows = self.grid_rows;
        let column_blocks = self.column_blocks();
        let column_height = self.column_height();
        let rows_per_group = self.rows_per_physical_group();
        let payloads_per_column = self.preprocessing.payloads_per_column::<P>();
        let payloads_per_grid_row = payloads_per_column * cols;
        let digits_per = P::EXPONENT;

        // The shard spans grid rows `[gr_lo, gr_hi]`, hence physical row groups
        // `[rg_lo, rg_hi]` and the contiguous matrix range `[m_lo, m_hi)`. Only
        // these matrices can receive a write, so workers scan just this window.
        let gr_lo = start / payloads_per_grid_row;
        let gr_hi = (end - 1) / payloads_per_grid_row;
        let m_lo = (gr_lo / rows_per_group) * column_blocks;
        let m_hi = (((gr_hi / rows_per_group) + 1) * column_blocks).min(self.matrices.len());
        let count = m_hi - m_lo;
        let workers = crate::parallel::num_threads(count);
        let per = count.div_ceil(workers);

        let slice = &mut self.matrices[m_lo..m_hi];
        std::thread::scope(|scope| {
            for (chunk_idx, chunk) in slice.chunks_mut(per).enumerate() {
                let base_m = m_lo + chunk_idx * per;
                scope.spawn(move || {
                    let mut digits = vec![0i16; digits_per];
                    for (local, matrix) in chunk.iter_mut().enumerate() {
                        let m = base_m + local;
                        let row_group = m / column_blocks;
                        let block = m % column_blocks;
                        // Valid columns in this (possibly narrower last) block.
                        let width = (cols - block * n).min(n);
                        for local_row in 0..rows_per_group {
                            let grid_row = row_group * rows_per_group + local_row;
                            if grid_row >= grid_rows {
                                break; // partial last physical group
                            }
                            let row_base = local_row * column_height;
                            for pic in 0..payloads_per_column {
                                // Payload index of column c = 0 in this row run.
                                let base = grid_row * payloads_per_grid_row + pic * cols + block * n;
                                // In-range column sub-interval within `[0, width)`.
                                let c_lo = start.saturating_sub(base).min(width);
                                let c_hi = end.saturating_sub(base).min(width);
                                if c_lo >= c_hi {
                                    continue;
                                }
                                let row_off = row_base + pic * digits_per;
                                for c in c_lo..c_hi {
                                    P::encode(&mut digits, payloads[(base + c) - start]);
                                    for (k, &d) in digits.iter().enumerate() {
                                        matrix.row_mut(row_off + k)[c] = d;
                                    }
                                }
                            }
                        }
                    }
                });
            }
        });
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
