use crate::circuit::AggregateLWE;
use crate::encoding::{U256_BASE65535_DIGITS, encode_u256_base65535};
use crate::interpolation::{interpolate_columns, interpolate_tmp_bytes};
use poulpy_core::{
    GLWEExpandLWEMatrix, LWEMatrixMul, ScratchArenaTakeCore,
    layouts::{
        Base2K, CoeffMatrix, CoeffMatrixLayout, Degree, GLWECompressedSeedMut,
        GLWECompressedToBackendRef, GLWEInfos, GLWEToBackendMut, LWEInfos, LWEMatrix,
        LWEMatrixInfos, LWEMatrixLayout, LWEMatrixToBackendMut, ModuleCoreAlloc, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxAddAssignBackend, VecZnxCopyBackend, VecZnxCopyRangeBackend,
        VecZnxFillUniformSourceBackend, VecZnxRotateBackend, VecZnxSubBackend, VecZnxZeroBackend,
    },
    layouts::{
        Backend, HostDataMut, Module, ScratchArena, VecZnx, VecZnxToBackendMut, VecZnxToBackendRef,
        ZnxViewMut,
    },
    source::Source,
};
use std::fmt;

/// Byte size of a single U256 (256-bit) payload.
pub const U256_PAYLOAD_BYTES: usize = 32;

/// Storage layout for a U256-payload PIR database.
///
/// Computes every dimension of a [`Database`] from four user-controlled inputs:
///
/// * `n`            — ring degree of the module (power of two, divisible by 16).
/// * `k_blocks`     — first-dimension column-block count (the `T` in `cols = T·n`).
/// * `base2k`       — torus base used by the underlying `VecZnx` storage.
/// * `num_payloads` — number of 256-bit entries the database must hold.
///
/// The struct is purely a calculator: every field except the four inputs is
/// derived in [`DatabaseLayout::new`]. Call [`DatabaseLayout::instantiate`] to
/// allocate a [`Database`] that exactly matches it.
///
/// # Example
///
/// ```ignore
/// // 32 GB database of 32-byte entries, ring degree 2048, two first-dim blocks.
/// let layout = DatabaseLayout::from_total_bytes(2048, 2, 16, 32 << 30);
/// println!("{layout}");
/// // -> nb_matrices = 2048, cols = 4096, interpolation t = 2048, …
/// let mut db = layout.instantiate(&module);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DatabaseLayout {
    // --- inputs ---
    /// Ring degree.
    pub n: usize,
    /// First-dimension column-block count (`cols / n`).
    pub k_blocks: usize,
    /// Torus base.
    pub base2k: usize,
    /// 256-bit entries the database must hold.
    pub num_payloads: usize,

    // --- derived ---
    /// First-dimension contraction width in i16 slots (`k_blocks · n`).
    pub cols: usize,
    /// Payloads stacked along the row-out axis of a single matrix (`n / 16`).
    pub payloads_per_column: usize,
    /// Total payload slots per matrix (`payloads_per_column · cols`).
    pub payloads_per_matrix: usize,
    /// Matrices on the interpolation axis.
    pub nb_matrices: usize,
    /// IDFT padding: `nb_matrices.next_power_of_two()` (= `t` in the paper).
    pub interpolation_t: usize,
}

impl DatabaseLayout {
    /// Computes the layout for at least `num_payloads` U256 entries.
    pub fn new(n: usize, k_blocks: usize, base2k: usize, num_payloads: usize) -> Self {
        assert!(n.is_power_of_two(), "n ({n}) must be a power of two");
        assert!(
            n.is_multiple_of(U256_BASE65535_DIGITS),
            "n ({n}) must be a multiple of {U256_BASE65535_DIGITS} (U256 digit count)",
        );
        assert!(k_blocks >= 1, "k_blocks must be ≥ 1");
        assert!(base2k >= 16, "base2k must be ≥ 16");

        let cols = k_blocks * n;
        let payloads_per_column = n / U256_BASE65535_DIGITS;
        let payloads_per_matrix = payloads_per_column * cols;
        let nb_matrices = if num_payloads == 0 {
            0
        } else {
            num_payloads.div_ceil(payloads_per_matrix)
        };
        let interpolation_t = nb_matrices.max(1).next_power_of_two();

        // The second dimension is addressed by IDFT roots of unity in
        // Z[X]/(X^n+1), whose monomial group has order 2n. The interpolation
        // degree (number of matrix-axis ciphertexts) therefore cannot exceed
        // 2n: beyond that ω = X^{2n/t} collapses to 1 and the roots stop being
        // distinct. Cap the second dimension at 2n; if a database needs more
        // matrices, raise `n` or add column blocks (`k_blocks`) to shrink it.
        assert!(
            interpolation_t <= 2 * n,
            "second dimension (interpolation degree {interpolation_t}) exceeds 2n = {} \
             for n = {n}; raise n or increase k_blocks to keep nb_matrices \u{2264} 2n",
            2 * n,
        );

        Self {
            n,
            k_blocks,
            base2k,
            num_payloads,
            cols,
            payloads_per_column,
            payloads_per_matrix,
            nb_matrices,
            interpolation_t,
        }
    }

    /// Layout for a database carrying `total_bytes` of payload (must be a
    /// multiple of [`U256_PAYLOAD_BYTES`] = 32).
    pub fn from_total_bytes(n: usize, k_blocks: usize, base2k: usize, total_bytes: usize) -> Self {
        assert!(
            total_bytes.is_multiple_of(U256_PAYLOAD_BYTES),
            "total_bytes ({total_bytes}) must be a multiple of {U256_PAYLOAD_BYTES} (U256 entry size)",
        );
        Self::new(n, k_blocks, base2k, total_bytes / U256_PAYLOAD_BYTES)
    }

    /// Total i16 slots the database occupies (`nb_matrices · n · cols`).
    pub fn total_i16_slots(&self) -> usize {
        self.nb_matrices * self.n * self.cols
    }

    /// Total payload byte capacity (`nb_matrices · payloads_per_matrix · 32`).
    ///
    /// Always ≥ `num_payloads · 32`; the slack lives in the last matrix.
    pub fn total_payload_bytes(&self) -> usize {
        self.nb_matrices * self.payloads_per_matrix * U256_PAYLOAD_BYTES
    }

    /// Unused payload slots in the last allocated matrix.
    pub fn unused_payload_slots(&self) -> usize {
        self.nb_matrices * self.payloads_per_matrix - self.num_payloads
    }

    /// Conceptual byte-matrix shape `(rows, cols)` for the layout — each cell
    /// is one payload byte, so `rows · cols = total_payload_bytes()`.
    ///
    /// `cols = n · k_blocks` (= the paper's `T · N`); `rows` is whatever makes
    /// the product equal `total_payload_bytes()`.
    pub fn byte_matrix_shape(&self) -> (usize, usize) {
        let cols_bytes = self.n * self.k_blocks;
        let rows_bytes = self.total_payload_bytes() / cols_bytes.max(1);
        (rows_bytes, cols_bytes)
    }

    /// Allocates a fresh [`Database`] with these dimensions.
    pub fn instantiate<BE: Backend>(&self, module: &Module<BE>) -> Database<BE>
    where
        Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
    {
        assert_eq!(
            module.n(),
            self.n,
            "module degree {} does not match layout n = {}",
            module.n(),
            self.n,
        );
        Database::with_u256_payload_count(module, self.num_payloads, self.base2k, self.k_blocks)
    }
}

impl fmt::Display for DatabaseLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_bytes = self.total_payload_bytes();
        let gib = total_bytes as f64 / (1u64 << 30) as f64;
        writeln!(f, "DatabaseLayout {{")?;
        writeln!(f, "  n               = {}", self.n)?;
        writeln!(f, "  k_blocks (T)    = {}", self.k_blocks)?;
        writeln!(f, "  base2k          = {}", self.base2k)?;
        writeln!(f, "  cols (T·N)      = {}", self.cols)?;
        writeln!(f, "  payloads/col    = {}", self.payloads_per_column)?;
        writeln!(f, "  payloads/matrix = {}", self.payloads_per_matrix)?;
        writeln!(f, "  nb_matrices D   = {}", self.nb_matrices)?;
        writeln!(f, "  interpolation t = {}", self.interpolation_t)?;
        writeln!(f, "  num_payloads    = {}", self.num_payloads)?;
        writeln!(f, "  unused slots    = {}", self.unused_payload_slots())?;
        writeln!(f, "  i16 slots       = {}", self.total_i16_slots())?;
        writeln!(f, "  capacity bytes  = {} ({:.3} GiB)", total_bytes, gib,)?;
        write!(f, "}}")
    }
}

pub struct Database<BE: Backend> {
    entries: usize,
    /// Number of `n`-row matrices on the matrix (interpolation) axis.
    nb_matrices: usize,
    /// Column-block count `k = cols / n`. Each matrix spans `cols = k·n` input
    /// columns, tiled into `k` blocks of `n` so the existing `n`-capped
    /// `lwe_matrix_mul` can be reused per block and the products summed.
    blocks: usize,
    /// Torus base, shared by every matrix/result layout.
    base2k: usize,
    /// Coefficient matrices, indexed `matrix * blocks + block`; each is `n × n`.
    db: Vec<CoeffMatrix<BE::OwnedBuf, i16>>,
    /// Raw per-matrix query results (`nb_matrices` entries), already summed over
    /// the `k` column blocks. Masks are cached per `precomp_seed`; bodies are
    /// refreshed on every query.
    precomp: Vec<LWEMatrix<BE::OwnedBuf>>,
    /// Interpolated query results, padded to `nb_matrices.next_power_of_two()`
    /// entries. The matrix index is carried on the `Z` axis of a polynomial
    /// `h(Z)` whose evaluations at the `t`-th roots of unity are the per-matrix
    /// results; entry `k` holds the (unnormalized, see [`interpolate_columns`])
    /// coefficient `c_k`. Interpolated masks are cached per `interp_seed`.
    interp: Vec<LWEMatrix<BE::OwnedBuf>>,
    precomp_seed: Option<[u8; 32]>,
    interp_seed: Option<[u8; 32]>,
}

impl<BE: Backend> Database<BE> {
    /// Builds a database holding `db_entries` scalar entries laid out as a tall
    /// `(n·D) × cols` matrix: `D` matrices on the interpolation axis, each
    /// `n` rows by `cols` input columns.
    ///
    /// `cols` is the per-query contraction width and trades off against the
    /// first-dimension collapse cost (a wider query packs more of the first
    /// dimension directly, leaving less to collapse). It must be a positive
    /// multiple of `n`; internally each matrix is tiled into `cols / n` blocks of
    /// `n` columns.
    pub fn new(module: &Module<BE>, db_entries: usize, base2k: usize, cols: usize) -> Self {
        assert!(base2k >= 16);

        let n = module.n();
        assert!(
            cols >= n && cols.is_multiple_of(n),
            "cols ({cols}) must be a positive multiple of n ({n})"
        );
        let blocks = cols / n;
        let nb_matrices = db_entries.div_ceil(n * cols);
        // Polynomial interpolation across matrices is a radix-2 IDFT, so the
        // matrix count is padded up to a power of two; the extra slots stay zero
        // and decode to non-existent matrix indices.
        let interp_len = nb_matrices.max(1).next_power_of_two();

        let u_infos = CoeffMatrixLayout {
            n: n.into(),
            rows_out: n.into(),
            base2k: Base2K(base2k as u32),
            k: TorusPrecision(base2k as u32),
        };
        let res_infos = LWEMatrixLayout {
            rows: n,
            n: Degree(n as u32),
            base2k: Base2K(base2k as u32),
            k: TorusPrecision(base2k as u32),
        };

        let mut db = Vec::with_capacity(nb_matrices * blocks);
        let mut precomp = Vec::with_capacity(nb_matrices);

        for _ in 0..nb_matrices {
            for _ in 0..blocks {
                db.push(module.coeff_matrix_alloc_from_infos(&u_infos));
            }
            precomp.push(module.lwe_matrix_alloc_from_infos(&res_infos));
        }

        let mut interp = Vec::with_capacity(interp_len);
        for _ in 0..interp_len {
            interp.push(module.lwe_matrix_alloc_from_infos(&res_infos));
        }

        Self {
            entries: db_entries,
            nb_matrices,
            blocks,
            base2k,
            db,
            precomp,
            interp,
            precomp_seed: None,
            interp_seed: None,
        }
    }

    /// Layout of a per-matrix query result / interpolation slot.
    fn res_layout(&self, module: &Module<BE>) -> LWEMatrixLayout {
        let n = module.n();
        LWEMatrixLayout {
            rows: n,
            n: Degree(n as u32),
            base2k: Base2K(self.base2k as u32),
            k: TorusPrecision(self.base2k as u32),
        }
    }

    pub fn encode_shard(&mut self, module: &Module<BE>, start_idx: usize, shard: &[i16])
    where
        BE::OwnedBuf: HostDataMut,
    {
        let n = module.n();
        let cols = self.blocks * n;
        let per_matrix = n * cols;
        let shard_end = start_idx
            .checked_add(shard.len())
            .expect("database shard index overflow");

        assert!(
            shard_end <= self.entries,
            "database shard writes past the configured number of entries"
        );

        for (offset, &value) in shard.iter().enumerate() {
            let idx = start_idx + offset;
            // Row-major within a matrix: entry (row_out, col) at row_out·cols + col.
            let mat_idx = idx / per_matrix;
            let local_idx = idx % per_matrix;
            let row_out = local_idx / cols;
            let col = local_idx % cols;
            let block = col / n;
            let row_in = col % n;

            self.db[mat_idx * self.blocks + block]
                .data_mut()
                .at_mut(row_out, 0)[row_in] = value as i64;
        }

        self.precomp_seed = None;
        self.interp_seed = None;
    }

    /// Allocates a database sized to hold exactly `num_payloads` 256-bit
    /// entries, tiled with `k_blocks` first-dimension column blocks.
    ///
    /// Convenience wrapper around [`Database::new`] that computes the i16-slot
    /// `db_entries = num_payloads · 16` and `cols = k_blocks · n` for you. The
    /// resulting database holds the entries as a tall `(N · D) × (k_blocks · N)`
    /// i16 matrix — equivalently a `(2 · N · D) × (k_blocks · N)` *byte*
    /// matrix, since each i16 carries one base-65535 digit ≈ 2 payload bytes.
    ///
    /// For a 32 GB database of 32 B entries with `N = 2048` and `k_blocks = 2`,
    /// this yields `D = 2048` matrices, each `2048 × 4096` i16 slots.
    pub fn with_u256_payload_count(
        module: &Module<BE>,
        num_payloads: usize,
        base2k: usize,
        k_blocks: usize,
    ) -> Self {
        let n = module.n();
        assert!(
            n.is_multiple_of(U256_BASE65535_DIGITS),
            "u256 packing requires module degree n ({n}) divisible by {U256_BASE65535_DIGITS}",
        );
        let cols = k_blocks * n;
        let db_entries = num_payloads
            .checked_mul(U256_BASE65535_DIGITS)
            .expect("num_payloads · 16 overflows usize");
        Self::new(module, db_entries, base2k, cols)
    }

    /// Maximum number of 256-bit payloads the database can hold given the
    /// current shape. Each payload occupies `16` row coefficients at a single
    /// column (one base-65535 digit per row), so the capacity is
    /// `nb_matrices · (n / 16) · cols`.
    ///
    /// Panics if the module degree is not a multiple of
    /// [`U256_BASE65535_DIGITS`] = 16.
    pub fn u256_payload_capacity(&self, module: &Module<BE>) -> usize {
        let n = module.n();
        assert!(
            n.is_multiple_of(U256_BASE65535_DIGITS),
            "u256 packing requires module degree n ({n}) divisible by {U256_BASE65535_DIGITS}",
        );
        let cols = self.blocks * n;
        self.nb_matrices * (n / U256_BASE65535_DIGITS) * cols
    }

    /// Encodes a shard of 256-bit payloads directly into the database.
    ///
    /// Each payload becomes 16 base-65535 digits (one per row coefficient of a
    /// single column), via [`encode_u256_base65535`]. Payloads are laid out
    /// row-block-major within each matrix: payload `e_local = rb · cols + c`
    /// sits at column `c`, in the 16-row stripe starting at `row_out = rb · 16`.
    /// This is the same row-major-with-fattened-rows layout as
    /// [`Database::encode_shard`], one level up.
    ///
    /// Every payload must be `< 65535^16 ≈ 2^256 − 2^244`; values past that
    /// bound debug-assert (see [`encode_u256_base65535`]).
    pub fn encode_u256_shard(
        &mut self,
        module: &Module<BE>,
        start_payload: usize,
        payloads: &[[u8; 32]],
    ) where
        BE::OwnedBuf: HostDataMut,
    {
        let n = module.n();
        assert!(
            n.is_multiple_of(U256_BASE65535_DIGITS),
            "u256 packing requires module degree n ({n}) divisible by {U256_BASE65535_DIGITS}",
        );

        let cols = self.blocks * n;
        let payloads_per_column = n / U256_BASE65535_DIGITS;
        let payloads_per_matrix = payloads_per_column * cols;
        let capacity = self.nb_matrices * payloads_per_matrix;

        let end = start_payload
            .checked_add(payloads.len())
            .expect("u256 shard index overflow");
        assert!(
            end <= capacity,
            "u256 shard writes past the configured capacity ({end} > {capacity})",
        );

        for (offset, payload) in payloads.iter().enumerate() {
            let e = start_payload + offset;
            let mat_idx = e / payloads_per_matrix;
            let e_local = e % payloads_per_matrix;
            // Row-block-major within a matrix: `rb` (which 16-row stripe)
            // varies slowest, `c` (column) varies fastest.
            let rb = e_local / cols;
            let c = e_local % cols;
            let row_out_start = rb * U256_BASE65535_DIGITS;
            let block = c / n;
            let row_in = c % n;

            let digits = encode_u256_base65535(payload);
            let sub_matrix = &mut self.db[mat_idx * self.blocks + block];
            for (k, &digit) in digits.iter().enumerate() {
                sub_matrix.data_mut().at_mut(row_out_start + k, 0)[row_in] = digit as i64;
            }
        }

        self.precomp_seed = None;
        self.interp_seed = None;
    }

    pub fn preprocess_query_mask_tmp_bytes<Q>(&self, module: &Module<BE>, query: &Q) -> usize
    where
        Module<BE>: AggregateLWE<BE> + GLWEExpandLWEMatrix<BE> + LWEMatrixMul<BE>,
        Q: GLWEInfos,
    {
        let mask_infos = LWEMatrixLayout {
            rows: module.n(),
            n: Degree((module.n() * query.rank().as_usize()) as u32),
            base2k: query.base2k(),
            k: query.max_k(),
        };
        let default_mask_bytes = default_query_mask_tmp_bytes(module, &mask_infos, query);

        // Every block/matrix shares the same shapes, so one representative bounds
        // the per-block mask product and aggregation cost. The aggregate buffer
        // size matches the *input* (mask_product) size, which is `query.size()`
        // — not `res.mask().size()`, which is the precomp's storage size and may
        // be smaller.
        let mask_product_size = mask_infos.size();
        match (self.db.first(), self.precomp.first()) {
            (Some(u), Some(res)) => {
                let mask_mul_bytes = module.lwe_matrix_mul_tmp_bytes(res, u, &mask_infos);
                let aggregate_bytes =
                    VecZnx::<Vec<u8>>::bytes_of(module.n(), module.n(), mask_product_size)
                        + module.aggregate_lwe_tmp_bytes(mask_product_size);
                default_mask_bytes.max(mask_mul_bytes).max(aggregate_bytes)
            }
            _ => default_mask_bytes,
        }
    }

    /// Precomputes the per-matrix masks for a query keyed on the `crs` seed.
    ///
    /// Each column block `b` draws its own mask from the deterministic seed
    /// [`derive_block_seeds`]`(crs, blocks)[b]`; the per-block mask products are
    /// aggregated and summed into each matrix's result mask.
    pub fn preprocess_query_mask<Q>(
        &mut self,
        module: &Module<BE>,
        crs: [u8; 32],
        query_infos: &Q,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: AggregateLWE<BE>
            + GLWEExpandLWEMatrix<BE>
            + LWEMatrixMul<BE>
            + VecZnxAddAssignBackend<BE>
            + VecZnxCopyBackend<BE>
            + VecZnxCopyRangeBackend<BE>
            + VecZnxFillUniformSourceBackend<BE>
            + VecZnxRotateBackend<BE>
            + VecZnxZeroBackend<BE>,
        Q: GLWEInfos,
    {
        let mask_infos = LWEMatrixLayout {
            rows: module.n(),
            n: Degree((module.n() * query_infos.rank().as_usize()) as u32),
            base2k: query_infos.base2k(),
            k: query_infos.max_k(),
        };
        let blocks = self.blocks;
        let nb = self.nb_matrices;
        let seeds = derive_block_seeds(crs, blocks);

        let mut default_mask = module.lwe_matrix_alloc_from_infos(&mask_infos);
        let mut mask_product = module.lwe_matrix_alloc_from_infos(&mask_infos);

        for (b, seed) in seeds.iter().enumerate() {
            fill_default_query_mask(module, &mut default_mask, *seed, query_infos, scratch);
            for matrix in 0..nb {
                let u = &self.db[matrix * blocks + b];
                module.lwe_matrix_mul(&mut mask_product, u, &default_mask, scratch);
                // First block initializes the result mask; later blocks add in.
                aggregate_lwe_mask(
                    module,
                    &mut self.precomp[matrix],
                    &mask_product,
                    b != 0,
                    scratch,
                );
            }
        }

        self.precomp_seed = Some(crs);
        // Recomputed raw masks invalidate the interpolated mask cache.
        self.interp_seed = None;
    }

    /// Per-query body channel: `precomp[matrix].body = Σ_block U[matrix][block] · query_block`.
    fn compute_bodies<Q>(
        &mut self,
        module: &Module<BE>,
        blocks_q: &[Q],
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Module<BE>: LWEMatrixMul<BE> + VecZnxAddAssignBackend<BE>,
        Q: GLWECompressedToBackendRef<BE> + GLWEInfos,
        VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
    {
        let blocks = self.blocks;
        let nb = self.nb_matrices;
        let res_layout = self.res_layout(module);
        let mut tmp_result = module.lwe_matrix_alloc_from_infos(&res_layout);

        for (b, block_q) in blocks_q.iter().enumerate() {
            for matrix in 0..nb {
                let u = &self.db[matrix * blocks + b];
                if b == 0 {
                    module.lwe_matrix_mul_body(&mut self.precomp[matrix], u, block_q, scratch);
                } else {
                    module.lwe_matrix_mul_body(&mut tmp_result, u, block_q, scratch);
                    let src = tmp_result.body().to_backend_ref();
                    let mut dst = self.precomp[matrix].body_mut().to_backend_mut();
                    module.vec_znx_add_assign_backend(&mut dst, 0, &src, 0);
                }
            }
        }
    }

    pub fn query_tmp_bytes<Q>(&self, module: &Module<BE>, query: &Q) -> usize
    where
        Module<BE>: AggregateLWE<BE> + GLWEExpandLWEMatrix<BE> + LWEMatrixMul<BE>,
        Q: GLWEInfos,
    {
        let mask_bytes = self.preprocess_query_mask_tmp_bytes(module, query);
        let body_bytes = match (self.db.first(), self.precomp.first()) {
            (Some(u), Some(res)) => module.lwe_matrix_mul_body_tmp_bytes(res, u, query),
            _ => 0,
        };

        mask_bytes.max(body_bytes)
    }

    /// Answers a first-dimension query over all `cols = blocks·n` columns.
    ///
    /// `blocks_q` must hold one body per column block (`blocks_q.len() == blocks`);
    /// block `b`'s mask is derived from `crs` via [`derive_block_seeds`]. Returns
    /// the per-matrix results (one [`LWEMatrix`] per matrix).
    pub fn query<Q>(
        &mut self,
        module: &Module<BE>,
        crs: [u8; 32],
        blocks_q: &[Q],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> &[LWEMatrix<BE::OwnedBuf>]
    where
        Module<BE>: AggregateLWE<BE>
            + GLWEExpandLWEMatrix<BE>
            + LWEMatrixMul<BE>
            + VecZnxAddAssignBackend<BE>
            + VecZnxCopyBackend<BE>
            + VecZnxCopyRangeBackend<BE>
            + VecZnxFillUniformSourceBackend<BE>
            + VecZnxRotateBackend<BE>
            + VecZnxZeroBackend<BE>,
        Q: GLWECompressedToBackendRef<BE> + GLWEInfos,
        VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
    {
        assert_eq!(
            blocks_q.len(),
            self.blocks,
            "query must provide exactly one body per column block"
        );

        if self.precomp_seed != Some(crs) {
            self.preprocess_query_mask(module, crs, &blocks_q[0], scratch);
        }

        self.compute_bodies(module, blocks_q, scratch);

        &self.precomp
    }

    /// Scratch bytes required by [`Database::query_interpolate`].
    pub fn query_interpolate_tmp_bytes<Q>(&self, module: &Module<BE>, query: &Q) -> usize
    where
        Module<BE>: AggregateLWE<BE> + GLWEExpandLWEMatrix<BE> + LWEMatrixMul<BE>,
        Q: GLWEInfos,
    {
        let query_bytes = self.query_tmp_bytes(module, query);
        let interp_bytes = self
            .interp
            .first()
            .map(|res| interpolate_tmp_bytes(module.n(), res.size()))
            .unwrap_or(0);

        query_bytes.max(interp_bytes)
    }

    /// Answers a first-dimension query and interpolates the per-matrix results
    /// across the matrix axis, returning `nb_matrices.next_power_of_two()`
    /// interpolated [`LWEMatrix`] results.
    ///
    /// With a single matrix this is equivalent to [`Database::query`] (the
    /// interpolation degenerates to a copy). With several matrices the matrix
    /// index is encoded on the `Z` axis of a polynomial `h(Z)`: evaluating the
    /// returned coefficients at the `k`-th root of unity (e.g. via encrypted
    /// Horner with a GGSW-encrypted monomial) selects the result of matrix `k`.
    /// The `1/t` IDFT normalization is **not** applied here (see
    /// [`interpolate_columns`]); the caller absorbs it into the plaintext
    /// scaling or a `log2(t)`-bit shift.
    ///
    /// Interpolated masks depend only on `crs` and are cached across calls; only
    /// the body channel is re-interpolated per query.
    pub fn query_interpolate<Q>(
        &mut self,
        module: &Module<BE>,
        crs: [u8; 32],
        blocks_q: &[Q],
        scratch: &mut ScratchArena<'_, BE>,
    ) -> &[LWEMatrix<BE::OwnedBuf>]
    where
        Module<BE>: AggregateLWE<BE>
            + GLWEExpandLWEMatrix<BE>
            + LWEMatrixMul<BE>
            + VecZnxAddAssignBackend<BE>
            + VecZnxCopyBackend<BE>
            + VecZnxCopyRangeBackend<BE>
            + VecZnxFillUniformSourceBackend<BE>
            + VecZnxRotateBackend<BE>
            + VecZnxSubBackend<BE>
            + VecZnxZeroBackend<BE>,
        Q: GLWECompressedToBackendRef<BE> + GLWEInfos,
        VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
    {
        assert_eq!(
            blocks_q.len(),
            self.blocks,
            "query must provide exactly one body per column block"
        );

        if self.interp_seed != Some(crs) {
            if self.precomp_seed != Some(crs) {
                self.preprocess_query_mask(module, crs, &blocks_q[0], scratch);
            }
            self.interpolate_masks(module, scratch);
            self.interp_seed = Some(crs);
        }

        self.compute_bodies(module, blocks_q, scratch);
        self.interpolate_bodies(module, scratch);

        &self.interp
    }

    /// Thin [`Query`]-aware wrapper over [`Database::query`].
    ///
    /// Equivalent to `self.query(module, query.crs(), query.blocks(), scratch)`;
    /// available so call sites can pass the encrypted query object directly
    /// instead of unpacking `(crs, blocks)` themselves. The
    /// [`Query::selector`] (second-dim GGSW) is **not** consumed here — it is
    /// used downstream during the encrypted-Horner / matrix-selection step.
    pub fn query_with(
        &mut self,
        module: &Module<BE>,
        query: &crate::query::Query,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> &[LWEMatrix<BE::OwnedBuf>]
    where
        BE: Backend<OwnedBuf = Vec<u8>>,
        Module<BE>: AggregateLWE<BE>
            + GLWEExpandLWEMatrix<BE>
            + LWEMatrixMul<BE>
            + VecZnxAddAssignBackend<BE>
            + VecZnxCopyBackend<BE>
            + VecZnxCopyRangeBackend<BE>
            + VecZnxFillUniformSourceBackend<BE>
            + VecZnxRotateBackend<BE>
            + VecZnxZeroBackend<BE>,
        VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
    {
        self.query(module, query.crs(), query.blocks(), scratch)
    }

    /// Thin [`Query`]-aware wrapper over [`Database::query_interpolate`].
    ///
    /// Equivalent to `self.query_interpolate(module, query.crs(), query.blocks(), scratch)`.
    /// See [`Database::query_with`] for the rationale.
    pub fn query_interpolate_with(
        &mut self,
        module: &Module<BE>,
        query: &crate::query::Query,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> &[LWEMatrix<BE::OwnedBuf>]
    where
        BE: Backend<OwnedBuf = Vec<u8>>,
        Module<BE>: AggregateLWE<BE>
            + GLWEExpandLWEMatrix<BE>
            + LWEMatrixMul<BE>
            + VecZnxAddAssignBackend<BE>
            + VecZnxCopyBackend<BE>
            + VecZnxCopyRangeBackend<BE>
            + VecZnxFillUniformSourceBackend<BE>
            + VecZnxRotateBackend<BE>
            + VecZnxSubBackend<BE>
            + VecZnxZeroBackend<BE>,
        VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
    {
        self.query_interpolate(module, query.crs(), query.blocks(), scratch)
    }

    /// Stages the cached raw masks into [`Database::interp`] (zero-padding the
    /// power-of-two tail) and interpolates each mask column across the matrix axis.
    fn interpolate_masks(&mut self, module: &Module<BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        Module<BE>: VecZnxAddAssignBackend<BE>
            + VecZnxCopyBackend<BE>
            + VecZnxRotateBackend<BE>
            + VecZnxSubBackend<BE>
            + VecZnxZeroBackend<BE>,
        VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
    {
        let nb = self.nb_matrices;
        let lwe_n = self.interp[0].mask().cols();

        for (src, dst) in self.precomp.iter().zip(self.interp.iter_mut()) {
            let src_mask = src.mask().to_backend_ref();
            let mut dst_mask = dst.mask_mut().to_backend_mut();
            for col in 0..lwe_n {
                module.vec_znx_copy_backend(&mut dst_mask, col, &src_mask, col);
            }
        }
        for dst in self.interp[nb..].iter_mut() {
            let mut dst_mask = dst.mask_mut().to_backend_mut();
            for col in 0..lwe_n {
                module.vec_znx_zero_backend(&mut dst_mask, col);
            }
        }

        for col in 0..lwe_n {
            interpolate_columns(
                module,
                &mut self.interp,
                LWEMatrix::mask,
                LWEMatrix::mask_mut,
                col,
                scratch,
            );
        }
    }

    /// Stages the freshly computed raw bodies into [`Database::interp`]
    /// (zero-padding the power-of-two tail) and interpolates the body channel.
    fn interpolate_bodies(&mut self, module: &Module<BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        Module<BE>: VecZnxAddAssignBackend<BE>
            + VecZnxCopyBackend<BE>
            + VecZnxRotateBackend<BE>
            + VecZnxSubBackend<BE>
            + VecZnxZeroBackend<BE>,
        VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
    {
        let nb = self.nb_matrices;

        for (src, dst) in self.precomp.iter().zip(self.interp.iter_mut()) {
            let src_body = src.body().to_backend_ref();
            let mut dst_body = dst.body_mut().to_backend_mut();
            module.vec_znx_copy_backend(&mut dst_body, 0, &src_body, 0);
        }
        for dst in self.interp[nb..].iter_mut() {
            let mut dst_body = dst.body_mut().to_backend_mut();
            module.vec_znx_zero_backend(&mut dst_body, 0);
        }

        interpolate_columns(
            module,
            &mut self.interp,
            LWEMatrix::body,
            LWEMatrix::body_mut,
            0,
            scratch,
        );
    }

    pub fn matrices(&self) -> &[CoeffMatrix<BE::OwnedBuf, i16>] {
        &self.db
    }

    pub fn precomputed(&self) -> &[LWEMatrix<BE::OwnedBuf>] {
        &self.precomp
    }

    /// Interpolated query results from the most recent
    /// [`Database::query_interpolate`] call.
    pub fn interpolated(&self) -> &[LWEMatrix<BE::OwnedBuf>] {
        &self.interp
    }
}

pub fn set_default_query_mask_seed<Q>(query: &mut Q, seed: [u8; 32])
where
    Q: GLWECompressedSeedMut,
{
    *query.seed_mut() = seed;
}

pub fn default_query_mask_tmp_bytes<BE, R, G>(
    module: &Module<BE>,
    _dst_infos: &R,
    glwe_infos: &G,
) -> usize
where
    BE: Backend,
    Module<BE>: GLWEExpandLWEMatrix<BE>,
    R: LWEMatrixInfos,
    G: GLWEInfos,
{
    VecZnx::<Vec<u8>>::bytes_of(
        module.n(),
        glwe_infos.rank().as_usize() + 1,
        glwe_infos.size(),
    ) + module.glwe_expand_lwe_matrix_tmp_bytes(_dst_infos, glwe_infos)
}

pub fn fill_default_query_mask<BE, R, G>(
    module: &Module<BE>,
    dst: &mut R,
    seed: [u8; 32],
    glwe_infos: &G,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>:
        GLWEExpandLWEMatrix<BE> + VecZnxFillUniformSourceBackend<BE> + VecZnxZeroBackend<BE>,
    R: LWEMatrixToBackendMut<BE> + LWEMatrixInfos,
    G: GLWEInfos,
{
    let n = module.n();
    let rank = glwe_infos.rank().as_usize();
    let lwe_n = rank * n;
    let rows = dst.rows();

    assert_eq!(
        glwe_infos.n().as_usize(),
        n,
        "fill_default_query_mask: GLWE.n() != module.n()"
    );
    assert_eq!(
        dst.n().as_usize(),
        lwe_n,
        "fill_default_query_mask: destination LWE dimension != rank * module.n()"
    );
    assert!(rows <= n, "fill_default_query_mask: rows > module.n()");
    assert_eq!(
        dst.base2k(),
        glwe_infos.base2k(),
        "fill_default_query_mask: base2k mismatch"
    );

    let arena = scratch.borrow();
    let (mut default_glwe, mut arena) = arena.take_glwe_scratch(&glwe_infos.glwe_layout());

    {
        let mut default_glwe = default_glwe.to_backend_mut();
        module.vec_znx_zero_backend(default_glwe.data_mut(), 0);

        let mut source = Source::new(seed);
        for col in 0..rank {
            module.vec_znx_fill_uniform_source_backend(
                glwe_infos.base2k().as_usize(),
                default_glwe.data_mut(),
                col + 1,
                &mut source,
            );
        }
    }

    module.glwe_expand_lwe_matrix(dst, &default_glwe, &mut arena);
}

/// Aggregates `src.mask` (LWE dim `r·n`) down to `n` columns and either writes
/// it to `dst.mask` (`accumulate = false`) or adds it in (`accumulate = true`).
/// Used to sum per-block mask contributions in [`Database::preprocess_query_mask`].
fn aggregate_lwe_mask<BE>(
    module: &Module<BE>,
    dst: &mut LWEMatrix<BE::OwnedBuf>,
    src: &LWEMatrix<BE::OwnedBuf>,
    accumulate: bool,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: AggregateLWE<BE> + VecZnxAddAssignBackend<BE> + VecZnxCopyBackend<BE>,
{
    let n = module.n();
    let size = src.mask().size();

    let (mut aggregate, mut arena_1) = scratch.borrow().take_vec_znx_scratch(n, n, size);

    module.aggregate_lwe(
        &mut aggregate,
        src.base2k().as_usize(),
        src.mask(),
        &mut arena_1,
    );

    let mut mask_mut =
        <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(dst.mask_mut());
    let aggregate_ref = aggregate.to_backend_ref();
    for col in 0..n {
        if accumulate {
            module.vec_znx_add_assign_backend(&mut mask_mut, col, &aggregate_ref, col);
        } else {
            module.vec_znx_copy_backend(&mut mask_mut, col, &aggregate_ref, col);
        }
    }
}

/// Derives the per-block mask seeds from a single Common Reference Seed `crs`.
///
/// Block `b`'s mask is sampled from `derive_block_seeds(crs, blocks)[b]`. Both
/// client (when encrypting block masks) and server ([`Database::preprocess_query_mask`])
/// call this so the two sides agree on the public mask of every block while the
/// in-flight query only has to carry the single `crs`.
pub fn derive_block_seeds(crs: [u8; 32], blocks: usize) -> Vec<[u8; 32]> {
    let mut source = Source::new(crs);
    (0..blocks).map(|_| source.new_seed()).collect()
}
