use poulpy_core::{
    GLWEExpandLWEMatrix, GLWEMaskFill,
    layouts::{
        Base2K, Degree, GLWECompressed, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos,
        LWEMatrix, LWEMatrixInfos, LWEMatrixLayout, LWEMatrixToBackendMut, ModuleCoreAlloc,
        TorusPrecision,
    },
};
use poulpy_hal::{
    api::{VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxZeroBackend},
    layouts::{
        Backend, Data, HostDataMut, HostDataRef, Module, ScratchArena, VecZnx, ZnxView, ZnxViewMut,
    },
};

use std::borrow::Cow;

use poulpy_cpu_ref::reference::fft64::reim::ReimArith;

use crate::{database::CoeffMatrix, parameters::Parameters, payload::Payload, server::gemm::Gemm};

pub(super) fn default_query_mask_tmp_bytes<BE, R, GM>(
    module: &Module<BE>,
    dst_infos: &R,
    glwe_mask: &GM,
) -> usize
where
    BE: Backend,
    Module<BE>: GLWEExpandLWEMatrix<BE> + VecZnxNormalizeTmpBytes,
    R: LWEMatrixInfos,
    GM: GLWEInfos,
{
    module
        .vec_znx_normalize_tmp_bytes()
        .max(module.glwe_expand_lwe_matrix_tmp_bytes(dst_infos, glwe_mask))
}

/// Internal coarse mask-regime layout.
pub(super) fn mask_regime_infos<BE: Backend, P: Payload>(
    params: &Parameters<BE, P>,
) -> LWEMatrixLayout {
    let n = params.n();
    LWEMatrixLayout {
        rows: n,
        n: Degree(n as u32),
        base2k: Base2K(params.mask_base2k() as u32),
        k: TorusPrecision((params.size_at(params.mask_base2k()) * params.mask_base2k()) as u32),
    }
}

/// Fills a seed-derived query mask `A` into `dst` in the coarse mask regime.
pub(super) fn fill_default_query_mask<BE, R, GF, GM>(
    module: &Module<BE>,
    dst: &mut R,
    seed: [u8; 32],
    glwe_fill: &GF,
    glwe_mask: &GM,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>:
        GLWEExpandLWEMatrix<BE> + GLWEMaskFill<BE> + VecZnxZeroBackend<BE> + VecZnxNormalize<BE>,
    R: LWEMatrixToBackendMut<BE> + LWEMatrixInfos,
    GF: GLWEInfos,
    GM: GLWEInfos,
{
    assert_eq!(glwe_fill.n().as_usize(), module.n());
    assert_eq!(dst.n().as_usize(), glwe_fill.rank().as_usize() * module.n());
    assert!(dst.rows() <= module.n());
    assert_eq!(dst.base2k(), glwe_mask.base2k());

    let rank = glwe_fill.rank().as_usize();
    let mut fill_glwe = module.glwe_alloc_from_infos(glwe_fill);
    let mut coarse_glwe = module.glwe_alloc_from_infos(glwe_mask);

    {
        let mut fill_mut = GLWEToBackendMut::<BE>::to_backend_mut(&mut fill_glwe);
        module.vec_znx_zero_backend(fill_mut.data_mut(), 0);
    }
    module.fill_glwe_mask_from_seed(glwe_fill.base2k().as_usize(), &mut fill_glwe, 1, rank, seed);

    {
        normalize_glwe_mask(module, &fill_glwe, &mut coarse_glwe, scratch);
    }

    module.glwe_expand_lwe_matrix(dst, &coarse_glwe, &mut scratch.borrow());
}

fn normalize_glwe_mask<BE, GF, GM>(
    module: &Module<BE>,
    src: &GF,
    dst: &mut GM,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VecZnxNormalize<BE>,
    GF: GLWEToBackendRef<BE> + GLWEInfos,
    GM: GLWEToBackendMut<BE> + GLWEInfos,
{
    let src_ref = GLWEToBackendRef::<BE>::to_backend_ref(src);
    let dst_base2k = dst.base2k().as_usize();
    let src_base2k = src.base2k().as_usize();
    let mut dst_mut = GLWEToBackendMut::<BE>::to_backend_mut(dst);
    let rank = src.rank().as_usize();
    for col in 0..rank + 1 {
        module.vec_znx_normalize(
            dst_mut.data_mut(),
            dst_base2k,
            0,
            col,
            src_ref.data(),
            src_base2k,
            col,
            &mut scratch.borrow(),
        );
    }
}

/// Coefficient matrix `U` flattened once into a row-major **`i16`** panel
/// (`rows_out × rows_in`), the GEMM-ready contraction operand for both the mask
/// product (`U·A`, offline) and the body product (`U·b`, online).
///
/// Stored as `i16` (not the widened `f64`) to cut the prepared-panel cache to ¼
/// of its size — for a 1 GiB DB this is ~1 GiB instead of ~4 GiB. The `f64`
/// `private-gemm` kernel needs `f64` inputs, so each panel is widened into a
/// caller-owned reusable scratch buffer ([`widen_into`]) right before its GEMM.
/// The widen is `O(rows_out·rows_in)` and negligible against the `O(n³)` mask
/// GEMM; it adds one panel read+write to the (memory-bound) body GEMV.
pub(crate) struct PreparedF64<'a> {
    values: Cow<'a, [i16]>,
    rows_out: usize,
    rows_in: usize,
}

impl<'a> PreparedF64<'a> {
    /// **Owned** copy of `matrix`'s contiguous panel — for small operands that
    /// must be stored away from their source (the resp1 digit DB; the
    /// interpolation matrix DB if ever decoupled).
    pub(crate) fn new(matrix: &CoeffMatrix) -> PreparedF64<'static> {
        PreparedF64 {
            values: Cow::Owned(matrix.flat().to_vec()),
            rows_out: matrix.rows_out(),
            rows_in: matrix.rows_in(),
        }
    }

    /// Bytes this value owns. A zero-copy view over a database panel owns
    /// nothing and reports 0; only [`new`](Self::new) copies allocate.
    pub(crate) fn allocated_bytes(&self) -> usize {
        match &self.values {
            Cow::Owned(v) => size_of_val(v.as_slice()),
            Cow::Borrowed(_) => 0,
        }
    }

    /// **Zero-copy view** over `matrix`'s contiguous panel — for the recursion DB,
    /// which already lives in `self.database`, so no second copy is materialized.
    pub(crate) fn from_matrix(matrix: &'a CoeffMatrix) -> Self {
        PreparedF64 {
            values: Cow::Borrowed(matrix.flat()),
            rows_out: matrix.rows_out(),
            rows_in: matrix.rows_in(),
        }
    }

    /// Widens the `i16` panel into `dst` (resized to `rows_out·rows_in`) as the
    /// `f64` GEMM operand. `dst` is reused across panels so the peak `f64`
    /// footprint is one panel per worker, not the whole prepared cache.
    fn widen_into(&self, dst: &mut Vec<f64>) {
        dst.resize(self.values.len(), 0.0);
        crate::server::gemm::widen_i16_to_f64(&self.values, dst);
    }
}

/// Computes the fixed mask product `sum_i U_i · A_i` and encodes it into the pack
/// regime as an [`LWEMatrix`], via the dense full-torus `f64` GEMM.
pub(super) fn mask_product_to_pack<BE, I>(
    module: &Module<BE>,
    out_infos: &I,
    prepared: &[PreparedF64],
    masks: &[QueryMask],
    torus_bits: usize,
    mask_threads: usize,
    gemm: &dyn Gemm,
) -> LWEMatrix<BE::OwnedBuf>
where
    BE: Backend<OwnedBuf = Vec<u8>> + ReimArith,
    I: LWEMatrixInfos,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
{
    let mut out = module.lwe_matrix_alloc_from_infos(out_infos);
    full_torus_f64_mask_product::<BE>(&mut out, prepared, masks, torus_bits, mask_threads, gemm);
    out
}

/// Query mask `A`, decoded once into a row-major `f64` buffer (`rows × cols`,
/// scaled into `[-0.5, 0.5)`) so the dense GEMM consumes it directly with no
/// per-product decode.
pub(crate) struct QueryMask {
    values: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl QueryMask {
    /// Decodes a coarse-regime query mask into its `f64` working representation
    /// using `torus_bits` of precision.
    pub(crate) fn new(mask: LWEMatrix<Vec<u8>>, torus_bits: usize) -> Self {
        let rows = mask.mask().n();
        let cols = mask.mask().cols();
        let mut values = vec![0.0f64; rows * cols];
        decode_torus_mask_f64(&mut values, &mask, rows, cols, torus_bits);
        Self { values, rows, cols }
    }
}

/// Accumulates `acc += sum_{bc in range} U_bc · A_bc` over a contiguous range of
/// block columns, in ascending `bc` order (the per-group partial of the tiled
/// mask product). Single-threaded `private-gemm` per block.
fn accumulate_mask_range(
    acc: &mut [f64],
    prepared: &[PreparedF64],
    masks: &[QueryMask],
    rows_out: usize,
    lwe_n: usize,
    range: std::ops::Range<usize>,
    gemm: &dyn Gemm,
) {
    let mut wide: Vec<f64> = Vec::new();
    for bc in range {
        let u = &prepared[bc];
        let rhs = &masks[bc];
        u.widen_into(&mut wide);
        gemm.gemm_f64_add(acc, &wide, &rhs.values, rows_out, u.rows_in, lwe_n);
    }
}

fn full_torus_f64_mask_product<BE>(
    out: &mut LWEMatrix<BE::OwnedBuf>,
    prepared: &[PreparedF64],
    masks: &[QueryMask],
    torus_bits: usize,
    mask_threads: usize,
    gemm: &dyn Gemm,
) where
    BE: Backend<OwnedBuf = Vec<u8>> + ReimArith,
{
    assert_eq!(
        prepared.len(),
        masks.len(),
        "coefficient matrix and query mask counts differ"
    );
    assert!(!prepared.is_empty(), "cannot compute an empty mask product");

    let rows_out = out.mask().n();
    let lwe_n = out.mask().cols();
    for (u, rhs) in prepared.iter().zip(masks) {
        assert_eq!(
            u.rows_out, rows_out,
            "coefficient matrix output rows mismatch"
        );
        assert_eq!(rhs.cols, lwe_n, "query mask LWE dimension mismatch");
        assert_eq!(
            u.rows_in, rhs.rows,
            "coefficient matrix input rows and query mask rows differ"
        );
    }

    let acc = mask_product_acc(prepared, masks, rows_out, lwe_n, mask_threads, gemm);

    out.body_mut().raw_mut().fill(0);
    encode_torus_mask_f64::<BE>(out, &acc, rows_out, lwe_n, torus_bits);
}

/// Which axis [`mask_product_acc`] splits across its `mask_threads` workers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MaskSplit {
    /// Split the dst *columns* into contiguous strips (the default): every
    /// output element is a single full-depth ascending-`bc` fold owned by one
    /// thread, so the result is independent of `nt` (bit-identical to the
    /// sequential fold on width-independent kernels).
    Cols,
    /// Split the `bc` *blocks* into contiguous ranges ("one matrix per
    /// thread"): each worker contracts whole panels at full dst width, and the
    /// per-range partials are reduced in ascending order. Faster — full-width
    /// GEMMs (a 512-wide strip costs ~14% kernel throughput at PIR shape),
    /// each panel is widened once per group instead of once per strip thread,
    /// and no rhs strip copies — but the summation *grouping* now depends on
    /// `nt`, so results vary by a few ulps across thread counts (the same
    /// cryptographic-equivalence class as swapping the GEMM kernel; far below
    /// the torus rounding margin and FHE noise floor).
    Blocks,
}

/// Runtime selection: `PIR_MASK_SPLIT=blocks|cols` (default `cols`, the
/// bit-exact status quo). Read once — the choice must not change mid-run.
fn mask_split() -> MaskSplit {
    static SPLIT: std::sync::OnceLock<MaskSplit> = std::sync::OnceLock::new();
    *SPLIT.get_or_init(|| match std::env::var("PIR_MASK_SPLIT").as_deref() {
        Ok("blocks") => MaskSplit::Blocks,
        _ => MaskSplit::Cols,
    })
}

/// The pure-`f64` mask accumulation `sum_bc U_bc · A_bc`, optionally split
/// across `mask_threads` threads. `mask_threads <= 1` is the exact sequential
/// left-fold (reference order). For `nt > 1` the split axis is chosen by
/// [`mask_split`]: dst-column strips (default; `nt`-independent results) or
/// `bc` block ranges (faster; see [`MaskSplit`] for the trade-off).
///
/// Column strips: each worker runs the full ascending-`bc` contraction for its
/// own strip. Every output element is a single full-depth dot product
/// accumulated in the reference block order by exactly one thread — no partial
/// accumulators, no cross-thread reduction — so unlike a `bc`-range split the
/// result does not depend on `nt` (measured bit-identical to the sequential
/// fold on the private-gemm kernel; at worst a few ulps if a kernel's internal
/// order were width-dependent, far below the FHE noise floor).
fn mask_product_acc(
    prepared: &[PreparedF64],
    masks: &[QueryMask],
    rows_out: usize,
    lwe_n: usize,
    mask_threads: usize,
    gemm: &dyn Gemm,
) -> Vec<f64> {
    let k = prepared.len();
    let nt = mask_threads.clamp(1, lwe_n);
    if nt <= 1 {
        let mut acc = vec![0.0f64; rows_out * lwe_n];
        accumulate_mask_range(&mut acc, prepared, masks, rows_out, lwe_n, 0..k, gemm);
        return acc;
    }

    if mask_split() == MaskSplit::Blocks && k >= 2 {
        // One matrix per thread: contiguous `bc` ranges, each contracted at
        // full dst width into its own partial (via the same sequential
        // building block), then reduced in ascending range order.
        let ntk = nt.min(k);
        let base = k / ntk;
        let rem = k % ntk;
        let mut ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(ntk);
        let mut start = 0;
        for i in 0..ntk {
            let len = base + usize::from(i < rem);
            ranges.push(start..start + len);
            start += len;
        }
        let mut partials: Vec<Vec<f64>> = std::thread::scope(|scope| {
            let handles: Vec<_> = ranges
                .into_iter()
                .map(|range| {
                    scope.spawn(move || {
                        let mut part = vec![0.0f64; rows_out * lwe_n];
                        accumulate_mask_range(
                            &mut part, prepared, masks, rows_out, lwe_n, range, gemm,
                        );
                        part
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut acc = std::mem::take(&mut partials[0]);
        for part in &partials[1..] {
            for (a, p) in acc.iter_mut().zip(part.iter()) {
                *a += *p;
            }
        }
        return acc;
    }

    let base = lwe_n / nt;
    let rem = lwe_n % nt;
    let mut strips: Vec<(usize, usize)> = Vec::with_capacity(nt); // (col_start, width)
    let mut col = 0;
    for i in 0..nt {
        let width = base + usize::from(i < rem);
        strips.push((col, width));
        col += width;
    }
    let mut acc = vec![0.0f64; rows_out * lwe_n];
    std::thread::scope(|scope| {
        let handles: Vec<_> = strips
            .iter()
            .map(|&(col_start, width)| {
                scope.spawn(move || {
                    // Thread-local contiguous strip accumulator plus a
                    // contiguous copy of the rhs column strip, contracted with
                    // the *same* widen + dense-GEMM kernel as the sequential
                    // fold (the kernel's per-column accumulation order is
                    // width-independent, which is what makes the strips
                    // bit-identical to the reference; the trait's i16 entry
                    // point would delegate width 1 to the GEMV kernel and
                    // break that). The widen is O(panel) per thread and
                    // negligible against the O(n³/nt) strip GEMM.
                    let mut local = vec![0.0f64; rows_out * width];
                    let mut wide: Vec<f64> = Vec::new();
                    let mut rhs_strip: Vec<f64> = Vec::new();
                    for bc in 0..k {
                        let u = &prepared[bc];
                        let rhs = &masks[bc];
                        rhs_strip.resize(u.rows_in * width, 0.0);
                        for r in 0..u.rows_in {
                            let src = r * lwe_n + col_start;
                            rhs_strip[r * width..(r + 1) * width]
                                .copy_from_slice(&rhs.values[src..src + width]);
                        }
                        u.widen_into(&mut wide);
                        gemm.gemm_f64_add(
                            &mut local, &wide, &rhs_strip, rows_out, u.rows_in, width,
                        );
                    }
                    (col_start, width, local)
                })
            })
            .collect();
        for handle in handles {
            let (col_start, width, local) = handle.join().unwrap();
            for r in 0..rows_out {
                let dst = r * lwe_n + col_start;
                acc[dst..dst + width].copy_from_slice(&local[r * width..(r + 1) * width]);
            }
        }
    });
    acc
}

/// Computes the body product `sum_i U_i · b_i` (a GEMV, `lwe_n = 1`) and encodes it
/// directly into `out` (a single-column `VecZnx`) at `out_base2k`. The online
/// counterpart of [`mask_product_to_pack`]: reuses the cached `f64` panels, so no
/// `U` decode happens per query.
pub(super) fn full_torus_f64_body_product<BE>(
    out: &mut VecZnx<BE::OwnedBuf>,
    out_base2k: usize,
    prepared: &[PreparedF64],
    bodies: &[GLWECompressed<BE::OwnedBuf>],
    body_base2k: usize,
    torus_bits: usize,
    gemm: &dyn Gemm,
) where
    BE: Backend<OwnedBuf = Vec<u8>> + ReimArith,
{
    assert_eq!(
        prepared.len(),
        bodies.len(),
        "prepared matrix and query body counts differ"
    );
    assert!(
        !prepared.is_empty(),
        "cannot accumulate an empty body product"
    );

    let rows_out = prepared[0].rows_out;
    let mut acc = vec![0.0f64; rows_out];
    let mut rhs = Vec::new();

    for (u, body) in prepared.iter().zip(bodies) {
        assert_eq!(u.rows_out, rows_out, "body product output rows mismatch");
        rhs.resize(u.rows_in, 0.0);
        decode_torus_body_f64(&mut rhs, body.data(), u.rows_in, body_base2k, torus_bits);
        // GEMV `acc += U * b`: read `U` as i16 and widen in-register (no 32 MiB
        // f64 panel materialized) — the memory-bound online win.
        gemm.gemv_i16_f64_add(&mut acc, &u.values, &rhs, u.rows_out, u.rows_in);
    }

    encode_torus_body_f64::<BE>(out, out_base2k, &acc, rows_out, torus_bits);
}

/// Batched body product: `out_bodies[q] = sum_bc U_bc · b^q_bc` over a query
/// batch sharing the same `prepared` (`U_bc`) panels. Each block contributes a
/// single i16×f64 GEMM whose RHS stacks the `Q` queries' decoded bodies as
/// columns ([`Gemm::gemm_i16_f64_add`]), so every `U_bc` panel is read once and
/// amortized over the whole batch — the win over `Q` separate memory-bound GEMVs.
/// The per-query repack is **not** batched: the caller packs each `out_bodies[q]`
/// independently.
pub(super) fn full_torus_f64_body_product_batch<BE>(
    out_bodies: &mut [VecZnx<BE::OwnedBuf>],
    out_base2k: usize,
    prepared: &[PreparedF64],
    bodies_per_query: &[&[GLWECompressed<BE::OwnedBuf>]],
    body_base2k: usize,
    torus_bits: usize,
    gemm: &dyn Gemm,
) where
    BE: Backend<OwnedBuf = Vec<u8>> + ReimArith,
{
    let q = out_bodies.len();
    assert_eq!(
        bodies_per_query.len(),
        q,
        "output and query-body batch widths differ"
    );
    assert!(q > 0, "cannot run an empty body-product batch");
    assert!(
        !prepared.is_empty(),
        "cannot accumulate an empty body product"
    );
    let nblocks = prepared.len();
    for bodies in bodies_per_query {
        assert_eq!(
            bodies.len(),
            nblocks,
            "per-query block count differs from prepared panels"
        );
    }

    let rows_out = prepared[0].rows_out;
    // `acc` and `rhs` are row-major `rows × q` (the query index is the fastest
    // axis), the column layout `gemm_i16_f64_add` contracts against.
    let mut acc = vec![0.0f64; rows_out * q];
    let mut rhs: Vec<f64> = Vec::new();
    let mut tmp: Vec<i64> = Vec::new();

    for bc in 0..nblocks {
        let u = &prepared[bc];
        assert_eq!(u.rows_out, rows_out, "body product output rows mismatch");
        rhs.clear();
        rhs.resize(u.rows_in * q, 0.0);
        for (qi, bodies) in bodies_per_query.iter().enumerate() {
            decode_torus_body_into_col(
                &mut rhs,
                qi,
                q,
                bodies[bc].data(),
                u.rows_in,
                body_base2k,
                torus_bits,
                &mut tmp,
            );
        }
        gemm.gemm_i16_f64_add(&mut acc, &u.values, &rhs, rows_out, u.rows_in, q);
    }

    // Per-query repack: gather column `qi` of `acc` (de-interleave) and encode it
    // into `out_bodies[qi]`.
    let mut col = vec![0.0f64; rows_out];
    for qi in 0..q {
        for r in 0..rows_out {
            col[r] = acc[r * q + qi];
        }
        encode_torus_body_f64::<BE>(&mut out_bodies[qi], out_base2k, &col, rows_out, torus_bits);
    }
}

/// Decodes the single body column of `body` into column `col` of a row-major
/// `rows × ncols` buffer (`rhs[r*ncols + col]`) as torus reals in `[-0.5, 0.5)`.
/// `tmp` is reused scratch sized to the ring degree.
#[allow(clippy::too_many_arguments)]
fn decode_torus_body_into_col(
    rhs: &mut [f64],
    col: usize,
    ncols: usize,
    body: &VecZnx<Vec<u8>>,
    rows: usize,
    base2k: usize,
    torus_bits: usize,
    tmp: &mut Vec<i64>,
) {
    let scale = torus_modulus_f64(torus_bits).recip();
    tmp.resize(body.n(), 0);
    body.decode_vec_i64(base2k, 0, torus_bits, tmp);
    for r in 0..rows {
        rhs[r * ncols + col] = tmp[r] as f64 * scale;
    }
}

fn decode_torus_mask_f64(
    out: &mut [f64],
    mask: &LWEMatrix<Vec<u8>>,
    rows: usize,
    cols: usize,
    torus_bits: usize,
) {
    let base2k = mask.base2k().as_usize();
    let scale = torus_modulus_f64(torus_bits).recip();
    let mut col = vec![0i64; rows];
    for c in 0..cols {
        mask.mask().decode_vec_i64(base2k, c, torus_bits, &mut col);
        for r in 0..rows {
            out[r * cols + c] = col[r] as f64 * scale;
        }
    }
}

fn encode_torus_mask_f64<BE>(
    out: &mut LWEMatrix<BE::OwnedBuf>,
    values: &[f64],
    rows: usize,
    cols: usize,
    torus_bits: usize,
) where
    BE: Backend<OwnedBuf = Vec<u8>> + ReimArith,
{
    let base2k = out.base2k().as_usize();
    let scale = torus_modulus_f64(torus_bits);
    // `reim_to_znx` computes `(a / divisor).round()`; we want `(a * scale).round()`.
    let divisor = scale.recip();
    let mut col_real = vec![0.0f64; rows];
    let mut col = vec![0i64; rows];
    for c in 0..cols {
        // mod-1 pre-pass: reduce each accumulated value into `[-0.5, 0.5)` in f64,
        // keeping the magnitude small enough that the `* 2^torus_bits` scaling is
        // exact before rounding (the reduced result then already lies in the
        // centered range mod `2^torus_bits`, so no further integer reduction is
        // needed).
        for r in 0..rows {
            let v = values[r * cols + c];
            col_real[r] = v - (v + 0.5).floor();
        }
        // f64 -> i64 round, vectorized (the AVX `bnd63` variant covers ±2^53).
        BE::reim_to_znx(&mut col, divisor, &col_real);
        out.mask_mut().encode_vec_i64(base2k, c, torus_bits, &col);
    }
}

/// Parallel level-1 body accumulator over the whole DB. For each physical row
/// group `rg`, returns `acc[rg]` (row-major `rows_out × q`, query fastest) equal
/// to `Σ_bc U(rg,bc) · b_bc` for the query batch — the same quantity
/// [`full_torus_f64_body_product_batch`] computes per group, but:
///
/// * the shared query RHS (`bodies_per_query`, identical for every row group) is
///   decoded **once** instead of once per group, and
/// * the work is parallelized across `(row_group, output-row band)` tiles, so all
///   `nthreads` cores are used even when the row-group count is below the thread
///   count (the online first dimension's dominant `D·b0`).
///
/// Output rows are independent GEMV rows, so the band split needs no partial-sum
/// reduction and the result is bit-identical to the serial per-group order. When
/// `nthreads <= physical_rows` each group is a single tile (the original
/// whole-panel GEMM). The caller encodes/​splits each returned `acc[rg]`.
#[allow(clippy::type_complexity)]
pub(super) fn body_product_acc_parallel(
    db_views: &[Vec<PreparedF64>],
    bodies_per_query: &[&[GLWECompressed<Vec<u8>>]],
    body_base2k: usize,
    torus_bits: usize,
    nthreads: usize,
    gemm: &dyn Gemm,
) -> Vec<Vec<f64>> {
    let physical_rows = db_views.len();
    assert!(physical_rows > 0, "cannot accumulate an empty DB");
    let nblocks = db_views[0].len();
    let q = bodies_per_query.len();
    assert!(q > 0, "cannot run an empty body-product batch");
    for bodies in bodies_per_query {
        assert_eq!(bodies.len(), nblocks, "per-query block count differs");
    }
    let rows_out = db_views[0][0].rows_out;

    // Decode the query bodies once; `rhs_all[bc]` is row-major `rows_in × q`
    // (query fastest), the layout `gemm_i16_f64_add` contracts against. Shared
    // read-only across every row group and every output-row band below.
    let mut rhs_all: Vec<Vec<f64>> = Vec::with_capacity(nblocks);
    {
        let mut tmp: Vec<i64> = Vec::new();
        for bc in 0..nblocks {
            let rows_in = db_views[0][bc].rows_in;
            let mut rhs = vec![0.0f64; rows_in * q];
            for (qi, bodies) in bodies_per_query.iter().enumerate() {
                decode_torus_body_into_col(
                    &mut rhs,
                    qi,
                    q,
                    bodies[bc].data(),
                    rows_in,
                    body_base2k,
                    torus_bits,
                    &mut tmp,
                );
            }
            rhs_all.push(rhs);
        }
    }

    // One accumulator per row group; the tiles below borrow disjoint output-row
    // bands of these buffers, so their writes never alias across threads.
    let mut acc_all: Vec<Vec<f64>> = (0..physical_rows)
        .map(|_| vec![0.0f64; rows_out * q])
        .collect();

    // Split each group's `rows_out` into `tiles_per_group` bands so the total tile
    // count reaches `nthreads` even with few row groups.
    let tiles_per_group = nthreads.div_ceil(physical_rows).max(1);
    let band = rows_out.div_ceil(tiles_per_group).max(1);
    let mut tiles: Vec<(usize, usize, &mut [f64])> = Vec::new();
    for (rg, acc) in acc_all.iter_mut().enumerate() {
        let mut r0 = 0;
        let mut rest = acc.as_mut_slice();
        while r0 < rows_out {
            let len = band.min(rows_out - r0);
            let (head, tail) = rest.split_at_mut(len * q);
            tiles.push((rg, r0, head));
            rest = tail;
            r0 += len;
        }
    }

    let rhs_all = &rhs_all;
    let chunk = tiles.len().div_ceil(nthreads.max(1)).max(1);
    std::thread::scope(|scope| {
        for group in tiles.chunks_mut(chunk) {
            scope.spawn(move || {
                for tile in group.iter_mut() {
                    let (rg, r0) = (tile.0, tile.1);
                    let rows_band = tile.2.len() / q;
                    for bc in 0..nblocks {
                        let u = &db_views[rg][bc];
                        let rin = u.rows_in;
                        let u_band = &u.values[r0 * rin..(r0 + rows_band) * rin];
                        gemm.gemm_i16_f64_add(
                            &mut *tile.2,
                            u_band,
                            &rhs_all[bc],
                            rows_band,
                            rin,
                            q,
                        );
                    }
                }
            });
        }
    });

    acc_all
}

/// Decodes the single body column (`col 0`) of `body` into `out[0..rows]` as real
/// torus values in `[-0.5, 0.5)`.
fn decode_torus_body_f64(
    out: &mut [f64],
    body: &VecZnx<Vec<u8>>,
    rows: usize,
    base2k: usize,
    torus_bits: usize,
) {
    let scale = torus_modulus_f64(torus_bits).recip();
    let mut col = vec![0i64; body.n()];
    body.decode_vec_i64(base2k, 0, torus_bits, &mut col);
    for r in 0..rows {
        out[r] = col[r] as f64 * scale;
    }
}

/// Encodes the `rows`-long real body accumulator into `out`'s column 0 at
/// `out_base2k`; coefficients beyond `rows` (up to the ring degree) are zeroed.
pub(super) fn encode_torus_body_f64<BE>(
    out: &mut VecZnx<BE::OwnedBuf>,
    out_base2k: usize,
    acc: &[f64],
    rows: usize,
    torus_bits: usize,
) where
    BE: Backend<OwnedBuf = Vec<u8>> + ReimArith,
{
    let scale = torus_modulus_f64(torus_bits);
    let divisor = scale.recip();
    let mut col_real = vec![0.0f64; rows];
    for r in 0..rows {
        let v = acc[r];
        col_real[r] = v - (v + 0.5).floor();
    }
    // `encode_vec_i64` requires a full ring-degree slice, so the tail stays zero.
    let mut full = vec![0i64; out.n()];
    BE::reim_to_znx(&mut full[..rows], divisor, &col_real);
    out.encode_vec_i64(out_base2k, 0, torus_bits, &full);
}

fn torus_modulus_i128(torus_bits: usize) -> i128 {
    assert!(
        torus_bits <= 62,
        "full-torus f64 product expects torus precision to fit i64"
    );
    1i128 << torus_bits
}

fn torus_modulus_f64(torus_bits: usize) -> f64 {
    torus_modulus_i128(torus_bits) as f64
}

pub(super) fn copy_vec_znx_rows<D>(
    dst: &mut VecZnx<D>,
    dst_row_offset: usize,
    src: &VecZnx<D>,
    src_row_offset: usize,
    rows: usize,
) where
    D: Data + HostDataMut + HostDataRef,
{
    assert_eq!(dst.cols(), src.cols(), "VecZnx column count mismatch");
    assert_eq!(dst.size(), src.size(), "VecZnx limb count mismatch");
    assert!(
        dst_row_offset + rows <= dst.n(),
        "destination row slice out of bounds"
    );
    assert!(
        src_row_offset + rows <= src.n(),
        "source row slice out of bounds"
    );
    for col in 0..dst.cols() {
        for limb in 0..dst.size() {
            let src_rows = &src.at(col, limb)[src_row_offset..src_row_offset + rows];
            dst.at_mut(col, limb)[dst_row_offset..dst_row_offset + rows].copy_from_slice(src_rows);
        }
    }
}

pub(super) fn copy_lwe_matrix_mask_rows<D>(
    dst: &mut LWEMatrix<D>,
    dst_row_offset: usize,
    src: &LWEMatrix<D>,
    src_row_offset: usize,
    rows: usize,
) where
    D: Data + HostDataMut + HostDataRef,
{
    assert_eq!(dst.base2k(), src.base2k(), "LWE base2k mismatch");
    copy_vec_znx_rows(
        dst.mask_mut(),
        dst_row_offset,
        src.mask(),
        src_row_offset,
        rows,
    );
}

#[cfg(test)]
mod mask_product_tests {
    use super::{PreparedF64, QueryMask, mask_product_acc};
    use crate::server::gemm::PrivateGemmX86;

    /// Deterministic pseudo-random f64 in `[lo, hi)`.
    fn prng(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn synthetic(
        k: usize,
        rows_out: usize,
        rows_in: usize,
        lwe_n: usize,
    ) -> (Vec<PreparedF64<'static>>, Vec<QueryMask>) {
        let mut s = 0x1234_5678_9abc_def0u64;
        let prepared = (0..k)
            .map(|_| {
                // U entries are centered i16-range integers (the database operand).
                let values: Vec<i16> = (0..rows_out * rows_in)
                    .map(|_| (prng(&mut s) * 65536.0 - 32768.0).round() as i16)
                    .collect();
                PreparedF64 {
                    values: super::Cow::Owned(values),
                    rows_out,
                    rows_in,
                }
            })
            .collect();
        let masks = (0..k)
            .map(|_| {
                // A entries are torus reals in [-0.5, 0.5).
                let values: Vec<f64> = (0..rows_in * lwe_n).map(|_| prng(&mut s) - 0.5).collect();
                QueryMask {
                    values,
                    rows: rows_in,
                    cols: lwe_n,
                }
            })
            .collect();
        (prepared, masks)
    }

    #[test]
    fn tiled_matches_sequential_within_noise_floor() {
        let (rows_out, rows_in, lwe_n, k) = (16, 16, 4, 13);
        let (prepared, masks) = synthetic(k, rows_out, rows_in, lwe_n);
        let seq = mask_product_acc(&prepared, &masks, rows_out, lwe_n, 1, &PrivateGemmX86);
        // The accumulated magnitude here is ~rows_in * 2^15 * 0.5 * k ≈ 2^25; the
        // f64 ulp is ~2^-27, so any reorder gap is a few ulps. The torus encode
        // rounds at ~2^-(53-torus_bits) of that, far coarser. Assert the relative
        // gap is < 1e-9 (cryptographically equivalent).
        for nt in [2, 3, 4, 8, k, k + 5] {
            let tiled = mask_product_acc(&prepared, &masks, rows_out, lwe_n, nt, &PrivateGemmX86);
            assert_eq!(tiled.len(), seq.len());
            let max_abs: f64 = seq
                .iter()
                .zip(&tiled)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            let scale: f64 = seq.iter().map(|v| v.abs()).fold(1.0, f64::max);
            assert!(
                max_abs / scale < 1e-9,
                "nt={nt}: relative gap {} exceeds tolerance",
                max_abs / scale
            );
        }
    }

    #[test]
    fn single_block_is_thread_count_invariant() {
        // With k=1 there is nothing to reorder: every thread count is identical.
        let (rows_out, rows_in, lwe_n) = (8, 8, 3);
        let (prepared, masks) = synthetic(1, rows_out, rows_in, lwe_n);
        let seq = mask_product_acc(&prepared, &masks, rows_out, lwe_n, 1, &PrivateGemmX86);
        for nt in [2, 4, 16] {
            let tiled = mask_product_acc(&prepared, &masks, rows_out, lwe_n, nt, &PrivateGemmX86);
            assert_eq!(seq, tiled, "k=1 must be bit-identical for nt={nt}");
        }
    }
}
