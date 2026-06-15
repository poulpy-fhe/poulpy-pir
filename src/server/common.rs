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

use poulpy_cpu_ref::reference::fft64::reim::ReimArith;

use crate::{database::CoeffMatrix, parameters::Parameters, payload::Payload};

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
pub(super) fn mask_regime_infos<BE: Backend, P: Payload<[u8; 32]>>(
    params: &Parameters<BE, [u8; 32], P>,
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

/// Coefficient matrix `U` decoded once into a row-major `f64` panel
/// (`rows_out × rows_in`), the GEMM-ready form used by both the mask product
/// (`U·A`, offline) and the body product (`U·b`, online). Caching it here keeps
/// the online body product decode-free.
pub(crate) struct PreparedF64 {
    values: Vec<f64>,
    rows_out: usize,
    rows_in: usize,
}

impl PreparedF64 {
    /// Decodes `matrix` into its `f64` panel. Entries are stored as centered
    /// `i16` (the database / interpolation `U` operand), so the GEMM-ready value
    /// is a direct `i16 -> f64` widening.
    pub(crate) fn new(matrix: &CoeffMatrix) -> Self {
        let rows_out = matrix.rows_out();
        let rows_in = matrix.rows_in();
        let mut values = vec![0.0f64; rows_out * rows_in];
        for out_row in 0..rows_out {
            let row = matrix.row(out_row);
            for in_row in 0..rows_in {
                values[out_row * rows_in + in_row] = row[in_row] as f64;
            }
        }
        Self {
            values,
            rows_out,
            rows_in,
        }
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
) -> LWEMatrix<BE::OwnedBuf>
where
    BE: Backend<OwnedBuf = Vec<u8>> + ReimArith,
    I: LWEMatrixInfos,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
{
    let mut out = module.lwe_matrix_alloc_from_infos(out_infos);
    full_torus_f64_mask_product::<BE>(&mut out, prepared, masks, torus_bits, mask_threads);
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
) {
    for bc in range {
        let u = &prepared[bc];
        let rhs = &masks[bc];
        gemm_f64_add(acc, &u.values, &rhs.values, rows_out, u.rows_in, lwe_n);
    }
}

fn full_torus_f64_mask_product<BE>(
    out: &mut LWEMatrix<BE::OwnedBuf>,
    prepared: &[PreparedF64],
    masks: &[QueryMask],
    torus_bits: usize,
    mask_threads: usize,
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

    let acc = mask_product_acc(prepared, masks, rows_out, lwe_n, mask_threads);

    out.body_mut().raw_mut().fill(0);
    encode_torus_mask_f64::<BE>(out, &acc, rows_out, lwe_n, torus_bits);
}

/// The pure-`f64` mask accumulation `sum_bc U_bc · A_bc`, optionally block-tiled
/// across `mask_threads` threads. `mask_threads <= 1` is the exact sequential
/// left-fold (reference order). For `nt > 1` the `bc` range is split into `nt`
/// contiguous ascending groups summed in parallel, then the partials are reduced
/// in ascending group order — deterministic for a given `nt`, but not
/// bit-identical to the sequential fold across different `nt` (f64 addition is
/// non-associative; the gap is far below the FHE noise floor).
fn mask_product_acc(
    prepared: &[PreparedF64],
    masks: &[QueryMask],
    rows_out: usize,
    lwe_n: usize,
    mask_threads: usize,
) -> Vec<f64> {
    let k = prepared.len();
    let nt = mask_threads.clamp(1, k);
    if nt <= 1 {
        let mut acc = vec![0.0f64; rows_out * lwe_n];
        accumulate_mask_range(&mut acc, prepared, masks, rows_out, lwe_n, 0..k);
        return acc;
    }

    let base = k / nt;
    let rem = k % nt;
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(nt);
    let mut start = 0;
    for i in 0..nt {
        let len = base + usize::from(i < rem);
        ranges.push(start..start + len);
        start += len;
    }
    let mut partials: Vec<Vec<f64>> = (0..nt).map(|_| vec![0.0f64; rows_out * lwe_n]).collect();
    std::thread::scope(|scope| {
        for (part, range) in partials.iter_mut().zip(ranges.into_iter()) {
            scope.spawn(move || {
                accumulate_mask_range(part, prepared, masks, rows_out, lwe_n, range);
            });
        }
    });
    let mut acc = std::mem::take(&mut partials[0]);
    for part in &partials[1..] {
        for (a, p) in acc.iter_mut().zip(part.iter()) {
            *a += *p;
        }
    }
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
        // GEMV `acc += U * b` (single RHS column).
        gemm_f64_add(&mut acc, &u.values, &rhs, u.rows_out, u.rows_in, 1);
    }

    encode_torus_body_f64::<BE>(out, out_base2k, &acc, rows_out, torus_bits);
}

/// Picks the densest available x86 instruction set for the GEMM kernel. AVX2 is a
/// hard requirement of the AVX backend this crate runs on, so `Avx256` is the
/// floor; `Avx512` is selected at runtime when the CPU reports `avx512f`.
fn gemm_instr_set() -> private_gemm_x86::InstrSet {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return private_gemm_x86::InstrSet::Avx512;
        }
    }
    private_gemm_x86::InstrSet::Avx256
}

/// Single-threaded dense `f64` GEMM `dst += lhs * rhs` for contiguous row-major
/// matrices: `lhs` is `m x k`, `rhs` is `k x n`, `dst` is `m x n`. Uses the
/// `private-gemm-x86` kernel (the same one faer dispatches to), with the
/// instruction set auto-selected at runtime by [`gemm_instr_set`].
fn gemm_f64_add(dst: &mut [f64], lhs: &[f64], rhs: &[f64], m: usize, k: usize, n: usize) {
    assert_eq!(dst.len(), m * n, "dst must be m*n");
    assert_eq!(lhs.len(), m * k, "lhs must be m*k");
    assert_eq!(rhs.len(), k * n, "rhs must be k*n");

    let alpha = 1.0f64;
    // SAFETY: `dst`/`lhs`/`rhs` are contiguous row-major buffers sized exactly for
    // the `m x n`, `m x k`, `k x n` shapes asserted above; the element strides
    // below match that layout (row stride = number of columns, col stride = 1).
    // `dst_row_idx`/`dst_col_idx`/`real_diag` are unused for `DstKind::Full`, and
    // `alpha` points to a live `f64`. The kernel runs single-threaded.
    unsafe {
        private_gemm_x86::gemm(
            private_gemm_x86::DType::F64,
            private_gemm_x86::IType::U64,
            gemm_instr_set(),
            m,
            n,
            k,
            dst.as_mut_ptr().cast(),
            n as isize,
            1,
            core::ptr::null(),
            core::ptr::null(),
            private_gemm_x86::DstKind::Full,
            private_gemm_x86::Accum::Add,
            lhs.as_ptr().cast(),
            k as isize,
            1,
            false,
            core::ptr::null(),
            0,
            rhs.as_ptr().cast(),
            n as isize,
            1,
            false,
            (&raw const alpha).cast(),
            1,
        );
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
fn encode_torus_body_f64<BE>(
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

    /// Deterministic pseudo-random f64 in `[lo, hi)`.
    fn prng(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn synthetic(k: usize, rows_out: usize, rows_in: usize, lwe_n: usize) -> (Vec<PreparedF64>, Vec<QueryMask>) {
        let mut s = 0x1234_5678_9abc_def0u64;
        let prepared = (0..k)
            .map(|_| {
                // U entries are centered i16-range integers (the database operand).
                let values: Vec<f64> = (0..rows_out * rows_in)
                    .map(|_| (prng(&mut s) * 65536.0 - 32768.0).round())
                    .collect();
                PreparedF64 { values, rows_out, rows_in }
            })
            .collect();
        let masks = (0..k)
            .map(|_| {
                // A entries are torus reals in [-0.5, 0.5).
                let values: Vec<f64> = (0..rows_in * lwe_n).map(|_| prng(&mut s) - 0.5).collect();
                QueryMask { values, rows: rows_in, cols: lwe_n }
            })
            .collect();
        (prepared, masks)
    }

    #[test]
    fn tiled_matches_sequential_within_noise_floor() {
        let (rows_out, rows_in, lwe_n, k) = (16, 16, 4, 13);
        let (prepared, masks) = synthetic(k, rows_out, rows_in, lwe_n);
        let seq = mask_product_acc(&prepared, &masks, rows_out, lwe_n, 1);
        // The accumulated magnitude here is ~rows_in * 2^15 * 0.5 * k ≈ 2^25; the
        // f64 ulp is ~2^-27, so any reorder gap is a few ulps. The torus encode
        // rounds at ~2^-(53-torus_bits) of that, far coarser. Assert the relative
        // gap is < 1e-9 (cryptographically equivalent).
        for nt in [2, 3, 4, 8, k, k + 5] {
            let tiled = mask_product_acc(&prepared, &masks, rows_out, lwe_n, nt);
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
        let seq = mask_product_acc(&prepared, &masks, rows_out, lwe_n, 1);
        for nt in [2, 4, 16] {
            let tiled = mask_product_acc(&prepared, &masks, rows_out, lwe_n, nt);
            assert_eq!(seq, tiled, "k=1 must be bit-identical for nt={nt}");
        }
    }
}
