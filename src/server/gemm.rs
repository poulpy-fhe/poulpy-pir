//! Pluggable GEMM backend for the server's full-torus `f64` matrix products.
//!
//! The PIR server contracts the i16 database panel `U` against the query in two
//! shapes:
//! - OFFLINE the mask product `U·A` is a dense matrix×matrix **GEMM** (`A` has
//!   `lwe_n` columns); it is compute-bound, so `U` is widened to `f64` once and
//!   fed to a blocked microkernel.
//! - ONLINE the body product `U·b` is a matrix×vector **GEMV** (`b` is one
//!   column); it is memory-bound, so `U` is read as `i16` and widened
//!   in-register to halve the bytes streamed.
//!
//! Both live behind the [`Gemm`] trait so a caller can swap the kernel (a
//! different SIMD library, a GPU offload, …) without touching the FHE backend.
//! The default [`PrivateGemmX86`] keeps today's behavior exactly: the dense GEMM
//! dispatches to `private-gemm-x86`, the GEMV uses the hand-written AVX2 path.

/// A backend for the server's full-torus `f64` matrix products.
///
/// Implementors provide the dense GEMM; the i16/f64 GEMV has a provided default
/// (the portable + AVX2 CPU kernel) that a backend may override. The trait is
/// `Send + Sync` because the mask product fans the GEMM out across scoped
/// threads, sharing one `&dyn Gemm`.
pub trait Gemm: Send + Sync {
    /// Dense `f64` GEMM `dst += lhs · rhs` for contiguous row-major matrices:
    /// `lhs` is `m × k`, `rhs` is `k × n`, `dst` is `m × n`. Accumulating
    /// (`Add`), single-threaded — the server tiles parallelism above this call.
    fn gemm_f64_add(&self, dst: &mut [f64], lhs: &[f64], rhs: &[f64], m: usize, k: usize, n: usize);

    /// GEMV `acc[i] += sum_j U[i][j] · b[j]`, reading the `rows_out × rows_in`
    /// contraction operand `U` as row-major **`i16`** and `b` as `f64`. This is
    /// the memory-bound online body product (`n = 1`); reading `U` as i16 (¼ the
    /// bytes) is the win, so the default never materializes an `f64` panel.
    ///
    /// The default is the portable + AVX2 + AVX512F CPU kernel (densest path
    /// selected at runtime). A backend that contracts `i16·f64` more cheaply
    /// itself (e.g. on a device) may override it; otherwise the default's
    /// few-ulp reorder (vs. a strict left fold) is far below the torus rounding
    /// margin and FHE noise floor.
    fn gemv_i16_f64_add(
        &self,
        acc: &mut [f64],
        u: &[i16],
        b: &[f64],
        rows_out: usize,
        rows_in: usize,
    ) {
        default_gemv_i16_f64_add(acc, u, b, rows_out, rows_in)
    }

    /// GEMM `acc[rows_out × n] += U[rows_out × rows_in] · B[rows_in × n]`, with
    /// `U` row-major **`i16`** and `B`/`acc` row-major `f64`. This is the
    /// **batched** online body product: `n` is the number of queries in the
    /// batch, so one `U` read is amortized over `n` query bodies (the batch win
    /// over `n` separate memory-bound GEMVs).
    ///
    /// Default: `n == 1` delegates to the memory-bound [`Self::gemv_i16_f64_add`];
    /// for `n > 1` the panel is compute-bound, so `U` is widened to `f64` once and
    /// fed to [`Self::gemm_f64_add`] (the blocked dense kernel) — mirroring the
    /// offline mask product. A backend with a fused `i16·f64` matmul (e.g. a
    /// device kernel) may override this directly.
    fn gemm_i16_f64_add(
        &self,
        acc: &mut [f64],
        u: &[i16],
        b: &[f64],
        rows_out: usize,
        rows_in: usize,
        n: usize,
    ) {
        if n == 1 {
            self.gemv_i16_f64_add(acc, u, b, rows_out, rows_in);
            return;
        }
        // Tile over output rows so the widened f64 panel is bounded to one tile
        // instead of the whole `rows_out × rows_in` block. Materializing the full
        // block as f64 is 4× the i16 DB and, replicated per worker thread, blows
        // memory for large panels. Each tile is an independent row-block of the
        // GEMM, so the accumulation is exact.
        const ROW_TILE: usize = 512;
        let tile_rows = ROW_TILE.min(rows_out.max(1));
        let mut wide = vec![0.0f64; tile_rows * rows_in];
        let mut r = 0;
        while r < rows_out {
            let rt = ROW_TILE.min(rows_out - r);
            let u_tile = &u[r * rows_in..(r + rt) * rows_in];
            let wide_tile = &mut wide[..rt * rows_in];
            widen_i16_to_f64(u_tile, wide_tile);
            self.gemm_f64_add(&mut acc[r * n..(r + rt) * n], wide_tile, b, rt, rows_in, n);
            r += rt;
        }
    }
}

/// Default GEMM backend: the dense product dispatches to `private-gemm-x86`
/// (the same kernel faer dispatches to), the GEMV uses the in-register AVX2
/// path. Zero-sized — construct with `PrivateGemmX86`.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrivateGemmX86;

impl Gemm for PrivateGemmX86 {
    fn gemm_f64_add(
        &self,
        dst: &mut [f64],
        lhs: &[f64],
        rhs: &[f64],
        m: usize,
        k: usize,
        n: usize,
    ) {
        gemm_f64_add(dst, lhs, rhs, m, k, n)
    }
    // gemv_i16_f64_add: provided default is already the AVX2 CPU kernel.
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

/// GEMV `acc[i] += sum_j U[i][j] * b[j]`, reading the `rows_out × rows_in`
/// contraction operand `U` as row-major **`i16`** and widening to `f64`
/// **in-register** (never materializing the f64 panel). This is the
/// memory-bound online body product, so reading `U` as i16 (¼ the bytes) is the
/// win. The densest available kernel is picked at runtime: AVX512F, else
/// AVX2+FMA, else a scalar floor.
///
/// Determinism: the SIMD accumulation order differs from a strict left fold (and
/// across kernels — an AVX512 host and an AVX2 host are not bit-identical to each
/// other), so the result is *cryptographically equivalent* (a few f64 ulps, far
/// below the torus rounding margin and FHE noise floor), not byte-identical.
fn default_gemv_i16_f64_add(
    acc: &mut [f64],
    u: &[i16],
    b: &[f64],
    rows_out: usize,
    rows_in: usize,
) {
    debug_assert_eq!(u.len(), rows_out * rows_in);
    debug_assert_eq!(b.len(), rows_in);
    debug_assert_eq!(acc.len(), rows_out);
    #[cfg(target_arch = "x86_64")]
    {
        // AVX512F carries its own 512-bit FMA, so it is the only flag needed.
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f confirmed; lengths checked above.
            unsafe { gemv_i16_f64_add_avx512(acc, u, b, rows_out, rows_in) };
            return;
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: avx2+fma confirmed; lengths checked above.
            unsafe { gemv_i16_f64_add_avx2(acc, u, b, rows_out, rows_in) };
            return;
        }
    }
    for (i, a) in acc.iter_mut().enumerate() {
        let row = &u[i * rows_in..i * rows_in + rows_in];
        let mut s = 0.0f64;
        for (&uij, &bj) in row.iter().zip(b) {
            s += uij as f64 * bj;
        }
        *a += s;
    }
}

/// AVX2+FMA body of [`default_gemv_i16_f64_add`]: per row, load 8 × i16, widen to
/// two `__m256d` lanes in-register, FMA against `b`, then horizontal-sum.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_i16_f64_add_avx2(
    acc: &mut [f64],
    u: &[i16],
    b: &[f64],
    rows_out: usize,
    rows_in: usize,
) {
    use std::arch::x86_64::*;
    let bp = b.as_ptr();
    for i in 0..rows_out {
        let row = unsafe { u.as_ptr().add(i * rows_in) };
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();
        let mut j = 0;
        while j + 8 <= rows_in {
            let v16 = unsafe { _mm_loadu_si128(row.add(j).cast::<__m128i>()) };
            let v32 = _mm256_cvtepi16_epi32(v16);
            let lo = _mm256_cvtepi32_pd(_mm256_castsi256_si128(v32));
            let hi = _mm256_cvtepi32_pd(_mm256_extracti128_si256::<1>(v32));
            let b0 = unsafe { _mm256_loadu_pd(bp.add(j)) };
            let b1 = unsafe { _mm256_loadu_pd(bp.add(j + 4)) };
            acc0 = _mm256_fmadd_pd(lo, b0, acc0);
            acc1 = _mm256_fmadd_pd(hi, b1, acc1);
            j += 8;
        }
        // Horizontal sum of the 8 partial lanes.
        let summed = _mm256_add_pd(acc0, acc1);
        let lo128 = _mm256_castpd256_pd128(summed);
        let hi128 = _mm256_extractf128_pd::<1>(summed);
        let pair = _mm_add_pd(lo128, hi128);
        let mut s = _mm_cvtsd_f64(_mm_add_sd(pair, _mm_unpackhi_pd(pair, pair)));
        // Scalar tail.
        while j < rows_in {
            s += unsafe { *row.add(j) as f64 * *bp.add(j) };
            j += 1;
        }
        unsafe { *acc.as_mut_ptr().add(i) += s };
    }
}

/// AVX512F body of [`default_gemv_i16_f64_add`]: per row, process 16 × i16 per
/// iteration — sign-extend each 8-wide half `i16 → i32 → f64` and FMA the two
/// 8-lane `__m512d` against `b` into two accumulators — then an 8-wide cleanup
/// step, a `_mm512_reduce_add_pd`, and a scalar tail. (`avx2` is enabled too for
/// the `i16 → i32` widening helpers; it is always present alongside `avx512f`.)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2")]
unsafe fn gemv_i16_f64_add_avx512(
    acc: &mut [f64],
    u: &[i16],
    b: &[f64],
    rows_out: usize,
    rows_in: usize,
) {
    use std::arch::x86_64::*;
    let bp = b.as_ptr();
    for i in 0..rows_out {
        let row = unsafe { u.as_ptr().add(i * rows_in) };
        let mut acc0 = _mm512_setzero_pd();
        let mut acc1 = _mm512_setzero_pd();
        let mut j = 0;
        while j + 16 <= rows_in {
            // 16 × i16 → two halves of 8 × i32 (sign-extended).
            let v16 = unsafe { _mm256_loadu_si256(row.add(j).cast::<__m256i>()) };
            let lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v16));
            let hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v16));
            // → two lanes of 8 × f64.
            let lo = _mm512_cvtepi32_pd(lo32);
            let hi = _mm512_cvtepi32_pd(hi32);
            let b0 = unsafe { _mm512_loadu_pd(bp.add(j)) };
            let b1 = unsafe { _mm512_loadu_pd(bp.add(j + 8)) };
            acc0 = _mm512_fmadd_pd(lo, b0, acc0);
            acc1 = _mm512_fmadd_pd(hi, b1, acc1);
            j += 16;
        }
        // 8-wide cleanup (covers the common `rows_in % 16 == 8` case).
        if j + 8 <= rows_in {
            let v8 = unsafe { _mm_loadu_si128(row.add(j).cast::<__m128i>()) };
            let f = _mm512_cvtepi32_pd(_mm256_cvtepi16_epi32(v8));
            let bv = unsafe { _mm512_loadu_pd(bp.add(j)) };
            acc0 = _mm512_fmadd_pd(f, bv, acc0);
            j += 8;
        }
        let mut s = _mm512_reduce_add_pd(_mm512_add_pd(acc0, acc1));
        // Scalar tail.
        while j < rows_in {
            s += unsafe { *row.add(j) as f64 * *bp.add(j) };
            j += 1;
        }
        unsafe { *acc.as_mut_ptr().add(i) += s };
    }
}

/// `dst[i] = src[i] as f64`, AVX2-accelerated when available (the AVX backend
/// this crate runs on guarantees AVX2; the scalar path is a portability floor).
/// Shared by the prepared-panel widen ([`crate::server::common`]) and the default
/// batched [`Gemm::gemm_i16_f64_add`].
pub(super) fn widen_i16_to_f64(src: &[i16], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 confirmed at runtime; lengths checked equal above.
            unsafe { widen_i16_to_f64_avx2(src, dst) };
            return;
        }
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = s as f64;
    }
}

/// AVX2 i16→f64: 8 lanes/iteration via `cvtepi16_epi32` then two `cvtepi32_pd`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn widen_i16_to_f64_avx2(src: &[i16], dst: &mut [f64]) {
    use std::arch::x86_64::*;
    let n = src.len();
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut i = 0;
    while i + 8 <= n {
        // 8 × i16 → 8 × i32 (sign-extended)
        let v16 = unsafe { _mm_loadu_si128(sp.add(i).cast::<__m128i>()) };
        let v32 = _mm256_cvtepi16_epi32(v16);
        // 8 × i32 → two lanes of 4 × f64
        let lo = _mm256_castsi256_si128(v32);
        let hi = _mm256_extracti128_si256::<1>(v32);
        let f_lo = _mm256_cvtepi32_pd(lo);
        let f_hi = _mm256_cvtepi32_pd(hi);
        unsafe {
            _mm256_storeu_pd(dp.add(i), f_lo);
            _mm256_storeu_pd(dp.add(i + 4), f_hi);
        }
        i += 8;
    }
    while i < n {
        unsafe { *dp.add(i) = *sp.add(i) as f64 };
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{Gemm, PrivateGemmX86, default_gemv_i16_f64_add};

    /// Strict left-fold reference for `acc += U · b`.
    fn scalar_gemv(acc: &mut [f64], u: &[i16], b: &[f64], rows_in: usize) {
        for (i, a) in acc.iter_mut().enumerate() {
            let mut s = 0.0f64;
            for j in 0..rows_in {
                s += u[i * rows_in + j] as f64 * b[j];
            }
            *a += s;
        }
    }

    /// Strict reference for the batched GEMM `acc[m×n] += U[m×k] · B[k×n]`,
    /// row-major.
    fn scalar_gemm_i16(acc: &mut [f64], u: &[i16], b: &[f64], m: usize, k: usize, n: usize) {
        for i in 0..m {
            for col in 0..n {
                let mut s = 0.0f64;
                for p in 0..k {
                    s += u[i * k + p] as f64 * b[p * n + col];
                }
                acc[i * n + col] += s;
            }
        }
    }

    /// Whatever SIMD kernel the host dispatches to (AVX512F / AVX2 / scalar) must
    /// match the scalar reference to within the few-ulp reorder tolerance, across
    /// `rows_in` that exercise the 16-wide body, the 8-wide cleanup, and the
    /// scalar tail.
    #[test]
    fn dispatched_gemv_matches_scalar_reference() {
        let rows_out = 5;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        for rows_in in [1usize, 7, 8, 9, 15, 16, 17, 23, 24, 31, 33, 64, 1024, 1031] {
            // U: centered i16-range integers (the database operand).
            let u: Vec<i16> = (0..rows_out * rows_in)
                .map(|_| (next() >> 48) as i16)
                .collect();
            // b: torus reals in [-0.5, 0.5).
            let b: Vec<f64> = (0..rows_in)
                .map(|_| (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
                .collect();
            // Non-zero baseline to also exercise the `+=` accumulation.
            let mut want = vec![1.0f64; rows_out];
            let mut got = want.clone();
            scalar_gemv(&mut want, &u, &b, rows_in);
            default_gemv_i16_f64_add(&mut got, &u, &b, rows_out, rows_in);
            for (w, g) in want.iter().zip(&got) {
                let scale = w.abs().max(1.0);
                assert!(
                    (w - g).abs() / scale < 1e-9,
                    "rows_in={rows_in}: {w} vs {g}"
                );
            }
        }
    }

    /// The batched `gemm_i16_f64_add` (default = widen + dense GEMM for `n > 1`,
    /// GEMV delegation for `n == 1`) must match the scalar matrix×matrix
    /// reference. `n` is the query-batch width.
    #[test]
    fn batched_gemm_i16_matches_scalar_reference() {
        let g = PrivateGemmX86;
        let (rows_out, rows_in) = (7usize, 40usize);
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let u: Vec<i16> = (0..rows_out * rows_in)
            .map(|_| (next() >> 48) as i16)
            .collect();
        for n in [1usize, 2, 3, 4, 8, 16] {
            // B: rows_in × n, torus reals in [-0.5, 0.5).
            let b: Vec<f64> = (0..rows_in * n)
                .map(|_| (next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
                .collect();
            let mut want = vec![0.5f64; rows_out * n];
            let mut got = want.clone();
            scalar_gemm_i16(&mut want, &u, &b, rows_out, rows_in, n);
            g.gemm_i16_f64_add(&mut got, &u, &b, rows_out, rows_in, n);
            for (w, gv) in want.iter().zip(&got) {
                let scale = w.abs().max(1.0);
                assert!((w - gv).abs() / scale < 1e-9, "n={n}: {w} vs {gv}");
            }
        }
    }
}
