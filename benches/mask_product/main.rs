//! Benchmarks one full `n x n` fixed mask product `U * A`.
//!
//! Each timed path starts from the same canonical inputs:
//! - `U`: `CoeffMatrix<i16>` in the normal PIR coefficient-matrix layout.
//! - `A`: `LWEMatrix` mask in the 2x32 coarse torus layout, with 54 meaningful
//!   torus bits.
//!
//! Timings mirror the production f64 path: the fixed query mask `A` is decoded to
//! its working representation once up front (as the server does at SETUP, where it
//! is cached in `QueryMask`), so only the per-product work is timed — the
//! coefficient decode (`U`), the `dgemm`, and the final output encoding into the
//! 3x18 pack regime. The f64<->i64 conversions go through the backend `reim`
//! kernels (`reim_from_znx`/`reim_to_znx`) exactly as production does. Work buffers
//! are reused across iterations, so allocation is intentionally excluded.

use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

use faer::{Accum, Par, linalg::matmul::matmul, mat::MatMut, mat::MatRef};
use matrixmultiply::dgemm;
use poulpy_core::layouts::{
    Base2K, Degree, LWEInfos, LWEMatrix, LWEMatrixLayout, ModuleCoreAlloc, TorusPrecision,
};
use poulpy_cpu_avx::FFT64Avx;
use poulpy_cpu_ref::reference::fft64::reim::ReimArith;
use poulpy_hal::layouts::{Backend, HostDataMut, Module, ZnxViewMut};
use poulpy_pir::database::CoeffMatrix;

type BE = FFT64Avx;

const DEFAULT_N: usize = 2048;
const DEFAULT_ITERS: usize = 5;
const MASK_BASE2K: usize = 32;
const PACK_BASE2K: usize = 18;
const TORUS_BITS: usize = 54;
const SPLIT_BITS: usize = 27;
const DEFAULT_TILES: usize = 8;

fn main() {
    let n = env_usize("PIR_MASK_BENCH_N", DEFAULT_N);
    let iters = env_usize("PIR_MASK_BENCH_ITERS", DEFAULT_ITERS);

    let mut bench = Bench::new(n);
    bench.full_f64();
    bench.faer_f64();
    bench.split27_f64();

    // `split27_f64` keeps the full 54-bit torus precision exactly, so it is the
    // reference for the lossy single-`f64` paths.
    let ref_out = decode_output(&bench.split_out);
    let full_err = diff_stats(&ref_out, &decode_output(&bench.full_out));
    let faer_err = diff_stats(&ref_out, &decode_output(&bench.faer_out));

    let full_avg = time_average(iters, || {
        bench.full_f64();
        black_box(&bench.full_out);
    });
    let faer_avg = time_average(iters, || {
        bench.faer_f64();
        black_box(&bench.faer_out);
    });
    let split_avg = time_average(iters, || {
        bench.split27_f64();
        black_box(&bench.split_out);
    });

    println!(
        "mask product bench (n={n}, iters={iters}, torus_bits={TORUS_BITS}, output={}x{})",
        bench.out_infos.size(),
        PACK_BASE2K
    );
    println!(
        "  per-product timing: U decode + dgemm + output encode (reim kernels); \
         the fixed mask A is decoded once at setup; allocation is excluded"
    );
    println!(
        "  {:<18}{:>12}{:>18}{:>18}",
        "path", "ms", "max diff units", "max diff log2"
    );
    print_row("full_f64", full_avg, full_err);
    print_row("faer_f64_seq", faer_avg, faer_err);
    print_row("split27_f64 (ref)", split_avg, DiffStats::default());

    // Shape study: the DB-scale mask product `U*A = sum_i U_i*A_i` over `tiles`
    // block-columns, computed two ways with faer — split (one `n x n` matmul per
    // tile, accumulated) vs full (one `m x (tiles*n) * (tiles*n) x n` matmul).
    let tiles = env_usize("PIR_MASK_BENCH_TILES", DEFAULT_TILES);
    let m = env_usize("PIR_MASK_BENCH_M", n);
    bench_faer_shapes(m, n, tiles, iters);
}

/// Compares faer on the two equivalent layouts of `U*A = sum_i U_i*A_i`:
/// - `faer_split`: `tiles` separate `m x n` * `n x n` matmuls, accumulated.
/// - `faer_full` : one `m x (tiles*n)` * `(tiles*n) x n` matmul.
///
/// Both read the *same* stacked operands (`U_i` are column-blocks of `lhs`, `A_i`
/// the matching row-blocks of `rhs`), so they must agree numerically. Same FLOPs
/// (`2*m*n*tiles*n`); only the GEMM blocking/overhead differs. (This is `U*A`, the
/// mask side only; the body `b` would add a single RHS column.)
fn bench_faer_shapes(m: usize, n: usize, tiles: usize, iters: usize) {
    assert!(tiles > 0 && m > 0 && n > 0);
    let kdim = tiles * n;
    let mut lhs = vec![0.0f64; m * kdim];
    let mut rhs = vec![0.0f64; kdim * n];
    fill_lhs_f64(&mut lhs, 0xA1A1_A1A1);
    fill_rhs_f64(&mut rhs, 0xB2B2_B2B2);
    let mut acc_full = vec![0.0f64; m * n];
    let mut acc_split = vec![0.0f64; m * n];

    let lhs_mat = MatRef::from_row_major_slice(&lhs, m, kdim);
    let rhs_mat = MatRef::from_row_major_slice(&rhs, kdim, n);

    let do_full = |acc: &mut [f64]| {
        let acc = MatMut::from_row_major_slice_mut(acc, m, n);
        matmul(acc, Accum::Replace, lhs_mat, rhs_mat, 1.0, Par::Seq);
    };
    let do_split = |acc: &mut [f64]| {
        acc.fill(0.0);
        for i in 0..tiles {
            let acc = MatMut::from_row_major_slice_mut(acc, m, n);
            matmul(
                acc,
                Accum::Add,
                lhs_mat.subcols(i * n, n),
                rhs_mat.subrows(i * n, n),
                1.0,
                Par::Seq,
            );
        }
    };

    // Warm up once and check the two layouts agree.
    do_full(&mut acc_full);
    do_split(&mut acc_split);
    let max_diff = acc_full
        .iter()
        .zip(&acc_split)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    let full_avg = time_average(iters, || {
        do_full(&mut acc_full);
        black_box(&acc_full);
    });
    let split_avg = time_average(iters, || {
        do_split(&mut acc_split);
        black_box(&acc_split);
    });

    let gflop = 2.0 * (m as f64) * (n as f64) * (kdim as f64) / 1e9;
    println!();
    println!(
        "faer shape study (single-thread): U*A = sum_i U_i*A_i, m={m}, n={n}, tiles={tiles}, K={kdim}"
    );
    println!("  split vs full agree to max abs diff {max_diff:.3e}");
    println!("  {:<18}{:>12}{:>14}", "path", "ms", "GFLOP/s");
    let row = |name: &str, d: Duration| {
        println!(
            "  {:<18}{:>12.3}{:>14.2}",
            name,
            millis(d),
            gflop / d.as_secs_f64()
        );
    };
    row("faer_full", full_avg);
    row("faer_split", split_avg);
}

fn fill_lhs_f64(buf: &mut [f64], seed: u64) {
    for (i, x) in buf.iter_mut().enumerate() {
        let r = splitmix64(i as u64 ^ seed);
        *x = ((r % 65_535) as i64 - 32_767) as f64;
    }
}

fn fill_rhs_f64(buf: &mut [f64], seed: u64) {
    for (i, x) in buf.iter_mut().enumerate() {
        let r = splitmix64(i as u64 ^ seed);
        *x = (r as f64 / u64::MAX as f64) - 0.5;
    }
}

struct Bench {
    module: Module<BE>,
    out_infos: LWEMatrixLayout,
    u: CoeffMatrix,
    full_out: LWEMatrix<<BE as Backend>::OwnedBuf>,
    faer_out: LWEMatrix<<BE as Backend>::OwnedBuf>,
    split_out: LWEMatrix<<BE as Backend>::OwnedBuf>,
    lhs: Vec<f64>,
    rhs: Vec<f64>,
    rhs_hi: Vec<f64>,
    rhs_lo: Vec<f64>,
    acc: Vec<f64>,
    acc_hi: Vec<f64>,
    acc_lo: Vec<f64>,
}

impl Bench {
    fn new(n: usize) -> Self {
        assert!(n > 0, "n must be non-zero");
        let module = Module::<BE>::new(n as u64);
        let out_infos = LWEMatrixLayout {
            rows: n,
            n: Degree(n as u32),
            base2k: Base2K(PACK_BASE2K as u32),
            k: TorusPrecision(TORUS_BITS as u32),
        };
        let coarse_infos = LWEMatrixLayout {
            rows: n,
            n: Degree(n as u32),
            base2k: Base2K(MASK_BASE2K as u32),
            k: TorusPrecision((TORUS_BITS.div_ceil(MASK_BASE2K) * MASK_BASE2K) as u32),
        };

        let mut u = CoeffMatrix::zeros(n, n);
        fill_u(&mut u);

        let mut a = module.lwe_matrix_alloc_from_infos(&coarse_infos);
        fill_a(&mut a);

        let full_out = module.lwe_matrix_alloc_from_infos(&out_infos);
        let faer_out = module.lwe_matrix_alloc_from_infos(&out_infos);
        let split_out = module.lwe_matrix_alloc_from_infos(&out_infos);
        let cells = n * n;

        // The query mask `A` is fixed, so its f64 working representation is decoded
        // once here, mirroring production where it is cached in `QueryMask` at SETUP
        // and never re-decoded per product.
        let mut rhs = vec![0.0; cells];
        decode_torus_mask_f64(&mut rhs, &a, n);
        let mut rhs_hi = vec![0.0; cells];
        let mut rhs_lo = vec![0.0; cells];
        decode_torus_mask_split27_f64(&mut rhs_hi, &mut rhs_lo, &a, n);

        Self {
            module,
            out_infos,
            u,
            full_out,
            faer_out,
            split_out,
            lhs: vec![0.0; cells],
            rhs,
            rhs_hi,
            rhs_lo,
            acc: vec![0.0; cells],
            acc_hi: vec![0.0; cells],
            acc_lo: vec![0.0; cells],
        }
    }

    fn full_f64(&mut self) {
        let n = self.module.n();
        self.acc.fill(0.0);
        // `U` is decoded per product (as in `full_torus_f64_mask_product`); the
        // mask `A` (`self.rhs`) was decoded once at construction.
        decode_coeff_matrix_f64(&mut self.lhs, &self.u, n);
        unsafe {
            dgemm(
                n,
                n,
                n,
                1.0,
                self.lhs.as_ptr(),
                n as isize,
                1,
                self.rhs.as_ptr(),
                n as isize,
                1,
                0.0,
                self.acc.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        encode_torus_mask_f64(&mut self.full_out, &self.acc, n);
    }

    /// Same full-torus product as [`Self::full_f64`], but the dense `f64` GEMM is
    /// `faer`'s `matmul` forced to sequential (`Par::Seq`, single core) instead of
    /// `matrixmultiply::dgemm`. Conversions/encode are identical to production.
    fn faer_f64(&mut self) {
        let n = self.module.n();
        decode_coeff_matrix_f64(&mut self.lhs, &self.u, n);
        let lhs = MatRef::from_row_major_slice(&self.lhs, n, n);
        let rhs = MatRef::from_row_major_slice(&self.rhs, n, n);
        let acc = MatMut::from_row_major_slice_mut(&mut self.acc, n, n);
        matmul(acc, Accum::Replace, lhs, rhs, 1.0, Par::Seq);
        encode_torus_mask_f64(&mut self.faer_out, &self.acc, n);
    }

    fn split27_f64(&mut self) {
        let n = self.module.n();
        self.acc_hi.fill(0.0);
        self.acc_lo.fill(0.0);
        // Mask split parts (`self.rhs_hi`/`self.rhs_lo`) were decoded once at
        // construction; only the per-product `U` decode is timed here.
        decode_coeff_matrix_f64(&mut self.lhs, &self.u, n);
        unsafe {
            dgemm(
                n,
                n,
                n,
                1.0,
                self.lhs.as_ptr(),
                n as isize,
                1,
                self.rhs_hi.as_ptr(),
                n as isize,
                1,
                0.0,
                self.acc_hi.as_mut_ptr(),
                n as isize,
                1,
            );
            dgemm(
                n,
                n,
                n,
                1.0,
                self.lhs.as_ptr(),
                n as isize,
                1,
                self.rhs_lo.as_ptr(),
                n as isize,
                1,
                0.0,
                self.acc_lo.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        encode_torus_mask_split27_f64(&mut self.split_out, &self.acc_hi, &self.acc_lo, n);
    }
}

fn fill_u(u: &mut CoeffMatrix) {
    let n = u.rows_in();
    for out_row in 0..u.rows_out() {
        let row = u.row_mut(out_row);
        for (in_row, cell) in row.iter_mut().enumerate().take(n) {
            let x = splitmix64((out_row as u64) << 32 ^ in_row as u64 ^ 0x9e37_79b9_7f4a_7c15);
            *cell = ((x % 65_535) as i64 - 32_767) as i16;
        }
    }
}

fn fill_a(a: &mut LWEMatrix<<BE as Backend>::OwnedBuf>) {
    let n = a.mask().n();
    let mut col = vec![0i64; n];
    for c in 0..a.mask().cols() {
        for (r, value) in col.iter_mut().enumerate() {
            let x = splitmix64((c as u64) << 32 ^ r as u64 ^ 0xd1b5_4a32_d192_ed03);
            *value = centered_torus_units_i128(
                (x as i128) & ((1i128 << TORUS_BITS) - 1),
                torus_modulus_i128(),
            ) as i64;
        }
        a.mask_mut()
            .encode_vec_i64(MASK_BASE2K, c, TORUS_BITS, &col);
    }
    a.body_mut().raw_mut().fill(0);
}

fn decode_coeff_matrix_f64(out: &mut [f64], u: &CoeffMatrix, n: usize) {
    let mut row = vec![0i64; n];
    for out_row in 0..n {
        for (slot, &v) in row.iter_mut().zip(u.row(out_row)) {
            *slot = v as i64;
        }
        // i16-bounded entries widen to their centered values; use the backend
        // kernel for the i64 -> f64 cast (as the production path does).
        FFT64Avx::reim_from_znx(&mut out[out_row * n..(out_row + 1) * n], &row);
    }
}

fn decode_torus_mask_f64(out: &mut [f64], a: &LWEMatrix<<BE as Backend>::OwnedBuf>, n: usize) {
    let scale = (torus_modulus_i128() as f64).recip();
    let mut col = vec![0i64; n];
    for c in 0..n {
        a.mask()
            .decode_vec_i64(MASK_BASE2K, c, TORUS_BITS, &mut col);
        for r in 0..n {
            out[r * n + c] = col[r] as f64 * scale;
        }
    }
}

fn decode_torus_mask_split27_f64(
    hi: &mut [f64],
    lo: &mut [f64],
    a: &LWEMatrix<<BE as Backend>::OwnedBuf>,
    n: usize,
) {
    let split = 1i64 << SPLIT_BITS;
    let mut col = vec![0i64; n];
    for c in 0..n {
        a.mask()
            .decode_vec_i64(MASK_BASE2K, c, TORUS_BITS, &mut col);
        for r in 0..n {
            let x = col[r];
            let low = centered_mod_i64(x, split);
            let high = (x - low) / split;
            hi[r * n + c] = high as f64;
            lo[r * n + c] = low as f64;
        }
    }
}

fn encode_torus_mask_f64(out: &mut LWEMatrix<<BE as Backend>::OwnedBuf>, values: &[f64], n: usize) {
    let scale = torus_modulus_i128() as f64;
    // `reim_to_znx` computes `(a / divisor).round()`; we want `(a * scale).round()`.
    let divisor = scale.recip();
    let mut col_real = vec![0.0f64; n];
    let mut col = vec![0i64; n];
    for c in 0..n {
        // mod-1 pre-pass in f64 -> [-0.5, 0.5), then vectorized f64 -> i64 round
        // (matches `encode_torus_mask_f64` in production).
        for r in 0..n {
            let v = values[r * n + c];
            col_real[r] = v - (v + 0.5).floor();
        }
        FFT64Avx::reim_to_znx(&mut col, divisor, &col_real);
        out.mask_mut()
            .encode_vec_i64(PACK_BASE2K, c, TORUS_BITS, &col);
    }
    out.body_mut().raw_mut().fill(0);
}

fn encode_torus_mask_split27_f64(
    out: &mut LWEMatrix<<BE as Backend>::OwnedBuf>,
    hi: &[f64],
    lo: &[f64],
    n: usize,
) {
    let modulus = torus_modulus_i128();
    let split = 1i128 << SPLIT_BITS;
    let mut col = vec![0i64; n];
    for c in 0..n {
        for r in 0..n {
            let high = hi[r * n + c].round() as i128;
            let low = lo[r * n + c].round() as i128;
            col[r] = centered_torus_units_i128(high * split + low, modulus) as i64;
        }
        out.mask_mut()
            .encode_vec_i64(PACK_BASE2K, c, TORUS_BITS, &col);
    }
    out.body_mut().raw_mut().fill(0);
}

fn decode_output(out: &LWEMatrix<<BE as Backend>::OwnedBuf>) -> Vec<i64> {
    let n = out.mask().n();
    let mut flat = vec![0i64; n * out.mask().cols()];
    let mut col = vec![0i64; n];
    for c in 0..out.mask().cols() {
        out.mask()
            .decode_vec_i64(PACK_BASE2K, c, TORUS_BITS, &mut col);
        for r in 0..n {
            flat[r * n + c] = col[r];
        }
    }
    flat
}

#[derive(Clone, Copy, Default)]
struct DiffStats {
    max_units: i128,
}

fn diff_stats(lhs: &[i64], rhs: &[i64]) -> DiffStats {
    let modulus = torus_modulus_i128();
    let max_units = lhs
        .iter()
        .zip(rhs)
        .map(|(&a, &b)| centered_torus_units_i128(b as i128 - a as i128, modulus).abs())
        .max()
        .unwrap_or(0);
    DiffStats { max_units }
}

fn print_row(name: &str, avg: Duration, diff: DiffStats) {
    let log2 = if diff.max_units == 0 {
        f64::NEG_INFINITY
    } else {
        (diff.max_units as f64 / torus_modulus_i128() as f64).log2()
    };
    println!(
        "  {:<18}{:>12.3}{:>18}{:>18.3}",
        name,
        millis(avg),
        diff.max_units,
        log2
    );
}

fn time_average(iterations: usize, mut f: impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed() / iterations as u32
}

fn millis(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000.0
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn torus_modulus_i128() -> i128 {
    1i128 << TORUS_BITS
}

fn centered_mod_i64(value: i64, modulus: i64) -> i64 {
    let half = modulus >> 1;
    let mut reduced = value.rem_euclid(modulus);
    if reduced >= half {
        reduced -= modulus;
    }
    reduced
}

fn centered_torus_units_i128(value: i128, modulus: i128) -> i128 {
    let half = modulus >> 1;
    let mut reduced = value.rem_euclid(modulus);
    if reduced >= half {
        reduced -= modulus;
    }
    reduced
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
