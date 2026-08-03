//! Self-contained scaling bench for the fixed mask product
//! `sum_bc U_bc · A_bc` — the offline `ua_mask` GEMM — reproduced over synthetic
//! `f64` panels at PIR shape. Mirrors `full_torus_f64_mask_product`'s block-tiled
//! accumulation (concurrency task M3): the `K` block columns are split into
//! contiguous groups, summed in parallel, then reduced in ascending group order.
//!
//! Demonstrates that the contraction is compute-bound and scales ~linearly with
//! threads (independent of the panel count `P`), which is the win M3 unlocks when
//! `P < cores`.

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

const ROWS_OUT: usize = 2048; // panel output rows (ring degree)
const ROWS_IN: usize = 2048; // contraction dim per block
const LWE_N: usize = 2048; // query-mask columns
const K: usize = 16; // block columns contracted per panel
const ITERS: usize = 5;

fn main() {
    let (prepared, masks) = synthetic();
    let flops = 2.0 * (K as f64) * (ROWS_OUT as f64) * (ROWS_IN as f64) * (LWE_N as f64);

    println!("mask_product bench (rows_out={ROWS_OUT}, rows_in={ROWS_IN}, lwe_n={LWE_N}, K={K})");

    // Bit-exactness: N-split and strip variants vs the sequential reference.
    let reference = mask_product_acc(&prepared, &masks, 1);
    for nt in [2usize, 7, 16, 32] {
        let got = mask_product_acc_nsplit(&prepared, &masks, nt);
        let diffs = reference
            .iter()
            .zip(&got)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!("  nsplit nt={nt:<3} vs sequential: {diffs} differing values");
    }
    for strip in [512usize, 256] {
        let got = mask_product_acc_strips(&prepared, &masks, strip);
        let diffs = reference
            .iter()
            .zip(&got)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!("  strips n={strip:<4} vs sequential: {diffs} differing values");
    }

    println!("== K-split (partials + reduction, status quo) ==");
    println!(
        "  {:<10}{:>12}{:>12}{:>12}",
        "threads", "ms", "GFLOP/s", "speedup"
    );
    let mut base_ms = 0.0;
    for nt in [1usize, 2, 4, 8, 16] {
        let avg = time_average(ITERS, || {
            let acc = mask_product_acc(&prepared, &masks, nt);
            black_box(acc[0]);
        });
        let ms = millis(avg);
        if nt == 1 {
            base_ms = ms;
        }
        let gflops = flops / (avg.as_secs_f64() * 1e9);
        println!(
            "  {:<10}{:>12.3}{:>12.1}{:>12.2}",
            nt,
            ms,
            gflops,
            base_ms / ms
        );
    }

    println!("== N-split (disjoint dst column strips, no reduction) ==");
    println!(
        "  {:<10}{:>12}{:>12}{:>12}",
        "threads", "ms", "GFLOP/s", "speedup"
    );
    for nt in [1usize, 2, 4, 8, 16, 32] {
        let avg = time_average(ITERS, || {
            let acc = mask_product_acc_nsplit(&prepared, &masks, nt);
            black_box(acc[0]);
        });
        let ms = millis(avg);
        let gflops = flops / (avg.as_secs_f64() * 1e9);
        println!(
            "  {:<10}{:>12.3}{:>12.1}{:>12.2}",
            nt,
            ms,
            gflops,
            base_ms / ms
        );
    }

    println!("== single-thread dst strip width (cache-blocking probe) ==");
    println!("  {:<10}{:>12}{:>12}", "strip_n", "ms", "GFLOP/s");
    for strip in [LWE_N, 1024, 512, 256, 128] {
        let avg = time_average(ITERS, || {
            let acc = mask_product_acc_strips(&prepared, &masks, strip);
            black_box(acc[0]);
        });
        let gflops = flops / (avg.as_secs_f64() * 1e9);
        println!("  {:<10}{:>12.3}{:>12.1}", strip, millis(avg), gflops);
    }
}

struct Panel {
    values: Vec<f64>,
    rows_out: usize,
    rows_in: usize,
}

struct Mask {
    values: Vec<f64>,
}

fn synthetic() -> (Vec<Panel>, Vec<Mask>) {
    let mut s = 0x1234_5678_9abc_def0u64;
    let prepared = (0..K)
        .map(|_| Panel {
            values: (0..ROWS_OUT * ROWS_IN)
                .map(|_| (prng(&mut s) * 65536.0 - 32768.0).round())
                .collect(),
            rows_out: ROWS_OUT,
            rows_in: ROWS_IN,
        })
        .collect();
    let masks = (0..K)
        .map(|_| Mask {
            values: (0..ROWS_IN * LWE_N).map(|_| prng(&mut s) - 0.5).collect(),
        })
        .collect();
    (prepared, masks)
}

fn prng(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Block-tiled accumulation matching `common::mask_product_acc`.
fn mask_product_acc(prepared: &[Panel], masks: &[Mask], mask_threads: usize) -> Vec<f64> {
    let k = prepared.len();
    let nt = mask_threads.clamp(1, k);
    let acc_len = ROWS_OUT * LWE_N;
    if nt <= 1 {
        let mut acc = vec![0.0f64; acc_len];
        accumulate_range(&mut acc, prepared, masks, 0..k);
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
    let mut partials: Vec<Vec<f64>> = (0..nt).map(|_| vec![0.0f64; acc_len]).collect();
    std::thread::scope(|scope| {
        for (part, range) in partials.iter_mut().zip(ranges) {
            scope.spawn(move || accumulate_range(part, prepared, masks, range));
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

fn accumulate_range(
    acc: &mut [f64],
    prepared: &[Panel],
    masks: &[Mask],
    range: std::ops::Range<usize>,
) {
    for bc in range {
        let u = &prepared[bc];
        gemm_f64_add(
            acc,
            &u.values,
            &masks[bc].values,
            u.rows_out,
            u.rows_in,
            LWE_N,
        );
    }
}

/// N-split: each thread owns a contiguous strip of dst *columns* and runs the
/// full K contraction for it in ascending block order. No partial buffers, no
/// reduction, and every output value is produced by exactly one thread with a
/// fixed accumulation order, so the result is bit-identical for every `nt`.
fn mask_product_acc_nsplit(prepared: &[Panel], masks: &[Mask], nt: usize) -> Vec<f64> {
    let nt = nt.clamp(1, LWE_N);
    let mut acc = vec![0.0f64; ROWS_OUT * LWE_N];
    let base = LWE_N / nt;
    let rem = LWE_N % nt;
    struct SendPtr(*mut f64);
    unsafe impl Send for SendPtr {}
    unsafe impl Sync for SendPtr {}
    let acc_ptr = SendPtr(acc.as_mut_ptr());
    let acc_ref = &acc_ptr;
    std::thread::scope(|scope| {
        let mut col = 0;
        for i in 0..nt {
            let width = base + usize::from(i < rem);
            let col_start = col;
            col += width;
            scope.spawn(move || {
                let dst = unsafe { acc_ref.0.add(col_start) };
                for bc in 0..K {
                    let u = &prepared[bc];
                    gemm_f64_add_cols(
                        dst,
                        &u.values,
                        &masks[bc].values[col_start..],
                        u.rows_out,
                        u.rows_in,
                        width,
                        LWE_N,
                    );
                }
            });
        }
    });
    acc
}

/// Sequential, but the dst (and rhs) columns are processed in strips of
/// `strip_n`, shrinking the dst tile per GEMM call.
fn mask_product_acc_strips(prepared: &[Panel], masks: &[Mask], strip_n: usize) -> Vec<f64> {
    let mut acc = vec![0.0f64; ROWS_OUT * LWE_N];
    let mut col = 0;
    while col < LWE_N {
        let width = strip_n.min(LWE_N - col);
        for bc in 0..K {
            let u = &prepared[bc];
            gemm_f64_add_cols(
                unsafe { acc.as_mut_ptr().add(col) },
                &u.values,
                &masks[bc].values[col..],
                u.rows_out,
                u.rows_in,
                width,
                LWE_N,
            );
        }
        col += width;
    }
    acc
}

/// `dst[.., 0..n_cols] += lhs * rhs[.., 0..n_cols]` where dst and rhs are
/// column strips of row-major matrices with row stride `stride`.
fn gemm_f64_add_cols(
    dst: *mut f64,
    lhs: &[f64],
    rhs: &[f64],
    m: usize,
    k: usize,
    n_cols: usize,
    stride: usize,
) {
    let alpha = 1.0f64;
    unsafe {
        private_gemm_x86::gemm(
            private_gemm_x86::DType::F64,
            private_gemm_x86::IType::U64,
            gemm_instr_set(),
            m,
            n_cols,
            k,
            dst.cast(),
            stride as isize,
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
            stride as isize,
            1,
            false,
            (&raw const alpha).cast(),
            1,
        );
    }
}

fn gemm_instr_set() -> private_gemm_x86::InstrSet {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return private_gemm_x86::InstrSet::Avx512;
        }
    }
    private_gemm_x86::InstrSet::Avx256
}

/// `dst += lhs * rhs`, row-major, single-threaded — same call as the server's
/// `gemm_f64_add`.
fn gemm_f64_add(dst: &mut [f64], lhs: &[f64], rhs: &[f64], m: usize, k: usize, n: usize) {
    let alpha = 1.0f64;
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
