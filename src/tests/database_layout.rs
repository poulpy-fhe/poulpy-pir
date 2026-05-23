use crate::{
    database::{DatabaseLayout, U256_PAYLOAD_BYTES},
    encoding::U256_BASE65535_DIGITS,
};
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::layouts::Module;

type BE = FFT64Ref;

/// Pins down the layout math for the working example used throughout the docs:
/// 32 GB of 32-byte payloads, ring degree `n = 2048`, first-dim block count
/// `k_blocks = 2`. This is the exact `(2^35 / (T·N)) × (N·T)` shape framing.
#[test]
fn database_layout_32gb_32b_n2048_t2() {
    const N: usize = 2048;
    const T: usize = 2;
    const TOTAL_BYTES: usize = 32 << 30; // 32 GB

    let layout = DatabaseLayout::from_total_bytes(N, T, 16, TOTAL_BYTES);

    // num_payloads = 2^35 / 32 = 2^30.
    assert_eq!(layout.num_payloads, 1 << 30);

    // cols = T·N = 4096.
    assert_eq!(layout.cols, T * N);

    // payloads/col = N / 16 = 128, payloads/matrix = 128 · 4096 = 2^19.
    assert_eq!(layout.payloads_per_column, N / U256_BASE65535_DIGITS);
    assert_eq!(layout.payloads_per_matrix, 1 << 19);

    // D = 2^30 / 2^19 = 2^11 = 2048 matrices, t = 2048.
    assert_eq!(layout.nb_matrices, 1 << 11);
    assert_eq!(layout.interpolation_t, 1 << 11);

    // Total i16 = 2^34 (because each i16 carries one ≈16-bit base-65535 digit).
    assert_eq!(layout.total_i16_slots(), 1usize << 34);
    // Total payload bytes = 2^35 = 32 GB (no slack on this divisible case).
    assert_eq!(layout.total_payload_bytes(), TOTAL_BYTES);
    assert_eq!(layout.unused_payload_slots(), 0);

    // Byte-matrix view matches the user-facing `(2^35 / (T·N)) × (T·N)` shape.
    let (rows_bytes, cols_bytes) = layout.byte_matrix_shape();
    assert_eq!(cols_bytes, T * N);
    assert_eq!(rows_bytes, TOTAL_BYTES / (T * N));
    assert_eq!(rows_bytes * cols_bytes, TOTAL_BYTES);
}

/// `instantiate` must give a database whose own capacity matches the layout.
#[test]
fn database_layout_instantiate_matches_capacity() {
    let n: usize = 16; // small ring so the test is cheap
    let module = Module::<BE>::new(n as u64);

    let layout = DatabaseLayout::new(
        n, /* k_blocks */ 2, /* base2k */ 16, /* payloads */ 96,
    );
    // 96 = 3 · payloads_per_matrix (= 1 · 32) at n=16, T=2: 3 matrices, slack 0.
    assert_eq!(layout.payloads_per_column, 1);
    assert_eq!(layout.payloads_per_matrix, 32);
    assert_eq!(layout.nb_matrices, 3);
    assert_eq!(layout.interpolation_t, 4);
    assert_eq!(layout.unused_payload_slots(), 0);

    let db = layout.instantiate(&module);
    assert_eq!(db.u256_payload_capacity(&module), 96);
    // 3 matrices, each tiled into k_blocks = 2 column-blocks ⇒ 6 sub-matrices.
    assert_eq!(db.matrices().len(), 3 * 2);
}

/// A non-divisible payload count rounds up to the next full matrix, leaving
/// slack capacity in the trailing matrix.
#[test]
fn database_layout_rounds_up_and_reports_slack() {
    let n: usize = 16;
    let layout = DatabaseLayout::new(n, 2, 16, /* payloads */ 33);
    // payloads_per_matrix = 32, so 33 payloads need 2 matrices with 31 unused.
    assert_eq!(layout.nb_matrices, 2);
    assert_eq!(layout.payloads_per_matrix, 32);
    assert_eq!(layout.unused_payload_slots(), 2 * 32 - 33);
    assert_eq!(layout.total_payload_bytes(), 2 * 32 * U256_PAYLOAD_BYTES);
}

/// The second dimension (matrix-axis interpolation degree) cannot exceed `2n`,
/// the count of distinct roots of unity in `Z[X]/(X^n+1)`. A layout whose
/// `nb_matrices` would push `interpolation_t` past `2n` must panic.
#[test]
#[should_panic(expected = "second dimension")]
fn database_layout_rejects_second_dimension_over_2n() {
    let n: usize = 16; // 2n = 32
    // k_blocks = 1 → payloads_per_matrix = (n/16)·n = 16, so 33 matrices' worth
    // of payloads forces interpolation_t = next_pow2(33) = 64 > 2n = 32.
    let payloads = 33 * 16;
    let _ = DatabaseLayout::new(n, 1, 16, payloads);
}

/// `nb_matrices == 2n` (interpolation_t == 2n) is the largest allowed second
/// dimension and must be accepted.
#[test]
fn database_layout_accepts_second_dimension_exactly_2n() {
    let n: usize = 16; // 2n = 32
    let payloads = 32 * 16; // exactly 32 matrices
    let layout = DatabaseLayout::new(n, 1, 16, payloads);
    assert_eq!(layout.nb_matrices, 32);
    assert_eq!(layout.interpolation_t, 32);
}

/// Sweep `k_blocks` across powers of two for a fixed 32 GB target and verify
/// the (T, D) trade-off matches the paper's `D · T = constant` relation.
#[test]
fn database_layout_t_d_tradeoff_is_constant() {
    const N: usize = 2048;
    const TOTAL_BYTES: usize = 32 << 30;
    let mut t_d_products = Vec::new();
    for k_blocks in [1, 2, 4, 8, 16, 32] {
        let layout = DatabaseLayout::from_total_bytes(N, k_blocks, 16, TOTAL_BYTES);
        t_d_products.push(layout.k_blocks * layout.nb_matrices);
        // The shape always sums back to the same 2^35 capacity.
        assert_eq!(layout.total_payload_bytes(), TOTAL_BYTES);
    }
    assert!(t_d_products.windows(2).all(|w| w[0] == w[1]));
    // T · D · payloads_per_column = num_payloads, with
    // payloads_per_column = n / 16 = 128. So T · D = 2^30 / 128 = 2^23 / 2^11 = 2^12.
    assert_eq!(t_d_products[0], 1 << 12);
}
