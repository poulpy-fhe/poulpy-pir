use crate::{
    database::{DatabaseInfos, DatabaseLayout},
    payload::{Payload, U256P65535},
};
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::layouts::Module;

type BE = FFT64Ref;
/// Layout for full 256-bit payloads (17 base-65535 digits).
type L = DatabaseLayout<U256P65535>;
const DIGITS: usize = U256P65535::EXPONENT; // 17

/// Shape-driven construction: the `block_rows × block_cols` grid is the input;
/// coefficient dims and capacity are derived. Pins the payload-packing math.
#[test]
fn layout_shape_and_capacity() {
    const N: usize = 2048;
    let layout = L::new(N, /* block_rows */ 4, /* block_cols */ 32);

    assert_eq!(layout.payload_digits(), DIGITS);
    assert_eq!(layout.payload_digits(), 17); // full 2^256, not the 16-digit bound
    assert_eq!(layout.p(), 65535);

    assert_eq!(layout.rows(), 4 * N);
    assert_eq!(layout.cols(), 32 * N);
    // payloads/col = floor(N / 17) = 120 (8 rows per column are unused slack).
    assert_eq!(layout.payloads_per_column(), N / DIGITS);
    assert_eq!(layout.payloads_per_column(), 120);
    assert_eq!(layout.payloads_per_block_row(), (N / DIGITS) * 32 * N);
    assert_eq!(layout.num_payloads(), 4 * (N / DIGITS) * 32 * N);
    assert_eq!(layout.interpolation_t(), 4); // next_pow2(block_rows)
    assert_eq!(layout.total_i16_slots(), layout.rows() * layout.cols());
    assert_eq!(layout.total_payload_bytes(), layout.num_payloads() * 32);
}

/// `with_capacity` sizes the block-rows to just fit a target payload count.
#[test]
fn with_capacity_is_tight() {
    const N: usize = 2048;
    let target = 1usize << 30;
    let layout = L::with_capacity(N, /* block_cols */ 2, target);
    assert!(layout.num_payloads() >= target);
    // one block-row fewer would not fit.
    assert!((layout.block_rows() - 1) * layout.payloads_per_block_row() < target);
}

/// `from_total_bytes` covers the requested byte budget and stays within `2n`.
#[test]
fn from_total_bytes_covers_budget() {
    const N: usize = 2048;
    const TOTAL_BYTES: usize = 32 << 30; // 32 GB of 32-byte payloads
    let layout = L::from_total_bytes(N, 2, TOTAL_BYTES);
    assert!(layout.total_payload_bytes() >= TOTAL_BYTES);
    assert!(layout.interpolation_t() <= 2 * N);
}

/// `instantiate` gives a database whose capacity matches the layout.
#[test]
fn instantiate_matches_capacity() {
    let n: usize = 32; // small ring (n ≥ 17 so a payload fits a column)
    let module = Module::<BE>::new(n as u64);
    let layout = L::new(n, /* block_rows */ 3, /* block_cols */ 2);
    let db = layout.instantiate(&module, /* base2k */ 16);
    assert_eq!(db.payload_capacity(), layout.num_payloads());
    // 3 block-rows × 2 block-cols ⇒ 6 sub-matrices.
    assert_eq!(db.matrices().len(), 3 * 2);
}

/// The second dimension (`interpolation_t = next_pow2(block_rows)`) cannot exceed
/// `2n`, the count of distinct roots of unity in `Z[X]/(X^n+1)`.
#[test]
#[should_panic(expected = "second dimension")]
fn rejects_second_dimension_over_2n() {
    let n: usize = 32; // 2n = 64
    let layout = L::new(n, /* block_rows */ 65, 1); // next_pow2(65) = 128 > 64
    let _ = layout.interpolation_t(); // the bound is enforced here
}

/// `interpolation_t == 2n` is the largest allowed second dimension.
#[test]
fn accepts_second_dimension_exactly_2n() {
    let n: usize = 32; // 2n = 64
    let layout = L::new(n, /* block_rows */ 64, 1); // next_pow2(64) = 64
    assert_eq!(layout.interpolation_t(), 64);
}

/// A payload must fit within one column: `payload_digits ≤ n`.
#[test]
#[should_panic(expected = "must fit within one column")]
fn rejects_payload_wider_than_column() {
    let _ = L::new(/* n */ 16, 1, 1); // 17 digits > 16 rows
}
