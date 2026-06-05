use crate::{
    config::{Collapse, INSPIRE_REC_32B},
    database::DatabaseLayout,
    payload::{Payload, U256P65535, U256P65536},
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
    let layout = L::new(4 * N, 32 * N);

    assert_eq!(layout.payload_digits(), DIGITS);
    assert_eq!(layout.payload_digits(), 17); // full 2^256, not the 16-digit bound
    assert_eq!(layout.p(), 65535);

    assert_eq!(layout.rows(), 4 * N);
    assert_eq!(layout.cols(), 32 * N);
    // payloads/col = floor(N / 17) = 120 (8 rows per column are unused slack).
    assert_eq!(layout.payloads_per_column(N), N / DIGITS);
    assert_eq!(layout.payloads_per_column(N), 120);
    assert_eq!(layout.payloads_per_block_row(N), (N / DIGITS) * 32 * N);
    assert_eq!(layout.num_payloads(N), 4 * (N / DIGITS) * 32 * N);
    assert_eq!(layout.interpolation_t(N), 4); // next_pow2(block_rows)
    assert_eq!(layout.total_i16_slots(), layout.rows() * layout.cols());
    assert_eq!(layout.total_payload_bytes(N), layout.num_payloads(N) * 32);
}

/// `with_capacity` sizes the block-rows to just fit a target payload count.
#[test]
fn with_capacity_is_tight() {
    const N: usize = 2048;
    let target = 1usize << 30;
    let block_cols = 2;
    let per_block_row = (N / DIGITS) * (block_cols * N);
    let block_rows = target.div_ceil(per_block_row.max(1)).max(1);
    let layout = L::new(block_rows * N, block_cols * N);
    assert!(layout.num_payloads(N) >= target);
    // one block-row fewer would not fit.
    assert!((layout.block_rows(N) - 1) * layout.payloads_per_block_row(N) < target);
}

/// `from_total_bytes` covers the requested byte budget and stays within `2n`.
#[test]
fn from_total_bytes_covers_budget() {
    const N: usize = 2048;
    const TOTAL_BYTES: usize = 32 << 30; // 32 GB of 32-byte payloads
    let block_cols = 2;
    let target = TOTAL_BYTES / size_of::<[u8; 32]>();
    let per_block_row = (N / DIGITS) * (block_cols * N);
    let block_rows = target.div_ceil(per_block_row.max(1)).max(1);
    let layout = L::new(block_rows * N, block_cols * N);
    assert!(layout.total_payload_bytes(N) >= TOTAL_BYTES);
    assert!(layout.interpolation_t(N) <= 2 * N);
}

/// `instantiate` gives a database whose capacity matches the layout.
#[test]
fn instantiate_matches_capacity() {
    let n: usize = 32; // small ring (n ≥ 17 so a payload fits a column)
    let module = Module::<BE>::new(n as u64);
    let layout = L::new(3 * n, 2 * n);
    let db = layout.instantiate(&module, /* base2k */ 16, n);
    assert_eq!(db.payload_capacity(), layout.num_payloads(n));
    // 3 block-rows × 2 block-cols ⇒ 6 sub-matrices.
    assert_eq!(db.matrices().len(), 3 * 2);
}

/// The second dimension (`interpolation_t = next_pow2(block_rows)`) cannot exceed
/// `2n`, the count of distinct roots of unity in `Z[X]/(X^n+1)`.
#[test]
#[should_panic(expected = "second dimension")]
fn rejects_second_dimension_over_2n() {
    let n: usize = 32; // 2n = 64
    let layout = L::new(65 * n, n); // next_pow2(65) = 128 > 64
    let _ = layout.interpolation_t(n); // the bound is enforced here
}

/// `interpolation_t == 2n` is the largest allowed second dimension.
#[test]
fn accepts_second_dimension_exactly_2n() {
    let n: usize = 32; // 2n = 64
    let layout = L::new(64 * n, n); // next_pow2(64) = 64
    assert_eq!(layout.interpolation_t(n), 64);
}

/// The default 32-byte InsPIRe² shape: the project default doubles the paper's
/// Table 8 γ0=16 baseline to γ0=32 / γ1=1024 / γ2=32, keeping N/t=8192 and the
/// same 1 GiB parameterization (doubling γ0 halves `grid_rows_for` and doubles
/// `payloads_per_column`, so total payloads and bytes are unchanged).
#[test]
fn recursion_default_matches_paper_32_byte_payloads() {
    let params = INSPIRE_REC_32B.new::<BE>();
    let layout = DatabaseLayout::<U256P65536>::new(2048 * 32, 8192);
    assert_eq!(
        params.collapse(),
        Collapse::Recursion {
            gamma0: 32,
            gamma1: 1024,
            gamma2: 32
        }
    );
    assert_eq!(params.n(), 2048);
    assert_eq!(layout.grid_rows_for(32), 2048);
    assert_eq!(layout.cols(), 8192);
    assert_eq!(layout.payloads_per_column(32), 2);
    assert_eq!(layout.num_payloads(32), 1usize << 25);
    assert_eq!(layout.total_payload_bytes(32), 1usize << 30);
    assert_eq!(layout.column_blocks(params.n()), 4);
}

/// A payload must fit within one column: `payload_digits ≤ n`.
#[test]
#[should_panic(expected = "must fit within one column")]
fn rejects_payload_wider_than_column() {
    let layout = L::new(/* rows */ 16, 1);
    let _ = layout.payloads_per_column(16); // 17 digits > 16 rows
}
