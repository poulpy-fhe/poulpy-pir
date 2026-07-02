use crate::{
    config::{Collapse, DefaultPirConfig32B, DefaultPirParameters32B, DefaultScheme},
    database::DatabaseLayout,
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
    let default = DefaultPirParameters32B::canonical(DefaultScheme::Recursion { gamma0: 32 }, 1)
        .recursion()
        .expect("canonical InsPIRe² γ0=32 1 GiB must resolve to recursion params");
    let params = default.config.new::<BE>();
    let layout = default.layout;
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

#[test]
fn default_32b_grid_is_valid_and_covers_table2() {
    let all = DefaultPirParameters32B::all();

    // 27 layout shapes (see cols_window sums) × 4 schemes.
    assert_eq!(all.len(), 108, "grid size");

    // Names are unique and round-trip through from_name.
    let mut names = std::collections::BTreeSet::new();
    for &p in &all {
        assert!(names.insert(p.name()), "duplicate name {}", p.name());
        assert_eq!(DefaultPirParameters32B::from_name(&p.name()), Some(p));
    }

    for &p in &all {
        // Exact DB fill: rows·cols == 2^29·db_gib, both powers of two.
        assert_eq!(p.rows() * p.cols(), p.total_u16(), "{}", p.name());
        assert!(p.cols().is_power_of_two() && p.rows().is_power_of_two());

        // cols lies in the size's window.
        let (lo, hi) = DefaultPirParameters32B::cols_window(p.db_gib);
        let c = p.cols().trailing_zeros();
        assert!((lo..=hi).contains(&c), "{}: cols 2^{c} out of window", p.name());

        // Backend-free size accessors resolve and return nonzero sizes.
        assert!(p.query_bytes() > 0 && p.response_bytes() > 0, "{}", p.name());

        // Resolves, and the layout math holds (asserts fire inside on bad shapes).
        match p.resolve() {
            DefaultPirConfig32B::Interpolation(r) => {
                assert!(matches!(p.scheme, DefaultScheme::Interpolation));
                assert_eq!(p.gamma0(), None);
                assert_eq!(r.db_size_gib, p.db_gib);
                assert_eq!(r.layout.rows(), p.rows());
                assert_eq!(r.layout.cols(), p.cols());
                // interpolation_t <= 2n is asserted inside interpolation_t.
                let _ = r.layout.interpolation_t(r.config.n());
                assert!(r.layout.num_payloads(r.config.n()) > 0);
                assert!(p.interpolation().is_some() && p.recursion().is_none());
            }
            DefaultPirConfig32B::Recursion(r) => {
                let DefaultScheme::Recursion { gamma0 } = p.scheme else {
                    panic!("recursion resolve for non-recursion scheme")
                };
                assert_eq!(p.gamma0(), Some(gamma0));
                assert_eq!(p.gamma1(), Some(1024));
                assert_eq!(p.gamma2(), Some(gamma0));
                assert_eq!(r.gamma0, gamma0);
                assert_eq!(r.gamma1, 1024);
                assert_eq!(r.gamma2, gamma0);
                assert_eq!(r.layout.rows(), p.rows());
                assert_eq!(r.layout.cols(), p.cols());
                assert_eq!(r.layout.grid_rows_for(gamma0), p.rows() / gamma0);
                assert!(r.layout.num_payloads(gamma0) > 0);
                assert!(p.recursion().is_some() && p.interpolation().is_none());
            }
        }
    }

    // Every paper Table 2 (scheme, db_gib, cols) point is present (rows derived).
    let table2: &[(DefaultScheme, usize, usize)] = &[
        // InsPIRe
        (DefaultScheme::Interpolation, 1, 8192),
        (DefaultScheme::Interpolation, 1, 16384),
        (DefaultScheme::Interpolation, 1, 32768),
        (DefaultScheme::Interpolation, 8, 65536),
        (DefaultScheme::Interpolation, 8, 131072),
        (DefaultScheme::Interpolation, 32, 131072),
        (DefaultScheme::Interpolation, 32, 262144),
        // InsPIRe² γ0=64
        (DefaultScheme::Recursion { gamma0: 64 }, 1, 4096),
        (DefaultScheme::Recursion { gamma0: 64 }, 1, 8192),
        (DefaultScheme::Recursion { gamma0: 64 }, 1, 16384),
        (DefaultScheme::Recursion { gamma0: 64 }, 1, 32768),
        (DefaultScheme::Recursion { gamma0: 64 }, 8, 8192),
        (DefaultScheme::Recursion { gamma0: 64 }, 8, 16384),
        (DefaultScheme::Recursion { gamma0: 64 }, 8, 32768),
        (DefaultScheme::Recursion { gamma0: 64 }, 8, 65536),
        (DefaultScheme::Recursion { gamma0: 64 }, 32, 32768),
        (DefaultScheme::Recursion { gamma0: 64 }, 32, 65536),
        (DefaultScheme::Recursion { gamma0: 64 }, 32, 131072),
        // InsPIRe² γ0=32
        (DefaultScheme::Recursion { gamma0: 32 }, 1, 16384),
        (DefaultScheme::Recursion { gamma0: 32 }, 1, 32768),
        (DefaultScheme::Recursion { gamma0: 32 }, 8, 131072),
        (DefaultScheme::Recursion { gamma0: 32 }, 32, 65536),
        (DefaultScheme::Recursion { gamma0: 32 }, 32, 131072),
        (DefaultScheme::Recursion { gamma0: 32 }, 32, 262144),
    ];
    for &(scheme, db_gib, cols) in table2 {
        let want = DefaultPirParameters32B {
            scheme,
            db_gib,
            cols,
        };
        assert!(all.contains(&want), "missing Table-2 point {}", want.name());
    }

    // Canonical points (rows = 2^16) exist for every scheme/size.
    for scheme in DefaultPirParameters32B::SCHEMES {
        for db in DefaultPirParameters32B::DB_SIZES_GIB {
            let canon = DefaultPirParameters32B::canonical(scheme, db);
            assert_eq!(canon.rows(), 1 << 16, "canonical rows for {}", canon.name());
            assert!(all.contains(&canon), "canonical {} missing", canon.name());
        }
    }
}

/// A payload must fit within one column: `payload_digits ≤ n`.
#[test]
#[should_panic(expected = "must fit within one column")]
fn rejects_payload_wider_than_column() {
    let layout = L::new(/* rows */ 16, 1);
    let _ = layout.payloads_per_column(16); // 17 digits > 16 rows
}
