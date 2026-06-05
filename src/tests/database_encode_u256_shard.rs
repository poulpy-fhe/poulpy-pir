use crate::{
    database::DatabaseLayout,
    payload::{Payload, U256P65535},
};
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::layouts::{Module, ZnxView};

type BE = FFT64Ref;
type L = DatabaseLayout<U256P65535>;

/// `Database::encode_shard` must lay each payload out as `P::EXPONENT` consecutive
/// row coefficients at a single column (one base-65535 digit per row). The test
/// fills the DB with known full-range payloads, then walks the stored matrices and
/// reconstructs each payload from the 17 digits at the layout's computed position.
#[test]
fn database_encode_shard_roundtrips_via_matrix_inspection() {
    let n: usize = 32; // power of two; 32 / 17 = 1 payload per column (15 rows slack)
    let base2k: usize = 16;
    let block_cols: usize = 2;
    let block_rows: usize = 3;

    let module = Module::<BE>::new(n as u64);
    let layout = L::new(block_rows * n, block_cols * n);
    let mut db = layout.instantiate(&module, base2k, n);

    let capacity = db.payload_capacity();
    assert_eq!(capacity, layout.num_payloads(n));
    assert_eq!(
        capacity,
        block_rows * (n / U256P65535::EXPONENT) * (block_cols * n)
    );

    // Deterministic distinct **full-range** payloads (no top-bit masking).
    let mut payloads: Vec<[u8; 32]> = (0..capacity)
        .map(|e| {
            let mut v = [0u8; 32];
            for (i, byte) in v.iter_mut().enumerate() {
                *byte = ((e.wrapping_mul(31).wrapping_add(i * 17 + 5)) & 0xFF) as u8;
            }
            v
        })
        .collect();
    payloads[0] = [0u8; 32];
    payloads[1] = [0xFFu8; 32]; // 2^256 − 1, the full-range extreme
    payloads[2] = [0xAAu8; 32];

    db.encode_shard(0, &payloads);

    let digits_per = U256P65535::EXPONENT; // 17
    let payloads_per_column = n / digits_per; // 2
    let payloads_per_matrix = payloads_per_column * (block_cols * n);
    let cols = block_cols * n;

    for (e, expected) in payloads.iter().enumerate() {
        assert_eq!(
            db.payload(e),
            *expected,
            "payload {e} did not round-trip through Database::payload"
        );

        let m = e / payloads_per_matrix;
        let e_local = e % payloads_per_matrix;
        let c = e_local % cols;
        let row_out_start = (e_local / cols) * digits_per;
        let block = c / n;
        let row_in = c % n;

        let sub = &db.matrices()[m * block_cols + block];
        let mut digits = vec![0i16; digits_per];
        for (k, slot) in digits.iter_mut().enumerate() {
            let stored = sub.data().at(row_out_start + k, 0)[row_in];
            assert!(
                (-32767..=32767).contains(&stored),
                "stored value {stored} outside centred Z_65535 range at e={e}, k={k}",
            );
            *slot = stored as i16;
        }
        let mut got = [0u8; 32];
        U256P65535::decode(&mut got, &digits);
        assert_eq!(
            &got, expected,
            "payload {e} did not round-trip through encode_shard"
        );
    }
}

/// Out-of-range writes must be caught by the capacity assert.
#[test]
#[should_panic(expected = "shard writes past the configured capacity")]
fn database_encode_shard_rejects_overflow() {
    let n: usize = 32;
    let module = Module::<BE>::new(n as u64);
    let layout = L::new(n, n);
    let mut db = layout.instantiate(&module, /* base2k */ 16, n);

    let capacity = db.payload_capacity();
    let too_many = vec![[0u8; 32]; capacity + 1];
    db.encode_shard(0, &too_many);
}
