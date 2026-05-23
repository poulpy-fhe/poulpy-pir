use crate::{
    database::Database,
    encoding::{U256_BASE65535_DIGITS, decode_u256_base65535},
};
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::layouts::{Module, ZnxView};

type BE = FFT64Ref;

/// `Database::encode_u256_shard` must lay each 256-bit payload out as 16
/// consecutive row coefficients at a single column (one base-65535 digit per
/// row). The test populates a DB with a known set of payloads, then walks the
/// stored coefficient matrices and reconstructs each payload by reading the 16
/// digits at the layout's computed position.
#[test]
fn database_encode_u256_shard_roundtrips_via_matrix_inspection() {
    let n: usize = 16; // smallest n that fits one U256 per column (16 / 16 = 1).
    let base2k: usize = 16;
    let cols: usize = 2 * n; // k_blocks = 2
    let nb_matrices: usize = 3; // -> t = 4 after interpolation padding
    let db_entries: usize = nb_matrices * n * cols;

    let module = Module::<BE>::new(n as u64);
    let mut db = Database::<BE>::new(&module, db_entries, base2k, cols);

    // Capacity = nb · (n / 16) · cols = 3 · 1 · 32 = 96 payloads.
    let capacity = db.u256_payload_capacity(&module);
    assert_eq!(capacity, nb_matrices * (n / U256_BASE65535_DIGITS) * cols);
    assert_eq!(capacity, 96);

    // Deterministic distinct payloads. Top 12 bits zero -> safely below 65535^16.
    let mut payloads: Vec<[u8; 32]> = (0..capacity)
        .map(|e| {
            let mut v = [0u8; 32];
            // Spread the index across the bytes so digits are non-trivial.
            for (i, byte) in v.iter_mut().enumerate() {
                *byte = ((e.wrapping_mul(31).wrapping_add(i * 17 + 5)) & 0xFF) as u8;
            }
            v[31] = 0;
            v[30] &= 0x0F;
            v
        })
        .collect();
    // Sprinkle in a few corner cases.
    payloads[0] = [0u8; 32];
    payloads[1] = {
        let mut v = [0u8; 32];
        v[0] = 1;
        v
    };
    payloads[2] = [0xAAu8; 32];
    payloads[2][31] = 0;
    payloads[2][30] &= 0x0F;

    db.encode_u256_shard(&module, 0, &payloads);

    let payloads_per_column = n / U256_BASE65535_DIGITS; // = 1 here
    let payloads_per_matrix = payloads_per_column * cols; // = 32 here
    let blocks = cols / n; // = 2 here

    for (e, expected) in payloads.iter().enumerate() {
        let m = e / payloads_per_matrix;
        let e_local = e % payloads_per_matrix;
        let rb = e_local / cols;
        let c = e_local % cols;
        let row_out_start = rb * U256_BASE65535_DIGITS;
        let block = c / n;
        let row_in = c % n;

        let sub_matrix = &db.matrices()[m * blocks + block];
        let mut digits = [0i16; U256_BASE65535_DIGITS];
        for (k, slot) in digits.iter_mut().enumerate() {
            let stored = sub_matrix.data().at(row_out_start + k, 0)[row_in];
            assert!(
                (-32767..=32767).contains(&stored),
                "stored value {stored} outside centred Z_65535 range at e={e}, k={k}",
            );
            *slot = stored as i16;
        }
        let got = decode_u256_base65535(&digits);
        assert_eq!(
            &got, expected,
            "payload {e} did not round-trip through encode_u256_shard",
        );
    }
}

/// Out-of-range writes must be caught by the capacity assert.
#[test]
#[should_panic(expected = "u256 shard writes past the configured capacity")]
fn database_encode_u256_shard_rejects_overflow() {
    let n: usize = 16;
    let base2k: usize = 16;
    let cols: usize = n;
    let nb_matrices: usize = 1;
    let db_entries: usize = nb_matrices * n * cols;

    let module = Module::<BE>::new(n as u64);
    let mut db = Database::<BE>::new(&module, db_entries, base2k, cols);

    let capacity = db.u256_payload_capacity(&module);
    let too_many = vec![[0u8; 32]; capacity + 1];
    db.encode_u256_shard(&module, 0, &too_many);
}
