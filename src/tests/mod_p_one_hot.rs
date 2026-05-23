use crate::encoding::ModPEncoder;
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::layouts::{Module, VecZnx, ZnxViewMut};

type BE = FFT64Ref;

/// The pre-computed `encoded_unit` shortcut must agree with the generic
/// `encode_coeff_i64(1)` path it bypasses (and `encode_coeff_i64(0)` must be 0).
#[test]
fn encoded_unit_matches_general_encode_of_one() {
    for &(p, k) in &[
        (65535i64, 16usize),
        (65535, 36),
        (65535, 54),
        (65537, 18),
        (257, 30),
    ] {
        let encoder = ModPEncoder::new(p, k);
        assert_eq!(
            encoder.encoded_unit(),
            encoder.encode_coeff_i64(1),
            "encoded_unit ≠ encode_coeff_i64(1) for p={p}, k={k}",
        );
        assert_eq!(encoder.encode_coeff_i64(0), 0);
    }
}

/// `encode_one_hot_into` round-trips: decoding the resulting plaintext must
/// yield `1` at the chosen `hot_slot` and `0` everywhere else.
#[test]
fn encode_one_hot_into_decodes_to_one_at_hot_slot() {
    let n: usize = 8;
    let base2k: usize = 18;
    let k: usize = 36;
    let size: usize = k.div_ceil(base2k);

    let module = Module::<BE>::new(n as u64);
    let encoder = ModPEncoder::new(65535, k);

    for hot_slot in [0, 1, n / 2, n - 1] {
        let mut pt: VecZnx<Vec<u8>> = module.vec_znx_alloc(1, size);
        // Stamp garbage in every limb to confirm the helper clears it.
        for limb in 0..size {
            for slot in pt.at_mut(0, limb).iter_mut() {
                *slot = 12345;
            }
        }

        encoder.encode_one_hot_into(&mut pt, base2k, 0, hot_slot);

        let mut decoded = vec![0i64; n];
        encoder.decode_vec_i64(&pt, base2k, 0, &mut decoded);
        for (i, &v) in decoded.iter().enumerate() {
            let want = i64::from(i == hot_slot);
            assert_eq!(v, want, "hot_slot={hot_slot}, coeff={i}");
        }
    }
}

/// `encode_zero_into` must zero every limb of the target column and leave
/// untouched columns alone.
#[test]
fn encode_zero_into_zeros_only_the_target_column() {
    let n: usize = 8;
    let base2k: usize = 18;
    let k: usize = 36;
    let size: usize = k.div_ceil(base2k);

    let module = Module::<BE>::new(n as u64);
    let encoder = ModPEncoder::new(65535, k);
    let mut pt: VecZnx<Vec<u8>> = module.vec_znx_alloc(2, size);

    for col in 0..2 {
        for limb in 0..size {
            for slot in pt.at_mut(col, limb).iter_mut() {
                *slot = 7777;
            }
        }
    }

    encoder.encode_zero_into(&mut pt, 0);

    for limb in 0..size {
        for &v in pt.at_mut(0, limb).iter() {
            assert_eq!(v, 0, "column 0 not fully zeroed at limb {limb}");
        }
        for &v in pt.at_mut(1, limb).iter() {
            assert_eq!(v, 7777, "column 1 was modified at limb {limb}");
        }
    }

    let mut decoded = vec![0i64; n];
    encoder.decode_vec_i64(&pt, base2k, 0, &mut decoded);
    assert!(
        decoded.iter().all(|&v| v == 0),
        "zero plaintext did not decode to all zeros",
    );
}
