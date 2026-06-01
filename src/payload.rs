use std::marker::PhantomData;

pub type U256P65535 = P65535<[u8; 32]>;

pub trait Payload<B> {
    const BASIS: u16;
    const EXPONENT: usize;
    fn encode(digits: &mut [i16], a: B);
    fn decode(a: &mut B, digits: &[i16]);
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct P65535<B> {
    _phantom: PhantomData<B>,
}

impl Payload<[u8; 32]> for P65535<[u8; 32]> {
    const BASIS: u16 = 65535;
    const EXPONENT: usize = 17;

    fn encode(digits: &mut [i16], value: [u8; 32]) {
        debug_assert!(digits.len() == Self::EXPONENT);
        let mut limbs = u256_le_to_limbs(&value);

        for d in digits {
            *d = digit_to_centred_i16(divmod_65535_in_place(&mut limbs));
        }
        debug_assert!(
            limbs == [0u64; 4],
            "encode_u256_base65535: value ≥ 65535^16 (overflows 16-digit base-65535)",
        );
    }

    fn decode(a: &mut [u8; 32], digits: &[i16]) {
        debug_assert!(digits.len() == Self::EXPONENT);
        let mut acc = [0u64; 4];
        // Horner over base 65535, most-significant digit first.
        for &stored in digits.iter().rev() {
            let digit = centred_i16_to_digit(stored) as u64;
            let mut carry: u128 = digit as u128;
            for limb in acc.iter_mut() {
                let prod = (*limb as u128) * 65535u128 + carry;
                *limb = prod as u64;
                carry = prod >> 64;
            }
            debug_assert_eq!(carry, 0, "decode_u256_base65535: accumulator overflow");
        }

        for (i, limb) in acc.iter().enumerate() {
            a[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
    }
}

#[inline]
fn u256_le_to_limbs(value: &[u8; 32]) -> [u64; 4] {
    [
        u64::from_le_bytes(value[0..8].try_into().unwrap()),
        u64::from_le_bytes(value[8..16].try_into().unwrap()),
        u64::from_le_bytes(value[16..24].try_into().unwrap()),
        u64::from_le_bytes(value[24..32].try_into().unwrap()),
    ]
}

/// In-place `limbs := limbs / 65535`, returning `limbs % 65535`.
///
/// `limbs` is a little-endian U256 (`limbs[0]` is the least-significant
/// `u64`). LLVM rewrites the `/ 65535` and `% 65535` in release builds as a
/// multiply + shift via the standard magic constant, so each limb is a small
/// constant number of cycles.
#[inline]
fn divmod_65535_in_place(limbs: &mut [u64; 4]) -> u16 {
    let mut r: u128 = 0;
    for limb in limbs.iter_mut().rev() {
        let cur: u128 = (r << 64) | (*limb as u128);
        *limb = (cur / 65535) as u64;
        r = cur % 65535;
    }
    r as u16
}

/// Maps a digit `d ∈ [0, 65534]` to its centred-`i16` form `[-32767, 32767]`.
#[inline]
fn digit_to_centred_i16(digit: u16) -> i16 {
    debug_assert!(digit < 65535);
    if digit > 32767 {
        // 32768..=65534  →  -32767..=-1
        (digit as i32 - 65535) as i16
    } else {
        digit as i16
    }
}

/// Inverse of [`digit_to_centred_i16`]. The `i16::MIN` value is not a valid
/// centred representation and is debug-asserted out.
#[inline]
fn centred_i16_to_digit(stored: i16) -> u16 {
    debug_assert!(stored != i16::MIN);
    if stored < 0 {
        (stored as i32 + 65535) as u16
    } else {
        stored as u16
    }
}

#[cfg(test)]
mod u256_base65535_tests {
    use super::*;

    fn u256_le_from_u64(low: u64) -> [u8; 32] {
        let mut v = [0u8; 32];
        v[0..8].copy_from_slice(&low.to_le_bytes());
        v
    }

    #[test]
    fn roundtrips_zero_and_small_values() {
        for n in [0u64, 1, 2, 65534, 65535, 65536, 65537, 1_000_000, u64::MAX] {
            let v = u256_le_from_u64(n);
            let mut digits = [0i16; U256P65535::EXPONENT];
            U256P65535::encode(&mut digits, v);
            let mut decoded = [0u8; 32];
            U256P65535::decode(&mut decoded, &digits);
            assert_eq!(decoded, v, "round-trip failed for n = {n}");
        }
    }

    #[test]
    fn digit_extremes_match_centred_mod_p() {
        assert_eq!(digit_to_centred_i16(0), 0);
        assert_eq!(digit_to_centred_i16(32767), 32767);
        assert_eq!(digit_to_centred_i16(32768), -32767);
        assert_eq!(digit_to_centred_i16(65534), -1);
        assert_eq!(centred_i16_to_digit(0), 0);
        assert_eq!(centred_i16_to_digit(32767), 32767);
        assert_eq!(centred_i16_to_digit(-32767), 32768);
        assert_eq!(centred_i16_to_digit(-1), 65534);
    }
}
