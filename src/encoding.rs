use poulpy_hal::layouts::{HostDataMut, HostDataRef, VecZnx, ZnxViewMut};

/// BFV-style embedding of `Z_p` coefficients into a `2^K` torus grid.
///
/// This is intentionally not modular inverse scaling. A plaintext coefficient
/// `m` is first centered modulo the odd plaintext modulus `p`, then embedded as
/// `round(m * 2^K / p)`. The resulting torus integer is then encoded with
/// Poulpy's native base-`2^base2k` limb decomposition at torus precision `K`.
/// Decoding reverses the native decomposition, applies `round(x * p / 2^K)`,
/// and recenters modulo `p`.
///
/// Every value the encoder represents (`p`, `2^K`, centred plaintexts, encoded
/// torus integers) fits in `i64`: the constructor pins `K ≤ 54`, so
/// `2^K ≤ 2^54 < i64::MAX`. `i128` is used **only** as a transient widening
/// inside [`mul`](Self::mul), [`inv`](Self::inv), [`encode_coeff_i64`](Self::encode_coeff_i64)
/// and [`decode_coeff_i64`](Self::decode_coeff_i64), where a single product
/// `(p/2) · 2^K` or `(p/2)²` can briefly exceed `i64::MAX` before the modular
/// reduction or rounding division brings the result back into `i64`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModPEncoder {
    p: i64,
    torus_bits: usize,
    torus_scale: i64,
    torus_half: i64,
    /// Pre-computed `round(2^K / p)` — the encoded torus integer for plaintext
    /// `1`. Cached so the only two values a first-dimension PIR query needs
    /// (`0/p` and `1/p`) can be written without invoking the generic
    /// `encode_coeff_i64` path.
    encoded_unit: i64,
}

impl ModPEncoder {
    pub fn new(p: i64, torus_bits: usize) -> Self {
        assert!(p > 1, "plaintext modulus must be greater than one");
        assert!(
            (1..=54).contains(&torus_bits),
            "torus precision must fit the application encoder range"
        );
        let torus_scale: i64 = 1i64 << torus_bits;
        assert!(
            p < torus_scale,
            "plaintext modulus must be smaller than the torus grid"
        );

        // `round(2^K / p)`: `torus_scale = q · p + r` with `0 ≤ r < p`, then
        // round-to-nearest gives `q` or `q + 1`. All operands fit `i64`
        // (`torus_scale ≤ 2^54`, `q · p = torus_scale − r ≤ torus_scale`).
        let q = torus_scale / p;
        let r = torus_scale - q * p;
        let encoded_unit = if 2 * r >= p { q + 1 } else { q };

        Self {
            p,
            torus_bits,
            torus_scale,
            torus_half: torus_scale >> 1,
            encoded_unit,
        }
    }

    pub fn modulus(&self) -> i64 {
        self.p
    }

    pub fn torus_bits(&self) -> usize {
        self.torus_bits
    }

    pub fn normalize(&self, value: i64) -> i64 {
        centered_mod_i64(value, self.p)
    }

    pub fn add(&self, lhs: i64, rhs: i64) -> i64 {
        // |lhs|, |rhs| ≤ (p − 1)/2 < 2^53, so the sum fits i64 with margin.
        centered_mod_i64(lhs + rhs, self.p)
    }

    pub fn sub(&self, lhs: i64, rhs: i64) -> i64 {
        centered_mod_i64(lhs - rhs, self.p)
    }

    pub fn mul(&self, lhs: i64, rhs: i64) -> i64 {
        // |lhs · rhs| < (p/2)² can reach ~2^106 for p near 2^54, so widen to
        // i128 just for the product and the modular reduction.
        let prod: i128 = (lhs as i128) * (rhs as i128);
        let reduced = prod.rem_euclid(self.p as i128) as i64; // ∈ [0, p) ⊂ i64.
        center_in_place(reduced, self.p)
    }

    pub fn inv(&self, value: i64) -> i64 {
        let value = value.rem_euclid(self.p);
        assert!(value != 0, "zero has no inverse modulo p");
        mod_inv_i64(value, self.p)
    }

    pub fn encode_coeff_i64(&self, value: i64) -> i64 {
        let centered = self.normalize(value);
        // |centered · 2^K| ≤ (p/2) · 2^K can reach ~2^107 for the largest legal
        // parameters; widen only here. The final rounded division lands back
        // in (-2^K/2, 2^K/2] ⊂ i64.
        let scaled: i128 = (centered as i128) * (self.torus_scale as i128);
        div_round_nearest_i128(scaled, self.p as i128) as i64
    }

    pub fn decode_coeff_i64(&self, value: i64) -> i64 {
        let centered = centered_mod_torus_i64(value, self.torus_scale, self.torus_half);
        // Same wide-product reason as `encode_coeff_i64`.
        let scaled: i128 = (centered as i128) * (self.p as i128);
        let decoded = div_round_nearest_i128(scaled, self.torus_scale as i128) as i64;
        centered_mod_i64(decoded, self.p)
    }

    /// Pre-computed encoded torus integer for plaintext `1`,
    /// `Δ = round(2^K / p)`. Every non-zero coefficient of a one-hot
    /// first-dimension PIR query is encoded with this value.
    #[inline]
    pub fn encoded_unit(&self) -> i64 {
        self.encoded_unit
    }

    /// Writes a one-hot first-dimension PIR query plaintext into column `col`:
    /// coefficient `hot_slot` is set to `Δ = round(2^K / p)`, every other
    /// coefficient is zero (across all limbs).
    ///
    /// Faster than [`Self::encode_vec_i64`] for one-hot inputs because the
    /// limb decomposition is computed for *one* coefficient only — the rest of
    /// the column is filled with raw zeros, bypassing the per-coefficient
    /// wide-product encode path.
    pub fn encode_one_hot_into<D: HostDataMut>(
        &self,
        out: &mut VecZnx<D>,
        base2k: usize,
        col: usize,
        hot_slot: usize,
    ) {
        assert!(base2k > 0, "base2k must be non-zero");
        let native_size = self.torus_bits.div_ceil(base2k);
        assert!(
            native_size <= out.size(),
            "output has {} limbs but K={} with base2k={} requires {} limbs",
            out.size(),
            self.torus_bits,
            base2k,
            native_size,
        );
        assert!(col < out.cols(), "column out of range");
        assert!(hot_slot < out.n(), "hot_slot out of range");

        // Zero every limb of column `col`, then encode `Δ` at `hot_slot`.
        // `VecZnx::encode_coeff_i64` rewrites only the `hot_slot` position
        // across the limb stack (it does not touch other coefficient
        // positions), so the pre-zero step is required to clear stale state.
        self.encode_zero_into(out, col);
        out.encode_coeff_i64(base2k, col, self.torus_bits, hot_slot, self.encoded_unit);
    }

    /// Writes an all-zero first-dimension PIR query plaintext into column
    /// `col` of `out`: every limb at every coefficient of that column is set
    /// to zero. Cheap.
    pub fn encode_zero_into<D: HostDataMut>(&self, out: &mut VecZnx<D>, col: usize) {
        assert!(col < out.cols(), "column out of range");
        let size = out.size();
        for limb in 0..size {
            out.at_mut(col, limb).fill(0);
        }
    }

    pub fn encode_vec_i64<D: HostDataMut>(
        &self,
        out: &mut VecZnx<D>,
        base2k: usize,
        col: usize,
        data: &[i64],
    ) {
        assert!(base2k > 0, "base2k must be non-zero");
        let native_size = self.torus_bits.div_ceil(base2k);
        assert!(
            native_size <= out.size(),
            "output has {} limbs but K={} with base2k={} requires {} limbs",
            out.size(),
            self.torus_bits,
            base2k,
            native_size
        );
        assert!(col < out.cols(), "column out of range");
        assert_eq!(data.len(), out.n(), "input coefficient count mismatch");

        let torus: Vec<i64> = data
            .iter()
            .map(|&value| self.encode_coeff_i64(value))
            .collect();
        out.encode_vec_i64(base2k, col, self.torus_bits, &torus);
    }

    pub fn decode_vec_i64<D: HostDataRef>(
        &self,
        src: &VecZnx<D>,
        base2k: usize,
        col: usize,
        data: &mut [i64],
    ) {
        assert!(base2k > 0, "base2k must be non-zero");
        let native_size = self.torus_bits.div_ceil(base2k);
        assert!(
            native_size <= src.size(),
            "source has {} limbs but K={} with base2k={} requires {} limbs",
            src.size(),
            self.torus_bits,
            base2k,
            native_size
        );
        assert!(col < src.cols(), "column out of range");
        assert!(
            data.len() >= src.n(),
            "output coefficient count must cover the source polynomial"
        );

        let mut torus = vec![0i64; src.n()];
        src.decode_vec_i64(base2k, col, self.torus_bits, &mut torus);
        for idx in 0..src.n() {
            data[idx] = self.decode_coeff_i64(torus[idx]);
        }
    }
}

/// Centred mod-`p` reduction in `i64`. The inputs the encoder feeds in are
/// either already-bounded sums of two centred values (`< 2p < 2^55`) or the
/// result of an `i128.rem_euclid(p)` downcast — both fit `i64`.
fn centered_mod_i64(value: i64, p: i64) -> i64 {
    let mut reduced = value.rem_euclid(p);
    if reduced > p / 2 {
        reduced -= p;
    }
    reduced
}

/// Centred reduction of an `i64` against a power-of-two torus modulus
/// (`modulus = 2^K`, `half = modulus >> 1`). Same in-range guarantees as
/// [`centered_mod_i64`].
fn centered_mod_torus_i64(value: i64, modulus: i64, half: i64) -> i64 {
    let mut reduced = value.rem_euclid(modulus);
    if reduced >= half {
        reduced -= modulus;
    }
    reduced
}

/// Centring step on an already-non-negative residue `value ∈ [0, p)`. Skips the
/// redundant `rem_euclid` that [`centered_mod_i64`] would otherwise do.
#[inline]
fn center_in_place(value: i64, p: i64) -> i64 {
    if value > p / 2 { value - p } else { value }
}

/// Round-to-nearest division kept in `i128` because the numerator is the wide
/// product `(p/2) · 2^K` produced by the encode/decode kernels.
fn div_round_nearest_i128(num: i128, den: i128) -> i128 {
    assert!(den > 0);
    if num >= 0 {
        (num + den / 2) / den
    } else {
        -((-num + den / 2) / den)
    }
}

fn mod_inv_i64(value: i64, modulus: i64) -> i64 {
    // The extended-Euclidean Bézout coefficients (`old_s`, `s`) grow up to
    // `modulus` in magnitude, so `q · s` reaches up to `modulus² < 2^108` for
    // the largest legal `p`. Hold them in `i128` for that range; everything
    // else flows back to `i64` on exit.
    let mut old_r: i128 = value as i128;
    let mut r: i128 = modulus as i128;
    let mut old_s: i128 = 1;
    let mut s: i128 = 0;

    while r != 0 {
        let q = old_r / r;

        let next_r = old_r - q * r;
        old_r = r;
        r = next_r;

        let next_s = old_s - q * s;
        old_s = s;
        s = next_s;
    }

    assert_eq!(old_r, 1, "value is not invertible modulo p");
    old_s.rem_euclid(modulus as i128) as i64
}

// =============================================================================
// 256-bit ↔ 16 × Z_65535 (centred i16) encoding.
// =============================================================================
//
// A `Database` matrix entry stores one centred-`Z_p` coefficient as `i16`
// (`p = 65535`, range `[-32767, 32767]`). A 256-bit payload packs into 16
// consecutive entries using base-65535 positional notation: any payload
// strictly less than `65535^16 ≈ 2^256 − 2^244` is encoded losslessly.

/// Number of base-`65535` digits used to encode a 256-bit payload.
pub const U256_BASE65535_DIGITS: usize = 16;

/// `65535^16` as four little-endian `u64` limbs — the exclusive upper bound of
/// the payload range that can be encoded losslessly. Approximately
/// `2^256 − 2^244 + 2^231 − …`.
pub const U256_BASE65535_BOUND_EXCLUSIVE: [u64; 4] = u256_pow_65535_16();

const fn u256_pow_65535_16() -> [u64; 4] {
    // Compile-time `65535^16`, computed as 16 limb-wise `* 65535`.
    let mut acc: [u64; 4] = [1, 0, 0, 0];
    let mut step = 0;
    while step < 16 {
        let mut carry: u128 = 0;
        let mut j = 0;
        while j < 4 {
            let prod = acc[j] as u128 * 65535u128 + carry;
            acc[j] = prod as u64;
            carry = prod >> 64;
            j += 1;
        }
        // `65535^16 < 2^256`, so the top carry is always zero.
        debug_assert!(carry == 0);
        step += 1;
    }
    acc
}

/// `true` iff `value` (little-endian U256) fits in 16 base-`65535` digits.
///
/// Equivalent to `value < 65535^16`.
pub fn u256_base65535_encodable(value: &[u8; 32]) -> bool {
    let v = u256_le_to_limbs(value);
    for i in (0..4).rev() {
        if v[i] != U256_BASE65535_BOUND_EXCLUSIVE[i] {
            return v[i] < U256_BASE65535_BOUND_EXCLUSIVE[i];
        }
    }
    false
}

/// Encodes a 256-bit little-endian unsigned integer as 16 base-`65535` digits
/// in centred-`i16` representation (`[-32767, 32767]`, matching
/// [`ModPEncoder::normalize`] at `p = 65535`).
///
/// The encoding is bijective on `[0, 65535^16)`. Inputs at or above
/// `65535^16` overflow the representable range; the function debug-asserts and
/// silently produces wrapped output in release builds — call
/// [`u256_base65535_encodable`] first if the payload distribution is not known
/// to be bounded.
#[inline]
pub fn encode_u256_base65535(value: &[u8; 32]) -> [i16; U256_BASE65535_DIGITS] {
    let mut limbs = u256_le_to_limbs(value);
    let mut digits = [0i16; U256_BASE65535_DIGITS];
    for d in &mut digits {
        *d = digit_to_centred_i16(divmod_65535_in_place(&mut limbs));
    }
    debug_assert!(
        limbs == [0u64; 4],
        "encode_u256_base65535: value ≥ 65535^16 (overflows 16-digit base-65535)",
    );
    digits
}

/// Inverse of [`encode_u256_base65535`].
///
/// Accepts any `[i16; 16]` whose entries are in `[-32767, 32767]` and
/// reconstructs the unique little-endian U256 in `[0, 65535^16)` it represents.
#[inline]
pub fn decode_u256_base65535(digits: &[i16; U256_BASE65535_DIGITS]) -> [u8; 32] {
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
    let mut bytes = [0u8; 32];
    for (i, limb) in acc.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    bytes
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
            let digits = encode_u256_base65535(&v);
            assert_eq!(
                decode_u256_base65535(&digits),
                v,
                "round-trip failed for n = {n}"
            );
        }
    }

    #[test]
    fn roundtrips_the_largest_legal_value() {
        // 65535^16 − 1 has all 16 base-65535 digits equal to 65534, i.e.
        // centred `-1` in every slot.
        let max_digits = [-1i16; 16];
        let v = decode_u256_base65535(&max_digits);
        assert_eq!(encode_u256_base65535(&v), max_digits);
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

    #[test]
    fn encodable_bound_is_exclusive() {
        // 65535^16 itself is the first non-encodable value.
        let mut bound = [0u8; 32];
        for (i, limb) in U256_BASE65535_BOUND_EXCLUSIVE.iter().enumerate() {
            bound[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
        assert!(!u256_base65535_encodable(&bound));

        // 65535^16 − 1 is the largest encodable value.
        let mut one_less = bound;
        // Subtract 1 (no borrow past byte 0 since bound[0] is nonzero).
        let mut i = 0;
        while i < 32 {
            if one_less[i] > 0 {
                one_less[i] -= 1;
                break;
            } else {
                one_less[i] = 0xFF;
                i += 1;
            }
        }
        assert!(u256_base65535_encodable(&one_less));
    }

    #[test]
    fn roundtrips_pseudo_random_payloads_below_2_pow_244() {
        // Clear the top 12 bits, guaranteeing v < 2^244 ≪ 65535^16.
        let mut state: u64 = 0xcafef00dbaadf00d;
        for _ in 0..1024 {
            let mut v = [0u8; 32];
            for byte in &mut v {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *byte = (state >> 33) as u8;
            }
            v[31] = 0;
            v[30] &= 0x0F;
            assert!(u256_base65535_encodable(&v));
            let digits = encode_u256_base65535(&v);
            assert_eq!(decode_u256_base65535(&digits), v);
        }
    }
}
