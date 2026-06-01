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
