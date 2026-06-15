//! Dense `i16` coefficient matrix — the database / interpolation `U` operand.
//!
//! Since the `U·A` mask product and `U·b` body product are now local `f64`
//! GEMMs (the homomorphic coefficient-matrix product was removed from poulpy),
//! the operand no longer needs poulpy's base2k/VecZnx-encoded `CoeffMatrix`. It
//! is just a `rows_out × rows_in` block of `i16` values, stored row-major.

/// A `rows_out × rows_in` matrix of `i16` coefficients, stored **contiguously**
/// row-major in one allocation. Contiguity makes it directly GEMM-ready, so the
/// `U` operand of the mask/body products can be a zero-copy view over
/// [`flat`](Self::flat) — no separate prepared-panel copy is needed.
#[derive(Clone, Debug)]
pub struct CoeffMatrix {
    data: Vec<i16>,
    rows_out: usize,
    rows_in: usize,
}

impl CoeffMatrix {
    /// A zeroed `rows_out × rows_in` matrix.
    pub fn zeros(rows_out: usize, rows_in: usize) -> Self {
        Self {
            data: vec![0i16; rows_out * rows_in],
            rows_out,
            rows_in,
        }
    }

    /// Output rows (the matmul `U`'s output dimension; the former `data().cols()`).
    pub fn rows_out(&self) -> usize {
        self.rows_out
    }

    /// Input rows (the matmul `U`'s contraction dimension; the former `data().n()`).
    pub fn rows_in(&self) -> usize {
        self.rows_in
    }

    /// Row `out` (`rows_in` coefficients).
    pub fn row(&self, out: usize) -> &[i16] {
        &self.data[out * self.rows_in..(out + 1) * self.rows_in]
    }

    /// Mutable row `out` (`rows_in` coefficients).
    pub fn row_mut(&mut self, out: usize) -> &mut [i16] {
        let start = out * self.rows_in;
        &mut self.data[start..start + self.rows_in]
    }

    /// The whole matrix as one contiguous row-major `i16` slice — the GEMM-ready
    /// `U` panel, consumed directly by `PreparedF64::from_matrix`.
    pub fn flat(&self) -> &[i16] {
        &self.data
    }

    /// Reset every coefficient to zero.
    pub fn zero(&mut self) {
        self.data.fill(0);
    }
}
