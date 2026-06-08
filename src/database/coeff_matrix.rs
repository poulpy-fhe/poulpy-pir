//! Dense `i16` coefficient matrix — the database / interpolation `U` operand.
//!
//! Since the `U·A` mask product and `U·b` body product are now local `f64`
//! GEMMs (the homomorphic coefficient-matrix product was removed from poulpy),
//! the operand no longer needs poulpy's base2k/VecZnx-encoded `CoeffMatrix`. It
//! is just a `rows_out × rows_in` block of `i16` values, stored row-major.

/// A `rows_out × rows_in` matrix of `i16` coefficients (`rows[out][in]`).
#[derive(Clone, Debug)]
pub struct CoeffMatrix {
    rows: Vec<Vec<i16>>,
    rows_in: usize,
}

impl CoeffMatrix {
    /// A zeroed `rows_out × rows_in` matrix.
    pub fn zeros(rows_out: usize, rows_in: usize) -> Self {
        Self {
            rows: vec![vec![0i16; rows_in]; rows_out],
            rows_in,
        }
    }

    /// Output rows (the matmul `U`'s output dimension; the former `data().cols()`).
    pub fn rows_out(&self) -> usize {
        self.rows.len()
    }

    /// Input rows (the matmul `U`'s contraction dimension; the former `data().n()`).
    pub fn rows_in(&self) -> usize {
        self.rows_in
    }

    /// Row `out` (`rows_in` coefficients).
    pub fn row(&self, out: usize) -> &[i16] {
        &self.rows[out]
    }

    /// Mutable row `out` (`rows_in` coefficients).
    pub fn row_mut(&mut self, out: usize) -> &mut [i16] {
        &mut self.rows[out]
    }

    /// Reset every coefficient to zero.
    pub fn zero(&mut self) {
        for row in &mut self.rows {
            row.fill(0);
        }
    }
}
