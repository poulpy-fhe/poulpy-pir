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

    /// Write one value per 4 KiB page so the physical pages are committed on
    /// the NUMA node of the *calling thread* (first-touch). The freshly
    /// allocated matrix is zeroed but lazily mapped, so the writes preserve
    /// every value. Volatile because the allocation is known-zeroed and a
    /// plain `= 0` store could legally be elided — the page fault is the
    /// entire point.
    pub(crate) fn first_touch(&mut self) {
        const I16_PER_PAGE: usize = 4096 / size_of::<i16>();
        let ptr = self.data.as_mut_ptr();
        let mut i = 0;
        while i < self.data.len() {
            // SAFETY: `i < len`, so the pointer is in bounds.
            unsafe { std::ptr::write_volatile(ptr.add(i), 0) };
            i += I16_PER_PAGE;
        }
    }
}
