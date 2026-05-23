use poulpy_core::layouts::{
    GGSWInfos, GGSWPreparedToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef,
};
use poulpy_hal::layouts::{
    Backend, ScratchArena, VecZnxToBackendMut, VecZnxToBackendRef, ZnxInfos,
};

pub trait MonomialInterpolation<BE: Backend> {
    fn monomial_interpolate_tmp_bytes(&self, size: usize) -> usize;

    /// In-place radix-2 Cooley-Tukey IDFT over the cyclotomic ring `R = Z[X]/(X^n + 1)`
    /// using the unit monomial `omega = X^(2n/t)` as the primitive `t`-th root of unity.
    ///
    /// Given `t` (a power of two) input polynomials `y_0, ..., y_{t-1}` interpreted as the
    /// evaluations of a polynomial `h(Z) = sum_j c_j Z^j ∈ R[Z]` at `z_k = omega^k`, this
    /// routine overwrites the slice with the unnormalized coefficients
    /// ```text
    /// y_j  <-  t · c_j   =  sum_k y_k · omega^{-jk}     (mod X^n + 1)
    /// ```
    ///
    /// The final `1/t` scaling is intentionally NOT applied: in a Torus / `base2k`
    /// encoding it is either absorbed into the message scaling factor `Δ` chosen at
    /// encryption time, or realised as a right shift by `log2(t)` bits.
    ///
    /// Because all twiddles are unit monomials the routine is just an integer linear
    /// combination of monomial-rotated copies of the inputs — no NTT / no requirement
    /// on the ambient `q` is involved. It therefore works identically whether the
    /// `y_k` carry plaintexts in `Z_p`, lifts in `Z_q`, or Torus-encoded values.
    ///
    /// # Layout
    /// Every `y_k` must have `n` equal to the module degree, `cols == 1`, and the
    /// same `size`. Operates in-place and uses one extra polynomial of scratch
    /// (see [`interpolate_tmp_bytes`]).
    fn monomial_interpolate<Y>(&self, y: &mut [Y], col: usize, scratch: &mut ScratchArena<'_, BE>)
    where
        Y: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos;
}

pub trait HornerHelper<BE: Backend, C: GLWEToBackendRef<BE>> {
    fn nb_coeffs(&self) -> usize;
    fn get_coeff(&self, idx: usize) -> &C;
}

pub trait HornerEvaluation<BE: Backend> {
    fn horner_evaluate_tmp_bytes<A, G>(&self, glwe_infos: &A, ggsw_infos: &G) -> usize
    where
        A: GLWEInfos,
        G: GGSWInfos;

    /// Evaluates `h(selector) = Σ_k c_k · selector^k` by encrypted Horner.
    ///
    /// Initialises `res` to the last (highest-degree) coefficient, then folds
    /// `res = c_k + res · selector` from `k = t-2` down to `k = 0`. The
    /// result is `h(X^i)` where `X^i` is the plaintext encrypted by `selector`
    /// — for a length-`t` `HornerCoeffs` built from
    /// [`crate::database::Database::query_interpolate`], this selects the
    /// matrix indexed by the GGSW-encrypted root of unity.
    fn horner_evaluate<R, H, C, G>(
        &self,
        res: &mut R,
        coeffs: &H,
        power: &G,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        H: HornerHelper<BE, C>,
        C: GLWEToBackendRef<BE>,
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos;
}
