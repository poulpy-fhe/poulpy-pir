mod api;
mod default;
mod delegates;
mod oep;
mod strategy;

pub use api::*;
pub use default::{impl_horner_evaluation_default, impl_monomial_interpolation_default};
pub use strategy::{
    Interpolated, Interpolation, InterpolationKeys, InterpolationQuery, InterpolationResponse,
    interpolation_root_exponent,
};

use poulpy_cpu_avx::FFT64Avx;

impl_monomial_interpolation_default!(FFT64Avx);
impl_horner_evaluation_default!(FFT64Avx);

#[cfg(test)]
mod tests;

use poulpy_core::layouts::{GLWEInfos, GLWEToBackendRef};
use poulpy_hal::layouts::Backend;

/// Matrix-axis coefficient polynomial as a vector of GLWE ciphertexts.
///
/// Each `coeffs[k]` is the GLWE form of the `k`-th interpolated coefficient
/// `t · c_k` of `h(Z) = Σ_k c_k · Z^k`, produced by collapsing the matching
/// `interp[k]` LWEMatrix. The Horner evaluation [`Self::horner`] selects one
/// matrix by evaluating `h` at the GGSW-encrypted root `X^i`.
pub struct HornerCoeffs<'a, G>(pub &'a [G]);

impl<'a, BE: Backend, G> HornerHelper<BE, G> for HornerCoeffs<'a, G>
where
    G: GLWEToBackendRef<BE> + GLWEInfos,
{
    fn nb_coeffs(&self) -> usize {
        self.0.len()
    }

    fn get_coeff(&self, idx: usize) -> &G {
        &self.0[idx]
    }
}
