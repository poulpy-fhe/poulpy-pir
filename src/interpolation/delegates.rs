use poulpy_core::layouts::{
    GGSWInfos, GGSWPreparedToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef,
};
use poulpy_hal::layouts::{
    Backend, Module, ScratchArena, VecZnxToBackendMut, VecZnxToBackendRef, ZnxInfos,
};

use crate::interpolation::{
    HornerEvaluation, HornerHelper,
    api::MonomialInterpolation,
    oep::{HornerEvaluationImpl, MonomialInterpolationImpl},
};

impl<BE> MonomialInterpolation<BE> for Module<BE>
where
    BE: Backend + MonomialInterpolationImpl<BE>,
{
    fn monomial_interpolate<Y>(&self, y: &mut [Y], col: usize, scratch: &mut ScratchArena<'_, BE>)
    where
        Y: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    {
        BE::monomial_interpolate_impl(self, y, col, scratch);
    }

    fn monomial_interpolate_tmp_bytes(&self, size: usize) -> usize {
        BE::monomial_interpolate_tmp_bytes_impl(self, size)
    }
}

impl<BE> HornerEvaluation<BE> for Module<BE>
where
    BE: Backend + HornerEvaluationImpl<BE>,
{
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
        G: GGSWPreparedToBackendRef<BE> + GGSWInfos,
    {
        BE::horner_evaluate(self, res, coeffs, power, scratch);
    }

    fn horner_evaluate_tmp_bytes<A, G>(&self, glwe_infos: &A, ggsw_infos: &G) -> usize
    where
        A: GLWEInfos,
        G: GGSWInfos,
    {
        BE::horner_evaluate_tmp_bytes_impl(self, glwe_infos, ggsw_infos)
    }
}
