use poulpy_core::layouts::{
    GGSWInfos, GGSWPreparedToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef,
};
use poulpy_hal::{
    api::{ModuleN, VecZnxAddAssignBackend, VecZnxRotateBackend, VecZnxSubBackend},
    layouts::{Backend, Module, ScratchArena, VecZnxToBackendMut, VecZnxToBackendRef, ZnxInfos},
};

use crate::interpolation::{
    HornerHelper,
    default::{HornerEvaluationDefault, MonomialInterpolationDefault},
};

#[doc(hidden)]
#[allow(private_bounds)]
pub unsafe trait MonomialInterpolationImpl<BE: Backend>: Backend {
    fn monomial_interpolate_tmp_bytes_impl(module: &Module<BE>, size: usize) -> usize;
    fn monomial_interpolate_impl<Y>(
        module: &Module<BE>,
        y: &mut [Y],
        col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Y: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos;
}

unsafe impl<BE: Backend> MonomialInterpolationImpl<BE> for BE
where
    Module<BE>: MonomialInterpolationDefault<BE>
        + ModuleN
        + VecZnxAddAssignBackend<BE>
        + VecZnxRotateBackend<BE>
        + VecZnxSubBackend<BE>,
{
    fn monomial_interpolate_tmp_bytes_impl(module: &Module<BE>, size: usize) -> usize {
        module.monomial_interpolate_tmp_bytes_default(size)
    }

    fn monomial_interpolate_impl<Y>(
        module: &Module<BE>,
        y: &mut [Y],
        col: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        Y: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    {
        module.monomial_interpolate_default(y, col, scratch);
    }
}

#[doc(hidden)]
#[allow(private_bounds)]
pub unsafe trait HornerEvaluationImpl<BE: Backend> {
    fn horner_evaluate_tmp_bytes_impl<G, H>(
        module: &Module<BE>,
        glwe_infos: &G,
        ggsw_infos: &H,
    ) -> usize
    where
        G: GLWEInfos,
        H: GGSWInfos;
    fn horner_evaluate<R, H, C, G>(
        module: &Module<BE>,
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

unsafe impl<BE: Backend> HornerEvaluationImpl<BE> for BE
where
    Module<BE>: HornerEvaluationDefault<BE>,
{
    fn horner_evaluate_tmp_bytes_impl<G, H>(
        module: &Module<BE>,
        glwe_infos: &G,
        ggsw_infos: &H,
    ) -> usize
    where
        G: GLWEInfos,
        H: GGSWInfos,
    {
        module.horner_evaluate_tmp_bytes_default(glwe_infos, ggsw_infos)
    }

    fn horner_evaluate<R, H, C, G>(
        module: &Module<BE>,
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
        module.horner_evaluate_default(res, coeffs, power, scratch);
    }
}
