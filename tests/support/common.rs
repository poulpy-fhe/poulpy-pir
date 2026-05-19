use poulpy_core::layouts::{
    GGLWEAtViewMut, GGLWEAtViewRef, GGLWEInfos, GGLWEPreparedFactory, GLWEToBackendMut,
    GLWEToBackendRef, ModuleCoreAlloc, Rank, prepared::GGLWEPrepared,
};
use poulpy_hal::{
    api::{ScratchOwnedBorrow, VecZnxCopyBackend},
    layouts::{
        Backend, Module, ScalarZnx, ScalarZnxBackendMut, ScalarZnxBackendRef,
        ScalarZnxToBackendMut, ScalarZnxToBackendRef, ScratchOwned, VecZnx, VecZnxBackendMut,
        VecZnxBackendRef, VecZnxToBackendMut, VecZnxToBackendRef,
    },
};

pub fn scalar_ref<BE: Backend>(scalar: &ScalarZnx<BE::OwnedBuf>) -> ScalarZnxBackendRef<'_, BE>
where
    ScalarZnx<BE::OwnedBuf>: ScalarZnxToBackendRef<BE>,
{
    ScalarZnxToBackendRef::<BE>::to_backend_ref(scalar)
}

pub fn scalar_mut<BE: Backend>(scalar: &mut ScalarZnx<BE::OwnedBuf>) -> ScalarZnxBackendMut<'_, BE>
where
    ScalarZnx<BE::OwnedBuf>: ScalarZnxToBackendMut<BE>,
{
    ScalarZnxToBackendMut::<BE>::to_backend_mut(scalar)
}

pub fn vec_ref<BE: Backend>(vec: &VecZnx<BE::OwnedBuf>) -> VecZnxBackendRef<'_, BE>
where
    VecZnx<BE::OwnedBuf>: VecZnxToBackendRef<BE>,
{
    VecZnxToBackendRef::<BE>::to_backend_ref(vec)
}

pub fn vec_mut<BE: Backend>(vec: &mut VecZnx<BE::OwnedBuf>) -> VecZnxBackendMut<'_, BE>
where
    VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE>,
{
    VecZnxToBackendMut::<BE>::to_backend_mut(vec)
}

pub fn split_output_key<BE, K>(
    module: &Module<BE>,
    key: &K,
    output_col: usize,
    scratch: &mut ScratchOwned<BE>,
) -> GGLWEPrepared<BE::OwnedBuf, BE>
where
    BE: Backend,
    Module<BE>:
        GGLWEPreparedFactory<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + VecZnxCopyBackend<BE>,
    K: GGLWEAtViewRef<BE> + GGLWEInfos,
{
    let mut split = module.gglwe_alloc(
        key.base2k(),
        key.max_k(),
        key.rank_in(),
        Rank(0),
        key.dnum(),
        key.dsize(),
    );

    for row in 0..key.dnum().as_usize() {
        for col in 0..key.rank_in().as_usize() {
            let src = GGLWEAtViewRef::<BE>::at_view(key, row, col);
            let mut dst = GGLWEAtViewMut::<BE>::at_view_mut(&mut split, row, col);
            let src_ref = src.to_backend_ref();
            let mut dst_mut = dst.to_backend_mut();
            module.vec_znx_copy_backend(dst_mut.data_mut(), 0, src_ref.data(), output_col);
        }
    }

    let mut prepared = module.gglwe_prepared_alloc_from_infos(&split);
    module.gglwe_prepare(&mut prepared, &split, &mut scratch.borrow());
    prepared
}
