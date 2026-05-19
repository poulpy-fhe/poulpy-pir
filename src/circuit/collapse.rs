use poulpy_core::{
    GLWEKeyswitch, GLWEZero,
    layouts::{
        GGLWEInfos, GGLWEPreparedVmpPMatRef, GLWE, GLWEInfos, GLWELayout, GLWEToBackendMut,
        GLWEToBackendRef, ModuleCoreAlloc, Rank, prepared::GGLWEPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxAddAssignBackend, VecZnxAutomorphismBackend,
        VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxCopyBackend, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{
        Backend, GaloisElement, Module, ScratchArena, VecZnx, VecZnxToBackendMut,
        VecZnxToBackendRef, ZnxInfos,
    },
};

use super::collapse_precompute::{
    SequentialCollapseMaskPrecompute, fixed_mask_1x1_vmp_body_addend,
    fixed_mask_1x1_vmp_body_addend_tmp_bytes, fixed_mask_1x1_vmp_keyswitch_body,
};

pub fn sequential_keyswitch_collapse_aggregate_mask_tmp_bytes<BE, R, A, K, H>(
    module: &Module<BE>,
    res: &R,
    aggregate_mask: &A,
    key_g: &K,
    key_h: &H,
) -> usize
where
    BE: Backend,
    Module<BE>: GLWEKeyswitch<BE>,
    R: GLWEInfos,
    A: ZnxInfos,
    K: GGLWEInfos,
    H: GGLWEInfos,
{
    assert_eq!(
        key_g.rank_in().as_usize(),
        1,
        "aggregate collapse consumes one aggregate mask column at a time"
    );
    assert_eq!(
        key_g.rank_out().as_usize(),
        1,
        "Kg-derived keys must collapse one source share into one target share"
    );
    assert_eq!(
        key_h.rank_in().as_usize(),
        1,
        "Kh must consume the tau_h share"
    );
    assert_eq!(
        key_h.rank_out(),
        res.rank(),
        "Kh output rank must match the result rank"
    );
    assert_eq!(
        aggregate_mask.n(),
        module.n(),
        "aggregate mask degree must match module degree"
    );
    assert_eq!(
        aggregate_mask.cols(),
        module.n(),
        "aggregate collapse expects one collapse per aggregate column"
    );

    let term_infos = GLWELayout {
        n: res.n(),
        base2k: res.base2k(),
        k: res.max_k(),
        rank: key_g.rank_in(),
    };

    module
        .glwe_keyswitch_tmp_bytes(&term_infos, &term_infos, key_g)
        .max(module.glwe_keyswitch_tmp_bytes(res, &term_infos, key_h))
}

#[allow(clippy::too_many_arguments)]
pub fn sequential_keyswitch_collapse_aggregate_mask<BE, R, B, A, K, H>(
    module: &Module<BE>,
    res: &mut R,
    body: &B,
    aggregate_mask: &A,
    key_g: &K,
    key_h: &H,
    key_g_size: usize,
    key_h_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: GLWEKeyswitch<BE>
        + GLWEZero<BE>
        + GaloisElement
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + VecZnxAddAssignBackend<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxCopyBackend<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    B: VecZnxToBackendRef<BE> + ZnxInfos,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
    K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
    H: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    let n = module.n();
    let half = n >> 1;
    assert_eq!(key_g.rank_in().as_usize(), 1);
    assert_eq!(key_g.rank_out().as_usize(), 1);
    assert_eq!(body.n(), module.n(), "body degree must match module degree");
    assert!(
        body.cols() >= 1,
        "body VecZnx must contain at least one column"
    );
    assert_eq!(
        aggregate_mask.n(),
        module.n(),
        "aggregate mask degree must match module degree"
    );
    assert_eq!(
        aggregate_mask.cols(),
        module.n(),
        "aggregate collapse expects one collapse per aggregate column"
    );
    assert_eq!(
        aggregate_mask.size(),
        res.size(),
        "aggregate mask size must match result size"
    );
    assert_eq!(key_h.rank_in().as_usize(), 1);
    assert_eq!(key_h.rank_out(), res.rank());

    let term_infos = GLWELayout {
        n: res.n(),
        base2k: res.base2k(),
        k: res.max_k(),
        rank: Rank(1),
    };
    let mut term = module.glwe_alloc_from_infos(&term_infos);
    let mut switched = module.glwe_alloc_from_infos(&term_infos);
    let mut switched_auto = module.glwe_alloc_from_infos(&term_infos);
    let mut body_work = module.vec_znx_alloc(1, body.size());
    let mut half_work = module.vec_znx_alloc(half, aggregate_mask.size());
    let aggregate_ref = aggregate_mask.to_backend_ref();

    {
        let body_ref = body.to_backend_ref();
        let mut body_mut =
            <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut body_work);
        module.vec_znx_copy_backend(&mut body_mut, 0, &body_ref, 0);
    }
    copy_aggregate_half(module, &mut half_work, &aggregate_ref, 0);
    collapse_half(
        module,
        &mut half_work,
        &mut body_work,
        false,
        key_g,
        key_g_size,
        &mut term,
        &mut switched,
        &mut switched_auto,
        scratch,
    );
    let mut first_share = module.vec_znx_alloc(1, aggregate_mask.size());
    {
        let half_ref = <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(&half_work);
        let mut first_mut =
            <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut first_share);
        module.vec_znx_copy_backend(&mut first_mut, 0, &half_ref, 0);
    }

    copy_aggregate_half(module, &mut half_work, &aggregate_ref, half);
    collapse_half(
        module,
        &mut half_work,
        &mut body_work,
        true,
        key_g,
        key_g_size,
        &mut term,
        &mut switched,
        &mut switched_auto,
        scratch,
    );

    module.glwe_zero(&mut term);
    {
        let body_ref = <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(&body_work);
        let half_ref = <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(&half_work);
        let mut term_ref = <_ as GLWEToBackendMut<BE>>::to_backend_mut(&mut term);
        module.vec_znx_copy_backend(term_ref.data_mut(), 0, &body_ref, 0);
        module.vec_znx_copy_backend(term_ref.data_mut(), 1, &half_ref, 0);
    }
    module.glwe_keyswitch(res, &term, key_h, key_h_size, scratch);
    {
        let first_ref =
            <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(&first_share);
        let mut res_ref = res.to_backend_mut();
        module.vec_znx_add_assign_backend(res_ref.data_mut(), 1, &first_ref, 0);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn sequential_keyswitch_collapse_aggregate_mask_split<
    BE,
    R,
    B,
    A,
    KGBody,
    KGMask,
    KHBody,
    KHMask,
>(
    module: &Module<BE>,
    res: &mut R,
    body: &B,
    aggregate_mask: &A,
    key_g_body: &KGBody,
    key_g_mask: &KGMask,
    key_h_body: &KHBody,
    key_h_mask: &KHMask,
    key_g_size: usize,
    key_h_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: GaloisElement
        + VecZnxAddAssignBackend<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VmpApplyDftToDft<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    B: VecZnxToBackendRef<BE> + ZnxInfos,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
    KGBody: poulpy_core::layouts::GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    KGMask: poulpy_core::layouts::GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    KHBody: poulpy_core::layouts::GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    KHMask: poulpy_core::layouts::GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBasic<'a, BE>,
{
    let n = module.n();
    let half = n >> 1;
    assert_eq!(res.rank().as_usize(), 1);
    assert_eq!(key_g_body.rank_in().as_usize(), 1);
    assert_eq!(key_g_mask.rank_in().as_usize(), 1);
    assert_eq!(key_h_body.rank_in().as_usize(), 1);
    assert_eq!(key_h_mask.rank_in().as_usize(), 1);
    assert_eq!(key_g_body.rank_out().as_usize(), 0);
    assert_eq!(key_g_mask.rank_out().as_usize(), 0);
    assert_eq!(key_h_body.rank_out().as_usize(), 0);
    assert_eq!(key_h_mask.rank_out().as_usize(), 0);
    assert_eq!(body.n(), module.n(), "body degree must match module degree");
    assert!(
        body.cols() >= 1,
        "body VecZnx must contain at least one column"
    );
    assert_eq!(
        aggregate_mask.n(),
        module.n(),
        "aggregate mask degree must match module degree"
    );
    assert_eq!(
        aggregate_mask.cols(),
        module.n(),
        "aggregate collapse expects one collapse per aggregate column"
    );
    assert_eq!(
        aggregate_mask.size(),
        res.size(),
        "aggregate mask size must match result size"
    );

    let res_base2k = res.base2k().as_usize();
    let scratch_local = scratch.borrow();
    let (mut body_work, scratch_local) = scratch_local.take_vec_znx_scratch(n, 1, body.size());
    let (mut half_work, scratch_local) =
        scratch_local.take_vec_znx_scratch(n, half, aggregate_mask.size());
    let (mut first_share, scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, aggregate_mask.size());
    let (mut term_body, scratch_local) = scratch_local.take_vec_znx_scratch(n, 1, body.size());
    let (mut term_mask, scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, aggregate_mask.size());
    let (mut switched_body, scratch_local) = scratch_local.take_vec_znx_scratch(n, 1, res.size());
    let (mut switched_mask, mut scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, res.size());

    {
        let mut body_mut = body_work.to_backend_mut();
        module.vec_znx_copy_backend(&mut body_mut, 0, &body.to_backend_ref(), 0);
    }

    let aggregate_ref = aggregate_mask.to_backend_ref();
    copy_aggregate_half(module, &mut half_work, &aggregate_ref, 0);
    collapse_half_split(
        module,
        &mut half_work,
        &mut body_work,
        false,
        key_g_body,
        key_g_mask,
        key_g_size,
        res_base2k,
        &mut term_body,
        &mut term_mask,
        &mut switched_body,
        &mut switched_mask,
        &mut scratch_local,
    );

    {
        let half_ref = half_work.to_backend_ref();
        let mut first_mut = first_share.to_backend_mut();
        module.vec_znx_copy_backend(&mut first_mut, 0, &half_ref, 0);
    }

    copy_aggregate_half(module, &mut half_work, &aggregate_ref, half);
    collapse_half_split(
        module,
        &mut half_work,
        &mut body_work,
        true,
        key_g_body,
        key_g_mask,
        key_g_size,
        res_base2k,
        &mut term_body,
        &mut term_mask,
        &mut switched_body,
        &mut switched_mask,
        &mut scratch_local,
    );

    {
        let body_ref = body_work.to_backend_ref();
        let half_ref = half_work.to_backend_ref();
        let mut term_body_mut = term_body.to_backend_mut();
        let mut term_mask_mut = term_mask.to_backend_mut();
        module.vec_znx_copy_backend(&mut term_body_mut, 0, &body_ref, 0);
        module.vec_znx_copy_backend(&mut term_mask_mut, 0, &half_ref, 0);
    }
    fixed_mask_1x1_vmp_keyswitch_body(
        module,
        &mut switched_body,
        res_base2k,
        &term_mask,
        0,
        &term_body,
        0,
        res_base2k,
        key_h_body,
        key_h_size,
        &mut scratch_local.borrow(),
    );
    fixed_mask_1x1_vmp_body_addend(
        module,
        &mut switched_mask,
        res_base2k,
        &term_mask,
        0,
        key_h_mask,
        key_h_size,
        &mut scratch_local.borrow(),
    );

    {
        let body_ref = switched_body.to_backend_ref();
        let mask_ref = switched_mask.to_backend_ref();
        let first_ref = first_share.to_backend_ref();
        let mut res_ref = res.to_backend_mut();
        module.vec_znx_copy_backend(res_ref.data_mut(), 0, &body_ref, 0);
        module.vec_znx_copy_backend(res_ref.data_mut(), 1, &mask_ref, 0);
        module.vec_znx_add_assign_backend(res_ref.data_mut(), 1, &first_ref, 0);
    }
}

/// Scratch estimate for [`sequential_keyswitch_collapse_aggregate_mask_precomputed`].
pub fn sequential_keyswitch_collapse_aggregate_mask_precomputed_tmp_bytes<
    BE,
    R,
    B,
    KGBody,
    KHBody,
>(
    module: &Module<BE>,
    res: &R,
    body: &B,
    precompute: &SequentialCollapseMaskPrecompute<BE::OwnedBuf>,
    key_g_body: &KGBody,
    key_h_body: &KHBody,
    key_g_size: usize,
    key_h_size: usize,
) -> usize
where
    BE: Backend,
    Module<BE>: VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + VmpApplyDftToDftTmpBytes,
    R: GLWEInfos,
    B: ZnxInfos,
    KGBody: GGLWEInfos,
    KHBody: GGLWEInfos,
{
    let align = |len: usize| len.next_multiple_of(BE::SCRATCH_ALIGN);
    let vec_scratch = align(VecZnx::<Vec<u8>>::bytes_of(module.n(), 1, body.size()))
        + 2 * align(VecZnx::<Vec<u8>>::bytes_of(module.n(), 1, res.size()));
    let key_g_scratch = fixed_mask_1x1_vmp_body_addend_tmp_bytes::<BE, _, _, _>(
        module,
        precompute.body_vmp_masks(),
        key_g_body,
        key_g_size,
    );
    let key_h_scratch = fixed_mask_1x1_vmp_body_addend_tmp_bytes::<BE, _, _, _>(
        module,
        precompute.body_vmp_masks(),
        key_h_body,
        key_h_size,
    );

    vec_scratch + key_g_scratch.max(key_h_scratch)
}

/// Collapses a preprocessed aggregate mask using only online body-side VMPs.
///
/// The mask schedule and final result mask are produced offline by
/// [`precompute_sequential_keyswitch_collapse_aggregate_mask`](crate::circuit::precompute_sequential_keyswitch_collapse_aggregate_mask).
/// At query time this routine replays the same collapse order as the split
/// baseline, but each step uses the stored fixed mask input with the
/// query-dependent body part of the key-switching key.
#[allow(clippy::too_many_arguments)]
pub fn sequential_keyswitch_collapse_aggregate_mask_precomputed<BE, R, B, KGBody, KHBody>(
    module: &Module<BE>,
    res: &mut R,
    body: &B,
    precompute: &SequentialCollapseMaskPrecompute<BE::OwnedBuf>,
    key_g_body: &KGBody,
    key_h_body: &KHBody,
    key_g_size: usize,
    key_h_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: GaloisElement
        + VecZnxAutomorphismBackend<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VmpApplyDftToDft<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    B: VecZnxToBackendRef<BE> + ZnxInfos,
    KGBody: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    KHBody: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBasic<'a, BE>,
{
    let n = module.n();
    let half = n >> 1;
    assert_eq!(precompute.steps(), n - 1);
    assert_eq!(precompute.rank().as_usize(), 1);
    assert_eq!(precompute.body_vmp_masks().n(), n);
    assert_eq!(precompute.body_vmp_masks().cols(), precompute.steps());
    assert_eq!(precompute.body_vmp_masks().size(), precompute.size());
    assert_eq!(precompute.final_mask().n(), n);
    assert_eq!(precompute.final_mask().cols(), 1);
    assert_eq!(precompute.final_mask().size(), precompute.size());
    assert_eq!(res.rank().as_usize(), 1);
    assert_eq!(res.size(), precompute.size());
    assert_eq!(res.base2k().as_usize(), precompute.base2k());
    assert_eq!(body.n(), n, "body degree must match module degree");
    assert!(
        body.cols() >= 1,
        "body VecZnx must contain at least one column"
    );
    assert_eq!(body.size(), precompute.size());
    assert_eq!(key_g_body.rank_in().as_usize(), 1);
    assert_eq!(key_h_body.rank_in().as_usize(), 1);
    assert_eq!(key_g_body.rank_out().as_usize(), 0);
    assert_eq!(key_h_body.rank_out().as_usize(), 0);
    assert_eq!(key_g_body.base2k().as_usize(), precompute.base2k());
    assert_eq!(key_h_body.base2k().as_usize(), precompute.base2k());

    let scratch_local = scratch.borrow();
    let (mut body_work, scratch_local) = scratch_local.take_vec_znx_scratch(n, 1, body.size());
    let (mut term_body, scratch_local) = scratch_local.take_vec_znx_scratch(n, 1, body.size());
    let (mut switched_body, mut scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, res.size());

    {
        let mut body_mut = body_work.to_backend_mut();
        module.vec_znx_copy_backend(&mut body_mut, 0, &body.to_backend_ref(), 0);
    }

    let mut step = 0usize;
    collapse_half_body_precomputed(
        module,
        &mut body_work,
        false,
        precompute,
        key_g_body,
        key_g_size,
        &mut step,
        &mut term_body,
        &mut switched_body,
        &mut scratch_local,
    );
    assert_eq!(step, half - 1);

    collapse_half_body_precomputed(
        module,
        &mut body_work,
        true,
        precompute,
        key_g_body,
        key_g_size,
        &mut step,
        &mut term_body,
        &mut switched_body,
        &mut scratch_local,
    );
    assert_eq!(step, 2 * (half - 1));

    {
        let body_ref = body_work.to_backend_ref();
        let mut term_body_mut = term_body.to_backend_mut();
        module.vec_znx_copy_backend(&mut term_body_mut, 0, &body_ref, 0);
    }
    fixed_mask_1x1_vmp_keyswitch_body(
        module,
        &mut switched_body,
        precompute.base2k(),
        precompute.body_vmp_masks(),
        step,
        &term_body,
        0,
        precompute.base2k(),
        key_h_body,
        key_h_size,
        &mut scratch_local.borrow(),
    );

    {
        let body_ref = switched_body.to_backend_ref();
        let mask_ref = <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(
            precompute.final_mask(),
        );
        let mut res_ref = res.to_backend_mut();
        module.vec_znx_copy_backend(res_ref.data_mut(), 0, &body_ref, 0);
        module.vec_znx_copy_backend(res_ref.data_mut(), 1, &mask_ref, 0);
    }
    step += 1;
    assert_eq!(step, precompute.steps());
}

#[allow(clippy::too_many_arguments)]
fn collapse_half_body_precomputed<BE, Body, TermBody, SwitchedBody, KBody>(
    module: &Module<BE>,
    body: &mut Body,
    use_tau_h: bool,
    precompute: &SequentialCollapseMaskPrecompute<BE::OwnedBuf>,
    key_body: &KBody,
    key_size: usize,
    step: &mut usize,
    term_body: &mut TermBody,
    switched_body: &mut SwitchedBody,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: GaloisElement
        + VecZnxAutomorphismBackend<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VmpApplyDftToDft<BE>,
    Body: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    TermBody: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    SwitchedBody: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    KBody: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBasic<'a, BE>,
{
    let half = module.n() >> 1;
    for target_col in (0..half - 1).rev() {
        let tau_g_j = module.galois_element(target_col as i64);
        let secret_view = if use_tau_h { -tau_g_j } else { tau_g_j };
        let alpha = module.galois_element_inv(secret_view);
        let alpha_inv = secret_view;

        module.vec_znx_automorphism_backend(
            alpha_inv,
            &mut term_body.to_backend_mut(),
            0,
            &body.to_backend_ref(),
            0,
        );

        fixed_mask_1x1_vmp_keyswitch_body(
            module,
            switched_body,
            precompute.base2k(),
            precompute.body_vmp_masks(),
            *step,
            term_body,
            0,
            precompute.base2k(),
            key_body,
            key_size,
            &mut scratch.borrow(),
        );

        module.vec_znx_automorphism_backend(
            alpha,
            &mut body.to_backend_mut(),
            0,
            &switched_body.to_backend_ref(),
            0,
        );

        *step += 1;
    }
}

fn copy_aggregate_half<BE, M, D>(
    module: &M,
    dst: &mut D,
    src: &poulpy_hal::layouts::VecZnxBackendRef<'_, BE>,
    offset: usize,
) where
    BE: Backend,
    M: VecZnxCopyBackend<BE>,
    D: VecZnxToBackendMut<BE> + ZnxInfos,
{
    let cols = dst.cols();
    let mut dst_mut = dst.to_backend_mut();
    for col in 0..cols {
        module.vec_znx_copy_backend(&mut dst_mut, col, src, offset + col);
    }
}

#[allow(clippy::too_many_arguments)]
fn collapse_half_split<
    BE,
    M,
    Mask,
    Body,
    TermBody,
    TermMask,
    SwitchedBody,
    SwitchedMask,
    KBody,
    KMask,
>(
    module: &M,
    mask: &mut Mask,
    body: &mut Body,
    use_tau_h: bool,
    key_body: &KBody,
    key_mask: &KMask,
    key_size: usize,
    base2k: usize,
    term_body: &mut TermBody,
    term_mask: &mut TermMask,
    switched_body: &mut SwitchedBody,
    switched_mask: &mut SwitchedMask,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GaloisElement
        + VecZnxAddAssignBackend<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VmpApplyDftToDft<BE>,
    Mask: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    Body: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    TermBody: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    TermMask: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    SwitchedBody: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    SwitchedMask: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    KBody: poulpy_core::layouts::GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    KMask: poulpy_core::layouts::GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBasic<'a, BE>,
{
    for target_col in (0..mask.cols() - 1).rev() {
        let source_col = target_col + 1;
        let tau_g_j = module.galois_element(target_col as i64);
        let secret_view = if use_tau_h { -tau_g_j } else { tau_g_j };
        let alpha = module.galois_element_inv(secret_view);
        let alpha_inv = secret_view;

        {
            let body_ref = body.to_backend_ref();
            let mask_ref = mask.to_backend_ref();
            let mut term_body_mut = term_body.to_backend_mut();
            let mut term_mask_mut = term_mask.to_backend_mut();
            module.vec_znx_automorphism_backend(alpha_inv, &mut term_body_mut, 0, &body_ref, 0);
            module.vec_znx_automorphism_backend(
                alpha_inv,
                &mut term_mask_mut,
                0,
                &mask_ref,
                source_col,
            );
        }

        fixed_mask_1x1_vmp_keyswitch_body(
            module,
            switched_body,
            base2k,
            term_mask,
            0,
            term_body,
            0,
            base2k,
            key_body,
            key_size,
            &mut scratch.borrow(),
        );
        fixed_mask_1x1_vmp_body_addend(
            module,
            switched_mask,
            base2k,
            term_mask,
            0,
            key_mask,
            key_size,
            &mut scratch.borrow(),
        );

        {
            let switched_body_ref = switched_body.to_backend_ref();
            let switched_mask_ref = switched_mask.to_backend_ref();
            let mut body_mut = body.to_backend_mut();
            let mut term_mask_mut = term_mask.to_backend_mut();
            module.vec_znx_automorphism_backend(alpha, &mut body_mut, 0, &switched_body_ref, 0);
            module.vec_znx_automorphism_backend(
                alpha,
                &mut term_mask_mut,
                0,
                &switched_mask_ref,
                0,
            );
        }
        {
            let term_mask_ref = term_mask.to_backend_ref();
            let mut mask_mut = mask.to_backend_mut();
            module.vec_znx_add_assign_backend(&mut mask_mut, target_col, &term_mask_ref, 0);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn collapse_half<BE, M, K>(
    module: &M,
    mask: &mut VecZnx<BE::OwnedBuf>,
    body: &mut VecZnx<BE::OwnedBuf>,
    use_tau_h: bool,
    key_g: &K,
    key_size: usize,
    term: &mut GLWE<BE::OwnedBuf>,
    switched: &mut GLWE<BE::OwnedBuf>,
    switched_auto: &mut GLWE<BE::OwnedBuf>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: GLWEKeyswitch<BE>
        + GLWEZero<BE>
        + GaloisElement
        + VecZnxAddAssignBackend<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxCopyBackend<BE>,
    K: GGLWEPreparedToBackendRef<BE> + GGLWEInfos,
{
    for target_col in (0..mask.cols() - 1).rev() {
        let source_col = target_col + 1;
        let tau_g_j = module.galois_element(target_col as i64);
        let secret_view = if use_tau_h { -tau_g_j } else { tau_g_j };
        let alpha = module.galois_element_inv(secret_view);
        let alpha_inv = secret_view;

        module.glwe_zero(term);
        {
            let body_ref = <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(body);
            let mask_ref = <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(mask);
            let mut term_ref = <_ as GLWEToBackendMut<BE>>::to_backend_mut(term);
            module.vec_znx_automorphism_backend(alpha_inv, term_ref.data_mut(), 0, &body_ref, 0);
            module.vec_znx_automorphism_backend(
                alpha_inv,
                term_ref.data_mut(),
                1,
                &mask_ref,
                source_col,
            );
        }
        module.glwe_keyswitch(switched, term, key_g, key_size, scratch);
        {
            let switched_ref = <_ as GLWEToBackendRef<BE>>::to_backend_ref(switched);
            let mut switched_auto_ref = <_ as GLWEToBackendMut<BE>>::to_backend_mut(switched_auto);
            module.vec_znx_automorphism_backend(
                alpha,
                switched_auto_ref.data_mut(),
                0,
                switched_ref.data(),
                0,
            );
            module.vec_znx_automorphism_backend(
                alpha,
                switched_auto_ref.data_mut(),
                1,
                switched_ref.data(),
                1,
            );
        }
        {
            let switched_ref = <_ as GLWEToBackendRef<BE>>::to_backend_ref(switched_auto);
            let mut body_mut =
                <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(body);
            let mut mask_mut =
                <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(mask);
            module.vec_znx_copy_backend(&mut body_mut, 0, switched_ref.data(), 0);
            module.vec_znx_add_assign_backend(&mut mask_mut, target_col, switched_ref.data(), 1);
        }
    }
}
