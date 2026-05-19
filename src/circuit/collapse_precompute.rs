use poulpy_core::layouts::{GGLWEInfos, GGLWEPreparedVmpPMatRef, Rank};
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxAddAssignBackend, VecZnxAutomorphismBackend,
        VecZnxBigAddSmallAssign, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxCopyBackend, VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftBytesOf, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes, VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{
        Backend, Data, GaloisElement, Module, ScratchArena, VecZnx, VecZnxBigToBackendMut,
        VecZnxBigToBackendRef, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxToBackendMut,
        VecZnxToBackendRef, ZnxInfos,
    },
};

/// State produced by the strict fixed-mask collapse precompute.
///
/// This is intentionally PIR-local: the generic core key-switch computes a full
/// `1 x 2` product, while this layout stores only the pieces needed by the
/// specialized collapse path. Only the mask side is stored here: the key body
/// changes per query and must be handled online. `body_vmp_masks` stores the
/// fixed per-step mask inputs that the online body-side `1 x 1` products must
/// use; these are masks, not body addends. `final_mask` holds the fixed mask
/// state after all precomputed mask updates have been applied.
pub struct SequentialCollapseMaskPrecompute<D: Data> {
    body_vmp_masks: VecZnx<D>,
    final_mask: VecZnx<D>,
    base2k: usize,
    key_base2k: usize,
    size: usize,
    steps: usize,
    rank: Rank,
}

impl<D: Data> SequentialCollapseMaskPrecompute<D> {
    /// Per-step fixed mask inputs for the online body-side `1 x 1` VMPs.
    ///
    /// Column `step` is the already-automorphed mask share used by the matching
    /// online collapse step. The query-dependent key body is intentionally not
    /// multiplied here.
    pub fn body_vmp_masks(&self) -> &VecZnx<D> {
        &self.body_vmp_masks
    }

    /// Mutable access used by the precompute routine while recording the mask schedule.
    pub fn body_vmp_masks_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.body_vmp_masks
    }

    /// Final precomputed GLWE mask to copy into the online collapse result.
    pub fn final_mask(&self) -> &VecZnx<D> {
        &self.final_mask
    }

    /// Mutable access used by the precompute routine while filling the mask.
    pub fn final_mask_mut(&mut self) -> &mut VecZnx<D> {
        &mut self.final_mask
    }

    pub fn base2k(&self) -> usize {
        self.base2k
    }

    pub fn key_base2k(&self) -> usize {
        self.key_base2k
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn rank(&self) -> Rank {
        self.rank
    }
}

/// Allocates the PIR-local storage for the fixed-mask collapse precompute.
///
/// `steps` is the number of logical mask updates in the collapse schedule.
/// `rank` describes the final GLWE mask rank and controls the number of
/// columns in `final_mask`. The precompute also keeps one mask input per step
/// for the online body-side products.
pub fn sequential_collapse_mask_precompute_alloc<BE: Backend>(
    module: &Module<BE>,
    steps: usize,
    size: usize,
    base2k: usize,
    key_base2k: usize,
    rank: Rank,
) -> SequentialCollapseMaskPrecompute<BE::OwnedBuf> {
    SequentialCollapseMaskPrecompute {
        body_vmp_masks: module.vec_znx_alloc(steps, size),
        final_mask: module.vec_znx_alloc(rank.as_usize(), size),
        base2k,
        key_base2k,
        size,
        steps,
        rank,
    }
}

/// Scratch estimate for [`precompute_sequential_keyswitch_collapse_aggregate_mask`].
#[allow(clippy::too_many_arguments)]
pub fn precompute_sequential_keyswitch_collapse_aggregate_mask_tmp_bytes<BE, A, KGMask, KHMask>(
    module: &Module<BE>,
    aggregate_mask: &A,
    key_g_mask: &KGMask,
    key_h_mask: &KHMask,
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
    A: ZnxInfos,
    KGMask: GGLWEInfos,
    KHMask: GGLWEInfos,
{
    let n = module.n();
    let half = n >> 1;
    let size = aggregate_mask.size();
    let align = |len: usize| len.next_multiple_of(BE::SCRATCH_ALIGN);
    let vec_scratch = align(VecZnx::<Vec<u8>>::bytes_of(n, half, size))
        + 4 * align(VecZnx::<Vec<u8>>::bytes_of(n, 1, size));
    let key_g_scratch = fixed_mask_1x1_vmp_body_addend_tmp_bytes::<BE, _, _, _>(
        module,
        aggregate_mask,
        key_g_mask,
        key_g_size,
    );
    let key_h_scratch = fixed_mask_1x1_vmp_body_addend_tmp_bytes::<BE, _, _, _>(
        module,
        aggregate_mask,
        key_h_mask,
        key_h_size,
    );

    vec_scratch + key_g_scratch.max(key_h_scratch)
}

/// Precomputes the fixed-mask side of the sequential split collapse.
///
/// The loop order and automorphisms mirror
/// `sequential_keyswitch_collapse_aggregate_mask_split`: the only difference is
/// that the input mask is fixed, so the mask-key `1 x 1` products are applied
/// to an offline mask state. Before each offline mask product, the routine also
/// records the fixed mask input required by the corresponding online body-side
/// product. The key body is query-dependent and is not part of this precompute.
#[allow(clippy::too_many_arguments)]
pub fn precompute_sequential_keyswitch_collapse_aggregate_mask<BE, A, KGMask, KHMask>(
    module: &Module<BE>,
    precompute: &mut SequentialCollapseMaskPrecompute<BE::OwnedBuf>,
    aggregate_mask: &A,
    key_g_mask: &KGMask,
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
    A: VecZnxToBackendRef<BE> + ZnxInfos,
    KGMask: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    KHMask: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBasic<'a, BE>,
{
    let n = module.n();
    let half = n >> 1;
    assert_eq!(aggregate_mask.n(), n);
    assert_eq!(aggregate_mask.cols(), n);
    assert_eq!(aggregate_mask.size(), precompute.size());
    assert_eq!(precompute.steps(), n - 1);
    assert_eq!(precompute.body_vmp_masks().cols(), precompute.steps());
    assert_eq!(precompute.rank().as_usize(), 1);
    assert_eq!(precompute.key_base2k(), key_g_mask.base2k().as_usize());
    assert_eq!(precompute.key_base2k(), key_h_mask.base2k().as_usize());
    assert_eq!(key_g_mask.rank_in().as_usize(), 1);
    assert_eq!(key_h_mask.rank_in().as_usize(), 1);
    assert_eq!(key_g_mask.rank_out().as_usize(), 0);
    assert_eq!(key_h_mask.rank_out().as_usize(), 0);

    let scratch_local = scratch.borrow();
    let (mut half_work, scratch_local) =
        scratch_local.take_vec_znx_scratch(n, half, aggregate_mask.size());
    let (mut first_share, scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, aggregate_mask.size());
    let (mut term_mask, scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, aggregate_mask.size());
    let (mut mask_addend, scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, aggregate_mask.size());
    let (mut mask_addend_auto, mut scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, aggregate_mask.size());

    let aggregate_ref = aggregate_mask.to_backend_ref();
    let mut step = 0usize;

    copy_aggregate_half(module, &mut half_work, &aggregate_ref, 0);
    precompute_collapse_half(
        module,
        precompute,
        &mut half_work,
        false,
        key_g_mask,
        key_g_size,
        &mut step,
        &mut term_mask,
        &mut mask_addend,
        &mut mask_addend_auto,
        &mut scratch_local,
    );

    {
        let half_ref = half_work.to_backend_ref();
        let mut first_mut = first_share.to_backend_mut();
        module.vec_znx_copy_backend(&mut first_mut, 0, &half_ref, 0);
    }

    copy_aggregate_half(module, &mut half_work, &aggregate_ref, half);
    precompute_collapse_half(
        module,
        precompute,
        &mut half_work,
        true,
        key_g_mask,
        key_g_size,
        &mut step,
        &mut term_mask,
        &mut mask_addend,
        &mut mask_addend_auto,
        &mut scratch_local,
    );

    {
        let half_ref = half_work.to_backend_ref();
        let mut term_mut = term_mask.to_backend_mut();
        module.vec_znx_copy_backend(&mut term_mut, 0, &half_ref, 0);
    }
    store_body_vmp_mask(module, precompute, step, &term_mask);
    fixed_mask_1x1_vmp_body_addend(
        module,
        &mut mask_addend,
        precompute.base2k(),
        &term_mask,
        0,
        key_h_mask,
        key_h_size,
        &mut scratch_local.borrow(),
    );
    {
        let mask_ref = mask_addend.to_backend_ref();
        let first_ref = first_share.to_backend_ref();
        let mut final_mut = <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(
            precompute.final_mask_mut(),
        );
        module.vec_znx_copy_backend(&mut final_mut, 0, &mask_ref, 0);
        module.vec_znx_add_assign_backend(&mut final_mut, 0, &first_ref, 0);
    }
    step += 1;
    assert_eq!(step, precompute.steps());
}

/// Scratch estimate for
/// [`precompute_sequential_keyswitch_collapse_aggregate_mask_dft_accumulated`].
///
/// This is intentionally conservative: the DFT-accumulated variant reuses the
/// strict collapse-half scratch and adds a terminal DFT accumulator plus one big
/// materialization buffer.
#[allow(clippy::too_many_arguments)]
pub fn precompute_sequential_keyswitch_collapse_aggregate_mask_dft_accumulated_tmp_bytes<
    BE,
    A,
    KGMask,
    KHMask,
>(
    module: &Module<BE>,
    aggregate_mask: &A,
    key_g_mask: &KGMask,
    key_h_mask: &KHMask,
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
    A: ZnxInfos,
    KGMask: GGLWEInfos,
    KHMask: GGLWEInfos,
{
    let key_h_size = key_h_mask.size().min(key_h_size);
    let align = |len: usize| len.next_multiple_of(BE::SCRATCH_ALIGN);
    precompute_sequential_keyswitch_collapse_aggregate_mask_tmp_bytes::<BE, _, _, _>(
        module,
        aggregate_mask,
        key_g_mask,
        key_h_mask,
        key_g_size,
        key_h_size,
    ) + fixed_mask_1x1_vmp_dft_product_tmp_bytes::<BE, _, _, _>(
        module,
        aggregate_mask,
        key_h_mask,
        key_h_size,
    ) + 2 * align(module.bytes_of_vec_znx_dft(1, key_h_size))
        + align(module.bytes_of_vec_znx_big(1, key_h_size))
}

/// Precomputes the fixed-mask collapse with a terminal DFT-domain accumulation.
///
/// This variant is deliberately separate from
/// [`precompute_sequential_keyswitch_collapse_aggregate_mask`]. The strict
/// routine preserves bit equality by normalizing every mask-side product at the
/// same point as the baseline. This routine keeps the final `Kh` mask product
/// in the DFT domain, adds the fixed first-half share there, then performs one
/// IDFT/normalization at the end. That changes the rounding point, so tests for
/// this path use decryption equality rather than final-GLWE byte equality.
#[allow(clippy::too_many_arguments)]
pub fn precompute_sequential_keyswitch_collapse_aggregate_mask_dft_accumulated<
    BE,
    A,
    KGMask,
    KHMask,
>(
    module: &Module<BE>,
    precompute: &mut SequentialCollapseMaskPrecompute<BE::OwnedBuf>,
    aggregate_mask: &A,
    key_g_mask: &KGMask,
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
        + VecZnxDftAddAssign<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VmpApplyDftToDft<BE>,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
    KGMask: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    KHMask: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBasic<'a, BE>,
{
    let n = module.n();
    let half = n >> 1;
    assert_eq!(aggregate_mask.n(), n);
    assert_eq!(aggregate_mask.cols(), n);
    assert_eq!(aggregate_mask.size(), precompute.size());
    assert_eq!(precompute.steps(), n - 1);
    assert_eq!(precompute.body_vmp_masks().cols(), precompute.steps());
    assert_eq!(precompute.rank().as_usize(), 1);
    assert_eq!(precompute.base2k(), precompute.key_base2k());
    assert_eq!(precompute.key_base2k(), key_g_mask.base2k().as_usize());
    assert_eq!(precompute.key_base2k(), key_h_mask.base2k().as_usize());
    assert_eq!(key_g_mask.rank_in().as_usize(), 1);
    assert_eq!(key_h_mask.rank_in().as_usize(), 1);
    assert_eq!(key_g_mask.rank_out().as_usize(), 0);
    assert_eq!(key_h_mask.rank_out().as_usize(), 0);

    let scratch_local = scratch.borrow();
    let (mut half_work, scratch_local) =
        scratch_local.take_vec_znx_scratch(n, half, aggregate_mask.size());
    let (mut first_share, scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, aggregate_mask.size());
    let (mut term_mask, scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, aggregate_mask.size());
    let (mut mask_addend, scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, aggregate_mask.size());
    let (mut mask_addend_auto, mut scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, aggregate_mask.size());

    let aggregate_ref = aggregate_mask.to_backend_ref();
    let mut step = 0usize;

    copy_aggregate_half(module, &mut half_work, &aggregate_ref, 0);
    precompute_collapse_half(
        module,
        precompute,
        &mut half_work,
        false,
        key_g_mask,
        key_g_size,
        &mut step,
        &mut term_mask,
        &mut mask_addend,
        &mut mask_addend_auto,
        &mut scratch_local,
    );

    {
        let half_ref = half_work.to_backend_ref();
        let mut first_mut = first_share.to_backend_mut();
        module.vec_znx_copy_backend(&mut first_mut, 0, &half_ref, 0);
    }

    copy_aggregate_half(module, &mut half_work, &aggregate_ref, half);
    precompute_collapse_half(
        module,
        precompute,
        &mut half_work,
        true,
        key_g_mask,
        key_g_size,
        &mut step,
        &mut term_mask,
        &mut mask_addend,
        &mut mask_addend_auto,
        &mut scratch_local,
    );

    {
        let half_ref = half_work.to_backend_ref();
        let mut term_mut = term_mask.to_backend_mut();
        module.vec_znx_copy_backend(&mut term_mut, 0, &half_ref, 0);
    }
    store_body_vmp_mask(module, precompute, step, &term_mask);

    let key_h_size = key_h_mask.size().min(key_h_size);
    let (mut final_dft, scratch_local) =
        scratch_local.take_vec_znx_dft_scratch(module, 1, key_h_size);
    let (mut first_dft, scratch_local) =
        scratch_local.take_vec_znx_dft_scratch(module, 1, key_h_size);
    let (mut final_big, mut scratch_local) =
        scratch_local.take_vec_znx_big_scratch(module, 1, key_h_size);

    fixed_mask_1x1_vmp_dft_product(
        module,
        &mut final_dft,
        &term_mask,
        0,
        key_h_mask,
        key_h_size,
        &mut scratch_local.borrow(),
    );
    module.vec_znx_dft_apply(
        1,
        0,
        &mut first_dft.to_backend_mut(),
        0,
        &first_share.to_backend_ref(),
        0,
    );
    module.vec_znx_dft_add_assign(
        &mut final_dft.to_backend_mut(),
        0,
        &first_dft.to_backend_ref(),
        0,
    );
    module.vec_znx_idft_apply(
        &mut final_big.to_backend_mut(),
        0,
        &final_dft.to_backend_ref(),
        0,
        &mut scratch_local.borrow(),
    );
    let base2k = precompute.base2k();
    module.vec_znx_big_normalize(
        &mut <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(
            precompute.final_mask_mut(),
        ),
        base2k,
        0,
        0,
        &final_big.to_backend_ref(),
        key_h_mask.base2k().as_usize(),
        0,
        &mut scratch_local,
    );

    step += 1;
    assert_eq!(step, precompute.steps());
}

fn copy_aggregate_half<BE, D>(
    module: &Module<BE>,
    dst: &mut D,
    src: &poulpy_hal::layouts::VecZnxBackendRef<'_, BE>,
    offset: usize,
) where
    BE: Backend,
    Module<BE>: VecZnxCopyBackend<BE>,
    D: VecZnxToBackendMut<BE> + ZnxInfos,
{
    let cols = dst.cols();
    let mut dst_mut = dst.to_backend_mut();
    for col in 0..cols {
        module.vec_znx_copy_backend(&mut dst_mut, col, src, offset + col);
    }
}

fn store_body_vmp_mask<BE, A>(
    module: &Module<BE>,
    precompute: &mut SequentialCollapseMaskPrecompute<BE::OwnedBuf>,
    step: usize,
    mask: &A,
) where
    BE: Backend,
    Module<BE>: VecZnxCopyBackend<BE>,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
{
    assert!(step < precompute.steps());
    let mask_ref = mask.to_backend_ref();
    let mut masks_mut = <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(
        precompute.body_vmp_masks_mut(),
    );
    module.vec_znx_copy_backend(&mut masks_mut, step, &mask_ref, 0);
}

#[allow(clippy::too_many_arguments)]
fn precompute_collapse_half<BE, Mask, TermMask, MaskAddend, MaskAddendAuto, KMask>(
    module: &Module<BE>,
    precompute: &mut SequentialCollapseMaskPrecompute<BE::OwnedBuf>,
    mask: &mut Mask,
    use_tau_h: bool,
    key_mask: &KMask,
    key_size: usize,
    step: &mut usize,
    term_mask: &mut TermMask,
    mask_addend: &mut MaskAddend,
    mask_addend_auto: &mut MaskAddendAuto,
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
    Mask: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    TermMask: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    MaskAddend: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    MaskAddendAuto: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    KMask: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBasic<'a, BE>,
{
    for target_col in (0..mask.cols() - 1).rev() {
        let source_col = target_col + 1;
        let tau_g_j = module.galois_element(target_col as i64);
        let secret_view = if use_tau_h { -tau_g_j } else { tau_g_j };
        let alpha = module.galois_element_inv(secret_view);
        let alpha_inv = secret_view;

        module.vec_znx_automorphism_backend(
            alpha_inv,
            &mut term_mask.to_backend_mut(),
            0,
            &mask.to_backend_ref(),
            source_col,
        );
        store_body_vmp_mask(module, precompute, *step, term_mask);

        fixed_mask_1x1_vmp_body_addend(
            module,
            mask_addend,
            precompute.base2k(),
            term_mask,
            0,
            key_mask,
            key_size,
            &mut scratch.borrow(),
        );

        module.vec_znx_automorphism_backend(
            alpha,
            &mut mask_addend_auto.to_backend_mut(),
            0,
            &mask_addend.to_backend_ref(),
            0,
        );

        module.vec_znx_add_assign_backend(
            &mut mask.to_backend_mut(),
            target_col,
            &mask_addend_auto.to_backend_ref(),
            0,
        );

        *step += 1;
    }
}

/// Scratch estimate for [`fixed_mask_1x1_vmp_body_addend`].
///
/// This is deliberately scoped to the local helper rather than the future full
/// precompute routine. It covers the scratch-backed DFT input, DFT product,
/// coefficient-domain big product, and the largest HAL operation scratch used
/// by VMP, IDFT, or normalization.
pub fn fixed_mask_1x1_vmp_body_addend_tmp_bytes<BE, M, A, K>(
    module: &M,
    mask: &A,
    key: &K,
    key_size: usize,
) -> usize
where
    BE: Backend,
    M: VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + VmpApplyDftToDftTmpBytes,
    A: ZnxInfos,
    K: GGLWEInfos,
{
    let key_size = key.size().min(key_size);
    assert_eq!(
        key.dsize().as_usize(),
        1,
        "fixed-mask 1x1 VMP currently assumes dsize = 1"
    );

    let align = |len: usize| len.next_multiple_of(BE::SCRATCH_ALIGN);
    let lvl_0 = align(module.bytes_of_vec_znx_dft(1, mask.size()));
    let lvl_1 = align(module.bytes_of_vec_znx_dft(1, key_size));
    let lvl_2 = align(module.bytes_of_vec_znx_big(1, key_size));
    let lvl_ops = module
        .vmp_apply_dft_to_dft_tmp_bytes(
            key_size,
            mask.size(),
            key.dnum().as_usize(),
            key.rank_in().as_usize(),
            key.rank_out().as_usize() + 1,
            key.size(),
        )
        .max(module.vec_znx_idft_apply_tmp_bytes())
        .max(module.vec_znx_big_normalize_tmp_bytes());

    lvl_0 + lvl_1 + lvl_2 + lvl_ops
}

/// Scratch estimate for [`fixed_mask_1x1_vmp_dft_product`].
pub fn fixed_mask_1x1_vmp_dft_product_tmp_bytes<BE, M, A, K>(
    module: &M,
    mask: &A,
    key: &K,
    key_size: usize,
) -> usize
where
    BE: Backend,
    M: VecZnxDftBytesOf + VmpApplyDftToDftTmpBytes,
    A: ZnxInfos,
    K: GGLWEInfos,
{
    let key_size = key.size().min(key_size);
    assert_eq!(
        key.dsize().as_usize(),
        1,
        "fixed-mask 1x1 VMP currently assumes dsize = 1"
    );

    let align = |len: usize| len.next_multiple_of(BE::SCRATCH_ALIGN);
    align(module.bytes_of_vec_znx_dft(1, mask.size()))
        + module.vmp_apply_dft_to_dft_tmp_bytes(
            key_size,
            mask.size(),
            key.dnum().as_usize(),
            key.rank_in().as_usize(),
            key.rank_out().as_usize() + 1,
            key.size(),
        )
}

/// Computes the DFT-domain product for one fixed-mask `1 x 1` VMP.
///
/// Unlike [`fixed_mask_1x1_vmp_body_addend`], this helper deliberately stops
/// before IDFT and normalization so callers can accumulate compatible products
/// in the DFT domain and choose a later materialization point.
pub fn fixed_mask_1x1_vmp_dft_product<BE, M, R, A, K>(
    module: &M,
    res: &mut R,
    mask: &A,
    mask_col: usize,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: ModuleN + VecZnxDftApply<BE> + VecZnxDftBytesOf + VmpApplyDftToDft<BE>,
    R: VecZnxDftToBackendMut<BE> + ZnxInfos,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
    K: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBasic<'a, BE>,
{
    assert_eq!(key.rank_in().as_usize(), 1);
    assert_eq!(key.rank_out().as_usize(), 0);
    assert_eq!(
        key.dsize().as_usize(),
        1,
        "fixed-mask 1x1 VMP currently assumes dsize = 1"
    );
    assert!(mask_col < mask.cols());
    assert!(res.cols() >= 1);
    assert_eq!(res.n(), mask.n());
    assert_eq!(res.n(), key.n().as_usize());

    let key_size = key.size().min(key_size);
    assert_eq!(res.size(), key_size);

    let (mut mask_dft, mut scratch_1) =
        scratch
            .borrow()
            .take_vec_znx_dft_scratch(module, 1, mask.size());
    module.vec_znx_dft_apply(
        1,
        0,
        &mut mask_dft.to_backend_mut(),
        0,
        &mask.to_backend_ref(),
        mask_col,
    );
    module.vmp_apply_dft_to_dft(
        &mut res.to_backend_mut(),
        &mask_dft.to_backend_ref(),
        &key.vmp_pmat_backend_ref(),
        0,
        &mut scratch_1.borrow(),
    );
}

#[allow(clippy::too_many_arguments)]
fn fixed_mask_1x1_vmp_body_addend_impl<BE, M, R, A, B, K>(
    module: &M,
    res: &mut R,
    res_base2k: usize,
    mask: &A,
    mask_col: usize,
    body: Option<(&B, usize, usize)>,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: ModuleN
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VmpApplyDftToDft<BE>,
    R: VecZnxToBackendMut<BE> + ZnxInfos,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
    B: VecZnxToBackendRef<BE> + ZnxInfos,
    K: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBasic<'a, BE>,
{
    assert_eq!(key.rank_in().as_usize(), 1);
    assert_eq!(key.rank_out().as_usize(), 0);
    assert_eq!(
        key.dsize().as_usize(),
        1,
        "fixed-mask 1x1 VMP currently assumes dsize = 1"
    );
    assert!(mask_col < mask.cols());
    assert!(res.cols() >= 1);
    assert_eq!(res.n(), mask.n());
    assert_eq!(res.n(), key.n().as_usize());
    if let Some((body, body_col, body_base2k)) = body {
        assert_eq!(body.n(), mask.n());
        assert!(body_col < body.cols());
        assert_eq!(
            body_base2k,
            key.base2k().as_usize(),
            "1x1 key-switch body addend currently expects body and key base2k to match"
        );
    }

    let key_size = key.size().min(key_size);
    let key_base2k = key.base2k().as_usize();

    // Step 1: prepare the fixed mask column in the DFT domain.
    let (mut mask_dft, scratch_1) =
        scratch
            .borrow()
            .take_vec_znx_dft_scratch(module, 1, mask.size());
    module.vec_znx_dft_apply(
        1,
        0,
        &mut mask_dft.to_backend_mut(),
        0,
        &mask.to_backend_ref(),
        mask_col,
    );

    let (mut product_dft, mut scratch_2) = scratch_1.take_vec_znx_dft_scratch(module, 1, key_size);

    // Step 2: a single DFT vector-matrix product gives the full dsize = 1 product.
    module.vmp_apply_dft_to_dft(
        &mut product_dft.to_backend_mut(),
        &mask_dft.to_backend_ref(),
        &key.vmp_pmat_backend_ref(),
        0,
        &mut scratch_2.borrow(),
    );

    // Step 3: materialize the DFT product into the big coefficient domain.
    let (mut product_big, mut scratch_3) = scratch_2.take_vec_znx_big_scratch(module, 1, key_size);
    module.vec_znx_idft_apply(
        &mut product_big.to_backend_mut(),
        0,
        &product_dft.to_backend_ref(),
        0,
        &mut scratch_3.borrow(),
    );

    if let Some((body, body_col, _)) = body {
        module.vec_znx_big_add_small_assign(
            &mut product_big.to_backend_mut(),
            0,
            &body.to_backend_ref(),
            body_col,
        );
    }

    // Step 4: normalize the body addend into the requested base.
    module.vec_znx_big_normalize(
        &mut res.to_backend_mut(),
        res_base2k,
        0,
        0,
        &product_big.to_backend_ref(),
        key_base2k,
        0,
        &mut scratch_3,
    );
}

/// Computes one fixed-mask body addend with a `1 x 1` VMP.
///
/// This mirrors the mask-dependent part of the core key-switch pipeline, but it
/// specializes the output to a single body column. The helper:
///
/// 1. borrows scratch for `mask[mask_col]` and prepares it into the DFT domain,
/// 2. applies the immutable prepared `VmpPMat` exposed by the key,
/// 3. performs an IDFT into a scratch-backed big coefficient buffer,
/// 4. normalizes into `res[0]`.
///
/// The key is required to have `rank_in = 1` and `rank_out = 0`, which gives the
/// `1 x 1` VMP shape used by the fixed-mask precompute. The helper currently
/// assumes `dsize = 1`, matching the planned mask-resampling keys.
#[allow(clippy::too_many_arguments)]
pub fn fixed_mask_1x1_vmp_body_addend<BE, M, R, A, K>(
    module: &M,
    res: &mut R,
    res_base2k: usize,
    mask: &A,
    mask_col: usize,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: ModuleN
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VmpApplyDftToDft<BE>,
    R: VecZnxToBackendMut<BE> + ZnxInfos,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
    K: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBasic<'a, BE>,
{
    fixed_mask_1x1_vmp_body_addend_impl::<BE, M, R, A, A, K>(
        module, res, res_base2k, mask, mask_col, None, key, key_size, scratch,
    );
}

/// Computes the body output of a split `1 x 1` key-switch.
///
/// This is the same VMP path as [`fixed_mask_1x1_vmp_body_addend`], but it also
/// adds the input GLWE body into the big coefficient accumulator before the
/// final normalization. That matches the rounding point of the generic
/// `GLWEKeyswitch` implementation for the body column.
#[allow(clippy::too_many_arguments)]
pub fn fixed_mask_1x1_vmp_keyswitch_body<BE, M, R, A, B, K>(
    module: &M,
    res: &mut R,
    res_base2k: usize,
    mask: &A,
    mask_col: usize,
    body: &B,
    body_col: usize,
    body_base2k: usize,
    key: &K,
    key_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: ModuleN
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VmpApplyDftToDft<BE>,
    R: VecZnxToBackendMut<BE> + ZnxInfos,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
    B: VecZnxToBackendRef<BE> + ZnxInfos,
    K: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBasic<'a, BE>,
{
    fixed_mask_1x1_vmp_body_addend_impl(
        module,
        res,
        res_base2k,
        mask,
        mask_col,
        Some((body, body_col, body_base2k)),
        key,
        key_size,
        scratch,
    );
}
