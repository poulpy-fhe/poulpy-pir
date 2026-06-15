//! Fixed mask-side precomputations for packing.
//!
//! This module owns the server/query-independent part of packing. It first
//! records the coefficient-domain mask schedule for the sequential collapse,
//! then derives the DFT-hot BSGS mask cache consumed by the online packing
//! loop. Client-key-side prepared bodies are passed separately through the
//! packing API and are never stored here.

use std::any::Any;

use poulpy_core::layouts::{GGLWEInfos, GGLWEPreparedVmpPMatRef};
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxAddAssignBackend, VecZnxAutomorphismBackend,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend,
        VecZnxDftAlloc, VecZnxDftApply, VecZnxDftAutomorphismPlan, VecZnxDftBytesOf,
        VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VecZnxNormalize, VmpApplyDftToDft,
        VmpApplyDftToDftTmpBytes,
    },
    layouts::{
        Backend, GaloisElement, Module, ScratchArena, VecZnx, VecZnxBigToBackendMut,
        VecZnxBigToBackendRef, VecZnxDft, VecZnxDftToBackendMut, VecZnxDftToBackendRef,
        VecZnxToBackendMut, VecZnxToBackendRef, ZnxInfos,
    },
};

const PACK_PRECOMPUTE_ARITHMETIC_BASE2K: usize = 50;

/// Shape metadata for fixed mask-side packing precomputations.
///
/// This is the allocation-free description used by scratch estimators. The
/// owned [`PackingPrecomputations`] stores buffers and backend plans; callers
/// should not need to allocate those just to ask how much scratch precompute
/// will require.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackingPrecomputeInfos {
    /// Total stored mask steps. For full packing this is all `key_g` collapse
    /// steps plus the final `key_h` step; for partial packing it is the `γ-1`
    /// `key_g` steps only (no `key_h`).
    steps: usize,
    /// Number of limbs/components in each stored mask column.
    size: usize,
    /// Base used when normalizing big products back into coefficient buffers.
    base2k: usize,
    /// Number of baby steps per giant-step group.
    baby_size: usize,
    /// Partial packing (Algorithm 2): a single `key_g` collapse half over `γ`
    /// coefficients, with no second half and no final `key_h` step.
    partial: bool,
    /// Galois generator stride `e = (d/2)/γ`. The packing automorphisms use the
    /// generator `g_γ = 5^e` (order `γ`), so every `galois_element(x)` becomes
    /// `galois_element(e·x)`. Full packing has `γ = d/2`, hence `e = 1`.
    stride: usize,
}

impl PackingPrecomputeInfos {
    /// Creates full-packing metadata matching [`pack_precompute_alloc_default`].
    pub fn new(steps: usize, size: usize, base2k: usize, baby_size: usize) -> Self {
        Self {
            steps,
            size,
            base2k,
            baby_size,
            partial: false,
            stride: 1,
        }
    }

    /// Creates partial-packing metadata: `steps = γ-1` `key_g` steps, no `key_h`,
    /// with Galois generator stride `e = (d/2)/γ`.
    pub fn new_partial(
        steps: usize,
        size: usize,
        base2k: usize,
        baby_size: usize,
        stride: usize,
    ) -> Self {
        Self {
            steps,
            size,
            base2k,
            baby_size,
            partial: true,
            stride,
        }
    }

    /// Whether this is a partial (single-half, no-`key_h`) packing layout.
    pub fn partial(self) -> bool {
        self.partial
    }

    /// Galois generator stride `e = (d/2)/γ` (1 for full packing).
    pub fn stride(self) -> usize {
        self.stride
    }

    /// Total stored mask steps, including the final `key_h` step.
    pub fn steps(self) -> usize {
        self.steps
    }

    /// Number of limbs/components in each stored mask column.
    pub fn size(self) -> usize {
        self.size
    }

    /// Normalization base shared by fixed-mask precompute and online output.
    pub fn base2k(self) -> usize {
        self.base2k
    }

    /// Number of baby steps per giant-step group.
    pub fn baby_size(self) -> usize {
        self.baby_size
    }
}

/// Metadata used by the coefficient-domain fixed-mask precompute arithmetic.
///
/// The online packing layout remains at the caller-requested base. The purely
/// offline VecZnx arithmetic can use a wider base while still satisfying the
/// FFT input bound of the current fixed-mask VMP path.
pub(crate) fn arithmetic_precompute_metadata(
    metadata: PackingPrecomputeInfos,
) -> PackingPrecomputeInfos {
    let base2k = PACK_PRECOMPUTE_ARITHMETIC_BASE2K.max(metadata.base2k);
    let torus_bits = metadata.size * metadata.base2k;
    let size = torus_bits.div_ceil(base2k).max(1);
    if metadata.partial {
        PackingPrecomputeInfos::new_partial(
            metadata.steps,
            size,
            base2k,
            metadata.baby_size,
            metadata.stride,
        )
    } else {
        PackingPrecomputeInfos::new(metadata.steps, size, base2k, metadata.baby_size)
    }
}

/// Fixed mask-side state produced before online packing.
///
/// This is intentionally PIR-local: the generic core key-switch computes a full
/// `1 x 2` product, while this layout stores only the pieces needed by the
/// specialized collapse path. Only fixed, query-independent state is stored
/// here: the key body changes per query and must be handled online.
pub struct PackingPrecomputations<BE: Backend> {
    /// Coefficient-domain fixed mask inputs for each online body-side VMP.
    body_vmp_masks: VecZnx<BE::OwnedBuf>,
    /// Final GLWE mask copied into the packed result after online body work.
    final_mask: VecZnx<BE::OwnedBuf>,
    /// DFT columns derived from `body_vmp_masks` for BSGS baby-step products.
    bsgs_masks: Vec<VecZnxDft<BE::OwnedBuf, BE>>,
    /// Type-erased giant-step plans used after each baby-step group sum. Boxed
    /// as `Any + Send + Sync` so a `PackingPrecomputations` can be both produced
    /// on a worker thread (Send) and shared read-only across worker threads
    /// (Sync) during online packing. The concrete HAL plan types are plain
    /// `Send + Sync` data.
    bsgs_giant_plans: Vec<Box<dyn Any + Send + Sync>>,
    /// Number of baby steps per giant-step group.
    bsgs_baby_size: usize,
    /// Base used when normalizing big products back into coefficient buffers.
    base2k: usize,
    /// Total stored mask steps (includes the final `key_h` step iff `!partial`).
    steps: usize,
    /// Partial packing: single `key_g` collapse half, no second half, no `key_h`.
    partial: bool,
    /// Galois generator stride `e = (d/2)/γ` (1 for full packing); see
    /// [`PackingPrecomputeInfos::stride`].
    stride: usize,
}

impl<BE: Backend> PackingPrecomputations<BE> {
    /// Allocation-free metadata matching this concrete precompute container.
    pub fn metadata(&self) -> PackingPrecomputeInfos {
        if self.partial {
            PackingPrecomputeInfos::new_partial(
                self.steps,
                self.size(),
                self.base2k,
                self.bsgs_baby_size,
                self.stride,
            )
        } else {
            PackingPrecomputeInfos::new(self.steps, self.size(), self.base2k, self.bsgs_baby_size)
        }
    }

    /// Whether this is a partial (single-half, no-`key_h`) packing layout.
    pub(crate) fn partial(&self) -> bool {
        self.partial
    }

    /// Galois generator stride `e = (d/2)/γ` (1 for full packing).
    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    /// Per-step fixed mask inputs for the online body-side `1 x 1` VMPs.
    ///
    /// Column `step` is the already-automorphed mask share used by the matching
    /// online collapse step. The query-dependent key body is intentionally not
    /// multiplied here.
    pub(crate) fn body_vmp_masks(&self) -> &VecZnx<BE::OwnedBuf> {
        &self.body_vmp_masks
    }

    /// Mutable access used only while recording the coefficient-domain mask schedule.
    fn body_vmp_masks_mut(&mut self) -> &mut VecZnx<BE::OwnedBuf> {
        &mut self.body_vmp_masks
    }

    /// Final precomputed GLWE mask to copy into the online collapse result.
    pub(crate) fn final_mask(&self) -> &VecZnx<BE::OwnedBuf> {
        &self.final_mask
    }

    /// Mutable access used only by the fixed-mask precompute finalization.
    fn final_mask_mut(&mut self) -> &mut VecZnx<BE::OwnedBuf> {
        &mut self.final_mask
    }

    /// DFT mask column for one collapse step.
    ///
    /// Used by the online BSGS loop in `bsgs_pack` for `key_g` products and
    /// for the final `key_h` product.
    pub(crate) fn bsgs_col(&self, step: usize) -> &VecZnxDft<BE::OwnedBuf, BE> {
        &self.bsgs_masks[step]
    }

    /// Giant-step automorphism plan indexed by derived group index.
    ///
    /// Used by `bsgs_pack` after summing the baby-step products of a group.
    pub(crate) fn bsgs_giant_plan<Plan: 'static>(&self, group_idx: usize) -> &Plan {
        self.bsgs_giant_plans[group_idx]
            .downcast_ref()
            .expect("packing precomputations built for a different backend plan type")
    }

    /// Number of baby keys expected by the online `key_g` BSGS pass.
    pub(crate) fn bsgs_baby_size(&self) -> usize {
        self.bsgs_baby_size
    }

    /// Number of `key_g` collapse steps. For full packing the remaining stored
    /// step is the final `key_h` step (`steps - 1`); for partial packing all
    /// stored steps are `key_g` steps (`steps`).
    pub(crate) fn bsgs_kg_steps(&self) -> usize {
        if self.partial {
            self.steps
        } else {
            self.steps - 1
        }
    }

    /// Number of `key_g` steps in each half of the split collapse. Partial
    /// packing has a single half holding all `key_g` steps.
    ///
    /// Used to derive group starts without storing a per-group schedule.
    pub(crate) fn bsgs_half_steps(&self) -> usize {
        if self.partial {
            self.bsgs_kg_steps()
        } else {
            self.bsgs_kg_steps() / 2
        }
    }

    /// Number of giant-step groups in one half.
    ///
    /// At most the last group in a half is partial, so callers can derive its
    /// length from the group index instead of storing group metadata.
    pub(crate) fn bsgs_groups_per_half(&self) -> usize {
        self.bsgs_half_steps().div_ceil(self.bsgs_baby_size)
    }

    /// Total number of giant-step groups: two halves for full packing, a single
    /// half for partial packing.
    pub(crate) fn bsgs_group_count(&self) -> usize {
        if self.partial {
            self.bsgs_groups_per_half()
        } else {
            2 * self.bsgs_groups_per_half()
        }
    }

    /// First collapse step covered by `group_idx`.
    ///
    /// Used by both the DFT cache builder and online packing loop to translate
    /// a BSGS group index back to the original sequential collapse order.
    pub(crate) fn bsgs_group_start_step(&self, group_idx: usize) -> usize {
        let groups_per_half = self.bsgs_groups_per_half();
        let half_idx = group_idx / groups_per_half;
        let giant_idx = group_idx % groups_per_half;
        half_idx * self.bsgs_half_steps() + giant_idx * self.bsgs_baby_size
    }

    /// Number of live baby steps in `group_idx`.
    ///
    /// Full groups have `bsgs_baby_size` steps; the final group of each half may
    /// be shorter because the sequential collapse has `n / 2 - 1` steps per
    /// half.
    pub(crate) fn bsgs_group_len(&self, group_idx: usize) -> usize {
        let start_in_half = self.bsgs_group_start_step(group_idx) % self.bsgs_half_steps();
        (self.bsgs_half_steps() - start_in_half).min(self.bsgs_baby_size)
    }

    /// Normalization base shared by fixed-mask precompute and online output.
    pub(crate) fn base2k(&self) -> usize {
        self.base2k
    }

    /// Number of coefficients/components in each stored mask column.
    pub(crate) fn size(&self) -> usize {
        self.body_vmp_masks.size()
    }

    /// Total number of stored mask steps.
    ///
    /// Used by the BSGS DFT builder to allocate one DFT column per recorded
    /// collapse step.
    pub(crate) fn steps(&self) -> usize {
        self.steps
    }
}

/// Allocates fixed mask-side precomputation storage.
///
/// `steps` is the number of stored mask steps: all `key_g` collapse steps plus
/// the final `key_h` step. `baby_size` fixes the BSGS grouping used later by
/// [`Packing::pack_precompute`](crate::packing::Packing::pack_precompute).
pub(crate) fn pack_precompute_alloc_default<BE: Backend>(
    module: &Module<BE>,
    steps: usize,
    size: usize,
    base2k: usize,
    baby_size: usize,
) -> PackingPrecomputations<BE> {
    PackingPrecomputations {
        body_vmp_masks: module.vec_znx_alloc(steps, size),
        final_mask: module.vec_znx_alloc(1, size),
        bsgs_masks: Vec::new(),
        bsgs_giant_plans: Vec::new(),
        bsgs_baby_size: baby_size,
        base2k,
        steps,
        partial: false,
        stride: 1,
    }
}

/// Allocates fixed mask-side storage for **partial** packing: `steps = γ-1`
/// `key_g` collapse steps, a single half, and no final `key_h` step. `stride` is
/// the Galois generator stride `e = (d/2)/γ`.
pub(crate) fn pack_precompute_alloc_partial<BE: Backend>(
    module: &Module<BE>,
    steps: usize,
    size: usize,
    base2k: usize,
    baby_size: usize,
    stride: usize,
) -> PackingPrecomputations<BE> {
    PackingPrecomputations {
        body_vmp_masks: module.vec_znx_alloc(steps, size),
        final_mask: module.vec_znx_alloc(1, size),
        bsgs_masks: Vec::new(),
        bsgs_giant_plans: Vec::new(),
        bsgs_baby_size: baby_size,
        base2k,
        steps,
        partial: true,
        stride,
    }
}

/// Scratch estimate for the coefficient-domain fixed-mask precompute when the
/// work buffers use a different base/size than the input aggregate layout.
pub(crate) fn precompute_sequential_keyswitch_collapse_aggregate_mask_tmp_bytes_for_size<
    BE,
    KGMask,
    KHMask,
>(
    module: &Module<BE>,
    aggregate_size: usize,
    vmp_input_size: usize,
    key_g_mask: &KGMask,
    key_h_mask: &KHMask,
) -> usize
where
    BE: Backend,
    Module<BE>: VecZnxBigBytesOf
        + VecZnxBigNormalizeTmpBytes
        + VecZnxDftBytesOf
        + VecZnxIdftApplyTmpBytes
        + VmpApplyDftToDftTmpBytes,
    KGMask: GGLWEInfos,
    KHMask: GGLWEInfos,
{
    let n = module.n();
    let half = n >> 1;
    let size = aggregate_size;
    let align = |len: usize| len.next_multiple_of(BE::SCRATCH_ALIGN);
    let vec_scratch = align(VecZnx::<Vec<u8>>::bytes_of(n, half, size))
        + 4 * align(VecZnx::<Vec<u8>>::bytes_of(n, 1, size))
        + align(VecZnx::<Vec<u8>>::bytes_of(n, 1, vmp_input_size));
    let key_g_scratch = fixed_mask_1x1_vmp_body_addend_tmp_bytes_for_size::<BE, _>(
        module,
        vmp_input_size,
        key_g_mask,
    );
    let key_h_scratch = fixed_mask_1x1_vmp_body_addend_tmp_bytes_for_size::<BE, _>(
        module,
        vmp_input_size,
        key_h_mask,
    );

    vec_scratch + key_g_scratch.max(key_h_scratch)
}

/// Scratch estimate for [`sequential_collapse_bsgs_dft_build`].
///
/// The BSGS DFT builder only needs one coefficient-domain baby-mask scratch
/// buffer; DFT columns and giant-step plans are allocated in
/// [`PackingPrecomputations`].
pub(crate) fn sequential_collapse_bsgs_dft_build_tmp_bytes<BE>(
    module: &Module<BE>,
    metadata: PackingPrecomputeInfos,
) -> usize
where
    BE: Backend,
{
    VecZnx::<Vec<u8>>::bytes_of(module.n(), 1, metadata.size()).next_multiple_of(BE::SCRATCH_ALIGN)
}

/// Precomputes the fixed-mask side of the sequential collapse, unifying full and
/// partial packing.
///
/// This is the first internal phase of
/// [`Packing::pack_precompute`](crate::packing::Packing::pack_precompute). The
/// second phase turns the recorded schedule into DFT-hot BSGS columns and
/// automorphism plans. The loop order and automorphisms mirror
/// `sequential_keyswitch_collapse_aggregate_mask_split`: the only difference is
/// that the input mask is fixed, so the mask-key `1 x 1` products are applied
/// to an offline mask state. Before each offline mask product, the routine also
/// records the fixed mask input required by the corresponding online body-side
/// product. The key body is query-dependent and is not part of this precompute.
///
/// The collapse is a sequence of `key_g` half-collapses (`precompute_collapse_half`),
/// one per half, mirroring Algorithm 1's `COLLAPSE = COLLAPSEHALF + … + merge`:
///
/// * **Partial** packing runs a single half over `γ` columns (generator stride
///   `e = (d/2)/γ`), with `key_h_mask = None`. The surviving column 0 is the
///   result mask.
/// * **Full** packing runs two halves over `d/2` columns each (stride 1): the
///   first with `ρ = I` (its result is saved as the `first_share`), the second
///   with `ρ = τ_h`, then a final `key_h` merge adds the `key_h` product of the
///   second half's result to the `first_share`.
///
/// `key_g_mask` and `key_h_mask` share the same prepared-key type; partial
/// packing passes `None` for the latter.
#[allow(clippy::too_many_arguments)]
pub(crate) fn precompute_collapse_mask<BE, A, K>(
    module: &Module<BE>,
    precompute: &mut PackingPrecomputations<BE>,
    aggregate_mask: &A,
    vmp_input_base2k: usize,
    vmp_input_size: usize,
    key_g_mask: &K,
    key_h_mask: Option<&K>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: GaloisElement
        + VecZnxAddAssignBackend<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxNormalize<BE>
        + VmpApplyDftToDft<BE>,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
    K: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
{
    let n = module.n();
    // Each half collapses `per_half_cols = half_steps + 1` aggregate columns; full
    // packing has two halves of `d/2`, partial a single half of `γ`.
    let per_half_cols = precompute.bsgs_half_steps() + 1;
    let num_halves = if precompute.partial() { 1 } else { 2 };
    assert!(per_half_cols <= n >> 1, "collapse half exceeds d/2 columns");
    assert!(
        aggregate_mask.cols() >= num_halves * per_half_cols,
        "aggregate has fewer columns than the collapse schedule requires"
    );

    let scratch_local = scratch.borrow();
    let (mut work, scratch_local) =
        scratch_local.take_vec_znx_scratch(n, per_half_cols, aggregate_mask.size());
    let (mut first_share, scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, aggregate_mask.size());
    let (mut term_mask, scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, aggregate_mask.size());
    let (mut mask_addend, scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, aggregate_mask.size());
    let (mut mask_addend_auto, scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, aggregate_mask.size());
    let (mut term_mask_vmp, mut scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, vmp_input_size);

    let aggregate_ref = aggregate_mask.to_backend_ref();
    let mut step = 0usize;

    for half_idx in 0..num_halves {
        let use_tau_h = half_idx == 1;
        copy_aggregate_half(module, &mut work, &aggregate_ref, half_idx * per_half_cols);
        precompute_collapse_half(
            module,
            precompute,
            &mut work,
            use_tau_h,
            key_g_mask,
            vmp_input_base2k,
            &mut step,
            &mut term_mask,
            &mut term_mask_vmp,
            &mut mask_addend,
            &mut mask_addend_auto,
            &mut scratch_local,
        );

        // Save the first (`ρ = I`) half's result; the `key_h` merge needs it.
        if half_idx == 0 && num_halves == 2 {
            let work_ref = work.to_backend_ref();
            let mut first_mut = first_share.to_backend_mut();
            module.vec_znx_copy_backend(&mut first_mut, 0, &work_ref, 0);
        }
    }

    // `work` column 0 now holds the last half's collapsed mask.
    match key_h_mask {
        // Partial: the surviving column 0 is the result, no merge.
        None => {
            let work_ref = work.to_backend_ref();
            let mut final_mut = <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(
                precompute.final_mask_mut(),
            );
            module.vec_znx_copy_backend(&mut final_mut, 0, &work_ref, 0);
        }
        // Full: merge the τ_h half via key_h, then add the saved first_share.
        Some(key_h_mask) => {
            {
                let work_ref = work.to_backend_ref();
                let mut term_mut = term_mask.to_backend_mut();
                module.vec_znx_copy_backend(&mut term_mut, 0, &work_ref, 0);
            }
            store_body_vmp_mask(module, precompute, step, &term_mask);
            normalize_term_for_vmp(
                module,
                &mut term_mask_vmp,
                vmp_input_base2k,
                &term_mask,
                precompute.base2k(),
                &mut scratch_local.borrow(),
            );
            fixed_mask_1x1_vmp_body_addend(
                module,
                &mut mask_addend,
                precompute.base2k(),
                &term_mask_vmp,
                0,
                key_h_mask,
                &mut scratch_local.borrow(),
            );
            let mask_ref = mask_addend.to_backend_ref();
            let first_ref = first_share.to_backend_ref();
            let mut final_mut = <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(
                precompute.final_mask_mut(),
            );
            module.vec_znx_copy_backend(&mut final_mut, 0, &mask_ref, 0);
            module.vec_znx_add_assign_backend(&mut final_mut, 0, &first_ref, 0);
        }
    }
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
    // The fixed-mask collapse processes the aggregate as two half-sized
    // schedules. This helper keeps the split explicit and is used once for the
    // `tau_g` half and once for the `tau_h` half in the public precompute.
    let cols = dst.cols();
    let mut dst_mut = dst.to_backend_mut();
    for col in 0..cols {
        module.vec_znx_copy_backend(&mut dst_mut, col, src, offset + col);
    }
}

/// Records the fixed mask input for one online body-side VMP.
///
/// `precompute_collapse_half` calls this immediately before applying the
/// matching mask-side product offline. The online pack later reads the same
/// column through `PackingPrecomputations::bsgs_col` after DFT conversion.
fn store_body_vmp_mask<BE, A>(
    module: &Module<BE>,
    precompute: &mut PackingPrecomputations<BE>,
    step: usize,
    mask: &A,
) where
    BE: Backend,
    Module<BE>: VecZnxCopyBackend<BE>,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
{
    let mask_ref = mask.to_backend_ref();
    let mut masks_mut = <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(
        precompute.body_vmp_masks_mut(),
    );
    module.vec_znx_copy_backend(&mut masks_mut, step, &mask_ref, 0);
}

/// Runs the fixed-mask schedule for one half of the sequential collapse.
///
/// The public precompute calls this twice: first for the `tau_g` half, then for
/// the `tau_h` half. Each iteration records the mask input needed by the online
/// body product, computes the corresponding mask-key product offline, applies
/// the inverse automorphism, and folds it into the current mask state.
#[allow(clippy::too_many_arguments)]
fn precompute_collapse_half<BE, Mask, TermMask, TermMaskVmp, MaskAddend, MaskAddendAuto, KMask>(
    module: &Module<BE>,
    precompute: &mut PackingPrecomputations<BE>,
    mask: &mut Mask,
    use_tau_h: bool,
    key_mask: &KMask,
    vmp_input_base2k: usize,
    step: &mut usize,
    term_mask: &mut TermMask,
    term_mask_vmp: &mut TermMaskVmp,
    mask_addend: &mut MaskAddend,
    mask_addend_auto: &mut MaskAddendAuto,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: GaloisElement
        + VecZnxAddAssignBackend<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxNormalize<BE>
        + VmpApplyDftToDft<BE>,
    Mask: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    TermMask: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    TermMaskVmp: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    MaskAddend: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    MaskAddendAuto: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    KMask: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
{
    let stride = precompute.stride();
    for target_col in (0..mask.cols() - 1).rev() {
        let source_col = target_col + 1;
        let tau_g_j =
            module.galois_element_inv(module.galois_element((stride * target_col) as i64));
        let secret_view = if use_tau_h { -tau_g_j } else { tau_g_j };
        let alpha = module.galois_element_inv(secret_view);
        let alpha_inv = secret_view;

        // The online body product must see the same automorphed term. Store it
        // before the offline mask-only product mutates the local mask state.
        module.vec_znx_automorphism_backend(
            alpha_inv,
            &mut term_mask.to_backend_mut(),
            0,
            &mask.to_backend_ref(),
            source_col,
        );
        store_body_vmp_mask(module, precompute, *step, term_mask);
        normalize_term_for_vmp(
            module,
            term_mask_vmp,
            vmp_input_base2k,
            term_mask,
            precompute.base2k(),
            scratch,
        );

        fixed_mask_1x1_vmp_body_addend(
            module,
            mask_addend,
            precompute.base2k(),
            term_mask_vmp,
            0,
            key_mask,
            &mut scratch.borrow(),
        );

        // The mask addend is produced in the secret-key view; move it back to
        // the aggregate view before accumulating into this half.
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

fn normalize_term_for_vmp<BE, Dst, Src>(
    module: &Module<BE>,
    dst: &mut Dst,
    dst_base2k: usize,
    src: &Src,
    src_base2k: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VecZnxNormalize<BE>,
    Dst: VecZnxToBackendMut<BE> + ZnxInfos,
    Src: VecZnxToBackendRef<BE> + ZnxInfos,
{
    module.vec_znx_normalize(
        &mut dst.to_backend_mut(),
        dst_base2k,
        0,
        0,
        &src.to_backend_ref(),
        src_base2k,
        0,
        scratch,
    );
}

fn fixed_mask_1x1_vmp_body_addend_tmp_bytes_for_size<BE, K>(
    module: &(
         impl VecZnxBigBytesOf
         + VecZnxBigNormalizeTmpBytes
         + VecZnxDftBytesOf
         + VecZnxIdftApplyTmpBytes
         + VmpApplyDftToDftTmpBytes
     ),
    mask_size: usize,
    key: &K,
) -> usize
where
    BE: Backend,
    K: GGLWEInfos,
{
    let key_size = key.size();

    let align = |len: usize| len.next_multiple_of(BE::SCRATCH_ALIGN);
    let lvl_0 = align(module.bytes_of_vec_znx_dft(1, mask_size));
    let lvl_1 = align(module.bytes_of_vec_znx_dft(1, key_size));
    let lvl_2 = align(module.bytes_of_vec_znx_big(1, key_size));
    let lvl_ops = module
        .vmp_apply_dft_to_dft_tmp_bytes(
            key_size,
            mask_size,
            key.dnum().as_usize(),
            key.rank_in().as_usize(),
            key.rank_out().as_usize() + 1,
            key.size(),
        )
        .max(module.vec_znx_idft_apply_tmp_bytes())
        .max(module.vec_znx_big_normalize_tmp_bytes());

    lvl_0 + lvl_1 + lvl_2 + lvl_ops
}

/// Re-encodes an aggregate mask into the widened arithmetic base used by the
/// offline fixed-mask precompute.
pub(crate) fn normalize_precompute_aggregate<BE, A>(
    module: &Module<BE>,
    dst: &mut VecZnx<BE::OwnedBuf>,
    dst_base2k: usize,
    src: &A,
    src_base2k: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VecZnxNormalize<BE>,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
{
    assert_eq!(
        dst.cols(),
        src.cols(),
        "precompute aggregate column count mismatch"
    );

    let cols = dst.cols();
    let src_ref = src.to_backend_ref();
    let mut dst_mut = <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(dst);
    for col in 0..cols {
        module.vec_znx_normalize(
            &mut dst_mut,
            dst_base2k,
            0,
            col,
            &src_ref,
            src_base2k,
            col,
            scratch,
        );
    }
}

/// Converts widened coefficient-domain precompute state back to the public
/// packing base before the DFT-hot cache is built for the online pass.
pub(crate) fn normalize_precompute_coefficients<BE>(
    module: &Module<BE>,
    dst: &mut PackingPrecomputations<BE>,
    src: &PackingPrecomputations<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VecZnxNormalize<BE>,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendRef<BE>,
{
    assert_eq!(dst.steps, src.steps, "packing precompute step mismatch");
    assert_eq!(
        dst.bsgs_baby_size, src.bsgs_baby_size,
        "packing precompute baby-size mismatch"
    );

    {
        let src_ref = src.body_vmp_masks.to_backend_ref();
        let mut dst_mut = <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(
            &mut dst.body_vmp_masks,
        );
        for step in 0..dst.steps {
            module.vec_znx_normalize(
                &mut dst_mut,
                dst.base2k,
                0,
                step,
                &src_ref,
                src.base2k,
                step,
                scratch,
            );
        }
    }

    {
        let src_ref = src.final_mask.to_backend_ref();
        let mut dst_mut =
            <VecZnx<BE::OwnedBuf> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut dst.final_mask);
        module.vec_znx_normalize(
            &mut dst_mut,
            dst.base2k,
            0,
            0,
            &src_ref,
            src.base2k,
            0,
            scratch,
        );
    }
}

/// Computes the coefficient-domain body addend of a fixed-mask `1 x 1` VMP.
///
/// This is a local building block for fixed-mask precomputation. The public
/// precompute uses it to advance the mask-only state while saving the matching
/// body-side mask columns for the online pass. It stays private because callers
/// should use the higher-level mask precompute APIs instead of assembling this
/// partial key-switch product themselves.
#[allow(clippy::too_many_arguments)]
fn fixed_mask_1x1_vmp_body_addend<BE, M, R, A, K>(
    module: &M,
    res: &mut R,
    res_base2k: usize,
    mask: &A,
    mask_col: usize,
    key: &K,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    M: ModuleN
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VmpApplyDftToDft<BE>,
    R: VecZnxToBackendMut<BE> + ZnxInfos,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
    K: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
{
    let key_size = key.size();
    let key_base2k = key.base2k().as_usize();

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

    module.vmp_apply_dft_to_dft(
        &mut product_dft.to_backend_mut(),
        &mask_dft.to_backend_ref(),
        &key.vmp_pmat_backend_ref(),
        0,
        &mut scratch_2.borrow(),
    );

    let (mut product_big, mut scratch_3) = scratch_2.take_vec_znx_big_scratch(module, 1, key_size);
    module.vec_znx_idft_apply(
        &mut product_big.to_backend_mut(),
        0,
        &product_dft.to_backend_ref(),
        0,
        &mut scratch_3.borrow(),
    );

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

/// Giant-step automorphism for a derived BSGS group.
///
/// Both halves share the same arithmetic, but the second half is viewed through
/// `tau_h` and therefore flips the secret-view sign. The DFT builder calls this
/// once per group to create the plan later consumed by `bsgs_pack`.
fn sequential_collapse_bsgs_giant_alpha<BE>(
    module: &Module<BE>,
    half_steps: usize,
    baby_size: usize,
    group_idx: usize,
    stride: usize,
) -> i64
where
    BE: Backend,
    Module<BE>: GaloisElement,
{
    let groups_per_half = half_steps.div_ceil(baby_size);
    let use_tau_h = group_idx >= groups_per_half;
    let giant_idx = group_idx % groups_per_half;
    let target_col = half_steps - 1 - giant_idx * baby_size;
    let tau_g_j = module.galois_element_inv(module.galois_element((stride * target_col) as i64));
    let secret_view = if use_tau_h { -tau_g_j } else { tau_g_j };
    module.galois_element_inv(secret_view)
}

/// Builds the BSGS mask cache and giant plans for the DFT-hot collapse.
///
/// This is DB-side/pre-query state. Baby keys are query-key state and are
/// passed separately to the online collapse routine.
///
/// This is the implementation behind [`Packing::pack_precompute`](crate::packing::Packing::pack_precompute).
/// It consumes the coefficient-domain columns recorded by
/// [`precompute_collapse_mask`], applies the
/// baby-step automorphisms, converts them to DFT, and prepares the giant-step
/// plans used in `bsgs_pack`. The plans are stored type-erased so the public
/// [`PackingPrecomputations`] type does not require HAL automorphism-plan
/// bounds at the front-end API.
pub(crate) fn sequential_collapse_bsgs_dft_build<BE>(
    module: &Module<BE>,
    precompute: &mut PackingPrecomputations<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: GaloisElement
        + VecZnxAutomorphismBackend<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftAutomorphismPlan<BE>,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendRef<BE>,
    <Module<BE> as VecZnxDftAutomorphismPlan<BE>>::Plan: 'static + Send + Sync,
{
    let n = module.n();
    // Full packing: `2 * (n/2 - 1)` key_g steps + a final key_h step. Partial
    // packing: `γ-1` key_g steps and no key_h step.
    let kg_steps = precompute.bsgs_kg_steps();
    let baby_size = precompute.bsgs_baby_size();

    let stride = precompute.stride();
    let group_count = precompute.bsgs_group_count();
    let mut giant_plans: Vec<Box<dyn Any + Send + Sync>> = Vec::with_capacity(group_count);
    for group_idx in 0..group_count {
        let giant_alpha = sequential_collapse_bsgs_giant_alpha(
            module,
            precompute.bsgs_half_steps(),
            baby_size,
            group_idx,
            stride,
        );
        giant_plans.push(Box::new(module.vec_znx_dft_automorphism_plan(giant_alpha)));
    }

    let mut masks: Vec<VecZnxDft<BE::OwnedBuf, BE>> = (0..precompute.steps())
        .map(|_| module.vec_znx_dft_alloc(1, precompute.size()))
        .collect();
    let scratch_local = scratch.borrow();
    let (mut baby_mask, _scratch_local) =
        scratch_local.take_vec_znx_scratch(n, 1, precompute.size());

    {
        let src_ref = precompute.body_vmp_masks().to_backend_ref();
        for group_idx in 0..group_count {
            let start_step = precompute.bsgs_group_start_step(group_idx);
            for baby_idx in 0..precompute.bsgs_group_len(group_idx) {
                let step = start_step + baby_idx;
                let baby_alpha =
                    module.galois_element_inv(module.galois_element((stride * baby_idx) as i64));
                // Convert each sequential step into the baby-step view expected
                // by the corresponding client-key-side baby key.
                module.vec_znx_automorphism_backend(
                    baby_alpha,
                    &mut baby_mask.to_backend_mut(),
                    0,
                    &src_ref,
                    step,
                );
                module.vec_znx_dft_apply(
                    1,
                    0,
                    &mut masks[step].to_backend_mut(),
                    0,
                    &baby_mask.to_backend_ref(),
                    0,
                );
            }
        }

        // The final `key_h` product is not part of the BSGS `key_g` grouping,
        // but it uses the same DFT-hot mask storage. Partial packing has no
        // `key_h` step, so its mask storage ends at the last `key_g` column.
        if !precompute.partial() {
            module.vec_znx_dft_apply(
                1,
                0,
                &mut masks[kg_steps].to_backend_mut(),
                0,
                &src_ref,
                kg_steps,
            );
        }
    }

    precompute.bsgs_masks = masks;
    precompute.bsgs_giant_plans = giant_plans;
}
