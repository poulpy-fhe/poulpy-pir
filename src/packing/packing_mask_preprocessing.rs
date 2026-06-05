//! Algorithm body for packing-mask aggregation.
//!
//! The public API, default trait, OEP hook, and delegate live in the standard
//! packing module files. This file only contains the reusable default
//! implementation they call.

use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxAddAssignBackend, VecZnxAutomorphismBackend,
        VecZnxCopyBackend, VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxRotateAssignBackend,
        VecZnxRotateAssignTmpBytes, VecZnxRshAssignBackend, VecZnxRshTmpBytes,
        VecZnxTransposeBackend, VecZnxZeroBackend,
    },
    layouts::{
        Backend, GaloisElement, Module, ScratchArena, VecZnx, VecZnxBackendMut,
        VecZnxReborrowBackendRef, VecZnxToBackendMut, VecZnxToBackendRef, ZnxInfos,
    },
};

const MASK_PREPROCESSING_ARITHMETIC_BASE2K: usize = 60;

/// Default scratch estimate for packing-mask aggregation.
pub(crate) fn packing_mask_preprocessing_tmp_bytes_default<BE>(
    module: &Module<BE>,
    size: usize,
) -> usize
where
    BE: Backend,
    Module<BE>: VecZnxNormalizeTmpBytes + VecZnxRotateAssignTmpBytes + VecZnxRshTmpBytes,
{
    const ALIGN: usize = 64;
    let round = |x: usize| x.next_multiple_of(ALIGN);
    let n = module.n();
    let work = packing_mask_preprocessing_work_tmp_bytes(module, size);
    let temp_input = round(VecZnx::<Vec<u8>>::bytes_of(n, n, size));
    let temp_output = round(VecZnx::<Vec<u8>>::bytes_of(n, n, size));
    let internal = round(
        module
            .vec_znx_normalize_tmp_bytes()
            .max(module.vec_znx_rsh_tmp_bytes())
            .max(module.vec_znx_rotate_assign_tmp_bytes()),
    );
    temp_input + temp_output + work + internal
}

fn mask_preprocessing_arithmetic_base2k(base2k: usize) -> usize {
    MASK_PREPROCESSING_ARITHMETIC_BASE2K.max(base2k)
}

fn mask_preprocessing_arithmetic_size(size: usize, base2k: usize) -> usize {
    let arithmetic_base2k = mask_preprocessing_arithmetic_base2k(base2k);
    (size * base2k).div_ceil(arithmetic_base2k).max(1)
}

fn packing_mask_preprocessing_work_tmp_bytes<BE>(module: &Module<BE>, size: usize) -> usize
where
    BE: Backend,
    Module<BE>: VecZnxRotateAssignTmpBytes + VecZnxRshTmpBytes,
{
    const ALIGN: usize = 64;
    let round = |x: usize| x.next_multiple_of(ALIGN);
    let n = module.n();
    let log_n = n.trailing_zeros() as usize;
    let transposed = round(VecZnx::<Vec<u8>>::bytes_of(n, n, size));
    let one_col = round(VecZnx::<Vec<u8>>::bytes_of(n, 1, size));
    let tree = round(VecZnx::<Vec<u8>>::bytes_of(n, log_n, size));
    let internal = round(
        module
            .vec_znx_rsh_tmp_bytes()
            .max(module.vec_znx_rotate_assign_tmp_bytes()),
    );
    transposed + 2 * one_col + 2 * tree + internal
}

fn normalize_mask_preprocessing_input<BE, R, A>(
    module: &Module<BE>,
    dst: &mut R,
    dst_base2k: usize,
    src: &A,
    src_base2k: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VecZnxNormalize<BE>,
    R: VecZnxToBackendMut<BE> + ZnxInfos,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
{
    let cols = dst.cols();
    assert_eq!(
        cols,
        src.cols(),
        "mask preprocessing input column count mismatch"
    );

    let src_ref = src.to_backend_ref();
    let mut dst_mut = dst.to_backend_mut();
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

fn normalize_mask_preprocessing_output<BE, R, A>(
    module: &Module<BE>,
    dst: &mut R,
    dst_base2k: usize,
    src: &A,
    src_base2k: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VecZnxNormalize<BE>,
    R: VecZnxToBackendMut<BE> + ZnxInfos,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
{
    let cols = dst.cols();
    assert_eq!(
        cols,
        src.cols(),
        "mask preprocessing output column count mismatch"
    );

    let src_ref = src.to_backend_ref();
    let mut dst_mut = dst.to_backend_mut();
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

/// Default packing-mask aggregation implementation.
pub(crate) fn packing_mask_preprocessing_default<BE, R, A>(
    module: &Module<BE>,
    dst: &mut R,
    base2k: usize,
    a: &A,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VecZnxCopyBackend<BE>
        + VecZnxTransposeBackend<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxNormalize<BE>
        + VecZnxRotateAssignBackend<BE>
        + VecZnxRshAssignBackend<BE>
        + GaloisElement,
    R: VecZnxToBackendMut<BE> + ZnxInfos,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
{
    let n = module.n();
    assert!(
        n.is_power_of_two(),
        "InspiRING requires a power-of-two ring degree"
    );
    assert!(n >= 2, "InspiRING requires ring degree d >= 2");

    let size = dst.size();

    assert_eq!(
        dst.n(),
        n,
        "destination VecZnx degree must match module degree"
    );
    assert_eq!(dst.cols(), n, "destination VecZnx must have d columns");
    assert_eq!(
        dst.size(),
        a.size(),
        "destination size must match input A size"
    );
    assert_eq!(a.n(), n, "input A must have d rows");
    assert_eq!(a.cols(), n, "input A must have d columns");

    let arithmetic_base2k = mask_preprocessing_arithmetic_base2k(base2k);
    let arithmetic_size = mask_preprocessing_arithmetic_size(size, base2k);
    let scratch_local = scratch.borrow();
    let (mut arithmetic_input, scratch_next) =
        scratch_local.take_vec_znx_scratch(n, n, arithmetic_size);
    let (mut arithmetic_dst, mut scratch_next) =
        scratch_next.take_vec_znx_scratch(n, n, arithmetic_size);

    normalize_mask_preprocessing_input(
        module,
        &mut arithmetic_input,
        arithmetic_base2k,
        a,
        base2k,
        &mut scratch_next.borrow(),
    );

    packing_mask_preprocessing_work(
        module,
        &mut arithmetic_dst,
        arithmetic_base2k,
        &arithmetic_input,
        &mut scratch_next.borrow(),
    );

    normalize_mask_preprocessing_output(
        module,
        dst,
        base2k,
        &arithmetic_dst,
        arithmetic_base2k,
        &mut scratch_next.borrow(),
    );
}

fn packing_mask_preprocessing_work<BE, R, A>(
    module: &Module<BE>,
    dst: &mut R,
    base2k: usize,
    a: &A,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VecZnxCopyBackend<BE>
        + VecZnxTransposeBackend<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxRotateAssignBackend<BE>
        + VecZnxRshAssignBackend<BE>
        + GaloisElement,
    R: VecZnxToBackendMut<BE> + ZnxInfos,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
{
    let n = module.n();
    let n_half = n >> 1;
    let log_n = n.trailing_zeros() as usize;
    let size = dst.size();

    assert_eq!(dst.n(), n, "working destination degree mismatch");
    assert_eq!(dst.cols(), n, "working destination must have d columns");
    assert_eq!(
        dst.size(),
        a.size(),
        "working destination size must match working input size"
    );
    assert_eq!(a.n(), n, "working input degree mismatch");
    assert_eq!(a.cols(), n, "working input must have d columns");

    let h_list: Vec<i64> = (0..n_half)
        .map(|i| module.galois_element(i as i64))
        .collect();

    let arena = scratch.borrow();
    let (mut transposed, arena) = arena.take_vec_znx_scratch(n, n, size);
    let (mut shared, arena) = arena.take_vec_znx_scratch(n, 1, size);
    let (mut stage_a, arena) = arena.take_vec_znx_scratch(n, 1, size);
    let (mut tree_a, arena) = arena.take_vec_znx_scratch(n, log_n, size);
    let (mut tree_b, mut arena) = arena.take_vec_znx_scratch(n, log_n, size);

    {
        let a_ref = a.to_backend_ref();
        let mut t_mut = transposed.to_backend_mut();
        module.vec_znx_transpose_backend(&mut t_mut, &a_ref);
    }

    {
        let mut t_mut = transposed.to_backend_mut();
        for col in 0..n {
            module.vec_znx_rsh_assign_backend(base2k, log_n, &mut t_mut, col, &mut arena.borrow());
        }
    }

    let t_ref = transposed.to_backend_ref();
    let mut shared_mut = shared.to_backend_mut();
    let mut stage_a_mut = stage_a.to_backend_mut();
    let mut tree_a_mut = tree_a.to_backend_mut();
    let mut tree_b_mut = tree_b.to_backend_mut();
    let mut dst_mut = dst.to_backend_mut();

    let mut occupied_a = vec![false; log_n];
    let mut occupied_b = vec![false; log_n];

    for (j, &h) in h_list.iter().enumerate() {
        let col_a = j;
        let col_b = j + n_half;
        occupied_a.iter_mut().for_each(|x| *x = false);
        occupied_b.iter_mut().for_each(|x| *x = false);

        for k in 0..n {
            module.vec_znx_automorphism_backend(h, &mut shared_mut, 0, &t_ref, k);
            {
                let shared_ref = VecZnxReborrowBackendRef::<BE>::reborrow_backend_ref(&shared_mut);
                module.vec_znx_automorphism_backend(-1, &mut stage_a_mut, 0, &shared_ref, 0);
            }

            if k != 0 {
                module.vec_znx_rotate_assign_backend(
                    k as i64,
                    &mut shared_mut,
                    0,
                    &mut arena.borrow(),
                );
                module.vec_znx_rotate_assign_backend(
                    k as i64,
                    &mut stage_a_mut,
                    0,
                    &mut arena.borrow(),
                );
            }

            binary_tree_step(
                module,
                &mut stage_a_mut,
                &mut tree_a_mut,
                &mut occupied_a,
                &mut dst_mut,
                col_a,
            );
            binary_tree_step(
                module,
                &mut shared_mut,
                &mut tree_b_mut,
                &mut occupied_b,
                &mut dst_mut,
                col_b,
            );
        }

        debug_assert!(
            occupied_a.iter().all(|&x| !x) && occupied_b.iter().all(|&x| !x),
            "after d streamed leaves, both trees must be flushed into dst"
        );
    }
}

/// Default scratch estimate for **partial** packing-mask aggregation.
pub(crate) fn packing_mask_preprocessing_partial_tmp_bytes_default<BE>(
    module: &Module<BE>,
    gamma: usize,
    size: usize,
) -> usize
where
    BE: Backend,
    Module<BE>: VecZnxNormalizeTmpBytes + VecZnxRotateAssignTmpBytes + VecZnxRshTmpBytes,
{
    const ALIGN: usize = 64;
    let round = |x: usize| x.next_multiple_of(ALIGN);
    let n = module.n();
    let log_gamma = gamma.trailing_zeros() as usize;
    // Arithmetic-domain input (γ rows) and output (n × γ).
    let temp_input = round(VecZnx::<Vec<u8>>::bytes_of(gamma, n, size));
    let temp_output = round(VecZnx::<Vec<u8>>::bytes_of(n, gamma, size));
    // Work buffers: a single τ_g half, so one tree and one staging column.
    let transposed = round(VecZnx::<Vec<u8>>::bytes_of(n, gamma, size));
    let one_col = round(VecZnx::<Vec<u8>>::bytes_of(n, 1, size));
    let tree = round(VecZnx::<Vec<u8>>::bytes_of(n, log_gamma.max(1), size));
    let internal = round(
        module
            .vec_znx_normalize_tmp_bytes()
            .max(module.vec_znx_rsh_tmp_bytes())
            .max(module.vec_znx_rotate_assign_tmp_bytes()),
    );
    temp_input + temp_output + transposed + 2 * one_col + tree + internal
}

/// Partial packing-mask aggregation (Algorithm 2 `TransformPartial` + Stage 2).
///
/// Packs the masks of `γ ≤ n/2` LWEs into a `γ`-component τ_g IRCtx aggregate.
/// Unlike [`packing_mask_preprocessing_default`] this uses the `γ⁻¹`
/// normalization (`rsh` by `log₂γ`), streams only `γ` leaves, emits only the
/// `γ` τ_g components (`col_a`), and skips the τ_h half entirely. `γ` must be a
/// power of two.
///
/// `a` is the `γ`-row LWE-matrix mask (`a.n() == γ`, `a.cols() == n`); `dst` is
/// the `γ`-column aggregate (`dst.n() == n`, `dst.cols() == γ`).
pub(crate) fn packing_mask_preprocessing_partial_default<BE, R, A>(
    module: &Module<BE>,
    dst: &mut R,
    base2k: usize,
    gamma: usize,
    a: &A,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VecZnxCopyBackend<BE>
        + VecZnxTransposeBackend<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxNormalize<BE>
        + VecZnxRotateAssignBackend<BE>
        + VecZnxRshAssignBackend<BE>
        + VecZnxZeroBackend<BE>
        + GaloisElement,
    R: VecZnxToBackendMut<BE> + ZnxInfos,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
{
    let n = module.n();
    assert!(
        n.is_power_of_two(),
        "InspiRING requires a power-of-two ring degree"
    );
    assert!(
        gamma.is_power_of_two(),
        "partial packing requires a power-of-two γ"
    );
    assert!(gamma <= n >> 1, "partial packing requires γ ≤ n/2");

    let size = dst.size();
    assert_eq!(dst.n(), n, "destination degree must match module degree");
    assert_eq!(dst.cols(), gamma, "destination must have γ columns");
    assert_eq!(
        dst.size(),
        a.size(),
        "destination size must match input size"
    );
    assert_eq!(a.n(), gamma, "input A must have γ rows");
    assert_eq!(a.cols(), n, "input A must have n columns");

    let arithmetic_base2k = mask_preprocessing_arithmetic_base2k(base2k);
    let arithmetic_size = mask_preprocessing_arithmetic_size(size, base2k);
    let scratch_local = scratch.borrow();
    let (mut arithmetic_input, scratch_next) =
        scratch_local.take_vec_znx_scratch(gamma, n, arithmetic_size);
    let (mut arithmetic_dst, mut scratch_next) =
        scratch_next.take_vec_znx_scratch(n, gamma, arithmetic_size);

    normalize_mask_preprocessing_input(
        module,
        &mut arithmetic_input,
        arithmetic_base2k,
        a,
        base2k,
        &mut scratch_next.borrow(),
    );

    packing_mask_preprocessing_partial_work(
        module,
        &mut arithmetic_dst,
        arithmetic_base2k,
        gamma,
        &arithmetic_input,
        &mut scratch_next.borrow(),
    );

    normalize_mask_preprocessing_output(
        module,
        dst,
        base2k,
        &arithmetic_dst,
        arithmetic_base2k,
        &mut scratch_next.borrow(),
    );
}

fn packing_mask_preprocessing_partial_work<BE, R, A>(
    module: &Module<BE>,
    dst: &mut R,
    base2k: usize,
    gamma: usize,
    a: &A,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VecZnxCopyBackend<BE>
        + VecZnxTransposeBackend<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxRotateAssignBackend<BE>
        + VecZnxRshAssignBackend<BE>
        + VecZnxZeroBackend<BE>
        + GaloisElement,
    R: VecZnxToBackendMut<BE> + ZnxInfos,
    A: VecZnxToBackendRef<BE> + ZnxInfos,
{
    let n = module.n();
    let log_gamma = gamma.trailing_zeros() as usize;
    let size = dst.size();

    assert_eq!(dst.n(), n, "working destination degree mismatch");
    assert_eq!(dst.cols(), gamma, "working destination must have γ columns");
    assert_eq!(dst.size(), a.size(), "working destination size mismatch");
    assert_eq!(a.n(), gamma, "working input must have γ rows");
    assert_eq!(a.cols(), n, "working input must have n columns");

    // Only the τ_g half is needed: components â[j] = τ_{g_γ}^j(ã) for j = 0..γ,
    // using the order-γ generator g_γ = 5^e (stride e = (d/2)/γ). At γ = d/2 the
    // stride is 1 and g_γ = 5, matching the full pack's first half.
    let stride = (n >> 1) / gamma;
    let h_list: Vec<i64> = (0..gamma)
        .map(|i| module.galois_element((stride * i) as i64))
        .collect();

    let arena = scratch.borrow();
    let (mut transposed, arena) = arena.take_vec_znx_scratch(n, gamma, size);
    let (mut term, mut arena) = arena.take_vec_znx_scratch(n, 1, size);

    {
        let a_ref = a.to_backend_ref();
        let mut t_mut = transposed.to_backend_mut();
        module.vec_znx_transpose_backend(&mut t_mut, &a_ref);
    }

    {
        // γ⁻¹ normalization of ã (full packing uses log₂n for d⁻¹).
        let mut t_mut = transposed.to_backend_mut();
        for col in 0..gamma {
            module.vec_znx_rsh_assign_backend(
                base2k,
                log_gamma,
                &mut t_mut,
                col,
                &mut arena.borrow(),
            );
        }
    }

    let t_ref = transposed.to_backend_ref();
    let mut dst_mut = dst.to_backend_mut();
    let mut term_mut = term.to_backend_mut();
    for col in 0..gamma {
        module.vec_znx_zero_backend(&mut dst_mut, col);
    }

    for (j, &h) in h_list.iter().enumerate() {
        for k in 0..gamma {
            module.vec_znx_automorphism_backend(-h, &mut term_mut, 0, &t_ref, k);
            if k != 0 {
                module.vec_znx_rotate_assign_backend(
                    k as i64,
                    &mut term_mut,
                    0,
                    &mut arena.borrow(),
                );
            }
            let term_ref = VecZnxReborrowBackendRef::<BE>::reborrow_backend_ref(&term_mut);
            module.vec_znx_add_assign_backend(&mut dst_mut, j, &term_ref, 0);
        }
    }
}

#[inline]
fn binary_tree_step<BE, M>(
    module: &M,
    stage: &mut VecZnxBackendMut<'_, BE>,
    tree: &mut VecZnxBackendMut<'_, BE>,
    occupied: &mut [bool],
    dst: &mut VecZnxBackendMut<'_, BE>,
    dst_col: usize,
) where
    BE: Backend,
    M: VecZnxAddAssignBackend<BE> + VecZnxCopyBackend<BE>,
{
    let log_n = occupied.len();
    let mut level = 0;
    while level < log_n && occupied[level] {
        let tree_ref = VecZnxReborrowBackendRef::<BE>::reborrow_backend_ref(tree);
        module.vec_znx_add_assign_backend(stage, 0, &tree_ref, level);
        occupied[level] = false;
        level += 1;
    }
    let stage_ref = VecZnxReborrowBackendRef::<BE>::reborrow_backend_ref(stage);
    if level == log_n {
        module.vec_znx_copy_backend(dst, dst_col, &stage_ref, 0);
    } else {
        module.vec_znx_copy_backend(tree, level, &stage_ref, 0);
        occupied[level] = true;
    }
}
