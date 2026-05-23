//! Online BSGS DFT-hot packing.
//!
//! This file contains the hot path behind [`Packing::pack`](crate::packing::Packing::pack).
//! It consumes fixed mask-side DFT columns from
//! [`PackingPrecomputations`] and client-key-side prepared bodies through
//! [`PackingKeyPrecomputationsHelper`].

use poulpy_core::layouts::{GGLWEInfos, GGLWEPreparedVmpPMatRef, GLWEInfos, GLWEToBackendMut};
use poulpy_hal::{
    api::{
        ScratchArenaTakeBasic, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxCopyBackend,
        VecZnxDftAddAssign, VecZnxDftApply, VecZnxDftAutomorphism, VecZnxDftAutomorphismPlan,
        VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftApply, VmpApplyDftToDft,
    },
    layouts::{
        Backend, Module, ScratchArena, VecZnx, VecZnxBigToBackendMut, VecZnxBigToBackendRef,
        VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxToBackendRef, ZnxInfos,
    },
};

use crate::packing::{
    PackingKeyPrecomputationsHelper, collapse_precompute::PackingPrecomputations,
};

/// Online BSGS DFT-hot packing implementation.
///
/// `chunk_size` batches giant-step groups so the loop can reuse a loaded baby
/// key across several groups before summing and applying each group's giant
/// automorphism. Increasing it improves reuse but grows scratch roughly as
/// `chunk_size * baby_size * key_size * n * 8`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pack_default<BE, B, R, H, K>(
    module: &Module<BE>,
    res: &mut R,
    body: &B,
    precomputations: &PackingPrecomputations<BE>,
    key_precomputations: &H,
    chunk_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftAutomorphism<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VmpApplyDftToDft<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    B: VecZnxToBackendRef<BE> + ZnxInfos,
    H: PackingKeyPrecomputationsHelper<BE, K>,
    K: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
    <Module<BE> as VecZnxDftAutomorphismPlan<BE>>::Plan: 'static,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBasic<'a, BE>,
{
    let kg_steps = precomputations.bsgs_kg_steps();
    let baby_size = precomputations.bsgs_baby_size();
    assert!(chunk_size >= 1, "chunk_size must be >= 1");
    let key_h_body = key_precomputations.key_h();
    let acc_size = key_precomputations.key_size();
    let total_baby_buffers = chunk_size * baby_size;

    // All buffers come from the scratch arena -- no heap allocs in the hot
    // path. The slice helper allocates one Vec for the view handles; the
    // f64 payload itself is from the arena.
    let scratch_local = scratch.borrow();
    let (mut baby_products, scratch_local) =
        scratch_local.take_vec_znx_dft_slice_scratch(module, total_baby_buffers, 1, acc_size);
    let (mut body_acc_dft, scratch_local) =
        scratch_local.take_vec_znx_dft_scratch(module, 1, acc_size);
    let (mut group_acc_dft, scratch_local) =
        scratch_local.take_vec_znx_dft_scratch(module, 1, acc_size);
    let (mut group_acc_auto_dft, scratch_local) =
        scratch_local.take_vec_znx_dft_scratch(module, 1, acc_size);
    let (mut product_h_dft, scratch_local) =
        scratch_local.take_vec_znx_dft_scratch(module, 1, acc_size);
    let (mut body_big, mut scratch_local) =
        scratch_local.take_vec_znx_big_scratch(module, 1, acc_size);

    module.vec_znx_dft_apply(
        1,
        0,
        &mut body_acc_dft.to_backend_mut(),
        0,
        &body.to_backend_ref(),
        0,
    );

    // Groups are derived from the fixed schedule instead of stored as structs:
    // only the last group of each half can be partial, and its length is
    // recovered from the group index.
    let group_count = precomputations.bsgs_group_count();
    let num_chunks = group_count.div_ceil(chunk_size);

    for chunk_idx in 0..num_chunks {
        let chunk_start = chunk_idx * chunk_size;
        let chunk_end = ((chunk_idx + 1) * chunk_size).min(group_count);
        let chunk_len = chunk_end - chunk_start;

        // Phase 1: VMPs with outer loop on baby_idx, inner on chunk groups.
        // Each baby key is loaded once and reused for every live group in the
        // chunk. Tail groups skip baby indices beyond their derived length.
        for baby_idx in 0..baby_size {
            for group_idx in chunk_start..chunk_end {
                let group_len = precomputations.bsgs_group_len(group_idx);
                if baby_idx >= group_len {
                    continue;
                }
                let c = group_idx - chunk_start;
                let step = precomputations.bsgs_group_start_step(group_idx) + baby_idx;
                let mask_dft_step = precomputations.bsgs_col(step).to_backend_ref();
                module.vmp_apply_dft_to_dft(
                    &mut baby_products[c * baby_size + baby_idx].to_backend_mut(),
                    &mask_dft_step,
                    &key_precomputations
                        .baby_key_g(baby_idx)
                        .vmp_pmat_backend_ref(),
                    0,
                    &mut scratch_local.borrow(),
                );
            }
        }

        // Phase 2 + giant auto + add-to-body, per group in the chunk.
        // Reads `baby_products[c * baby_size + i]` which was written several
        // VMPs ago. The sum is in the baby-step view and must be moved back by
        // the precomputed giant automorphism before being added to the body.
        for group_idx in chunk_start..chunk_end {
            let c = group_idx - chunk_start;
            module.vec_znx_dft_zero(&mut group_acc_dft.to_backend_mut(), 0);
            for baby_idx in 0..precomputations.bsgs_group_len(group_idx) {
                module.vec_znx_dft_add_assign(
                    &mut group_acc_dft.to_backend_mut(),
                    0,
                    &baby_products[c * baby_size + baby_idx].to_backend_ref(),
                    0,
                );
            }
            let giant_plan = precomputations
                .bsgs_giant_plan::<<Module<BE> as VecZnxDftAutomorphismPlan<BE>>::Plan>(group_idx);
            module.vec_znx_dft_automorphism_with_plan(
                giant_plan,
                &mut group_acc_auto_dft.to_backend_mut(),
                0,
                &group_acc_dft.to_backend_ref(),
                0,
            );
            module.vec_znx_dft_add_assign(
                &mut body_acc_dft.to_backend_mut(),
                0,
                &group_acc_auto_dft.to_backend_ref(),
                0,
            );
        }
        let _ = chunk_len; // silence unused if no debug build
    }

    // The final `key_h` body product is outside the `key_g` BSGS groups. It
    // uses the final DFT mask column stored at `kg_steps`.
    {
        let mask_dft_final = precomputations.bsgs_col(kg_steps).to_backend_ref();
        module.vmp_apply_dft_to_dft(
            &mut product_h_dft.to_backend_mut(),
            &mask_dft_final,
            &key_h_body.vmp_pmat_backend_ref(),
            0,
            &mut scratch_local.borrow(),
        );
    }
    module.vec_znx_dft_add_assign(
        &mut body_acc_dft.to_backend_mut(),
        0,
        &product_h_dft.to_backend_ref(),
        0,
    );

    module.vec_znx_idft_apply(
        &mut body_big.to_backend_mut(),
        0,
        &body_acc_dft.to_backend_ref(),
        0,
        &mut scratch_local.borrow(),
    );
    let base2k = precomputations.base2k();
    let mask_ref = <VecZnx<BE::OwnedBuf> as VecZnxToBackendRef<BE>>::to_backend_ref(
        precomputations.final_mask(),
    );
    {
        // The online body accumulator is normalized into result body column 0;
        // the fixed mask precompute already produced result mask column 1.
        let mut res_ref = res.to_backend_mut();
        module.vec_znx_big_normalize(
            res_ref.data_mut(),
            base2k,
            0,
            0,
            &body_big.to_backend_ref(),
            base2k,
            0,
            &mut scratch_local,
        );
        module.vec_znx_copy_backend(res_ref.data_mut(), 1, &mask_ref, 0);
    }
    drop(baby_products);
}
