//! InsPIRe² packing and decomposition primitives.
//!
//! These are the low-level pieces below the two-level PIR protocol:
//! - partial-pack batches of `gamma` LWEs into RLWEs;
//! - modulus-switch packed RLWEs to `qtilde`;
//! - decompose the switched ciphertext at `base2k = 16`.

use poulpy_core::{
    EncryptionLayout, GLWENormalize,
    layouts::{Base2K, Degree, GLWE, GLWELayout, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision},
};
use poulpy_hal::{
    api::{ScratchOwnedBorrow, VecZnxNormalize},
    layouts::{
        Backend, Module, ScratchArena, ScratchOwned, VecZnx, VecZnxToBackendMut,
        VecZnxToBackendRef, ZnxViewMut,
    },
};

use crate::{
    packing::{Packing, PackingKeys, PackingPrecomputations},
    parallel::{assign_panels, num_threads_online, scoped_workers_pooled},
};

/// Number of base2k=16 decomposition digits for a `qtilde`-modulus plaintext.
pub fn decompose_digits(qtilde_bits: usize) -> usize {
    qtilde_bits.div_ceil(16)
}

/// Modulus-switches a packed RLWE from its native modulus `q` down to
/// `qtilde = 2^{16 * tau}` and writes the result at `base2k = 16`.
pub fn modulus_switch_to_digits<BE>(
    module: &Module<BE>,
    dst: &mut GLWE<BE::OwnedBuf>,
    src: &GLWE<BE::OwnedBuf>,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend<OwnedBuf = Vec<u8>>,
    Module<BE>: GLWENormalize<BE>,
{
    module.glwe_normalize(dst, src, scratch);
    balance_base2k16::<BE>(dst);
}

pub fn balance_base2k16<BE>(glwe: &mut GLWE<BE::OwnedBuf>)
where
    BE: Backend<OwnedBuf = Vec<u8>>,
{
    balance_base2k16_data(glwe.data_mut());
}

/// Centered-digit carry pass over a base2k = 16 `VecZnx`: rebalances every
/// digit into `[-2^15, 2^15)` so each limb reinterprets as a valid `i16`
/// decomposition digit. Idempotent.
pub(crate) fn balance_base2k16_data(data: &mut VecZnx<Vec<u8>>) {
    let tau = data.size();
    let cols = data.cols();
    let n = data.n();
    let mut carries = vec![0i64; n];
    for col in 0..cols {
        for limb in (1..tau).rev() {
            {
                let lo = data.at_mut(col, limb);
                for pos in 0..n {
                    let v = lo[pos];
                    let k = (v + 32768).div_euclid(65536);
                    lo[pos] = v - k * 65536;
                    carries[pos] = k;
                }
            }
            let hi = data.at_mut(col, limb - 1);
            for pos in 0..n {
                hi[pos] += carries[pos];
            }
        }
    }
}

/// Precomputes the `qtilde`-switched final mask consumed by the fused
/// reduced-precision pack ([`Packing::pack_to_qtilde`]): normalizes the
/// full-precision `final_mask` from the pack regime (`base2k`, `size` limbs)
/// into the qtilde regime (base2k = 16, `tau` limbs) and balances its digits —
/// exactly the switch `modulus_switch_to_digits` would apply to the mask
/// column online, moved to precompute time. The full-precision mask is kept;
/// other consumers (interpolation, the two-step path) are unaffected.
pub fn switch_final_mask_to_qtilde<BE>(
    module: &Module<BE>,
    precompute: &mut PackingPrecomputations<BE>,
    qtilde_bits: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend<OwnedBuf = Vec<u8>>,
    Module<BE>: VecZnxNormalize<BE>,
    VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
{
    let tau = decompose_digits(qtilde_bits);
    let mut switched = module.vec_znx_alloc(1, tau);
    {
        let src_ref =
            <VecZnx<Vec<u8>> as VecZnxToBackendRef<BE>>::to_backend_ref(precompute.final_mask());
        let mut dst_mut = <VecZnx<Vec<u8>> as VecZnxToBackendMut<BE>>::to_backend_mut(&mut switched);
        module.vec_znx_normalize(
            &mut dst_mut,
            16,
            0,
            0,
            &src_ref,
            precompute.base2k(),
            0,
            scratch,
        );
    }
    balance_base2k16_data(&mut switched);
    precompute.set_final_mask_qtilde(switched);
}

/// Allocation layout (`base2k = 16`, `tau` limbs, rank 1) of a mod-switched,
/// decomposed packed RLWE.
pub fn qtilde_glwe_layout(n: Degree, qtilde_bits: usize) -> EncryptionLayout<GLWELayout> {
    let tau = decompose_digits(qtilde_bits);
    EncryptionLayout::new_from_default_sigma(GLWELayout {
        n,
        base2k: Base2K(16),
        k: TorusPrecision((16 * tau) as u32),
        rank: Rank(1),
    })
    .unwrap()
}

/// Partial-packs `packed_inputs` into RLWEs, then modulus-switches every packed
/// RLWE to `qtilde` at `base2k = 16`.
#[allow(dead_code)] // used by recursion tests; superseded by `partial_pack_batch_pooled` in prod
pub(crate) fn partial_pack_batch<BE>(
    module: &Module<BE>,
    src_infos: &EncryptionLayout<GLWELayout>,
    qtilde_bits: usize,
    packed_inputs: &[(&PackingPrecomputations<BE>, &VecZnx<BE::OwnedBuf>)],
    key: &PackingKeys<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> Vec<GLWE<BE::OwnedBuf>>
where
    BE: Backend<OwnedBuf = Vec<u8>>,
    Module<BE>: Packing<BE> + GLWENormalize<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
{
    let qtilde_infos = qtilde_glwe_layout(src_infos.n(), qtilde_bits);

    let mut out = Vec::with_capacity(packed_inputs.len());
    for &(precompute, body) in packed_inputs {
        let mut packed = module.glwe_alloc_from_infos(src_infos);
        module.pack(&mut packed, body, precompute, key, 1, &mut scratch.borrow());

        let mut switched = module.glwe_alloc_from_infos(&qtilde_infos);
        modulus_switch_to_digits(module, &mut switched, &packed, &mut scratch.borrow());
        out.push(switched);
    }
    out
}

/// Parallel production counterpart of [`partial_pack_batch`]: each input packs
/// independently across workers, each borrowing a persistent [`ScratchOwned`]
/// from `pool` (no per-query allocation). Output is written by index ⇒
/// bit-identical to the sequential order.
///
/// Unlike the two-step test path, this uses the **fused** reduced-precision
/// pack ([`Packing::pack_to_qtilde`]): the BSGS accumulation runs at the
/// precision that survives the switch to `qtilde` and the switch itself is
/// folded into the pack's final normalize. Every input's precompute must carry
/// the qtilde-switched mask (see [`switch_final_mask_to_qtilde`]).
#[allow(clippy::type_complexity)]
pub(crate) fn partial_pack_batch_pooled<BE>(
    module: &Module<BE>,
    src_infos: &EncryptionLayout<GLWELayout>,
    qtilde_bits: usize,
    packed_inputs: &[(&PackingPrecomputations<BE>, &VecZnx<BE::OwnedBuf>)],
    key: &PackingKeys<BE>,
    pool: &mut [ScratchOwned<BE>],
) -> Vec<GLWE<BE::OwnedBuf>>
where
    BE: Backend<OwnedBuf = Vec<u8>>,
    Module<BE>: Packing<BE> + GLWENormalize<BE> + ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
{
    let count = packed_inputs.len();
    let qtilde_infos = qtilde_glwe_layout(src_infos.n(), qtilde_bits);
    let nthreads = num_threads_online(count).min(pool.len().max(1));
    let work = assign_panels(count, 1, nthreads);

    let mut outputs: Vec<Option<GLWE<BE::OwnedBuf>>> = (0..count).map(|_| None).collect();
    {
        let mut out_slabs: Vec<&mut [Option<GLWE<BE::OwnedBuf>>]> = Vec::with_capacity(work.len());
        let mut rest = outputs.as_mut_slice();
        for grp in &work {
            let (head, tail) = rest.split_at_mut(grp.len());
            out_slabs.push(head);
            rest = tail;
        }
        let scratch_slabs: Vec<&mut ScratchOwned<BE>> = pool[..work.len()].iter_mut().collect();
        let qtilde_infos = &qtilde_infos;
        scoped_workers_pooled::<BE, Option<GLWE<BE::OwnedBuf>>, _>(
            out_slabs,
            scratch_slabs,
            &work,
            |slab, grp, sc| {
                for (slot, w) in slab.iter_mut().zip(grp.iter()) {
                    let (precompute, body) = packed_inputs[w.panel];
                    let mut switched = module.glwe_alloc_from_infos(qtilde_infos);
                    module.pack_to_qtilde(
                        &mut switched,
                        body,
                        precompute,
                        key,
                        1,
                        &mut sc.borrow(),
                    );
                    balance_base2k16::<BE>(&mut switched);
                    *slot = Some(switched);
                }
            },
        );
    }
    outputs
        .into_iter()
        .map(|o| o.expect("pack worker did not fill its slot"))
        .collect()
}
