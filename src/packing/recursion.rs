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
    api::ScratchOwnedBorrow,
    layouts::{Backend, Module, ScratchArena, ScratchOwned, VecZnx, ZnxViewMut},
};

use crate::{
    packing::{Packing, PackingKeys, PackingPrecomputations},
    parallel::{assign_panels, num_threads, scoped_workers_pooled},
};

/// Number of base2k=16 decomposition digits for a `qtilde`-modulus plaintext.
pub(crate) fn decompose_digits(qtilde_bits: usize) -> usize {
    qtilde_bits.div_ceil(16)
}

/// Modulus-switches a packed RLWE from its native modulus `q` down to
/// `qtilde = 2^{16 * tau}` and writes the result at `base2k = 16`.
pub(crate) fn modulus_switch_to_digits<BE>(
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

fn balance_base2k16<BE>(glwe: &mut GLWE<BE::OwnedBuf>)
where
    BE: Backend<OwnedBuf = Vec<u8>>,
{
    let tau = glwe.data().size();
    let cols = glwe.data().cols();
    let n = glwe.data().n();
    let data = glwe.data_mut();
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

/// Allocation layout (`base2k = 16`, `tau` limbs, rank 1) of a mod-switched,
/// decomposed packed RLWE.
pub(crate) fn qtilde_glwe_layout(n: Degree, qtilde_bits: usize) -> EncryptionLayout<GLWELayout> {
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

/// Parallel [`partial_pack_batch`]: each input's `pack` + modulus-switch is
/// independent, so they run across workers, each borrowing a persistent
/// [`ScratchOwned`] from `pool` (no per-query allocation). Output is written by
/// index ⇒ bit-identical to the sequential order.
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
    let nthreads = num_threads(count).min(pool.len().max(1));
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
                    let mut packed = module.glwe_alloc_from_infos(src_infos);
                    module.pack(&mut packed, body, precompute, key, 1, &mut sc.borrow());
                    let mut switched = module.glwe_alloc_from_infos(qtilde_infos);
                    modulus_switch_to_digits::<BE>(module, &mut switched, &packed, &mut sc.borrow());
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
