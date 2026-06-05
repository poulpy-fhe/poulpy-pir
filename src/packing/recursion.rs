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
use poulpy_hal::layouts::{Backend, Module, ScratchArena, VecZnx, ZnxViewMut};

use crate::packing::{Packing, PackingKeys, PackingPrecomputations};

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
