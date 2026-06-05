use poulpy_core::{
    GLWEDecrypt,
    layouts::{Degree, GLWE, GLWESecretPrepared, ModuleCoreAlloc, TorusPrecision},
};
use poulpy_hal::{
    api::{ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, HostDataMut, HostDataRef, Module, ScratchOwned, ZnxViewMut, ZnxZero},
};

use crate::packing::recursion::{decompose_digits, qtilde_glwe_layout};

/// InsPIRe² response: the packed digit-RLWE sets produced by the two recursive
/// packing levels.
pub struct RecursionResponse<BE: Backend> {
    resp1: Vec<GLWE<BE::OwnedBuf>>,
    resp2: Vec<GLWE<BE::OwnedBuf>>,
}

impl<BE: Backend> RecursionResponse<BE> {
    pub(crate) fn new(resp1: Vec<GLWE<BE::OwnedBuf>>, resp2: Vec<GLWE<BE::OwnedBuf>>) -> Self {
        Self { resp1, resp2 }
    }

    /// Packed `gamma1` mask-digit RLWEs.
    pub fn resp1(&self) -> &[GLWE<BE::OwnedBuf>] {
        &self.resp1
    }

    /// Packed `gamma2` body-digit RLWEs.
    pub fn resp2(&self) -> &[GLWE<BE::OwnedBuf>] {
        &self.resp2
    }
}

/// Extracts an InsPIRe² response into the `gamma0` `Z_p` record.
#[allow(clippy::too_many_arguments)]
pub(crate) fn extract_response<BE: Backend<OwnedBuf = Vec<u8>>>(
    module: &Module<BE>,
    sk_dst_prep: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    gamma0: usize,
    gamma1: usize,
    gamma2: usize,
    k_pt: usize,
    qtilde_bits: usize,
    response: &RecursionResponse<BE>,
) -> Vec<i64>
where
    Module<BE>: ModuleN + ModuleCoreAlloc<OwnedBuf = Vec<u8>> + GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    let n = module.n();
    let qtilde_infos = qtilde_glwe_layout(Degree(n as u32), qtilde_bits);
    let recomposed = recompose_response(
        module,
        sk_dst_prep,
        gamma0,
        gamma1,
        gamma2,
        k_pt,
        qtilde_bits,
        response,
    );

    let mut sc = ScratchOwned::<BE>::alloc(module.glwe_decrypt_tmp_bytes(&qtilde_infos));
    let mut out_pt = module.glwe_plaintext_alloc_from_infos(&qtilde_infos);
    module.glwe_decrypt(&recomposed, &mut out_pt, sk_dst_prep, &mut sc.borrow());
    let mut got = vec![0i64; n];
    out_pt.decode_vec_i64(&mut got, TorusPrecision(k_pt as u32));
    got.truncate(gamma0);
    got
}

/// Rebuilds the final qtilde GLWE ciphertext from the two InsPIRe² packed digit
/// responses. This is the ciphertext decrypted by [`extract_response`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn recompose_response<BE: Backend<OwnedBuf = Vec<u8>>>(
    module: &Module<BE>,
    sk_dst_prep: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    gamma0: usize,
    gamma1: usize,
    gamma2: usize,
    k_pt: usize,
    qtilde_bits: usize,
    response: &RecursionResponse<BE>,
) -> GLWE<BE::OwnedBuf>
where
    Module<BE>: ModuleN + ModuleCoreAlloc<OwnedBuf = Vec<u8>> + GLWEDecrypt<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    let n = module.n();
    let tau = decompose_digits(qtilde_bits);
    let qtilde_infos = qtilde_glwe_layout(Degree(n as u32), qtilde_bits);
    let mut sc = ScratchOwned::<BE>::alloc(module.glwe_decrypt_tmp_bytes(&qtilde_infos));

    let mask_vals = unpack_response_digits(
        module,
        sk_dst_prep,
        response.resp1(),
        gamma1,
        n * tau,
        k_pt,
        &qtilde_infos,
        &mut sc,
    );
    let body_vals = unpack_response_digits(
        module,
        sk_dst_prep,
        response.resp2(),
        gamma2,
        gamma0 * tau,
        k_pt,
        &qtilde_infos,
        &mut sc,
    );

    let mut recomposed = module.glwe_alloc_from_infos(&qtilde_infos);
    recomposed.data_mut().zero();
    for c in 0..n {
        for l in 0..tau {
            recomposed.data_mut().at_mut(1, l)[c] = mask_vals[c * tau + l];
        }
    }
    for c in 0..gamma0 {
        for l in 0..tau {
            recomposed.data_mut().at_mut(0, l)[c] = body_vals[c * tau + l];
        }
    }
    recomposed
}

#[allow(clippy::too_many_arguments)]
fn unpack_response_digits<BE, I>(
    module: &Module<BE>,
    sk_dst_prep: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    response: &[GLWE<BE::OwnedBuf>],
    gamma: usize,
    total: usize,
    k_pt: usize,
    qtilde_infos: &I,
    scratch: &mut ScratchOwned<BE>,
) -> Vec<i64>
where
    BE: Backend<OwnedBuf = Vec<u8>>,
    Module<BE>: ModuleN + ModuleCoreAlloc<OwnedBuf = Vec<u8>> + GLWEDecrypt<BE>,
    I: poulpy_core::layouts::GLWEInfos,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    let n = module.n();
    let mut out = Vec::with_capacity(total);
    for glwe in response {
        let mut pt = module.glwe_plaintext_alloc_from_infos(qtilde_infos);
        module.glwe_decrypt(glwe, &mut pt, sk_dst_prep, &mut scratch.borrow());
        let mut dec = vec![0i64; n];
        pt.decode_vec_i64(&mut dec, TorusPrecision(k_pt as u32));
        out.extend_from_slice(&dec[..gamma]);
    }
    out.truncate(total);
    out
}
