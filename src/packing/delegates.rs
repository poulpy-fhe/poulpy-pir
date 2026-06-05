//! Delegate blanket impl: wire the user-facing [`Packing`] trait on
//! `Module<BE>` to the OEP `*Impl<BE>` surface.

#![allow(clippy::too_many_arguments)]

use poulpy_core::{
    GLWENoise, LWEMatrixDecrypt,
    layouts::{
        CoeffBound, CoeffMatrix, CoeffMatrixInfos, GGLWECompressedSeed, GGLWEInfos, GLWE,
        GLWEInfos, GLWELayout, GLWEPlaintext, GLWESecretPreparedFactory, GLWEToBackendMut,
        GLWEToBackendRef, GetGaloisElement, LWEInfos, LWEMatrix, LWEMatrixInfos,
        LWESecretToBackendRef, ModuleCoreAlloc, Rank, compressed::GGLWECompressedToBackendRef,
        prepared::GLWESecretPreparedToBackendRef,
    },
};
use poulpy_hal::layouts::{
    Backend, HostBackend, HostDataMut, HostDataRef, Module, ScratchArena, Stats,
    VecZnxToBackendMut, VecZnxToBackendRef, ZnxInfos, ZnxView, ZnxViewMut,
};

use crate::encoding::ModPEncoder;
use crate::packing::{
    GLWECoeffNoise, LWEMatrixCoeffNoise, Packing, PackingKeys, PackingMaskAggregation,
    oep::{PackingImpl, PackingMaskAggregationImpl},
    packing_precomputations::{PackingPrecomputations, PackingPrecomputeInfos},
};

impl<BE> PackingMaskAggregation<BE> for Module<BE>
where
    BE: Backend + PackingMaskAggregationImpl<BE>,
{
    fn packing_mask_preprocessing_tmp_bytes(&self, size: usize) -> usize {
        BE::packing_mask_preprocessing_tmp_bytes_impl(self, size)
    }

    fn packing_mask_preprocessing<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos,
    {
        BE::packing_mask_preprocessing_impl(self, dst, base2k, a, scratch);
    }

    fn pack_partial_mask_preprocessing_tmp_bytes(&self, gamma: usize, size: usize) -> usize {
        BE::pack_partial_mask_preprocessing_tmp_bytes_impl(self, gamma, size)
    }

    fn packing_partial_mask_preprocessing<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        gamma: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos,
    {
        BE::pack_partial_mask_preprocessing_impl(self, dst, base2k, gamma, a, scratch);
    }
}

impl<BE> Packing<BE> for Module<BE>
where
    BE: Backend + PackingImpl<BE>,
{
    fn pack_keys_precompute_tmp_bytes<KG, KH>(
        &self,
        key_g: &KG,
        key_h: &KH,
        baby_size: usize,
    ) -> usize
    where
        KG: GGLWEInfos,
        KH: GGLWEInfos,
    {
        BE::pack_keys_precompute_tmp_bytes_impl(self, key_g, key_h, baby_size)
    }

    fn pack_keys_precompute<KG, KH>(
        &self,
        key_g: &KG,
        key_h: &KH,
        baby_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> PackingKeys<BE>
    where
        KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
        KH: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
    {
        BE::pack_keys_precompute_impl(self, key_g, key_h, baby_size, scratch)
    }

    fn pack_partial_keys_precompute<KG>(
        &self,
        key_g: &KG,
        stride: usize,
        baby_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> PackingKeys<BE>
    where
        KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
    {
        BE::pack_partial_keys_precompute_impl(self, key_g, stride, baby_size, scratch)
    }

    fn pack_precompute_alloc(
        &self,
        steps: usize,
        size: usize,
        base2k: usize,
        baby_size: usize,
    ) -> PackingPrecomputations<BE> {
        BE::pack_precompute_alloc_impl(self, steps, size, base2k, baby_size)
    }

    fn pack_partial_precompute_alloc(
        &self,
        steps: usize,
        size: usize,
        base2k: usize,
        baby_size: usize,
        stride: usize,
    ) -> PackingPrecomputations<BE> {
        BE::pack_partial_precompute_alloc_impl(self, steps, size, base2k, baby_size, stride)
    }

    fn pack_precompute_tmp_bytes<A, KMask>(
        &self,
        metadata: PackingPrecomputeInfos,
        aggregate_mask: &A,
        key_mask_source: &KMask,
    ) -> usize
    where
        A: ZnxInfos,
        KMask: GGLWEInfos,
    {
        BE::pack_precompute_tmp_bytes_impl(self, metadata, aggregate_mask, key_mask_source)
    }

    fn pack_precompute<A, KMask>(
        &self,
        precomputations: &mut PackingPrecomputations<BE>,
        aggregate_mask: &A,
        key_mask_source: &KMask,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: VecZnxToBackendRef<BE> + ZnxInfos,
        KMask: GGLWECompressedSeed + GGLWEInfos,
    {
        BE::pack_precompute_impl(
            self,
            precomputations,
            aggregate_mask,
            key_mask_source,
            scratch,
        );
    }

    fn pack_partial_precompute<A, KMask>(
        &self,
        precomputations: &mut PackingPrecomputations<BE>,
        aggregate_mask: &A,
        key_mask_source: &KMask,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: VecZnxToBackendRef<BE> + ZnxInfos,
        KMask: GGLWECompressedSeed + GGLWEInfos,
    {
        BE::pack_partial_precompute_impl(
            self,
            precomputations,
            aggregate_mask,
            key_mask_source,
            scratch,
        );
    }

    /// Forwards the public module API to the backend OEP hook.
    ///
    /// Keeping the delegate this small makes the split explicit: API shape
    /// lives in `api.rs`, backend specialization in `oep.rs`, and algorithmic
    /// work in `default.rs`/`bsgs_pack.rs`.
    fn pack<R, B>(
        &self,
        res: &mut R,
        body: &B,
        precomputations: &PackingPrecomputations<BE>,
        key_precomputations: &PackingKeys<BE>,
        chunk_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        B: VecZnxToBackendRef<BE> + ZnxInfos,
    {
        BE::pack_impl(
            self,
            res,
            body,
            precomputations,
            key_precomputations,
            chunk_size,
            scratch,
        );
    }
}

impl<BE> LWEMatrixCoeffNoise<BE> for Module<BE>
where
    BE: Backend + HostBackend,
    Module<BE>: LWEMatrixDecrypt<BE>
        + GLWENoise<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + GLWESecretPreparedFactory<BE>,
    BE::OwnedBuf: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    fn lwe_matrix_coeff_noise_tmp_bytes<P>(&self, product: &P) -> usize
    where
        P: LWEMatrixInfos,
    {
        let glwe_layout = GLWELayout {
            n: product.n(),
            base2k: product.base2k(),
            k: product.max_k(),
            rank: Rank(1),
        };

        self.lwe_matrix_decrypt_tmp_bytes(product)
            .max(self.glwe_noise_tmp_bytes(&glwe_layout))
    }

    fn lwe_matrix_coeff_noise<BU, S>(
        &self,
        coeffs: &CoeffMatrix<BE::OwnedBuf, BU>,
        column: usize,
        product: &LWEMatrix<BE::OwnedBuf>,
        sk_lwe: &S,
        encoder: &ModPEncoder,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Stats
    where
        BU: CoeffBound,
        S: LWESecretToBackendRef<BE> + LWEInfos,
    {
        assert!(
            column < coeffs.n().as_usize(),
            "coefficient column out of range"
        );
        assert_eq!(
            coeffs.rows_out(),
            product.rows(),
            "coefficient rows must match LWE matrix rows"
        );

        let glwe_layout = GLWELayout {
            n: product.n(),
            base2k: product.base2k(),
            k: product.max_k(),
            rank: Rank(1),
        };

        let mut decrypted: GLWEPlaintext<_> = self.glwe_plaintext_alloc_from_infos(&glwe_layout);
        self.lwe_matrix_decrypt(product, &mut decrypted, sk_lwe, &mut scratch.borrow());

        let mut expected_values = vec![0i64; product.n().as_usize()];
        for (row, value) in expected_values.iter_mut().take(product.rows()).enumerate() {
            *value = coeffs.data().at(row, 0)[column];
        }

        let mut expected: GLWEPlaintext<_> = self.glwe_plaintext_alloc_from_infos(&glwe_layout);
        encoder.encode_vec_i64(
            &mut expected.data,
            product.base2k().as_usize(),
            0,
            &expected_values,
        );

        let decrypted_as_glwe = glwe_from_plaintext_body(self, &glwe_layout, &decrypted);
        let mut zero_sk = self.glwe_secret_alloc(Rank(1));
        zero_sk.fill_zero();
        let mut zero_sk_prepared = self.glwe_secret_prepared_alloc(Rank(1));
        self.glwe_secret_prepare(&mut zero_sk_prepared, &zero_sk);

        self.glwe_noise(
            &decrypted_as_glwe,
            &expected,
            &zero_sk_prepared,
            &mut scratch.borrow(),
        )
    }
}

impl<BE> GLWECoeffNoise<BE> for Module<BE>
where
    BE: Backend + HostBackend,
    Module<BE>: GLWENoise<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
    BE::OwnedBuf: HostDataMut,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    fn glwe_coeff_noise_tmp_bytes<G>(&self, glwe: &G) -> usize
    where
        G: GLWEInfos,
    {
        self.glwe_noise_tmp_bytes(glwe)
    }

    fn glwe_coeff_noise<BU, G, S>(
        &self,
        coeffs: &CoeffMatrix<BE::OwnedBuf, BU>,
        column: usize,
        glwe: &G,
        sk_glwe: &S,
        encoder: &ModPEncoder,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Stats
    where
        BU: CoeffBound,
        G: GLWEToBackendRef<BE> + GLWEInfos,
        S: GLWESecretPreparedToBackendRef<BE> + GLWEInfos,
    {
        assert!(
            column < coeffs.n().as_usize(),
            "coefficient column out of range"
        );
        assert!(
            coeffs.rows_out() <= glwe.n().as_usize(),
            "coefficient rows must fit the GLWE ring degree"
        );

        let glwe_layout = GLWELayout {
            n: glwe.n(),
            base2k: glwe.base2k(),
            k: glwe.max_k(),
            rank: glwe.rank(),
        };
        let expected = expected_coeff_plaintext(self, coeffs, column, &glwe_layout, encoder);
        self.glwe_noise(glwe, &expected, sk_glwe, scratch)
    }
}

fn expected_coeff_plaintext<BE, BU>(
    module: &Module<BE>,
    coeffs: &CoeffMatrix<BE::OwnedBuf, BU>,
    column: usize,
    glwe_layout: &GLWELayout,
    encoder: &ModPEncoder,
) -> GLWEPlaintext<BE::OwnedBuf>
where
    BE: Backend,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
    BE::OwnedBuf: HostDataMut,
    BU: CoeffBound,
{
    let mut expected_values = vec![0i64; glwe_layout.n.as_usize()];
    for (row, value) in expected_values
        .iter_mut()
        .take(coeffs.rows_out())
        .enumerate()
    {
        *value = coeffs.data().at(row, 0)[column];
    }

    let mut expected: GLWEPlaintext<_> = module.glwe_plaintext_alloc_from_infos(glwe_layout);
    encoder.encode_vec_i64(
        &mut expected.data,
        glwe_layout.base2k.as_usize(),
        0,
        &expected_values,
    );
    expected
}

fn glwe_from_plaintext_body<BE>(
    module: &Module<BE>,
    glwe_layout: &GLWELayout,
    pt: &GLWEPlaintext<BE::OwnedBuf>,
) -> GLWE<BE::OwnedBuf>
where
    BE: Backend,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>,
    BE::OwnedBuf: HostDataMut,
{
    let mut glwe = module.glwe_alloc_from_infos(glwe_layout);
    for col in 0..glwe.data().cols() {
        for limb in 0..glwe.data().size() {
            glwe.data_mut().at_mut(col, limb).fill(0);
        }
    }
    for limb in 0..pt.data.size() {
        glwe.data_mut()
            .at_mut(0, limb)
            .copy_from_slice(pt.data.at(0, limb));
    }
    glwe
}
