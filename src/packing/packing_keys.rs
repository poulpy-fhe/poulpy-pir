//! User-key-side packing precomputations plus fixed seed-mask helpers.
//!
//! The server receives compressed automorphism keys. Online packing needs their
//! user-dependent body columns. Fixed-mask precomputation only needs the public
//! mask seed, so the seed-derived mask helpers here only use the
//! [`GGLWECompressedSeed`] part of a compressed key.

use poulpy_core::{
    EncryptionInfos, GLWEAutomorphismKeyCompressedEncryptSk, GLWEMaskFillDefault,
    ScratchArenaTakeCore,
    layouts::{
        GGLWE, GGLWEAtViewMut, GGLWEAtViewRef, GGLWECompressedSeed, GGLWEInfos, GGLWELayout,
        GGLWEPreparedFactory, GGLWEToBackendMut, GGLWEToBackendRef, GLWEAutomorphismKeyCompressed,
        GLWESecret, GLWESecretToBackendMut, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement,
        LWEInfos, LWESecretToBackendRef, ModuleCoreAlloc, ModuleCoreCompressedAlloc, Rank,
        compressed::GGLWECompressedToBackendRef, prepared::GGLWEPrepared,
    },
};
use poulpy_hal::{
    api::{ScalarZnxAutomorphismBackend, VecZnxAutomorphismBackend, VecZnxCopyBackend},
    layouts::{Backend, GaloisElement, Module, ScratchArena},
    source::Source,
};

use crate::packing::PackingKeysGenerate;

/// Owned user-key-side precomputations used by packing.
///
/// This is the current concrete key-precompute container. It is produced by
/// [`Packing::pack_keys_precompute`] from the full `key_g`/`key_h` switching
/// keys and contains no seed-derived mask material.
pub struct PackingKeys<BE: Backend> {
    /// Prepared baby-step `key_g` body keys indexed by baby step.
    baby_key_g_bodies: Vec<GGLWEPrepared<BE::OwnedBuf, BE>>,
    /// Prepared final `key_h` body key.
    key_h_body: GGLWEPrepared<BE::OwnedBuf, BE>,
}

impl<BE: Backend> PackingKeys<BE> {
    /// Creates owned user-key-side body precomputations.
    pub fn new(
        baby_key_g_bodies: Vec<GGLWEPrepared<BE::OwnedBuf, BE>>,
        key_h_body: GGLWEPrepared<BE::OwnedBuf, BE>,
    ) -> Self {
        Self {
            baby_key_g_bodies,
            key_h_body,
        }
    }

    /// Returns the prepared baby-step `key_g` body at `idx`.
    pub fn baby_key_g(&self, idx: usize) -> &GGLWEPrepared<BE::OwnedBuf, BE> {
        &self.baby_key_g_bodies[idx]
    }

    /// Returns the prepared final `key_h` body.
    pub fn key_h(&self) -> &GGLWEPrepared<BE::OwnedBuf, BE> {
        &self.key_h_body
    }

    /// Output size of the key products, used to size online scratch buffers.
    pub fn key_size(&self) -> usize {
        self.key_h().size()
    }
}

impl<BE> PackingKeysGenerate<BE> for Module<BE>
where
    // Compressed keys are host-side `Vec<u8>` buffers (see the compressed alloc
    // and `GLWEAutomorphismKeyCompressedEncryptSk` impl bounds), consistent with
    // the rest of the crate's compressed-buffer code.
    BE: Backend<OwnedBuf = Vec<u8>>,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + ModuleCoreCompressedAlloc
        + GLWEAutomorphismKeyCompressedEncryptSk<BE>
        + GaloisElement
        + ScalarZnxAutomorphismBackend<BE>,
{
    fn pack_keys_generate_tmp_bytes<E>(&self, key_infos: &E) -> usize
    where
        E: GGLWEInfos,
    {
        self.glwe_automorphism_key_compressed_encrypt_sk_tmp_bytes(key_infos)
    }

    fn pack_keys_generate<E, S>(
        &self,
        key_infos: &E,
        sk_lwe: &S,
        key_seed: [u8; 32],
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> (
        GLWEAutomorphismKeyCompressed<Vec<u8>>,
        GLWEAutomorphismKeyCompressed<Vec<u8>>,
    )
    where
        E: EncryptionInfos + GGLWEInfos,
        S: LWESecretToBackendRef<BE>,
    {
        // The natural automorphism keys are signed under the raw LWE secret
        // wrapped (identity automorphism) into a rank-1 GLWE polynomial key.
        let sk_base = wrap_lwe_secret(self, sk_lwe);

        // These must match the rotations the server-side precompute realigns
        // against; see `pack_keys_precompute_default`.
        let key_g_rotation = self.galois_element_inv(self.galois_element(1));
        let key_h_rotation = -1i64;

        let mut key_g = self.glwe_automorphism_key_compressed_alloc_from_infos(key_infos);
        self.glwe_automorphism_key_compressed_encrypt_sk(
            &mut key_g,
            key_g_rotation,
            &sk_base,
            key_seed,
            key_infos,
            source_xe,
            &mut scratch.borrow(),
        );

        let mut key_h = self.glwe_automorphism_key_compressed_alloc_from_infos(key_infos);
        self.glwe_automorphism_key_compressed_encrypt_sk(
            &mut key_h,
            key_h_rotation,
            &sk_base,
            key_seed,
            key_infos,
            source_xe,
            &mut scratch.borrow(),
        );

        (key_g, key_h)
    }
}

/// Wraps a raw LWE secret into the rank-1 GLWE polynomial key (`sk_base`) that
/// the packing automorphism keys are signed under.
///
/// Unlike [`poulpy_core::SecretConversion::glwe_secret_from_lwe_secret`] (which
/// applies the `X -> X^{-1}` automorphism `p = -1`), packing keys are signed
/// under the secret in its natural orientation, so this uses the identity
/// automorphism `p = 1`.
fn wrap_lwe_secret<BE, S>(module: &Module<BE>, sk_lwe: &S) -> GLWESecret<BE::OwnedBuf>
where
    BE: Backend<OwnedBuf = Vec<u8>>,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + ScalarZnxAutomorphismBackend<BE>,
    S: LWESecretToBackendRef<BE>,
{
    let src = sk_lwe.to_backend_ref();
    assert_eq!(
        src.n().as_usize(),
        module.n(),
        "LWE secret degree must equal module degree"
    );
    let mut sk_base = module.glwe_secret_alloc(Rank(1));
    // `fill_zero` clears the data and, crucially, sets a non-`NONE` distribution
    // tag so the automorphism-key encryption guard accepts the key; the actual
    // secret coefficients are written over column 0 by the identity automorphism
    // below. There is no public setter to copy the LWE secret's exact
    // distribution, and the encryption uses the provided coefficients directly.
    sk_base.fill_zero();
    {
        let mut res_ref = GLWESecretToBackendMut::<BE>::to_backend_mut(&mut sk_base);
        module.scalar_znx_automorphism_backend(1, res_ref.data_mut(), 0, src.data(), 0);
    }
    sk_base
}

/// Scratch estimate for [`packing_keys_precompute`].
///
/// Splitting and baby automorphisms use owned temporary key buffers; scratch is
/// needed only for preparing one projected key at a time.
pub(crate) fn packing_keys_precompute_tmp_bytes<BE, KG, KH>(
    module: &Module<BE>,
    key_g: &KG,
    key_h: &KH,
    _baby_size: usize,
) -> usize
where
    BE: Backend,
    Module<BE>: GGLWEPreparedFactory<BE>,
    KG: GGLWEInfos,
    KH: GGLWEInfos,
{
    module
        .gglwe_prepare_tmp_bytes(key_g)
        .max(module.gglwe_prepare_tmp_bytes(key_h))
}

/// Builds the user-key-side body precomputations consumed by packing.
///
/// The compressed body column is prepared for online `key_g` / `key_h` products.
/// `key_g` body projections are additionally transformed into the baby-step
/// views used by the online BSGS loop. Seed-derived mask material is built by
/// [`packing_mask_keys_precompute`] during the fixed precompute phase.
///
/// Clients sign their natural automorphism keys with `sk_base` (their raw LWE
/// secret). The intermediate key-switching shape the collapse expects is
/// `sk_g -> sk_base` (and `sk_h -> sk_base`), where `sk_g`, `sk_h` are
/// galois-rotated views of `sk_base`. With `p = g^{-1}`, `sk = sk_base`, the
/// natural key encrypts `sk_base` under `sk_{g^1}`; rotating the whole key by
/// `g^{-1}` realigns it to `sk_g -> sk_base`. The same idea with `p = -1`
/// and rotation by `-1` realigns the `key_h` half. Both rotations are absorbed
/// once at precompute time and the rest of the packing pipeline stays the
/// same shape as the historical switching-key path.
pub(crate) fn pack_keys_precompute_default<BE, KG, KH>(
    module: &Module<BE>,
    key_g: &KG,
    key_h: &KH,
    baby_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> PackingKeys<BE>
where
    BE: Backend,
    Module<BE>: GGLWEPreparedFactory<BE>
        + GaloisElement
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxCopyBackend<BE>,
    KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
    KH: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
{
    let key_g_rotation = module.galois_element_inv(module.galois_element(1));
    let key_h_rotation = -1i64;
    assert_same_mask_seed(key_g, key_h);
    assert_eq!(
        key_g.p(),
        key_g_rotation,
        "packing key_g must be generated with p = galois_element_inv(galois_element(1))"
    );
    assert_eq!(
        key_h.p(),
        key_h_rotation,
        "packing key_h must be generated with p = -1"
    );

    let baby_key_g_bodies =
        prepare_baby_body_keys_from_compressed(module, key_g, key_g_rotation, baby_size, scratch);
    let key_h_body = prepare_body_key_from_compressed(module, key_h, key_h_rotation, scratch);

    PackingKeys::new(baby_key_g_bodies, key_h_body)
}

/// Scratch estimate for preparing both fixed seed-derived mask keys.
pub(crate) fn packing_mask_keys_precompute_tmp_bytes<BE, KMask>(
    module: &Module<BE>,
    key_mask_source: &KMask,
) -> usize
where
    BE: Backend,
    Module<BE>: GGLWEPreparedFactory<BE>,
    KMask: GGLWEInfos,
{
    let layout = mask_key_layout(key_mask_source);
    let split_bytes =
        GGLWE::<Vec<u8>>::bytes_of_from_infos(&layout).next_multiple_of(BE::SCRATCH_ALIGN);
    2 * split_bytes + module.gglwe_prepare_tmp_bytes(&layout)
}

/// Rank-0 layout of the compressed-key mask projection used by fixed precompute.
pub(crate) fn mask_key_layout<K: GGLWEInfos>(key: &K) -> GGLWELayout {
    GGLWELayout {
        n: key.n(),
        base2k: key.base2k(),
        k: key.max_k(),
        rank_in: key.rank_in(),
        rank_out: Rank(0),
        dnum: key.dnum(),
        dsize: key.dsize(),
    }
}

/// Prepares the fixed mask-side `key_g` and `key_h` projections.
///
/// These projections are deterministic functions of the public packing seed,
/// key layout, and the rotations required to align the plain automorphism-key
/// direction with the collapse schedule.
pub(crate) fn packing_mask_keys_precompute<BE, KMask>(
    module: &Module<BE>,
    key_mask_source: &KMask,
    scratch: &mut ScratchArena<'_, BE>,
) -> (
    GGLWEPrepared<BE::OwnedBuf, BE>,
    GGLWEPrepared<BE::OwnedBuf, BE>,
)
where
    BE: Backend,
    Module<BE>: GGLWEPreparedFactory<BE>
        + GLWEMaskFillDefault<BE>
        + GaloisElement
        + VecZnxAutomorphismBackend<BE>
        + VecZnxCopyBackend<BE>,
    KMask: GGLWECompressedSeed + GGLWEInfos,
{
    let key_g_rotation = module.galois_element_inv(module.galois_element(1));
    let key_h_rotation = -1i64;
    let key_g_mask = prepare_mask_key_from_seed(module, key_mask_source, key_g_rotation, scratch);
    let key_h_mask = prepare_mask_key_from_seed(module, key_mask_source, key_h_rotation, scratch);
    (key_g_mask, key_h_mask)
}

fn prepare_body_key_from_compressed<BE, K>(
    module: &Module<BE>,
    key: &K,
    rotation: i64,
    scratch: &mut ScratchArena<'_, BE>,
) -> GGLWEPrepared<BE::OwnedBuf, BE>
where
    BE: Backend,
    Module<BE>: GGLWEPreparedFactory<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxCopyBackend<BE>,
    K: GGLWECompressedToBackendRef<BE> + GGLWEInfos,
{
    let key_ref = key.to_backend_ref();
    let body_ref = key_ref.body_as_gglwe();
    let split = split_output_key_plain(module, &&body_ref, 0, rotation);
    let mut prepared = module.gglwe_prepared_alloc_from_infos(&split);
    module.gglwe_prepare(&mut prepared, &split, &mut scratch.borrow());
    prepared
}

fn prepare_baby_body_keys_from_compressed<BE, K>(
    module: &Module<BE>,
    key: &K,
    rotation: i64,
    baby_size: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> Vec<GGLWEPrepared<BE::OwnedBuf, BE>>
where
    BE: Backend,
    Module<BE>: GGLWEPreparedFactory<BE>
        + GaloisElement
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxCopyBackend<BE>,
    K: GGLWECompressedToBackendRef<BE> + GGLWEInfos,
{
    let mut baby_keys = Vec::with_capacity(baby_size);

    for baby_idx in 0..baby_size {
        let baby_rotation = rotation * module.galois_element(baby_idx as i64);
        baby_keys.push(prepare_body_key_from_compressed(
            module,
            key,
            baby_rotation,
            scratch,
        ));
    }

    baby_keys
}

fn prepare_mask_key_from_seed<BE, K>(
    module: &Module<BE>,
    key: &K,
    rotation: i64,
    scratch: &mut ScratchArena<'_, BE>,
) -> GGLWEPrepared<BE::OwnedBuf, BE>
where
    BE: Backend,
    Module<BE>: GGLWEPreparedFactory<BE>
        + GLWEMaskFillDefault<BE>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxCopyBackend<BE>,
    K: GGLWECompressedSeed + GGLWEInfos,
{
    let layout = mask_key_layout(key);
    let mut prepared = module.gglwe_prepared_alloc_from_infos(&layout);
    let scratch_local = scratch.borrow();
    let (mut split, scratch_local) = scratch_local.take_gglwe_scratch(&layout);

    let mut scratch_local = if rotation == 1 {
        fill_mask_key_from_seed(module, &mut split, key);
        scratch_local
    } else {
        let (mut tmp, scratch_next) = scratch_local.take_gglwe_scratch(&layout);
        fill_mask_key_from_seed(module, &mut tmp, key);
        automorph_mask_key(module, &mut split, &tmp, key, rotation);
        scratch_next
    };

    module.gglwe_prepare(&mut prepared, &split, &mut scratch_local.borrow());
    prepared
}

fn fill_mask_key_from_seed<BE, Dst, K>(module: &Module<BE>, dst: &mut Dst, key: &K)
where
    BE: Backend,
    Module<BE>: GLWEMaskFillDefault<BE>,
    Dst: GGLWEToBackendMut<BE> + GGLWEInfos,
    K: GGLWECompressedSeed + GGLWEInfos,
{
    let seeds = key.seed();
    let rank_in = key.rank_in().as_usize();
    let mut dst_backend = dst.to_backend_mut();
    for row in 0..key.dnum().as_usize() {
        for col in 0..rank_in {
            let mut dst_view = dst_backend.at_view_mut(row, col);
            module.fill_glwe_mask_from_seed_default(
                key.base2k().as_usize(),
                &mut dst_view,
                0,
                1,
                seeds[row * rank_in + col],
            );
        }
    }
}

fn automorph_mask_key<BE, Dst, Src, K>(
    module: &Module<BE>,
    dst: &mut Dst,
    src: &Src,
    key: &K,
    rotation: i64,
) where
    BE: Backend,
    Module<BE>: VecZnxAutomorphismBackend<BE>,
    Dst: GGLWEToBackendMut<BE> + GGLWEInfos,
    Src: GGLWEToBackendRef<BE> + GGLWEInfos,
    K: GGLWEInfos,
{
    let rank_in = key.rank_in().as_usize();
    let src_backend = src.to_backend_ref();
    let mut dst_backend = dst.to_backend_mut();
    for row in 0..key.dnum().as_usize() {
        for col in 0..rank_in {
            let src = src_backend.at_view(row, col);
            let mut dst = dst_backend.at_view_mut(row, col);
            let src_ref = src.to_backend_ref();
            let mut dst_mut = dst.to_backend_mut();
            module.vec_znx_automorphism_backend(rotation, dst_mut.data_mut(), 0, src_ref.data(), 0);
        }
    }
}

fn assert_same_mask_seed<KG, KH>(key_g: &KG, key_h: &KH)
where
    KG: GGLWECompressedSeed + GGLWEInfos,
    KH: GGLWECompressedSeed + GGLWEInfos,
{
    assert_eq!(key_g.n(), key_h.n(), "packing keys have different degrees");
    assert_eq!(
        key_g.base2k(),
        key_h.base2k(),
        "packing keys have different base2k"
    );
    assert_eq!(
        key_g.max_k(),
        key_h.max_k(),
        "packing keys have different k"
    );
    assert_eq!(
        key_g.rank_in(),
        key_h.rank_in(),
        "packing keys have different input ranks"
    );
    assert_eq!(
        key_g.dnum(),
        key_h.dnum(),
        "packing keys have different dnum"
    );
    assert_eq!(
        key_g.dsize(),
        key_h.dsize(),
        "packing keys have different dsize"
    );
    assert_eq!(
        key_g.seed().as_slice(),
        key_h.seed().as_slice(),
        "packing keys must be generated from the same mask seed"
    );
}

fn split_output_key_plain<BE, K>(
    module: &Module<BE>,
    key: &K,
    output_col: usize,
    rotation: i64,
) -> GGLWE<BE::OwnedBuf>
where
    BE: Backend,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
        + VecZnxAutomorphismBackend<BE>
        + VecZnxCopyBackend<BE>,
    K: GGLWEAtViewRef<BE> + GGLWEInfos,
{
    let mut split = module.gglwe_alloc(
        key.base2k(),
        key.max_k(),
        key.rank_in(),
        Rank(0),
        key.dnum(),
        key.dsize(),
    );

    for row in 0..key.dnum().as_usize() {
        for col in 0..key.rank_in().as_usize() {
            let src = GGLWEAtViewRef::<BE>::at_view(key, row, col);
            let mut dst = GGLWEAtViewMut::<BE>::at_view_mut(&mut split, row, col);
            let src_ref = src.to_backend_ref();
            let mut dst_mut = dst.to_backend_mut();
            if rotation == 1 {
                module.vec_znx_copy_backend(dst_mut.data_mut(), 0, src_ref.data(), output_col);
            } else {
                module.vec_znx_automorphism_backend(
                    rotation,
                    dst_mut.data_mut(),
                    0,
                    src_ref.data(),
                    output_col,
                );
            }
        }
    }

    split
}
