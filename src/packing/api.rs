//! User-facing packing trait on `Module<BE>`.
//!
//! The deep-batched, optionally group-chunked BSGS DFT-hot path separates fixed
//! mask-side state from user-key state. [`PackingPrecomputations`] contains the
//! collapse schedule derived from the already database-multiplied query mask,
//! while [`PackingKeyPrecomputations`] contains only the user-dependent prepared
//! key bodies used by the online pass.

#![allow(clippy::too_many_arguments)]

use poulpy_core::EncryptionInfos;
use poulpy_core::layouts::{
    GGLWECompressedSeed, GGLWEInfos, GLWEAutomorphismKeyCompressed, GLWEInfos, GLWEToBackendMut,
    GetGaloisElement, LWESecretToBackendRef, compressed::GGLWECompressedToBackendRef,
};
use poulpy_hal::layouts::{
    Backend, ScratchArena, VecZnxToBackendMut, VecZnxToBackendRef, ZnxInfos,
};
use poulpy_hal::source::Source;

use crate::packing::{
    PackingKeys,
    packing_precomputations::{PackingPrecomputations, PackingPrecomputeInfos},
};

/// Aggregates a DB-multiplied LWE mask matrix into the mask layout consumed by packing.
pub trait PackingMaskAggregation<BE: Backend> {
    /// Scratch estimate for [`PackingMaskAggregation::packing_mask_preprocessing`].
    fn packing_mask_preprocessing_tmp_bytes(&self, size: usize) -> usize;

    /// Aggregates `a` into `dst`.
    ///
    /// `a` is the `n x n` LWE mask matrix produced after query expansion and
    /// database multiplication. `dst` receives the `n` aggregate mask columns
    /// used by [`Packing::pack_precompute`].
    fn packing_mask_preprocessing<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos;

    /// Like [`PackingMaskAggregation::packing_mask_preprocessing`], but the
    /// `n/2`-leaf work loop runs across `intra_threads` threads. Bit-identical to
    /// the sequential path (the per-leaf arithmetic is independent). Implemented
    /// only for `Vec<u8>`-buffer backends.
    fn packing_mask_preprocessing_threaded<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        a: &A,
        intra_threads: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos;

    /// Scratch estimate for partial packing-mask aggregation.
    fn pack_partial_mask_preprocessing_tmp_bytes(&self, gamma: usize, size: usize) -> usize;

    /// Aggregates the first `gamma` LWE mask rows into the partial-packing mask
    /// layout consumed by [`Packing::pack_partial_precompute`].
    fn packing_partial_mask_preprocessing<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        gamma: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos;

    /// Like [`PackingMaskAggregation::packing_partial_mask_preprocessing`], but the
    /// `γ`-component work loop runs across `intra_threads` threads. Bit-identical
    /// to the sequential path. Implemented only for `Vec<u8>`-buffer backends.
    fn packing_partial_mask_preprocessing_threaded<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        gamma: usize,
        a: &A,
        intra_threads: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos;
}

/// Deep-batched BSGS DFT-hot packing.
///
/// The trait exposes the full front-end packing flow:
/// - [`Packing::pack_keys_precompute`] derives user-key-side body material.
/// - [`Packing::pack_precompute_alloc`] allocates fixed mask-side storage.
/// - [`Packing::pack_precompute`] fills all fixed mask-side precomputations
///   from the DB-multiplied aggregate mask and a compressed key seed.
/// - [`Packing::pack`] consumes those fixed precomputations plus separate
///   user-key-side precomputations to pack one query body.
pub trait Packing<BE: Backend> {
    /// Scratch estimate for [`Packing::pack_keys_precompute`].
    fn pack_keys_precompute_tmp_bytes<KG, KH>(
        &self,
        key_g: &KG,
        key_h: &KH,
        baby_size: usize,
    ) -> usize
    where
        KG: GGLWEInfos,
        KH: GGLWEInfos;

    /// Builds all user-key-side body material used by packing.
    ///
    /// This prepares the body columns stored in the compressed automorphism keys
    /// received by the server. The seed-derived mask columns are intentionally
    /// handled by [`Packing::pack_precompute`] instead.
    fn pack_keys_precompute<KG, KH>(
        &self,
        key_g: &KG,
        key_h: &KH,
        baby_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> PackingKeys<BE>
    where
        KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
        KH: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement;

    /// Builds user-key-side body material for partial packing.
    ///
    /// Partial packing (InsPIRe² Algorithm 2) uses a single `key_g`, generated
    /// for `galois_element(stride)`, and no `key_h`.
    fn pack_partial_keys_precompute<KG>(
        &self,
        key_g: &KG,
        stride: usize,
        baby_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> PackingKeys<BE>
    where
        KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement;

    /// Allocates fixed mask-side precomputation storage.
    ///
    /// This is the public entry point for creating [`PackingPrecomputations`];
    /// callers should not construct the PIR-local layout through free helper
    /// functions.
    fn pack_precompute_alloc(
        &self,
        steps: usize,
        size: usize,
        base2k: usize,
        baby_size: usize,
    ) -> PackingPrecomputations<BE>;

    /// Allocates fixed mask-side storage for partial packing.
    fn pack_partial_precompute_alloc(
        &self,
        steps: usize,
        size: usize,
        base2k: usize,
        baby_size: usize,
        stride: usize,
    ) -> PackingPrecomputations<BE>;

    /// Scratch estimate for [`Packing::pack_precompute`].
    fn pack_precompute_tmp_bytes<A, KMask>(
        &self,
        metadata: PackingPrecomputeInfos,
        aggregate_mask: &A,
        key_mask_source: &KMask,
    ) -> usize
    where
        A: ZnxInfos,
        KMask: GGLWEInfos;

    /// Fills all fixed mask-side precomputations.
    ///
    /// `aggregate_mask` is the already DB-multiplied/aggregated query mask
    /// (for PIR, the aggregation of `U * A`, not the direct expanded query LWE
    /// mask). Internally this records the coefficient-domain mask schedule,
    /// computes the final result mask, and derives the DFT-hot BSGS mask
    /// columns and giant-step plans used by [`Packing::pack`]. It does not
    /// consume client-key-side body material.
    fn pack_precompute<A, KMask>(
        &self,
        precomputations: &mut PackingPrecomputations<BE>,
        aggregate_mask: &A,
        key_mask_source: &KMask,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: VecZnxToBackendRef<BE> + ZnxInfos,
        KMask: GGLWECompressedSeed + GGLWEInfos;

    /// Fills fixed mask-side precomputations for partial packing.
    fn pack_partial_precompute<A, KMask>(
        &self,
        precomputations: &mut PackingPrecomputations<BE>,
        aggregate_mask: &A,
        key_mask_source: &KMask,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: VecZnxToBackendRef<BE> + ZnxInfos,
        KMask: GGLWECompressedSeed + GGLWEInfos;

    /// Packs `body` with precomputed fixed masks and client-key material.
    ///
    /// `precomputations` must already contain the coefficient-domain mask
    /// schedule and the DFT-hot BSGS mask cache produced by
    /// [`Packing::pack_precompute`]. `key_precomputations` supplies the
    /// client-key-side prepared bodies and is intentionally passed separately.
    ///
    /// `chunk_size` controls how many giant-step groups are processed together
    /// while reusing baby keys; larger chunks spend more scratch to improve key
    /// reuse in the online loop.
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
        B: VecZnxToBackendRef<BE> + ZnxInfos;
}

/// Client-side generation of the two compressed automorphism keys consumed by
/// packing.
///
/// The server side never generates these. A client holding its raw LWE secret
/// calls this to produce the `(key_g, key_h)` pair with exactly the galois
/// elements and shared mask seed that the server-side
/// [`Packing::pack_keys_precompute`] / [`Packing::pack_precompute`] path
/// requires:
/// - `key_g` is generated with `p = galois_element_inv(galois_element(1))`,
/// - `key_h` is generated with `p = -1`,
/// - both use `key_seed` as their common mask seed.
///
/// These are the invariants that [`Packing::pack_keys_precompute`] otherwise
/// only checks defensively via assertions; generating the keys through this
/// helper makes them impossible to get wrong. The LWE secret is wrapped into
/// the rank-1 GLWE polynomial key (`sk_base`) the natural automorphism keys are
/// signed under. The returned keys are compressed (seed-only mask), ready to be
/// sent to the server.
pub trait PackingKeysGenerate<BE: Backend> {
    /// Scratch estimate for [`PackingKeysGenerate::pack_keys_generate`].
    fn pack_keys_generate_tmp_bytes<E>(&self, key_infos: &E) -> usize
    where
        E: GGLWEInfos;

    /// Encrypts the `(key_g, key_h)` packing automorphism keys under `sk_lwe`.
    ///
    /// `key_infos` describes the shared automorphism-key layout, `key_seed` is
    /// the public mask seed shared by both keys, and `source_xe` supplies the
    /// encryption noise.
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
        S: LWESecretToBackendRef<BE>;

    /// Encrypts the single partial-packing automorphism key `K_{g_γ}` under
    /// `sk_lwe`, for the order-γ generator `g_γ = 5^stride` (stride `= (d/2)/γ`).
    /// Partial packing (Algorithm 2) uses only this `key_g` and no `key_h`.
    fn pack_partial_key_generate<E, S>(
        &self,
        key_infos: &E,
        sk_lwe: &S,
        key_seed: [u8; 32],
        stride: usize,
        source_xe: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> GLWEAutomorphismKeyCompressed<Vec<u8>>
    where
        E: EncryptionInfos + GGLWEInfos,
        S: LWESecretToBackendRef<BE>;
}
