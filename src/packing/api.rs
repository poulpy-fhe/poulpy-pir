//! User-facing packing trait on `Module<BE>`.
//!
//! The deep-batched, optionally group-chunked BSGS DFT-hot path separates fixed
//! mask-side state from user-key state. [`PackingPrecomputations`] contains the
//! collapse schedule derived from the already database-multiplied query mask,
//! while [`PackingKeyPrecomputations`] contains only the user-dependent prepared
//! key bodies used by the online pass.

#![allow(clippy::too_many_arguments)]

use poulpy_core::layouts::{
    GGLWECompressedSeed, GGLWEInfos, GLWEInfos, GLWEToBackendMut, GetGaloisElement,
    compressed::GGLWECompressedToBackendRef,
    prepared::{GGLWEPrepared, GGLWEPreparedVmpPMatRef},
};
use poulpy_hal::layouts::{
    Backend, ScratchArena, VecZnxToBackendMut, VecZnxToBackendRef, ZnxInfos,
};

use crate::packing::collapse_precompute::{PackingPrecomputations, PackingPrecomputeInfos};

/// Aggregates a DB-multiplied LWE mask matrix into the mask layout consumed by packing.
pub trait PackingMaskAggregation<BE: Backend> {
    /// Scratch estimate for [`PackingMaskAggregation::packing_mask_aggregate`].
    fn packing_mask_aggregate_tmp_bytes(&self, size: usize) -> usize;

    /// Aggregates `a` into `dst`.
    ///
    /// `a` is the `n x n` LWE mask matrix produced after query expansion and
    /// database multiplication. `dst` receives the `n` aggregate mask columns
    /// used by [`Packing::pack_precompute`].
    fn packing_mask_aggregate<R, A>(
        &self,
        dst: &mut R,
        base2k: usize,
        a: &A,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: VecZnxToBackendMut<BE> + ZnxInfos,
        A: VecZnxToBackendRef<BE> + ZnxInfos;
}

/// Accessor for client-key-side packing precomputations.
///
/// Packing only needs prepared body projections online: baby-step bodies for
/// `key_g` products and the final `key_h` body. Fixed key-mask material is
/// derived from a compressed key seed inside [`Packing::pack_precompute`].
pub trait PackingKeyPrecomputationsHelper<BE: Backend, K>
where
    K: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
{
    /// Prepared body key for baby step `idx`.
    ///
    /// Used by the online BSGS loop in `bsgs_pack` for every group that has a
    /// live baby step at this index.
    fn baby_key_g(&self, idx: usize) -> &K;

    /// Prepared body key for the final `key_h` product.
    ///
    /// Used once after all BSGS `key_g` groups have been accumulated.
    fn key_h(&self) -> &K;

    /// Output size of the key products, used to size online scratch buffers.
    fn key_size(&self) -> usize {
        self.key_h().size()
    }
}

/// Owned user-key-side precomputations used by packing.
///
/// This is the current concrete key-precompute container. It is produced by
/// [`Packing::pack_keys_precompute`] from the full `key_g`/`key_h` switching
/// keys and contains no seed-derived mask material.
pub struct PackingKeyPrecomputations<K> {
    /// Prepared baby-step `key_g` body keys indexed by baby step.
    baby_key_g_bodies: Vec<K>,
    /// Prepared final `key_h` body key.
    key_h_body: K,
}

impl<K> PackingKeyPrecomputations<K> {
    /// Creates owned user-key-side body precomputations.
    pub fn new(baby_key_g_bodies: Vec<K>, key_h_body: K) -> Self {
        Self {
            baby_key_g_bodies,
            key_h_body,
        }
    }
}

impl<BE, K> PackingKeyPrecomputationsHelper<BE, K> for PackingKeyPrecomputations<K>
where
    BE: Backend,
    K: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos,
{
    fn baby_key_g(&self, idx: usize) -> &K {
        &self.baby_key_g_bodies[idx]
    }

    fn key_h(&self) -> &K {
        &self.key_h_body
    }
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
    ) -> PackingKeyPrecomputations<GGLWEPrepared<BE::OwnedBuf, BE>>
    where
        KG: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement,
        KH: GGLWECompressedSeed + GGLWECompressedToBackendRef<BE> + GGLWEInfos + GetGaloisElement;

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
    fn pack<R, B, P, K>(
        &self,
        res: &mut R,
        body: &B,
        precomputations: &PackingPrecomputations<BE>,
        key_precomputations: &P,
        chunk_size: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        B: VecZnxToBackendRef<BE> + ZnxInfos,
        P: PackingKeyPrecomputationsHelper<BE, K>,
        K: GGLWEPreparedVmpPMatRef<BE> + GGLWEInfos;
}
