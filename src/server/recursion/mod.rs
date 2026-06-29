//! InsPIRe² (recursive partial-pack) server collapse.
//!
//! The work is split by phase across themed submodules:
//!
//! * [`keys`] / [`params`] — query/key types and parameter-derived helpers.
//! * [`setup`] — server construction and the fixed CRS query-mask expansion.
//! * [`offline`] — the `O(N·d)` query-independent mask-side precomputation.
//! * [`online`] — the light per-query body work in [`Server::respond`].
//! * [`packing`] — the shared mask/body partial-packing helpers and scratch.
//!
//! [`Server::respond`]: crate::server::Server::respond

mod keys;
mod offline;
mod online;
mod packing;
mod params;
mod setup;

#[cfg(test)]
mod tests;

use poulpy_core::{
    EncryptionLayout,
    layouts::{GLWEAutomorphismKeyCompressed, GLWELayout},
};
use poulpy_hal::layouts::Backend;
use std::time::Duration;

use crate::{
    packing::{PackingKeys, PackingPrecomputations},
    payload::Payload,
    server::{Server, ServerCollapse, common::PreparedF64, common::QueryMask},
};

pub use keys::RecursionQuery;
pub(crate) use keys::{CompressedKey, RecursionKeys, generate_recursion_key};
pub(crate) use params::{assert_params_valid, qtilde_bits, src_infos_for, tau};

/// Query-independent preprocessing produced by [`Server::offline`]: the
/// mask-side (`D·A0`, `D1·A1`) packing precomputes and the prepared database
/// matrices. This is the `O(N·d)` server-side work the construction amortizes; it
/// depends only on the database + CRS masks, never on a query. `respond` then does
/// only the light online body work.
///
/// [`Server::offline`]: crate::server::Server::offline
pub struct RecursionPrecomputation<BE: Backend> {
    /// Level-1 mask-side packing precomputes (`t`).
    l1_precompute: Vec<PackingPrecomputations<BE>>,
    /// `resp1` (mask-digit) `f64`-decoded `D1` matrices, reused for the online body GEMV.
    resp1_prep: Vec<Vec<PreparedF64<'static>>>,
    /// `resp1` mask-side packing precomputes.
    resp1_precompute: Vec<PackingPrecomputations<BE>>,
}

impl<BE: Backend> Default for RecursionPrecomputation<BE> {
    fn default() -> Self {
        Self {
            l1_precompute: Vec::new(),
            resp1_precompute: Vec::new(),
            resp1_prep: Vec::new(),
        }
    }
}

/// Server-side prepared partial-packing key: the received compressed key plus its
/// precomputed baby-step bodies.
struct KeyBundle<'a, BE: Backend> {
    key: &'a GLWEAutomorphismKeyCompressed<BE::OwnedBuf>,
    precomp: PackingKeys<BE>,
    stride: usize,
}

#[derive(Clone, Copy)]
struct RecursionOfflineShape {
    n: usize,
    size: usize,
    t: usize,
    gamma0: usize,
    gamma1: usize,
    base2k: usize,
    baby_size: usize,
    tau: usize,
}

#[derive(Clone, Copy)]
struct PackMaskPhaseNames {
    prepare_db: &'static str,
    mask_product: &'static str,
    mask_prep: &'static str,
    pack_precompute: &'static str,
}

#[derive(Default)]
struct PackMaskDurations {
    prepare_db: Duration,
    mask_product: Duration,
    mask_prep: Duration,
    pack_precompute: Duration,
}

/// Collapse-specific InsPIRe² state kept outside the common [`Server`] fields.
pub(crate) struct RecursionState<BE: Backend> {
    pub(crate) src_infos: EncryptionLayout<GLWELayout>,
    pub(crate) key0_mask: CompressedKey<BE>,
    pub(crate) key1_mask: CompressedKey<BE>,
    pub(crate) q0_masks: Vec<QueryMask>,
    pub(crate) q1_masks: Vec<QueryMask>,
}

impl<BE: Backend, P: Payload<[u8; 32]>> Server<BE, P> {
    fn recursion_state(&self) -> &RecursionState<BE> {
        let ServerCollapse::Recursion(state) = &self.collapse else {
            panic!("InsPIRe² state requested for non-InsPIRe² server");
        };
        state
    }

    fn recursion_state_mut(&mut self) -> &mut RecursionState<BE> {
        let ServerCollapse::Recursion(state) = &mut self.collapse else {
            panic!("InsPIRe² state requested for non-InsPIRe² server");
        };
        state
    }
}
