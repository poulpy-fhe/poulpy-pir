//! InsPIRe (interpolation) server collapse.
//!
//! The work is split by phase across themed submodules:
//!
//! * [`setup`] — server construction and per-block-column query-mask generation.
//! * [`offline`] — interpolate the DB into the matrix DB, then the per-panel
//!   mask-side packing precomputations.
//! * [`online`] — the per-query body product, pack, and Horner reduction.

mod offline;
mod online;
mod setup;

use poulpy_hal::layouts::Backend;

use crate::{
    database::Database,
    interpolation::Interpolation,
    packing::PackingPrecomputations,
    payload::Payload,
    server::common::{PreparedF64, QueryMask},
};

/// Collapse-specific interpolation state kept outside the common [`Server`]
/// fields.
///
/// [`Server`]: crate::server::Server
pub(crate) struct InterpolationState<BE: Backend, P: Payload<[u8; 32]>> {
    interpolation: Interpolation,
    /// Interpolated `U` matrices (`interpolation_t` block-rows), rebuilt by `offline`.
    matrix: Database<BE, P>,
}

/// Query-independent interpolation precomputation.
pub struct InterpolationPrecomputation<BE: Backend> {
    /// Query masks `A`, one per block-column (from `generate_query_mask`),
    /// pre-decoded to their `f64` working representation.
    pub(crate) masks: Vec<QueryMask>,
    /// `f64`-decoded `U` panels (`interpolation_t × block_cols`), from `offline`;
    /// reused for the offline mask product and the online body product.
    pub(crate) prepared_u: Vec<Vec<PreparedF64<'static>>>,
    /// Fixed mask-side packing precomputations (one per panel), from `offline`.
    pub(crate) precomputations: Vec<PackingPrecomputations<BE>>,
}

impl<BE: Backend> Default for InterpolationPrecomputation<BE> {
    fn default() -> Self {
        Self {
            masks: Vec::new(),
            prepared_u: Vec::new(),
            precomputations: Vec::new(),
        }
    }
}
