//! PIR server: owns the plaintext database and its interpolated matrix form,
//! materializes the query mask `A` from its public [`ServerSeed`], runs the
//! query-independent OFFLINE pre-processing, and answers a client [`Query`].
//!
//! Phases:
//! - SETUP — [`Server::generate_query_mask`]: materialize the fixed query masks
//!   from the public CRS seed. Depends only on the public seed + DB shape, so it
//!   is reused across both DB updates and queries.
//! - OFFLINE — [`Server::offline`]: interpolate the plaintext DB into the matrix
//!   DB, then per interpolation panel compute `U·A`, `packing_mask_preprocessing`
//!   and `pack_precompute`. Depends on DB content + masks, query-independent.
//! - ONLINE — [`Server::respond`]: per panel `U·b`, `pack`, then the Horner
//!   reduction at the query's GGSW root.
//!
//! Host backends only (`BE::OwnedBuf = Vec<u8>`).

#![allow(clippy::too_many_arguments)]

use std::time::Duration;

use poulpy_core::layouts::{
    GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEMatrix, LWEMatrixToBackendMut,
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAutomorphismPlan},
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, VecZnxToBackendMut,
        VecZnxToBackendRef,
    },
};

use crate::{
    client::{Response, ServerSeed},
    config::{Collapse, Config},
    database::{Database, DatabaseLayout},
    interpolation::InterpolationQuery,
    parameters::Parameters,
    payload::Payload,
};

mod api;
mod common;
mod default;
mod delegates;
pub mod gemm;
mod interpolation;
mod oep;
mod recursion;

use api::{InterpolationServerModule, RecursionServerModule};
pub use gemm::{Gemm, PrivateGemmX86};
pub use interpolation::InterpolationPrecomputation;
use interpolation::InterpolationState;
use recursion::RecursionState;
pub(crate) use recursion::{CompressedKey, RecursionKeys, generate_recursion_key, qtilde_bits};
pub use recursion::{RecursionPrecomputation, RecursionQuery};

/// One measured OFFLINE phase.
#[derive(Clone, Copy, Debug)]
pub struct OfflinePhaseTiming {
    name: &'static str,
    duration: Duration,
}

impl OfflinePhaseTiming {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn duration(&self) -> Duration {
        self.duration
    }
}

/// Per-step OFFLINE timing breakdown (query-independent pre-processing).
#[derive(Default, Clone, Debug)]
pub struct OfflineTimings {
    phases: Vec<OfflinePhaseTiming>,
    pub interpolation: Duration,
    pub prepare_u: Duration,
    pub ua_mask: Duration,
    pub mask_prep: Duration,
    pub pack_precompute: Duration,
}

impl OfflineTimings {
    pub fn phases(&self) -> &[OfflinePhaseTiming] {
        &self.phases
    }

    pub fn total(&self) -> Duration {
        if self.phases.is_empty() {
            return self.interpolation
                + self.prepare_u
                + self.ua_mask
                + self.mask_prep
                + self.pack_precompute;
        }
        self.phases
            .iter()
            .fold(Duration::default(), |sum, phase| sum + phase.duration)
    }

    pub(crate) fn record_phase(&mut self, name: &'static str, duration: Duration) {
        self.phases.push(OfflinePhaseTiming { name, duration });
    }

    pub(crate) fn add_interpolation(&mut self, name: &'static str, duration: Duration) {
        self.interpolation += duration;
        self.record_phase(name, duration);
    }

    pub(crate) fn add_prepare_u(&mut self, name: &'static str, duration: Duration) {
        self.prepare_u += duration;
        self.record_phase(name, duration);
    }

    pub(crate) fn add_ua_mask(&mut self, name: &'static str, duration: Duration) {
        self.ua_mask += duration;
        self.record_phase(name, duration);
    }

    pub(crate) fn add_mask_prep(&mut self, name: &'static str, duration: Duration) {
        self.mask_prep += duration;
        self.record_phase(name, duration);
    }

    pub(crate) fn add_pack_precompute(&mut self, name: &'static str, duration: Duration) {
        self.pack_precompute += duration;
        self.record_phase(name, duration);
    }
}

/// Per-step ONLINE timing breakdown (per query).
#[derive(Clone, Copy, Debug)]
pub struct OnlinePhaseTiming {
    name: &'static str,
    duration: Duration,
}

impl OnlinePhaseTiming {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn duration(&self) -> Duration {
        self.duration
    }
}

/// Per-step ONLINE timing breakdown (per query).
#[derive(Default, Clone, Debug)]
pub struct OnlineTimings {
    phases: Vec<OnlinePhaseTiming>,
    pub key_precompute: Duration,
    pub prepare_db: Duration,
    pub mask_product: Duration,
    pub body_product: Duration,
    pub mask_prep: Duration,
    pub pack_precompute: Duration,
    pub pack: Duration,
    pub decompose: Duration,
    pub reduce_precompute: Duration,
    pub reduce: Duration,
}

impl OnlineTimings {
    pub fn phases(&self) -> &[OnlinePhaseTiming] {
        &self.phases
    }

    pub fn total(&self) -> Duration {
        if !self.phases.is_empty() {
            return self
                .phases
                .iter()
                .fold(Duration::default(), |sum, phase| sum + phase.duration);
        }
        self.key_precompute
            + self.prepare_db
            + self.mask_product
            + self.body_product
            + self.mask_prep
            + self.pack_precompute
            + self.pack
            + self.decompose
            + self.reduce_precompute
            + self.reduce
    }

    pub(crate) fn record_phase(&mut self, name: &'static str, duration: Duration) {
        self.phases.push(OnlinePhaseTiming { name, duration });
    }

    pub(crate) fn add_key_precompute(&mut self, name: &'static str, duration: Duration) {
        self.key_precompute += duration;
        self.record_phase(name, duration);
    }

    pub(crate) fn add_prepare_db(&mut self, name: &'static str, duration: Duration) {
        self.prepare_db += duration;
        self.record_phase(name, duration);
    }

    pub(crate) fn add_mask_product(&mut self, name: &'static str, duration: Duration) {
        self.mask_product += duration;
        self.record_phase(name, duration);
    }

    pub(crate) fn add_body_product(&mut self, name: &'static str, duration: Duration) {
        self.body_product += duration;
        self.record_phase(name, duration);
    }

    pub(crate) fn add_mask_prep(&mut self, name: &'static str, duration: Duration) {
        self.mask_prep += duration;
        self.record_phase(name, duration);
    }

    pub(crate) fn add_pack_precompute(&mut self, name: &'static str, duration: Duration) {
        self.pack_precompute += duration;
        self.record_phase(name, duration);
    }

    pub(crate) fn add_pack(&mut self, name: &'static str, duration: Duration) {
        self.pack += duration;
        self.record_phase(name, duration);
    }

    pub(crate) fn add_decompose(&mut self, name: &'static str, duration: Duration) {
        self.decompose += duration;
        self.record_phase(name, duration);
    }

    pub(crate) fn add_reduce_precompute(&mut self, name: &'static str, duration: Duration) {
        self.reduce_precompute += duration;
        self.record_phase(name, duration);
    }

    pub(crate) fn add_reduce(&mut self, name: &'static str, duration: Duration) {
        self.reduce += duration;
        self.record_phase(name, duration);
    }
}

// =============================================================================
// Collapse-dispatching server (one type hosting both constructions).
// =============================================================================

/// A client query, one variant per second-dimension collapse — the input to
/// [`Server::respond`].
pub enum Query<BE: Backend> {
    Interpolation(InterpolationQuery<BE>),
    Recursion(RecursionQuery<BE>),
}

pub(crate) enum ServerCollapse<BE: Backend, P: Payload<[u8; 32]>> {
    Interpolation(InterpolationState<BE, P>),
    Recursion(RecursionState<BE>),
}

/// Query-independent state generated by [`Server::offline`].
pub enum ServerPrecomputation<BE: Backend> {
    Interpolation(InterpolationPrecomputation<BE>),
    Recursion(RecursionPrecomputation<BE>),
}

/// PIR server that hosts either construction, chosen by `params.collapse()` in
/// [`Server::new`]. Common state is stored directly on the server; only
/// collapse-specific strategy/key/precomputation state is held behind field
/// enums.
pub struct Server<BE: Backend, P: Payload<[u8; 32]>> {
    params: Parameters<BE, [u8; 32], P>,
    layout: DatabaseLayout<P>,
    server_seed: ServerSeed,
    database: Database<BE, P>,
    scratch: ScratchOwned<BE>,
    /// Persistent per-worker scratch arenas for the parallel online panel loop,
    /// allocated once (lazily, on first query) and reused — a fresh per-query
    /// `ScratchOwned::alloc` would fault a large arena in concurrently and swamp
    /// the memory-bound body product (plan M2′).
    scratch_pool: Vec<ScratchOwned<BE>>,
    collapse: ServerCollapse<BE, P>,
    precomputation: ServerPrecomputation<BE>,
    /// The GEMM backend driving the full-torus `f64` mask/body products. Defaults
    /// to [`PrivateGemmX86`]; swap it with [`Server::with_gemm`] to plug a custom
    /// kernel on top of the FHE backend `BE`.
    gemm: Box<dyn Gemm>,
}

impl<BE: Backend, P: Payload<[u8; 32]>> Server<BE, P> {
    /// The shared cryptosystem parameters (used, e.g., to size a received
    /// [`Query`] in [`Query::read_from`]).
    pub fn params(&self) -> &Parameters<BE, [u8; 32], P> {
        &self.params
    }

    /// Replaces the GEMM backend used for the full-torus `f64` mask and body
    /// products with a custom [`Gemm`] implementation, returning the server for
    /// chaining. The default is [`PrivateGemmX86`]; this is the customization
    /// point for a different SIMD library, a GPU offload, etc.
    pub fn with_gemm(mut self, gemm: impl Gemm + 'static) -> Self {
        self.gemm = Box::new(gemm);
        self
    }

    /// The active GEMM backend, as a `&dyn Gemm` for threading into the
    /// product helpers.
    pub(crate) fn gemm(&self) -> &dyn Gemm {
        &*self.gemm
    }
}

#[allow(private_bounds)]
impl<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload<[u8; 32]>> Server<BE, P>
where
    BE: poulpy_cpu_ref::reference::fft64::reim::ReimArith,
    Module<BE>: InterpolationServerModule<BE> + RecursionServerModule<BE>,
    <Module<BE> as VecZnxDftAutomorphismPlan<BE>>::Plan: 'static + Send + Sync,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    VecZnx<Vec<u8>>:
        VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + poulpy_hal::layouts::ZnxInfos,
    LWEMatrix<Vec<u8>>: LWEMatrixToBackendMut<BE>,
    GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    /// Build the PIR server from a config and database layout. Parameters are
    /// instantiated internally and the construction is selected by
    /// [`Parameters::collapse`].
    pub fn new(config: Config<[u8; 32], P>, layout: DatabaseLayout<P>) -> Self {
        Self::from_params(config.new::<BE>(), layout)
    }

    /// Compatibility/internal constructor for call sites that already own
    /// instantiated parameters.
    pub fn from_params(params: Parameters<BE, [u8; 32], P>, layout: DatabaseLayout<P>) -> Self {
        match params.collapse() {
            Collapse::Interpolation => Self::new_interpolation(params, layout),
            Collapse::Recursion { .. } => Self::new_recursion(params, layout),
        }
    }

    /// Compatibility helper for interpolation (InsPIRe).
    pub fn interpolation(config: Config<[u8; 32], P>, layout: DatabaseLayout<P>) -> Self {
        Self::new(config, layout)
    }

    /// Compatibility helper for InsPIRe².
    pub fn recursion(config: Config<[u8; 32], P>, layout: DatabaseLayout<P>) -> Self {
        Self::new(config, layout)
    }

    /// The public CRS seed. Generic clients currently derive the same default
    /// seed internally, but this remains useful for lower-level APIs.
    pub fn server_seed(&self) -> ServerSeed {
        self.server_seed
    }

    /// The database layout, shared by both constructions.
    pub fn layout(&self) -> DatabaseLayout<P> {
        self.layout
    }

    /// The owned plaintext database.
    pub fn database(&self) -> &Database<BE, P> {
        &self.database
    }

    /// Mutable access to the owned plaintext database.
    pub fn database_mut(&mut self) -> &mut Database<BE, P> {
        &mut self.database
    }

    /// Loads records directly into the owned database.
    pub fn encode(&mut self, records: &[Vec<i64>]) {
        self.database.encode(records);
    }

    /// SETUP: materialize the fixed query masks for the selected construction.
    pub fn generate_query_mask(&mut self) {
        match &self.collapse {
            ServerCollapse::Interpolation(_) => self.generate_interpolation_query_mask(),
            ServerCollapse::Recursion(_) => self.generate_recursion_query_mask(),
        }
    }

    /// OFFLINE (server-side) pre-processing: the query-independent `O(N·d)` work.
    /// Returns an ordered phase timing breakdown for the selected construction.
    pub fn offline(&mut self) -> OfflineTimings {
        match &self.collapse {
            ServerCollapse::Interpolation(_) => self.offline_interpolation(),
            ServerCollapse::Recursion(_) => self.offline_recursion(),
        }
    }

    /// Bulk-write payloads from index `start` using the database's preprocessing
    /// layout (`n` for interpolation, `gamma0` for InsPIRe²).
    pub fn update_shard(&mut self, start: usize, values: &[[u8; 32]]) {
        self.database.encode_shard(start, values);
    }

    /// Ground-truth payload at index `i` from the server's own plaintext DB.
    pub fn get(&self, i: usize) -> [u8; 32] {
        self.database.payload(i)
    }

    /// ONLINE: answer a query, dispatching on its collapse variant.
    pub fn respond(&mut self, query: &Query<BE>) -> Response<BE> {
        self.respond_timed(query).0
    }

    /// ONLINE: answer a query and return an ordered timing breakdown.
    pub fn respond_timed(&mut self, query: &Query<BE>) -> (Response<BE>, OnlineTimings) {
        match query {
            Query::Interpolation(q) => self.respond_interpolation(q),
            Query::Recursion(q) => self.respond_recursion(q),
        }
    }

    /// ONLINE (batched): answer a batch of queries against the same database,
    /// returning one [`Response`] per query in input order.
    ///
    /// For **interpolation** (InsPIRe) the per-panel body product is computed as a
    /// single i16×f64 GEMM over the whole batch — each database panel is read once
    /// for all queries (the win over `respond`-per-query), while the pack and
    /// Horner reduction remain per-query. Results are identical to calling
    /// [`respond`](Self::respond) on each query individually.
    ///
    /// For **recursion** (InsPIRe²) the multi-level online pipeline is not yet
    /// batch-accelerated, so this falls back to answering each query sequentially
    /// (correct, same result, no speedup).
    ///
    /// All queries in the batch must use the same construction as the server;
    /// passing a query of the other variant panics.
    pub fn respond_batch(&mut self, queries: &[Query<BE>]) -> Vec<Response<BE>> {
        if queries.is_empty() {
            return Vec::new();
        }
        let all_interpolation = queries
            .iter()
            .all(|q| matches!(q, Query::Interpolation(_)));
        if all_interpolation {
            let interp: Vec<&InterpolationQuery<BE>> = queries
                .iter()
                .map(|q| match q {
                    Query::Interpolation(q) => q,
                    Query::Recursion(_) => unreachable!("checked all-interpolation above"),
                })
                .collect();
            return self.respond_interpolation_batch(&interp);
        }
        // Recursion (or a mixed batch): no batched fast path yet — answer one by
        // one. `respond` panics on a query that mismatches the server construction.
        queries.iter().map(|q| self.respond(q)).collect()
    }
}
