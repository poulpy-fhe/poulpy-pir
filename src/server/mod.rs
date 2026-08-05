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

use std::sync::Arc;
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

pub(crate) mod api;
mod common;
mod default;
mod delegates;
pub mod gemm;
mod interpolation;
mod oep;
mod recursion;

use api::{InterpolationServerModule, RecursionServerModule};
#[cfg(feature = "cblas-gemm")]
pub use gemm::CblasDgemm;
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

    /// Fold another breakdown into this one, summing like-named phases (and the
    /// typed buckets). Used to aggregate a batch's per-query timings into one
    /// report so `phases()`/`total()` reflect the whole batch.
    pub(crate) fn accumulate(&mut self, other: &OnlineTimings) {
        for p in &other.phases {
            match self.phases.iter_mut().find(|e| e.name == p.name) {
                Some(existing) => existing.duration += p.duration,
                None => self.phases.push(*p),
            }
        }
        self.key_precompute += other.key_precompute;
        self.prepare_db += other.prepare_db;
        self.mask_product += other.mask_product;
        self.body_product += other.body_product;
        self.mask_prep += other.mask_prep;
        self.pack_precompute += other.pack_precompute;
        self.pack += other.pack;
        self.decompose += other.decompose;
        self.reduce_precompute += other.reduce_precompute;
        self.reduce += other.reduce;
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

pub(crate) enum ServerCollapse<BE: Backend, P: Payload> {
    Interpolation(InterpolationState<BE, P>),
    /// Behind an `Arc` so a [`PrecompContext`] can share the (immutable after
    /// SETUP) CRS masks and key masks without copying them.
    Recursion(Arc<RecursionState<BE>>),
}

/// Query-independent state generated by [`Server::offline`].
///
/// It is the mask side of one specific database, so it is only meaningful
/// paired with that database — see [`Server::install`], which swaps the two in
/// together.
pub enum ServerPrecomputation<BE: Backend> {
    Interpolation(InterpolationPrecomputation<BE>),
    Recursion(RecursionPrecomputation<BE>),
}

/// Byte counts for the server's large allocations, from
/// [`Server::memory_report`].
///
/// These are the allocations that actually scale — the ones worth watching when
/// sizing a deployment. Small fixed state (CRS masks, key masks, the per-server
/// scratch) is not counted, so `total()` is a lower bound on RSS, not an
/// accounting of it.
#[derive(Clone, Copy, Debug, Default)]
pub struct ServerMemory {
    /// The encoded coefficient database.
    pub database: usize,
    /// The query-independent precomputation — the mask side of `database`.
    pub precomputation: usize,
    /// Persistent per-worker online scratch arenas, sized by the offline worker
    /// count. Paid once per server, independent of database size.
    pub online_scratch_pool: usize,
}

impl ServerMemory {
    pub fn total(&self) -> usize {
        self.database + self.precomputation + self.online_scratch_pool
    }
}

/// A detached worker that runs the query-independent OFFLINE precomputation
/// against a database the live [`Server`] is *not* serving from (InsPIRe² only).
///
/// It shares the server's parameters, CRS masks and GEMM backend by `Arc` and
/// borrows nothing from it, so the multi-second precomputation runs fully
/// concurrently with [`Server::respond`] — only the final [`Server::install`]
/// needs exclusive access, and that is two moves.
///
/// It owns neither a database nor an online scratch pool: the caller's staging
/// database is swapped in for the duration of each
/// [`offline_for`](Self::offline_for) call, and the multi-GiB online pool is
/// pointless for a worker that answers no queries. Refresh loop:
///
/// ```text
///   let mut ctx = server.lock().precomp_context();      // brief lock
///   let (precomp, _) = ctx.offline_for(&mut staging);   // no lock held
///   staging = server.lock().install(staging, precomp);  // brief lock, atomic pair
/// ```
pub struct PrecompContext<BE: Backend, P: Payload> {
    inner: Server<BE, P>,
}

/// PIR server that hosts either construction, chosen by `params.collapse()` in
/// [`Server::new`]. Common state is stored directly on the server; only
/// collapse-specific strategy/key/precomputation state is held behind field
/// enums.
pub struct Server<BE: Backend, P: Payload> {
    /// Shared with any detached [`PrecompContext`]; immutable after SETUP.
    params: Arc<Parameters<BE, P>>,
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
    gemm: Arc<dyn Gemm>,
    /// Cached pack-scratch arena size (recursion only; see `scratch_for_pack`).
    /// Sizing is a pure function of the fixed parameters but *expensive* to
    /// compute (it builds a probe layout and queries the backend's tmp-bytes
    /// planners, ~30-60 ms) and was being recomputed twice per online query —
    /// measured as most of the ONLINE wall-clock vs sum-of-phases gap.
    /// `OnceLock` because online workers call it through `&self`.
    pack_scratch_bytes: std::sync::OnceLock<usize>,
}

impl<BE: Backend, P: Payload> Server<BE, P> {
    /// The shared cryptosystem parameters (used, e.g., to size a received
    /// [`Query`] in [`Query::read_from`]).
    pub fn params(&self) -> &Parameters<BE, P> {
        &self.params
    }

    /// Replaces the GEMM backend used for the full-torus `f64` mask and body
    /// products with a custom [`Gemm`] implementation, returning the server for
    /// chaining. The default is [`PrivateGemmX86`]; this is the customization
    /// point for a different SIMD library, a GPU offload, etc.
    pub fn with_gemm(mut self, gemm: impl Gemm + 'static) -> Self {
        self.gemm = Arc::new(gemm);
        self
    }

    /// The active GEMM backend, as a `&dyn Gemm` for threading into the
    /// product helpers.
    pub(crate) fn gemm(&self) -> &dyn Gemm {
        &*self.gemm
    }
}

#[allow(private_bounds)]
impl<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload> Server<BE, P>
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
    pub fn new(config: Config<P>, layout: DatabaseLayout<P>) -> Self {
        Self::from_params(config.new::<BE>(), layout)
    }

    /// Compatibility/internal constructor for call sites that already own
    /// instantiated parameters.
    pub fn from_params(params: Parameters<BE, P>, layout: DatabaseLayout<P>) -> Self {
        crate::parallel::tune_allocator();
        match params.collapse() {
            Collapse::Interpolation => Self::new_interpolation(params, layout),
            Collapse::Recursion { .. } => Self::new_recursion(params, layout),
        }
    }

    /// Compatibility helper for interpolation (InsPIRe).
    pub fn interpolation(config: Config<P>, layout: DatabaseLayout<P>) -> Self {
        Self::new(config, layout)
    }

    /// Compatibility helper for InsPIRe².
    pub fn recursion(config: Config<P>, layout: DatabaseLayout<P>) -> Self {
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
    pub fn update_shard(&mut self, start: usize, values: &[P::Block]) {
        self.database.encode_shard(start, values);
    }

    /// Allocate an empty database matching this server's shape, for use as a
    /// staging buffer with [`PrecompContext::offline_for`] / [`install`].
    ///
    /// [`install`]: Self::install
    pub fn new_database(&self) -> Database<BE, P> {
        self.layout.instantiate(
            self.params.module(),
            self.params.base2k(),
            self.params.column_height(),
        )
    }

    /// Detach a [`PrecompContext`] — a precomputation worker that shares this
    /// server's parameters, CRS masks and GEMM backend but borrows nothing from
    /// it, so it can run while this server keeps answering queries.
    ///
    /// SETUP must have run first ([`generate_query_mask`](Self::generate_query_mask)):
    /// the context reuses the materialized CRS masks rather than expanding its
    /// own. InsPIRe² only.
    pub fn precomp_context(&self) -> PrecompContext<BE, P> {
        let ServerCollapse::Recursion(state) = &self.collapse else {
            panic!("PrecompContext is InsPIRe²-only (Collapse::Recursion)");
        };
        assert!(
            !state.q0_masks.is_empty() && !state.q1_masks.is_empty(),
            "call Server::generate_query_mask() before detaching a PrecompContext"
        );
        // Carry the (expensive to derive) pack-scratch sizing across so the
        // context does not recompute it on its first run.
        let pack_scratch_bytes = std::sync::OnceLock::new();
        let _ = pack_scratch_bytes.set(self.scratch_for_pack());
        PrecompContext {
            inner: Server {
                params: self.params.clone(),
                layout: self.layout,
                server_seed: self.server_seed,
                // No database of its own — the caller's staging buffer is
                // swapped in per call, so this only has to be a well-formed
                // empty shell.
                database: Database::placeholder(
                    self.params.n(),
                    self.layout,
                    self.params.base2k(),
                    self.params.column_height(),
                ),
                // The InsPIRe² paths allocate their own scratch per parallel
                // region and never touch `Server::scratch`.
                scratch: ScratchOwned::<BE>::alloc(0),
                // Never warmed: this worker answers no queries.
                scratch_pool: Vec::new(),
                collapse: ServerCollapse::Recursion(state.clone()),
                precomputation: ServerPrecomputation::Recursion(RecursionPrecomputation::default()),
                gemm: self.gemm.clone(),
                pack_scratch_bytes,
            },
        }
    }

    /// Swap in a new database together with the precomputation that matches it,
    /// returning the retired database so it can be reused as the next staging
    /// buffer.
    ///
    /// The two **must** correspond: the precomputation is the mask side of one
    /// specific database, so installing one without the other silently serves
    /// wrong responses. Taking `&mut self` is what makes the pair swap atomic
    /// with respect to [`respond`](Self::respond).
    ///
    /// `database` is swapped, not moved: on return it holds the retired
    /// database, ready to be refilled as the next staging buffer.
    pub fn install(
        &mut self,
        database: &mut Database<BE, P>,
        precomputation: ServerPrecomputation<BE>,
    ) {
        self.precomputation = precomputation;
        std::mem::swap(&mut self.database, database);
    }

    /// Ground-truth payload at index `i` from the server's own plaintext DB.
    pub fn get(&self, i: usize) -> P::Block {
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
    /// For **recursion** (InsPIRe²) the level-1 body select `D·b0` is likewise a
    /// single batched GEMM — the plaintext DB is streamed once per chunk for all
    /// queries — and the per-query FHE tail (`resp0` pack, decompose, `resp1`,
    /// `resp2`) runs across the worker pool. Measured on a 64-core Granite
    /// Rapids host at the 2 GiB shape: 8.7 queries/s unbatched, 33.7 at batch 64,
    /// saturating near 40 at batch 128. Latency is the whole batch's wall time,
    /// so batch size is a throughput/latency dial, not a free win.
    ///
    /// All queries in the batch must use the same construction as the server;
    /// passing a query of the other variant panics.
    pub fn respond_batch(&mut self, queries: &[Query<BE>]) -> Vec<Response<BE>> {
        self.respond_batch_timed(queries).0
    }

    /// ONLINE (batched) with an aggregated per-phase timing breakdown summed over
    /// the whole batch (like [`respond_timed`](Self::respond_timed) but for a
    /// batch). Interpolation uses the batched i16×f64 GEMM path (each `U` panel
    /// read once for all queries); recursion has no batched fast path yet, so it
    /// answers each query sequentially and sums the per-query timings.
    pub fn respond_batch_timed(
        &mut self,
        queries: &[Query<BE>],
    ) -> (Vec<Response<BE>>, OnlineTimings) {
        if queries.is_empty() {
            return (Vec::new(), OnlineTimings::default());
        }
        let all_interpolation = queries.iter().all(|q| matches!(q, Query::Interpolation(_)));
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
        let all_recursion = queries.iter().all(|q| matches!(q, Query::Recursion(_)));
        if all_recursion {
            let rec: Vec<&RecursionQuery<BE>> = queries
                .iter()
                .map(|q| match q {
                    Query::Recursion(q) => q,
                    Query::Interpolation(_) => unreachable!("checked all-recursion above"),
                })
                .collect();
            return self.respond_recursion_batch(&rec);
        }
        // Mixed batch (both constructions): no batched fast path — answer one by one
        // and sum the per-query timings so the report still covers every step.
        let mut responses = Vec::with_capacity(queries.len());
        let mut timings = OnlineTimings::default();
        for q in queries {
            let (resp, t) = self.respond_timed(q);
            responses.push(resp);
            timings.accumulate(&t);
        }
        (responses, timings)
    }
}

#[allow(private_bounds)]
impl<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload> PrecompContext<BE, P>
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
    /// Allocate a staging database matching the originating server's shape.
    pub fn new_database(&self) -> Database<BE, P> {
        self.inner.new_database()
    }

    /// Run the query-independent precomputation against `db`.
    ///
    /// `db` is swapped in for the duration and swapped straight back out, so
    /// the caller keeps ownership of its staging buffer and the call leaves it
    /// unchanged. Pass the returned precomputation and the same `db` to
    /// [`Server::install`] — they only mean anything together.
    ///
    /// This is the phase that costs seconds, and it holds no borrow on the live
    /// server.
    pub fn offline_for(
        &mut self,
        db: &mut Database<BE, P>,
    ) -> (ServerPrecomputation<BE>, OfflineTimings) {
        std::mem::swap(&mut self.inner.database, db);
        let timings = self.inner.offline_recursion_precompute();
        std::mem::swap(&mut self.inner.database, db);
        let precomputation = std::mem::replace(
            &mut self.inner.precomputation,
            ServerPrecomputation::Recursion(RecursionPrecomputation::default()),
        );
        (precomputation, timings)
    }
}
