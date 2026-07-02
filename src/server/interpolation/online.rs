//! Per-query ONLINE work: the panel body product `U·b`, pack, and the Horner
//! reduction at the query's GGSW root.

use std::time::{Duration, Instant};

use poulpy_core::layouts::{
    GLWE, GLWECompressed, GLWEDecompress, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef,
    ModuleCoreAlloc,
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, VecZnxToBackendMut,
        VecZnxToBackendRef,
    },
};

use crate::{
    client::Response,
    interpolation::{InterpolationQuery, InterpolationResponse},
    packing::Packing,
    parallel::{assign_panels, num_threads, scoped_workers_pooled},
    payload::Payload,
    server::{
        Gemm, OnlineTimings, Server, ServerCollapse, ServerPrecomputation,
        api::InterpolationServerModule,
        common::{PreparedF64, full_torus_f64_body_product, full_torus_f64_body_product_batch},
        interpolation::setup::server_scratch_bytes,
    },
};

#[allow(private_bounds)]
impl<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload<[u8; 32]>> Server<BE, P>
where
    BE: poulpy_cpu_ref::reference::fft64::reim::ReimArith,
    Module<BE>: InterpolationServerModule<BE> + GLWEDecompress<Backend = BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    VecZnx<Vec<u8>>:
        VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + poulpy_hal::layouts::ZnxInfos,
    GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    /// ONLINE: answer `query`. Per panel computes `U·b`, packs it, then evaluates
    /// the interpolated polynomial at the query's GGSW root (Horner). Returns the
    /// [`Response`] and a per-step timing breakdown.
    pub(crate) fn respond_interpolation(
        &mut self,
        query: &InterpolationQuery<BE>,
    ) -> (Response<BE>, OnlineTimings) {
        let ServerPrecomputation::Interpolation(precomputation) = &self.precomputation else {
            panic!("interpolation respond requested for non-interpolation server");
        };
        assert!(
            !precomputation.precomputations.is_empty(),
            "call offline() before respond()"
        );
        let mut timings = OnlineTimings::default();
        let module = self.params.module();
        let glwe_pack = self.params.glwe_pack();
        let ServerCollapse::Interpolation(state) = &self.collapse else {
            panic!("interpolation respond requested for non-interpolation server");
        };
        let panels = state.interpolation.num_panels();

        let t = Instant::now();
        let key_precomputations = module.pack_keys_precompute(
            query.keys.key_g(),
            query.keys.key_h(),
            self.params.baby_size(),
            &mut self.scratch.borrow(),
        );
        timings.add_key_precompute("interpolation.key_precompute", t.elapsed());

        let body_size = self.params.size_at(self.params.base2k());
        let out_base2k = self.params.base2k();
        let body_base2k = self.params.matmul_base2k();
        let torus_bits = self.params.k();
        let k = self.layout.block_cols(self.params.n());
        let nthreads = num_threads(panels);

        // Lazily grow the persistent per-worker scratch pool (allocated once,
        // reused across queries — plan M2′) so the parallel region pays no
        // per-query allocation/first-touch fault.
        while self.scratch_pool.len() < nthreads {
            self.scratch_pool
                .push(ScratchOwned::<BE>::alloc(server_scratch_bytes(
                    &self.params,
                )));
        }

        // One panel per work item: GEMV body product + pack, fully independent
        // (own pooled scratch, read-only shared precomputes/keys). Output written
        // by panel index ⇒ bit-identical to the sequential order.
        type PanelOut<BE> = (Option<GLWE<<BE as Backend>::OwnedBuf>>, [Duration; 2]);
        let mut outputs: Vec<PanelOut<BE>> = (0..panels)
            .map(|_| (None, [Duration::default(); 2]))
            .collect();
        let work = assign_panels(panels, k, nthreads);

        let region = Instant::now();
        {
            let prepared_u = &precomputation.prepared_u;
            let precomputes = &precomputation.precomputations;
            let key_precomp = &key_precomputations;
            let blocks = &query.common.blocks;
            let glwe_pack_ref = &glwe_pack;
            // Borrow the `gemm` field directly (not via `self.gemm()`) so it stays
            // disjoint from the `&mut self.scratch_pool` taken just below.
            let gemm: &dyn Gemm = &*self.gemm;

            let mut out_slabs: Vec<&mut [PanelOut<BE>]> = Vec::with_capacity(work.len());
            let mut rest = outputs.as_mut_slice();
            for group in &work {
                let (head, tail) = rest.split_at_mut(group.len());
                out_slabs.push(head);
                rest = tail;
            }
            let scratch_slabs: Vec<&mut ScratchOwned<BE>> =
                self.scratch_pool[..work.len()].iter_mut().collect();

            scoped_workers_pooled::<BE, PanelOut<BE>, _>(
                out_slabs,
                scratch_slabs,
                &work,
                |slab, group, sc| {
                    for (slot, w) in slab.iter_mut().zip(group.iter()) {
                        let panel = w.panel;
                        let mut body = module.vec_znx_alloc(1, body_size);
                        let t = Instant::now();
                        accumulate_body_product::<BE>(
                            &mut body,
                            out_base2k,
                            body_base2k,
                            torus_bits,
                            &prepared_u[panel],
                            blocks,
                            gemm,
                        );
                        let bp = t.elapsed();
                        let mut packed = module.glwe_alloc_from_infos(glwe_pack_ref);
                        let t = Instant::now();
                        module.pack(
                            &mut packed,
                            &body,
                            &precomputes[panel],
                            key_precomp,
                            1,
                            &mut sc.borrow(),
                        );
                        let pk = t.elapsed();
                        *slot = (Some(packed), [bp, pk]);
                    }
                },
            );
        }
        let region_wall = region.elapsed();

        let mut body_product = Duration::default();
        let mut pack = Duration::default();
        for (_, [bp, pk]) in &outputs {
            body_product += *bp;
            pack += *pk;
        }
        // Rescale phase CPU times to the parallel region's wall-clock so the
        // online breakdown sums to true wall time.
        if nthreads > 1 {
            let cpu = body_product + pack;
            if !cpu.is_zero() {
                let scale = region_wall.as_secs_f64() / cpu.as_secs_f64();
                body_product = body_product.mul_f64(scale);
                pack = pack.mul_f64(scale);
            }
        }
        let packed_coeffs: Vec<GLWE<BE::OwnedBuf>> =
            outputs.into_iter().map(|(p, _)| p.unwrap()).collect();
        timings.add_body_product("interpolation.body_product", body_product);
        timings.add_pack("interpolation.pack", pack);

        let t = Instant::now();
        let root_prepared =
            state
                .interpolation
                .prepare_root(module, &query.root, &mut self.scratch);
        timings.add_reduce_precompute("interpolation.reduce_precompute", t.elapsed());
        let mut selected = module.glwe_alloc_from_infos(&glwe_pack);
        let t = Instant::now();
        state.interpolation.reduce(
            module,
            &packed_coeffs,
            &root_prepared,
            &mut selected,
            &mut self.scratch,
        );
        timings.add_reduce("interpolation.reduce", t.elapsed());
        (
            Response::Interpolation(InterpolationResponse::new(selected)),
            timings,
        )
    }

    /// ONLINE (batched): answer `nq` interpolation queries against the same
    /// database in one pass. The per-panel body product `U·[b^0 … b^{nq-1}]` runs
    /// as a single i16×f64 GEMM — each `U` panel read once for the whole batch
    /// (the win over `nq` separate memory-bound GEMVs). The pack and the per-query
    /// Horner reduction stay per-query (they can't be batched). Returns one
    /// [`Response`] per query, in input order — identical results to calling
    /// [`respond_interpolation`](Self::respond_interpolation) on each query.
    pub(crate) fn respond_interpolation_batch(
        &mut self,
        queries: &[&InterpolationQuery<BE>],
    ) -> (Vec<Response<BE>>, OnlineTimings) {
        let nq = queries.len();
        assert!(nq > 0, "empty interpolation batch");
        let mut timings = OnlineTimings::default();

        let ServerPrecomputation::Interpolation(precomputation) = &self.precomputation else {
            panic!("interpolation respond requested for non-interpolation server");
        };
        assert!(
            !precomputation.precomputations.is_empty(),
            "call offline() before respond()"
        );
        let module = self.params.module();
        let glwe_pack = self.params.glwe_pack();
        let ServerCollapse::Interpolation(state) = &self.collapse else {
            panic!("interpolation respond requested for non-interpolation server");
        };
        let panels = state.interpolation.num_panels();

        // Per-query key precompute (sequential; consumes `self.scratch`).
        let started = Instant::now();
        let key_precomputations: Vec<_> = queries
            .iter()
            .map(|query| {
                module.pack_keys_precompute(
                    query.keys.key_g(),
                    query.keys.key_h(),
                    self.params.baby_size(),
                    &mut self.scratch.borrow(),
                )
            })
            .collect();
        timings.add_key_precompute("interpolation.batch.key_precompute", started.elapsed());

        let body_size = self.params.size_at(self.params.base2k());
        let out_base2k = self.params.base2k();
        let body_base2k = self.params.matmul_base2k();
        let torus_bits = self.params.k();
        let k = self.layout.block_cols(self.params.n());
        let nthreads = num_threads(panels);

        while self.scratch_pool.len() < nthreads {
            self.scratch_pool
                .push(ScratchOwned::<BE>::alloc(server_scratch_bytes(
                    &self.params,
                )));
        }

        // One panel per work item: a batched body product (`nq` outputs) followed
        // by `nq` independent packs. Output is `Vec<GLWE>` of length `nq`,
        // panel-major; written by panel index ⇒ bit-identical to sequential order.
        type PanelOut<BE> = Option<Vec<GLWE<<BE as Backend>::OwnedBuf>>>;
        let mut outputs: Vec<PanelOut<BE>> = (0..panels).map(|_| None).collect();
        let work = assign_panels(panels, k, nthreads);

        let started = Instant::now();
        {
            let prepared_u = &precomputation.prepared_u;
            let precomputes = &precomputation.precomputations;
            let key_precomps = &key_precomputations;
            let glwe_pack_ref = &glwe_pack;
            // Field-level borrow (not `self.gemm()`) so it stays disjoint from the
            // `&mut self.scratch_pool` taken just below.
            let gemm: &dyn Gemm = &*self.gemm;
            // Shared `U` panels, per-query one-hot bodies.
            let blocks_per_query: Vec<&[GLWECompressed<BE::OwnedBuf>]> = queries
                .iter()
                .map(|qy| qy.common.blocks.as_slice())
                .collect();
            let blocks_per_query = &blocks_per_query;

            let mut out_slabs: Vec<&mut [PanelOut<BE>]> = Vec::with_capacity(work.len());
            let mut rest = outputs.as_mut_slice();
            for group in &work {
                let (head, tail) = rest.split_at_mut(group.len());
                out_slabs.push(head);
                rest = tail;
            }
            let scratch_slabs: Vec<&mut ScratchOwned<BE>> =
                self.scratch_pool[..work.len()].iter_mut().collect();

            scoped_workers_pooled::<BE, PanelOut<BE>, _>(
                out_slabs,
                scratch_slabs,
                &work,
                |slab, group, sc| {
                    for (slot, w) in slab.iter_mut().zip(group.iter()) {
                        let panel = w.panel;
                        let mut out_bodies: Vec<VecZnx<BE::OwnedBuf>> = (0..nq)
                            .map(|_| module.vec_znx_alloc(1, body_size))
                            .collect();
                        full_torus_f64_body_product_batch::<BE>(
                            &mut out_bodies,
                            out_base2k,
                            &prepared_u[panel],
                            blocks_per_query,
                            body_base2k,
                            torus_bits,
                            gemm,
                        );
                        let mut packed_q: Vec<GLWE<BE::OwnedBuf>> = Vec::with_capacity(nq);
                        for (qi, body) in out_bodies.iter().enumerate() {
                            let mut packed = module.glwe_alloc_from_infos(glwe_pack_ref);
                            module.pack(
                                &mut packed,
                                body,
                                &precomputes[panel],
                                &key_precomps[qi],
                                1,
                                &mut sc.borrow(),
                            );
                            packed_q.push(packed);
                        }
                        *slot = Some(packed_q);
                    }
                },
            );
        }

        timings.add_body_product("interpolation.batch.body_product+pack", started.elapsed());
        let started = Instant::now();

        // Transpose panel-major outputs → per-query packed-coefficient vectors.
        let mut per_query: Vec<Vec<GLWE<BE::OwnedBuf>>> =
            (0..nq).map(|_| Vec::with_capacity(panels)).collect();
        for panel_out in &mut outputs {
            let panel_vec = panel_out
                .take()
                .expect("panel worker did not fill its slot");
            for (qi, glwe) in panel_vec.into_iter().enumerate() {
                per_query[qi].push(glwe);
            }
        }

        // Per-query Horner reduction at each query's GGSW root.
        let mut responses = Vec::with_capacity(nq);
        for (qi, packed_coeffs) in per_query.into_iter().enumerate() {
            let root_prepared =
                state
                    .interpolation
                    .prepare_root(module, &queries[qi].root, &mut self.scratch);
            let mut selected = module.glwe_alloc_from_infos(&glwe_pack);
            state.interpolation.reduce(
                module,
                &packed_coeffs,
                &root_prepared,
                &mut selected,
                &mut self.scratch,
            );
            responses.push(Response::Interpolation(InterpolationResponse::new(
                selected,
            )));
        }
        timings.add_reduce("interpolation.batch.reduce", started.elapsed());
        (responses, timings)
    }
}

/// `out_body = sum_bc U[bc] · b[bc]` for one interpolation panel, via the dense
/// `f64` GEMV. Writes a small body `VecZnx` (online only needs the body, not the
/// mask) directly in the pack regime.
fn accumulate_body_product<BE>(
    out_body: &mut VecZnx<BE::OwnedBuf>,
    out_base2k: usize,
    body_base2k: usize,
    torus_bits: usize,
    u_panel: &[PreparedF64],
    bodies: &[GLWECompressed<BE::OwnedBuf>],
    gemm: &dyn Gemm,
) where
    BE: Backend<OwnedBuf = Vec<u8>> + poulpy_cpu_ref::reference::fft64::reim::ReimArith,
{
    full_torus_f64_body_product::<BE>(
        out_body,
        out_base2k,
        u_panel,
        bodies,
        body_base2k,
        torus_bits,
        gemm,
    );
}
