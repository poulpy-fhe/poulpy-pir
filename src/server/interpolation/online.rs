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
        common::{PreparedF64, full_torus_f64_body_product},
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
                .push(ScratchOwned::<BE>::alloc(server_scratch_bytes(&self.params)));
        }

        // One panel per work item: GEMV body product + pack, fully independent
        // (own pooled scratch, read-only shared precomputes/keys). Output written
        // by panel index ⇒ bit-identical to the sequential order.
        type PanelOut<BE> = (Option<GLWE<<BE as Backend>::OwnedBuf>>, [Duration; 2]);
        let mut outputs: Vec<PanelOut<BE>> =
            (0..panels).map(|_| (None, [Duration::default(); 2])).collect();
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
        out_body, out_base2k, u_panel, bodies, body_base2k, torus_bits, gemm,
    );
}
