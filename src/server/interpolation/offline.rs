//! Interpolate the plaintext DB into the matrix DB, then run each
//! query-independent packing precomputation phase (OFFLINE).

use std::time::{Duration, Instant};

use poulpy_core::layouts::{
    Degree, GLWE, GLWEAutomorphismKeyCompressed, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef,
    LWEInfos, LWEMatrix, LWEMatrixLayout, LWEMatrixToBackendMut, ModuleCoreAlloc,
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchArena, ScratchOwned, VecZnx,
        VecZnxToBackendMut, VecZnxToBackendRef, ZnxInfos,
    },
    source::Source,
};

use crate::{
    packing::{Packing, PackingKeysGenerate, PackingMaskAggregation, PackingPrecomputeInfos,
        PackingPrecomputations},
    parallel::{assign_panels, num_threads, scoped_workers},
    payload::Payload,
    server::{
        OfflineTimings, Server, ServerCollapse, ServerPrecomputation,
        api::InterpolationServerModule,
        common::{PreparedF64, QueryMask, mask_product_to_pack},
        interpolation::setup::server_scratch_bytes,
    },
};

#[allow(private_bounds)]
impl<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload<[u8; 32]>> Server<BE, P>
where
    BE: poulpy_cpu_ref::reference::fft64::reim::ReimArith,
    Module<BE>: InterpolationServerModule<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    VecZnx<Vec<u8>>:
        VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + poulpy_hal::layouts::ZnxInfos,
    LWEMatrix<Vec<u8>>: LWEMatrixToBackendMut<BE>,
    GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    /// OFFLINE: interpolate the plaintext DB into the matrix DB, then run each
    /// query-independent packing precomputation phase in sequence. Re-run after
    /// DB updates.
    pub(crate) fn offline_interpolation(&mut self) -> OfflineTimings {
        let mut timings = OfflineTimings::default();

        if let Some(duration) = self.offline_interpolation_query_mask() {
            timings.record_phase("interpolation.query_mask", duration);
        }
        timings.add_interpolation(
            "interpolation.interpolate",
            self.offline_interpolation_interpolate(),
        );
        timings.add_prepare_u(
            "interpolation.prepare_u",
            self.offline_interpolation_prepare_u(),
        );

        let t = Instant::now();
        let key_mask_src = self.interpolation_pack_key_mask();
        timings.record_phase("interpolation.pack_key_mask", t.elapsed());

        let (ua_mask, mask_prep, pack_precompute) =
            self.offline_interpolation_pack_precomputations(&key_mask_src);
        timings.add_ua_mask("interpolation.ua_mask", ua_mask);
        timings.add_mask_prep("interpolation.mask_prep", mask_prep);
        timings.add_pack_precompute("interpolation.pack_precompute", pack_precompute);

        // Warm the online per-worker scratch pool here (OFFLINE), so the first
        // query doesn't pay the per-worker arena allocation (plan M2′).
        let panels = match &self.collapse {
            ServerCollapse::Interpolation(state) => state.interpolation.num_panels(),
            ServerCollapse::Recursion(_) => unreachable!(),
        };
        let nthreads = num_threads(panels);
        while self.scratch_pool.len() < nthreads {
            self.scratch_pool
                .push(ScratchOwned::<BE>::alloc(server_scratch_bytes(&self.params)));
        }

        timings
    }

    fn offline_interpolation_query_mask(&mut self) -> Option<Duration> {
        let masks_empty = match &self.precomputation {
            ServerPrecomputation::Interpolation(precomputation) => precomputation.masks.is_empty(),
            ServerPrecomputation::Recursion(_) => {
                panic!("interpolation offline requested for non-interpolation server")
            }
        };
        if masks_empty {
            let t = Instant::now();
            self.generate_interpolation_query_mask();
            return Some(t.elapsed());
        }
        None
    }

    fn offline_interpolation_interpolate(&mut self) -> Duration {
        let encoder = self.params.encoder();
        let t = Instant::now();
        let ServerCollapse::Interpolation(state) = &mut self.collapse else {
            panic!("interpolation offline requested for non-interpolation server");
        };
        state.interpolation.interpolate_into(
            self.params.module(),
            &self.database,
            &mut state.matrix,
            &encoder,
            &mut self.scratch,
        );
        t.elapsed()
    }

    fn offline_interpolation_prepare_u(&mut self) -> Duration {
        let _module = self.params.module();
        let block_cols = self.layout.block_cols(self.params.n());
        let t = Instant::now();
        let prepared_u = {
            let ServerCollapse::Interpolation(state) = &self.collapse else {
                panic!("interpolation offline requested for non-interpolation server");
            };
            let panels = state.interpolation.num_panels();
            let mut prepared_u: Vec<Vec<PreparedF64>> = Vec::with_capacity(panels);
            for panel in 0..panels {
                let row: Vec<_> = (0..block_cols)
                    .map(|bc| PreparedF64::new(&state.matrix.matrices()[panel * block_cols + bc]))
                    .collect();
                prepared_u.push(row);
            }
            prepared_u
        };
        let ServerPrecomputation::Interpolation(precomputation) = &mut self.precomputation else {
            panic!("interpolation precomputation requested for non-interpolation server");
        };
        precomputation.prepared_u = prepared_u;
        t.elapsed()
    }

    /// A throwaway key whose mask is seeded by `server_seed.keys()`; its body is
    /// ignored by `pack_precompute`, which consumes only the mask seed.
    fn interpolation_pack_key_mask(&mut self) -> GLWEAutomorphismKeyCompressed<BE::OwnedBuf> {
        let module = self.params.module();
        let mut sk = module.lwe_secret_alloc(Degree(self.params.n() as u32));
        sk.fill_ternary_prob(0.5, &mut Source::new([0u8; 32]));
        let (key_g, _key_h) = module.pack_keys_generate(
            &self.params.key_layout(),
            &sk,
            self.server_seed.keys(),
            &mut Source::new([0u8; 32]),
            &mut self.scratch.borrow(),
        );
        key_g
    }

    fn offline_interpolation_pack_precomputations(
        &mut self,
        key_mask_src: &GLWEAutomorphismKeyCompressed<BE::OwnedBuf>,
    ) -> (Duration, Duration, Duration) {
        let module = self.params.module();
        let lwe_infos = self.params.lwe_matrix_infos();
        let precompute_metadata = self.params.packing_precompute_infos();
        let panels = {
            let ServerCollapse::Interpolation(state) = &self.collapse else {
                panic!("interpolation offline requested for non-interpolation server");
            };
            state.interpolation.num_panels()
        };

        // One panel per work item; panels are fully independent (own aggregate,
        // own per-worker scratch). Output written by panel index ⇒ bit-identical
        // to the sequential order regardless of thread count.
        type PanelOut<BE> = (Option<PackingPrecomputations<BE>>, [Duration; 3]);
        let mut outputs: Vec<PanelOut<BE>> =
            (0..panels).map(|_| (None, [Duration::default(); 3])).collect();

        let k = self.layout.block_cols(self.params.n());
        let nthreads = num_threads(panels);
        let work = assign_panels(panels, k, nthreads);
        let bytes = server_scratch_bytes(&self.params);
        // M3 block-tiling: when panels under-fill the machine (P < cores), give
        // each panel's mask-product GEMM the spare cores to tile its K-way
        // contraction. `mask_threads = 1` (incl. PIR_THREADS=1) keeps the exact
        // sequential fold. Nesting is safe: the mask product is scratch-free.
        let mask_threads = (num_threads(usize::MAX) / nthreads).max(1);
        // P-free scalars captured by the worker closure (avoids capturing
        // `&Parameters<_, _, P>`, whose `PhantomData<P>` would force `P: Sync`).
        let n = self.params.n();
        let base2k = self.params.base2k();
        let torus_bits = self.params.k();

        let region = Instant::now();
        {
            let ServerPrecomputation::Interpolation(precomputation) = &self.precomputation else {
                panic!("interpolation precomputation requested for non-interpolation server");
            };
            let prepared_u = &precomputation.prepared_u;
            let masks = &precomputation.masks;
            let lwe_infos = &lwe_infos;
            let precompute_metadata = &precompute_metadata;

            // Split the output buffer into group-aligned disjoint slabs.
            let mut slabs: Vec<&mut [PanelOut<BE>]> = Vec::with_capacity(work.len());
            let mut rest = outputs.as_mut_slice();
            for group in &work {
                let (head, tail) = rest.split_at_mut(group.len());
                slabs.push(head);
                rest = tail;
            }

            scoped_workers::<BE, PanelOut<BE>, _>(slabs, &work, bytes, |slab, group, sc| {
                for (slot, w) in slab.iter_mut().zip(group.iter()) {
                    let (precompute, ua, prep, pp) = compute_panel_precompute(
                        module,
                        n,
                        base2k,
                        torus_bits,
                        mask_threads,
                        lwe_infos,
                        precompute_metadata,
                        &prepared_u[w.panel],
                        masks,
                        key_mask_src,
                        &mut sc.borrow(),
                    );
                    *slot = (Some(precompute), [ua, prep, pp]);
                }
            });
        }
        let region_wall = region.elapsed();

        // Per-phase CPU times summed across panels. When threaded, rescale them
        // so the three phases sum to the parallel region's wall-clock — keeping
        // `OFFLINE total` (a sum of phases) a true wall-clock figure while the
        // relative breakdown is preserved.
        let mut ua_mask = Duration::default();
        let mut mask_prep = Duration::default();
        let mut pack_precompute = Duration::default();
        for (_, [ua, prep, pp]) in &outputs {
            ua_mask += *ua;
            mask_prep += *prep;
            pack_precompute += *pp;
        }
        if nthreads > 1 {
            let cpu = ua_mask + mask_prep + pack_precompute;
            if !cpu.is_zero() {
                let scale = region_wall.as_secs_f64() / cpu.as_secs_f64();
                ua_mask = ua_mask.mul_f64(scale);
                mask_prep = mask_prep.mul_f64(scale);
                pack_precompute = pack_precompute.mul_f64(scale);
            }
        }

        let precomputations: Vec<PackingPrecomputations<BE>> =
            outputs.into_iter().map(|(p, _)| p.unwrap()).collect();
        let ServerPrecomputation::Interpolation(precomputation) = &mut self.precomputation else {
            panic!("interpolation precomputation requested for non-interpolation server");
        };
        precomputation.precomputations = precomputations;
        (ua_mask, mask_prep, pack_precompute)
    }
}

/// One interpolation panel's query-independent precompute: the fixed mask
/// product `U·A` (GEMM), its mask preprocessing into a fresh per-call
/// `aggregate`, and the pack precompute. Pure w.r.t. shared state (own
/// `aggregate`, caller-supplied `scratch`) so it is safe to run one panel per
/// worker thread. Returns the precompute and the `(ua_mask, mask_prep,
/// pack_precompute)` sub-timings.
#[allow(clippy::too_many_arguments)]
fn compute_panel_precompute<BE>(
    module: &Module<BE>,
    n: usize,
    base2k: usize,
    torus_bits: usize,
    mask_threads: usize,
    lwe_infos: &LWEMatrixLayout,
    precompute_metadata: &PackingPrecomputeInfos,
    prepared_u_panel: &[PreparedF64],
    masks: &[QueryMask],
    key_mask_src: &GLWEAutomorphismKeyCompressed<BE::OwnedBuf>,
    scratch: &mut ScratchArena<'_, BE>,
) -> (PackingPrecomputations<BE>, Duration, Duration, Duration)
where
    BE: Backend<OwnedBuf = Vec<u8>> + poulpy_cpu_ref::reference::fft64::reim::ReimArith,
    Module<BE>: InterpolationServerModule<BE> + ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
    VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
    LWEMatrix<Vec<u8>>: LWEMatrixToBackendMut<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    let t = Instant::now();
    let product =
        mask_product_to_pack(module, lwe_infos, prepared_u_panel, masks, torus_bits, mask_threads);
    let ua = t.elapsed();

    // Per-call aggregate (was a shared buffer reused across panels) — this is
    // what makes the panel independent and thread-safe.
    let mut aggregate = module.vec_znx_alloc(n, lwe_infos.size());
    let t = Instant::now();
    if mask_threads > 1 {
        // Intra-op parallelism over the `n/2` `h`-leaves; same budget as the
        // mask-product tiling, so the panel × intra nesting stays balanced.
        module.packing_mask_preprocessing_threaded(
            &mut aggregate,
            base2k,
            product.mask(),
            mask_threads,
            scratch,
        );
    } else {
        module.packing_mask_preprocessing(&mut aggregate, base2k, product.mask(), scratch);
    }
    let prep = t.elapsed();

    let mut precompute = module.pack_precompute_alloc(
        precompute_metadata.steps(),
        precompute_metadata.size(),
        precompute_metadata.base2k(),
        precompute_metadata.baby_size(),
    );
    let t = Instant::now();
    module.pack_precompute(&mut precompute, &aggregate, key_mask_src, scratch);
    let pp = t.elapsed();

    (precompute, ua, prep, pp)
}
