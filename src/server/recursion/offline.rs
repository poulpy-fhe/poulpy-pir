//! Query-independent mask-side precomputation (the `O(N·d)` OFFLINE work):
//! `D·A0` (level 1) and `D1·A1` (`resp1`), plus the `resp0` mask decompose.

use std::time::{Duration, Instant};

use poulpy_core::layouts::GLWEAutomorphismKeyCompressed;
use poulpy_core::{
    GLWENormalize,
    layouts::{
        Degree, GLWE, LWEInfos, LWEMatrix, LWEMatrixLayout, LWEMatrixToBackendMut, ModuleCoreAlloc,
    },
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAutomorphismPlan},
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchArena, ScratchOwned, VecZnx,
        VecZnxToBackendMut, VecZnxToBackendRef, ZnxView, ZnxViewMut, ZnxZero,
    },
};

use crate::{
    config::Collapse,
    packing::{
        Packing, PackingMaskAggregation, PackingPrecomputations,
        recursion::{modulus_switch_to_digits, qtilde_glwe_layout},
    },
    parallel::{assign_panels, num_threads, scoped_workers},
    payload::Payload,
    server::{
        Gemm, OfflineTimings, Server, ServerCollapse, ServerPrecomputation,
        api::RecursionServerModule,
        common::{PreparedF64, QueryMask, copy_lwe_matrix_mask_rows, mask_product_to_pack},
    },
};

use super::{PackMaskPhaseNames, RecursionOfflineShape, RecursionPrecomputation, qtilde_bits, tau};

#[allow(private_bounds)]
impl<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload<[u8; 32]>> Server<BE, P>
where
    BE: poulpy_cpu_ref::reference::fft64::reim::ReimArith,
    Module<BE>: RecursionServerModule<BE>,
    <Module<BE> as VecZnxDftAutomorphismPlan<BE>>::Plan: 'static + Send + Sync,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
    LWEMatrix<Vec<u8>>: LWEMatrixToBackendMut<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    /// Server-side preprocessing (query-independent, the `O(N·d)` work). Computes
    /// the mask-side products `D·A0` (level 1) and `D1·A1` (`resp1`) from the fixed
    /// CRS masks, stores the resulting packing precomputes + prepared database
    /// matrices, and returns an ordered phase timing breakdown. Call once
    /// (re-run after a database update) before [`respond`](Self::respond_recursion).
    pub(crate) fn offline_recursion(&mut self) -> OfflineTimings {
        let mut timings = OfflineTimings::default();
        self.ensure_recursion_query_mask(&mut timings);
        let shape = self.recursion_offline_shape();
        let l1_precompute = self.offline_recursion_l1(shape, &mut timings);
        let mask_data = self.offline_recursion_resp0_mask_data(shape, &l1_precompute, &mut timings);
        let (resp1_prep, resp1_precompute) =
            self.offline_recursion_resp1(shape, &mask_data, &mut timings);

        self.precomputation = ServerPrecomputation::Recursion(RecursionPrecomputation {
            l1_precompute,
            resp1_prep,
            resp1_precompute,
        });

        // Warm the online per-worker scratch pool (plan M2′) so per-query packs
        // reuse it instead of allocating.
        let bytes = self.scratch_for_pack();
        let nthreads = num_threads(usize::MAX);
        while self.scratch_pool.len() < nthreads {
            self.scratch_pool.push(ScratchOwned::<BE>::alloc(bytes));
        }

        timings
    }

    fn recursion_offline_shape(&self) -> RecursionOfflineShape {
        let params = &self.params;
        let Collapse::Recursion {
            gamma0,
            gamma1,
            gamma2: _,
        } = params.collapse()
        else {
            panic!("Recursion offline requires Collapse::Recursion parameters");
        };
        RecursionOfflineShape {
            n: params.n(),
            size: self.recursion_state().src_infos.size(),
            t: self.database.t(),
            gamma0,
            gamma1,
            base2k: params.base2k(),
            baby_size: params.baby_size(),
            tau: tau(params),
        }
    }

    fn offline_recursion_l1(
        &mut self,
        shape: RecursionOfflineShape,
        timings: &mut OfflineTimings,
    ) -> Vec<PackingPrecomputations<BE>> {
        let module = self.params.module();
        let ServerCollapse::Recursion(state) = &self.collapse else {
            panic!("InsPIRe² key requested for non-InsPIRe² server");
        };
        assert!(
            !state.q0_masks.is_empty(),
            "call Server::generate_query_mask() before InsPIRe² offline"
        );
        let q0_masks = &state.q0_masks;
        let key0_mask_source = &state.key0_mask.key;
        let key0_stride = state.key0_mask.stride;
        let full_pack_infos = LWEMatrixLayout {
            rows: shape.n,
            n: Degree(shape.n as u32),
            base2k: self.recursion_state().src_infos.base2k(),
            k: self.recursion_state().src_infos.max_k(),
        };
        let partial_res_infos = LWEMatrixLayout {
            rows: shape.gamma0,
            ..full_pack_infos
        };
        let rows_per_group = self.database.rows_per_physical_group();
        let physical_rows = self.database.physical_rows();
        let torus_bits = self.params.k();

        // Phase 1 (near-free, sequential): build zero-copy `PreparedF64` **views**
        // over the contiguous plaintext DB blocks (no second copy — the DB lives in
        // `self.database`). Materialized up front so the parallel region captures
        // `db_views` (slices, hence `Send + Sync` and no `P: Sync`), not
        // `&self.database`.
        let started = Instant::now();
        let db_views: Vec<Vec<PreparedF64<'_>>> = (0..physical_rows)
            .map(|row_group| {
                (0..self.database.column_blocks())
                    .map(|block| {
                        PreparedF64::from_matrix(self.database.physical_block(row_group, block))
                    })
                    .collect()
            })
            .collect();
        let prepare_db = started.elapsed();

        // Phase 2 (parallel over row groups): `D·A0` mask product + per-row
        // partial mask-prep / pack-precompute. Each row group is independent
        // (own scratch, sequential per-batch ops ⇒ bit-identical).
        let bytes = self.scratch_for_pack();
        let nthreads = num_threads(physical_rows);
        // Spare cores tile each row group's mask-product contraction (balanced
        // nesting with the row-group parallelism, like the interpolation path).
        let mask_threads = (num_threads(usize::MAX) / nthreads).max(1);
        let work = assign_panels(physical_rows, 1, nthreads);
        type GroupOut<BE> = (Option<Vec<PackingPrecomputations<BE>>>, [Duration; 3]);
        let mut outputs: Vec<GroupOut<BE>> = (0..physical_rows)
            .map(|_| (None, [Duration::default(); 3]))
            .collect();

        let region = Instant::now();
        {
            let db_views = &db_views;
            let full_pack_infos = &full_pack_infos;
            let partial_res_infos = &partial_res_infos;
            let gemm: &dyn Gemm = &*self.gemm;
            let mut slabs: Vec<&mut [GroupOut<BE>]> = Vec::with_capacity(work.len());
            let mut rest = outputs.as_mut_slice();
            for group in &work {
                let (head, tail) = rest.split_at_mut(group.len());
                slabs.push(head);
                rest = tail;
            }
            scoped_workers::<BE, GroupOut<BE>, _>(slabs, &work, bytes, |slab, group, sc| {
                for (slot, w) in slab.iter_mut().zip(group.iter()) {
                    let (precomputes, d) = compute_l1_row_group(
                        module,
                        w.panel,
                        rows_per_group,
                        shape.t,
                        shape.gamma0,
                        shape.base2k,
                        shape.baby_size,
                        torus_bits,
                        mask_threads,
                        key0_stride,
                        full_pack_infos,
                        partial_res_infos,
                        shape.size,
                        &db_views[w.panel],
                        q0_masks,
                        key0_mask_source,
                        gemm,
                        &mut sc.borrow(),
                    );
                    *slot = (Some(precomputes), d);
                }
            });
        }
        let region_wall = region.elapsed();

        let mut mask_product = Duration::default();
        let mut mask_prep = Duration::default();
        let mut pack_precompute = Duration::default();
        for (_, d) in &outputs {
            mask_product += d[0];
            mask_prep += d[1];
            pack_precompute += d[2];
        }
        if nthreads > 1 {
            let cpu = mask_product + mask_prep + pack_precompute;
            if !cpu.is_zero() {
                let scale = region_wall.as_secs_f64() / cpu.as_secs_f64();
                mask_product = mask_product.mul_f64(scale);
                mask_prep = mask_prep.mul_f64(scale);
                pack_precompute = pack_precompute.mul_f64(scale);
            }
        }

        let mut l1_precompute = Vec::with_capacity(shape.t);
        for (slot, _) in outputs {
            l1_precompute.extend(slot.unwrap());
        }

        timings.add_prepare_u("recursion.l1.prepare_db", prepare_db);
        timings.add_ua_mask("recursion.l1.mask_product", mask_product);
        timings.add_mask_prep("recursion.l1.mask_prep", mask_prep);
        timings.add_pack_precompute("recursion.l1.pack_precompute", pack_precompute);
        l1_precompute
    }

    fn offline_recursion_resp0_mask_data(
        &self,
        shape: RecursionOfflineShape,
        l1_precompute: &[PackingPrecomputations<BE>],
        timings: &mut OfflineTimings,
    ) -> Vec<Vec<i16>> {
        // resp0 mask → decompose. The mask is already produced by the
        // query-independent pack precompute; materializing it directly avoids
        // running the online body-side BSGS path with zero bodies for every batch.
        let started = Instant::now();
        let resp0_mask = self.materialize_precomputed_masks(l1_precompute);
        timings.record_phase("recursion.resp0.materialize_mask", started.elapsed());

        let started = Instant::now();
        let mut mask_data: Vec<Vec<i16>> = vec![vec![0i16; shape.t]; shape.n * shape.tau];
        for (k, glwe) in resp0_mask.iter().enumerate() {
            let data = glwe.data();
            for c in 0..shape.n {
                for l in 0..shape.tau {
                    mask_data[c * shape.tau + l][k] = data.at(1, l)[c] as i16;
                }
            }
        }
        timings.record_phase("recursion.resp0.decompose_mask", started.elapsed());
        mask_data
    }

    fn offline_recursion_resp1(
        &self,
        shape: RecursionOfflineShape,
        mask_data: &[Vec<i16>],
        timings: &mut OfflineTimings,
    ) -> (
        Vec<Vec<PreparedF64<'static>>>,
        Vec<PackingPrecomputations<BE>>,
    ) {
        let state = self.recursion_state();
        assert!(
            !state.q1_masks.is_empty(),
            "call Server::generate_query_mask() before InsPIRe² offline"
        );

        self.precompute_pack_mask_timed(
            mask_data,
            &state.q1_masks,
            shape.gamma1,
            &state.key1_mask.key,
            state.key1_mask.stride,
            timings,
            PackMaskPhaseNames {
                prepare_db: "recursion.resp1.prepare_db",
                mask_product: "recursion.resp1.mask_product",
                mask_prep: "recursion.resp1.mask_prep",
                pack_precompute: "recursion.resp1.pack_precompute",
            },
        )
    }

    /// Materializes only the final GLWE masks already computed by
    /// `pack_precompute_partial`, then modulus-switches/decomposes them exactly as
    /// [`partial_pack_batch`](crate::packing::recursion::partial_pack_batch) would
    /// after copying the same mask into the packed result. This is the
    /// query-independent half of first-level packing.
    fn materialize_precomputed_masks(
        &self,
        precomputes: &[PackingPrecomputations<BE>],
    ) -> Vec<GLWE<BE::OwnedBuf>> {
        let module = self.params.module();
        let qtilde_infos =
            qtilde_glwe_layout(Degree(self.params.n() as u32), qtilde_bits(&self.params));
        let src_infos = &self.recursion_state().src_infos;
        let mut sc = ScratchOwned::<BE>::alloc(module.glwe_normalize_tmp_bytes());
        let mut out = Vec::with_capacity(precomputes.len());
        for precompute in precomputes {
            let mut packed = module.glwe_alloc_from_infos(src_infos);
            packed.data_mut().zero();
            let mask = precompute.final_mask();
            for limb in 0..src_infos.size() {
                packed
                    .data_mut()
                    .at_mut(1, limb)
                    .copy_from_slice(mask.at(0, limb));
            }

            let mut switched = module.glwe_alloc_from_infos(&qtilde_infos);
            modulus_switch_to_digits(module, &mut switched, &packed, &mut sc.borrow());
            out.push(switched);
        }
        out
    }
}

/// One level-1 row group's mask-side precompute: the full `D·A0` mask product,
/// then per local row the partial mask-prep / pack-precompute. Pure w.r.t. shared
/// state (own scratch, sequential per-batch ops ⇒ bit-identical), so it runs one
/// row group per worker thread. Returns the row group's precomputes (in `local`
/// order) and `[mask_product, mask_prep, pack_precompute]` sub-timings.
#[allow(clippy::too_many_arguments)]
fn compute_l1_row_group<BE>(
    module: &Module<BE>,
    row_group: usize,
    rows_per_group: usize,
    t: usize,
    gamma0: usize,
    base2k: usize,
    baby_size: usize,
    torus_bits: usize,
    mask_threads: usize,
    key0_stride: usize,
    full_pack_infos: &LWEMatrixLayout,
    partial_res_infos: &LWEMatrixLayout,
    size: usize,
    row_prep: &[PreparedF64],
    q0_masks: &[QueryMask],
    key0_mask_source: &GLWEAutomorphismKeyCompressed<BE::OwnedBuf>,
    gemm: &dyn Gemm,
    scratch: &mut ScratchArena<'_, BE>,
) -> (Vec<PackingPrecomputations<BE>>, [Duration; 3])
where
    BE: Backend<OwnedBuf = Vec<u8>> + poulpy_cpu_ref::reference::fft64::reim::ReimArith,
    Module<BE>: RecursionServerModule<BE> + ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
    VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
    LWEMatrix<Vec<u8>>: LWEMatrixToBackendMut<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    let started = Instant::now();
    let full_res_mask = mask_product_to_pack(
        module,
        full_pack_infos,
        row_prep,
        q0_masks,
        torus_bits,
        mask_threads,
        gemm,
    );
    let d_mask_product = started.elapsed();

    let mut precomputes: Vec<PackingPrecomputations<BE>> = Vec::with_capacity(rows_per_group);
    let mut d_mask_prep = Duration::default();
    let mut d_pack_precompute = Duration::default();
    for local in 0..rows_per_group {
        let batch = row_group * rows_per_group + local;
        if batch >= t {
            break;
        }
        let mut res_mask = module.lwe_matrix_alloc_from_infos(partial_res_infos);
        copy_lwe_matrix_mask_rows(&mut res_mask, 0, &full_res_mask, local * gamma0, gamma0);

        let mut aggregate = module.vec_znx_alloc(gamma0, size);
        let started = Instant::now();
        if mask_threads > 1 {
            module.packing_partial_mask_preprocessing_threaded(
                &mut aggregate,
                base2k,
                gamma0,
                res_mask.mask(),
                mask_threads,
                scratch,
            );
        } else {
            module.packing_partial_mask_preprocessing(
                &mut aggregate,
                base2k,
                gamma0,
                res_mask.mask(),
                scratch,
            );
        }
        d_mask_prep += started.elapsed();

        let mut precompute =
            module.pack_partial_precompute_alloc(gamma0 - 1, size, base2k, baby_size, key0_stride);
        let started = Instant::now();
        module.pack_partial_precompute(&mut precompute, &aggregate, key0_mask_source, scratch);
        d_pack_precompute += started.elapsed();

        precomputes.push(precompute);
    }

    (
        precomputes,
        [d_mask_product, d_mask_prep, d_pack_precompute],
    )
}
