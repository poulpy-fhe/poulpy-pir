//! Query-independent mask-side precomputation (the `O(N·d)` OFFLINE work):
//! `D·A0` (level 1) and `D1·A1` (`resp1`), plus the `resp0` mask decompose.

use std::time::{Duration, Instant};

use poulpy_core::{
    GLWENormalize,
    layouts::{
        Degree, GLWE, LWEInfos, LWEMatrix, LWEMatrixLayout, LWEMatrixToBackendMut, ModuleCoreAlloc,
    },
};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAutomorphismPlan},
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, VecZnxToBackendMut,
        VecZnxToBackendRef, ZnxView, ZnxViewMut, ZnxZero,
    },
};

use crate::{
    config::Collapse,
    packing::{
        Packing, PackingMaskAggregation, PackingPrecomputations,
        recursion::{modulus_switch_to_digits, qtilde_glwe_layout},
    },
    payload::Payload,
    server::{
        OfflineTimings, Server, ServerCollapse, ServerPrecomputation,
        api::RecursionServerModule,
        common::{PreparedF64, copy_lwe_matrix_mask_rows, mask_product_to_pack},
    },
};

use super::{PackMaskPhaseNames, RecursionOfflineShape, RecursionPrecomputation, qtilde_bits, tau};

#[allow(private_bounds)]
impl<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload<[u8; 32]>> Server<BE, P>
where
    BE: poulpy_cpu_ref::reference::fft64::reim::ReimArith,
    Module<BE>: RecursionServerModule<BE>,
    <Module<BE> as VecZnxDftAutomorphismPlan<BE>>::Plan: 'static,
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
        let (db_prep, l1_precompute) = self.offline_recursion_l1(shape, &mut timings);
        let mask_data = self.offline_recursion_resp0_mask_data(shape, &l1_precompute, &mut timings);
        let (resp1_prep, resp1_precompute) =
            self.offline_recursion_resp1(shape, &mask_data, &mut timings);

        self.precomputation = ServerPrecomputation::Recursion(RecursionPrecomputation {
            db_prep,
            l1_precompute,
            resp1_prep,
            resp1_precompute,
        });
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
    ) -> (Vec<Vec<PreparedF64>>, Vec<PackingPrecomputations<BE>>) {
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
        let mut prepare_db = Duration::default();
        let mut mask_product = Duration::default();
        let mut mask_prep = Duration::default();
        let mut pack_precompute = Duration::default();
        let mut db_prep = Vec::with_capacity(self.database.physical_rows());
        let mut l1_precompute = Vec::with_capacity(shape.t);
        for row_group in 0..self.database.physical_rows() {
            let started = Instant::now();
            let row_prep: Vec<PreparedF64> = (0..self.database.column_blocks())
                .map(|block| PreparedF64::new(self.database.physical_block(row_group, block)))
                .collect();
            prepare_db += started.elapsed();

            let started = Instant::now();
            let full_res_mask =
                mask_product_to_pack(module, &self.params, &full_pack_infos, &row_prep, q0_masks);
            mask_product += started.elapsed();

            db_prep.push(row_prep);
            for local in 0..rows_per_group {
                let batch = row_group * rows_per_group + local;
                if batch >= shape.t {
                    break;
                }
                let mut res_mask = module.lwe_matrix_alloc_from_infos(&partial_res_infos);
                copy_lwe_matrix_mask_rows(
                    &mut res_mask,
                    0,
                    &full_res_mask,
                    local * shape.gamma0,
                    shape.gamma0,
                );

                let mut aggregate = module.vec_znx_alloc(shape.gamma0, shape.size);
                let started = Instant::now();
                module.packing_partial_mask_preprocessing(
                    &mut aggregate,
                    shape.base2k,
                    shape.gamma0,
                    res_mask.mask(),
                    &mut self.scratch.borrow(),
                );
                mask_prep += started.elapsed();

                let mut precompute = module.pack_partial_precompute_alloc(
                    shape.gamma0 - 1,
                    shape.size,
                    shape.base2k,
                    shape.baby_size,
                    key0_stride,
                );
                let started = Instant::now();
                module.pack_partial_precompute(
                    &mut precompute,
                    &aggregate,
                    key0_mask_source,
                    &mut self.scratch.borrow(),
                );
                pack_precompute += started.elapsed();

                l1_precompute.push(precompute);
            }
        }
        timings.add_prepare_u("recursion.l1.prepare_db", prepare_db);
        timings.add_ua_mask("recursion.l1.mask_product", mask_product);
        timings.add_mask_prep("recursion.l1.mask_prep", mask_prep);
        timings.add_pack_precompute("recursion.l1.pack_precompute", pack_precompute);
        (db_prep, l1_precompute)
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
    ) -> (Vec<Vec<PreparedF64>>, Vec<PackingPrecomputations<BE>>) {
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
