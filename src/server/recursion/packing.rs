//! Shared partial-packing helpers used by both OFFLINE (`resp1`) and ONLINE
//! (`resp1`/`resp2`) phases: the mask-side `D·A` precompute, the online body
//! product + pack, and the packing scratch estimate.

use std::time::{Duration, Instant};

use poulpy_core::{
    GLWENormalize,
    layouts::{
        Degree, GLWE, GLWEAutomorphismKeyCompressed, GLWECompressed, LWEInfos, LWEMatrix,
        LWEMatrixLayout, LWEMatrixToBackendMut,
    },
};
use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAutomorphismPlan, VecZnxNormalizeTmpBytes,
    },
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchOwned, VecZnx, VecZnxToBackendMut,
        VecZnxToBackendRef,
    },
};

use crate::{
    config::Collapse,
    database::CoeffMatrix,
    packing::{
        Packing, PackingKeys, PackingMaskAggregation, PackingPrecomputations,
        recursion::partial_pack_batch,
    },
    payload::Payload,
    server::{
        OfflineTimings, OnlineTimings, Server,
        api::RecursionServerModule,
        common::{PreparedF64, QueryMask, full_torus_f64_body_product, mask_product_to_pack},
    },
};

use super::{KeyBundle, PackBodyPhaseNames, PackMaskDurations, PackMaskPhaseNames, qtilde_bits};

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
    /// Mask side of a digit-DB packing (query-independent given a CRS `q_mask`):
    /// build the `D1` matrices from the digits and compute `D1·A` → packing
    /// precomputes. Returns the prepared `D1` matrices (reused for the online body
    /// product) and the precomputes.
    pub(super) fn precompute_pack_mask_timed(
        &self,
        all_digits: &[Vec<i16>],
        q_masks: &[QueryMask],
        gamma: usize,
        key_mask_source: &GLWEAutomorphismKeyCompressed<BE::OwnedBuf>,
        key_stride: usize,
        timings: &mut OfflineTimings,
        phase_names: PackMaskPhaseNames,
    ) -> (Vec<Vec<PreparedF64>>, Vec<PackingPrecomputations<BE>>) {
        let (prepared, precomputes, durations) = self.precompute_pack_mask_inner(
            all_digits,
            q_masks,
            gamma,
            key_mask_source,
            key_stride,
        );
        timings.add_prepare_u(phase_names.prepare_db, durations.prepare_db);
        timings.add_ua_mask(phase_names.mask_product, durations.mask_product);
        timings.add_mask_prep(phase_names.mask_prep, durations.mask_prep);
        timings.add_pack_precompute(phase_names.pack_precompute, durations.pack_precompute);
        (prepared, precomputes)
    }

    fn precompute_pack_mask_online_timed(
        &self,
        all_digits: &[Vec<i16>],
        q_masks: &[QueryMask],
        gamma: usize,
        key_mask_source: &GLWEAutomorphismKeyCompressed<BE::OwnedBuf>,
        key_stride: usize,
        timings: &mut OnlineTimings,
        phase_names: PackMaskPhaseNames,
    ) -> (Vec<Vec<PreparedF64>>, Vec<PackingPrecomputations<BE>>) {
        let (prepared, precomputes, durations) = self.precompute_pack_mask_inner(
            all_digits,
            q_masks,
            gamma,
            key_mask_source,
            key_stride,
        );
        timings.add_prepare_db(phase_names.prepare_db, durations.prepare_db);
        timings.add_mask_product(phase_names.mask_product, durations.mask_product);
        timings.add_mask_prep(phase_names.mask_prep, durations.mask_prep);
        timings.add_pack_precompute(phase_names.pack_precompute, durations.pack_precompute);
        (prepared, precomputes)
    }

    fn precompute_pack_mask_inner(
        &self,
        all_digits: &[Vec<i16>],
        q_masks: &[QueryMask],
        gamma: usize,
        key_mask_source: &GLWEAutomorphismKeyCompressed<BE::OwnedBuf>,
        key_stride: usize,
    ) -> (
        Vec<Vec<PreparedF64>>,
        Vec<PackingPrecomputations<BE>>,
        PackMaskDurations,
    ) {
        let params = &self.params;
        let module = params.module();
        let n = params.n();
        let t = self.database.t();
        let base2k = params.base2k();
        let baby_size = params.baby_size();
        let total = all_digits.len();
        let nbatches = total.div_ceil(gamma);
        let src_infos = &self.recursion_state().src_infos;
        let res_infos = LWEMatrixLayout {
            rows: gamma,
            n: Degree(n as u32),
            base2k: src_infos.base2k(),
            k: src_infos.max_k(),
        };
        let size = res_infos.size();
        let mut sc = ScratchOwned::<BE>::alloc(self.scratch_for_pack());

        let mut prepared: Vec<Vec<PreparedF64>> = Vec::with_capacity(nbatches);
        let mut precomputes: Vec<PackingPrecomputations<BE>> = Vec::with_capacity(nbatches);
        let mut durations = PackMaskDurations::default();
        for m in 0..nbatches {
            let started = Instant::now();
            let mut row_prep: Vec<PreparedF64> = Vec::with_capacity(q_masks.len());
            for block in 0..q_masks.len() {
                let start = block * n;
                let mut db = CoeffMatrix::zeros(gamma, n);
                for j in 0..gamma {
                    let idx = m * gamma + j;
                    let row = db.row_mut(j);
                    for b in 0..n {
                        row[b] = if idx < total && start + b < t {
                            all_digits[idx][start + b]
                        } else {
                            0
                        };
                    }
                }
                row_prep.push(PreparedF64::new(&db));
            }
            durations.prepare_db += started.elapsed();

            let started = Instant::now();
            let res_mask = mask_product_to_pack(module, params, &res_infos, &row_prep, q_masks);
            durations.mask_product += started.elapsed();

            let mut aggregate = module.vec_znx_alloc(gamma, size);
            let started = Instant::now();
            module.packing_partial_mask_preprocessing(
                &mut aggregate,
                base2k,
                gamma,
                res_mask.mask(),
                &mut sc.borrow(),
            );
            durations.mask_prep += started.elapsed();

            let mut precompute = module.pack_partial_precompute_alloc(
                gamma - 1,
                size,
                base2k,
                baby_size,
                key_stride,
            );
            let started = Instant::now();
            module.pack_partial_precompute(
                &mut precompute,
                &aggregate,
                key_mask_source,
                &mut sc.borrow(),
            );
            durations.pack_precompute += started.elapsed();

            prepared.push(row_prep);
            precomputes.push(precompute);
        }
        (prepared, precomputes, durations)
    }

    pub(super) fn pack_bodies_timed(
        &self,
        prepared: &[Vec<PreparedF64>],
        precomputes: &[PackingPrecomputations<BE>],
        q_bodies: &[GLWECompressed<BE::OwnedBuf>],
        gamma: usize,
        key_precomp: &PackingKeys<BE>,
        timings: &mut OnlineTimings,
        phase_names: PackBodyPhaseNames,
    ) -> Vec<GLWE<BE::OwnedBuf>> {
        let (out, body_product, pack) =
            self.pack_bodies_inner(prepared, precomputes, q_bodies, gamma, key_precomp);
        timings.add_body_product(phase_names.body_product, body_product);
        timings.add_pack(phase_names.pack, pack);
        out
    }

    fn pack_bodies_inner(
        &self,
        prepared: &[Vec<PreparedF64>],
        precomputes: &[PackingPrecomputations<BE>],
        q_bodies: &[GLWECompressed<BE::OwnedBuf>],
        gamma: usize,
        key_precomp: &PackingKeys<BE>,
    ) -> (Vec<GLWE<BE::OwnedBuf>>, Duration, Duration) {
        let params = &self.params;
        let module = params.module();
        let n = params.n();
        let base2k = params.base2k();
        let src_infos = &self.recursion_state().src_infos;
        let res_infos = LWEMatrixLayout {
            rows: gamma,
            n: Degree(n as u32),
            base2k: src_infos.base2k(),
            k: src_infos.max_k(),
        };
        let size = res_infos.size();
        let mut sc = ScratchOwned::<BE>::alloc(self.scratch_for_pack());
        let mut bodies: Vec<VecZnx<BE::OwnedBuf>> = Vec::with_capacity(prepared.len());
        let mut body_product = Duration::default();
        for row_prep in prepared {
            let mut res_body = module.vec_znx_alloc(1, size);
            let started = Instant::now();
            full_torus_f64_body_product::<BE>(
                &mut res_body,
                base2k,
                row_prep,
                q_bodies,
                base2k,
                params.k(),
            );
            body_product += started.elapsed();
            bodies.push(res_body);
        }
        let started = Instant::now();
        let inputs: Vec<_> = precomputes.iter().zip(bodies.iter()).collect();
        let out = partial_pack_batch(
            module,
            src_infos,
            qtilde_bits(params),
            &inputs,
            key_precomp,
            &mut sc.borrow(),
        );
        (out, body_product, started.elapsed())
    }

    pub(super) fn pack_digits_timed(
        &self,
        all_digits: &[Vec<i16>],
        q_masks: &[QueryMask],
        q_bodies: &[GLWECompressed<BE::OwnedBuf>],
        gamma: usize,
        key: &KeyBundle<'_, BE>,
        timings: &mut OnlineTimings,
    ) -> Vec<GLWE<BE::OwnedBuf>> {
        let (prepared, precomputes) = self.precompute_pack_mask_online_timed(
            all_digits,
            q_masks,
            gamma,
            key.key,
            key.stride,
            timings,
            PackMaskPhaseNames {
                prepare_db: "recursion.resp2.prepare_db",
                mask_product: "recursion.resp2.mask_product",
                mask_prep: "recursion.resp2.mask_prep",
                pack_precompute: "recursion.resp2.pack_precompute",
            },
        );
        self.pack_bodies_timed(
            &prepared,
            &precomputes,
            q_bodies,
            gamma,
            &key.precomp,
            timings,
            PackBodyPhaseNames {
                body_product: "recursion.resp2.body_product",
                pack: "recursion.resp2.pack",
            },
        )
    }

    pub(super) fn scratch_for_pack(&self) -> usize {
        let params = &self.params;
        let module = params.module();
        let n = params.n();
        let base2k = params.base2k();
        let Collapse::Recursion {
            gamma0: _,
            gamma1,
            gamma2,
        } = params.collapse()
        else {
            panic!("Recursion scratch sizing requires Collapse::Recursion parameters");
        };
        let max_gamma = gamma1.max(gamma2);
        let src_infos = &self.recursion_state().src_infos;
        let size = src_infos.size();
        module
            .vec_znx_normalize_tmp_bytes()
            .max(module.pack_partial_mask_preprocessing_tmp_bytes(max_gamma, size))
            .max(module.glwe_normalize_tmp_bytes())
            .max(module.pack_precompute_tmp_bytes(
                crate::packing::PackingPrecomputeInfos::new(
                    n - 1,
                    size,
                    base2k,
                    params.baby_size(),
                ),
                &module.vec_znx_alloc(n, size),
                &params.key_layout(),
            ))
    }
}
