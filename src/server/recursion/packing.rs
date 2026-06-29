//! Shared partial-packing helpers used by both OFFLINE (`resp1`) and ONLINE
//! (`resp1`/`resp2`) phases: the mask-side `D·A` precompute, the online body
//! product + pack, and the packing scratch estimate.

use std::time::{Duration, Instant};

use poulpy_core::{
    EncryptionLayout, GLWENormalize,
    layouts::{
        Degree, GLWE, GLWEAutomorphismKeyCompressed, GLWECompressed, GLWELayout, LWEInfos,
        LWEMatrix, LWEMatrixLayout, LWEMatrixToBackendMut, ModuleCoreAlloc,
    },
};
use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxDftAutomorphismPlan, VecZnxNormalizeTmpBytes,
    },
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchArena, ScratchOwned, VecZnx,
        VecZnxToBackendMut, VecZnxToBackendRef,
    },
};

use crate::{
    config::Collapse,
    database::CoeffMatrix,
    packing::{
        Packing, PackingKeys, PackingMaskAggregation, PackingPrecomputations,
        recursion::partial_pack_batch_pooled,
    },
    parallel::{assign_panels, num_threads, scoped_workers},
    payload::Payload,
    server::{
        Gemm, OfflineTimings, OnlineTimings, Server,
        api::RecursionServerModule,
        common::{PreparedF64, QueryMask, full_torus_f64_body_product, mask_product_to_pack},
    },
};

use super::{KeyBundle, PackMaskDurations, PackMaskPhaseNames};

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
    ) -> (
        Vec<Vec<PreparedF64<'static>>>,
        Vec<PackingPrecomputations<BE>>,
    ) {
        // OFFLINE: full parallel budget across batches.
        let (prepared, precomputes, durations) = self.precompute_pack_mask_inner(
            all_digits,
            q_masks,
            gamma,
            key_mask_source,
            key_stride,
            num_threads(usize::MAX),
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
    ) -> (
        Vec<Vec<PreparedF64<'static>>>,
        Vec<PackingPrecomputations<BE>>,
    ) {
        // ONLINE (resp2): per-query, only `nbatches=2` here — the per-worker scratch
        // alloc + thread spawn dwarfs the tiny work, so run sequentially.
        let (prepared, precomputes, durations) = self.precompute_pack_mask_inner(
            all_digits,
            q_masks,
            gamma,
            key_mask_source,
            key_stride,
            1,
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
        max_threads: usize,
    ) -> (
        Vec<Vec<PreparedF64<'static>>>,
        Vec<PackingPrecomputations<BE>>,
        PackMaskDurations,
    ) {
        let module = self.params.module();
        let n = self.params.n();
        let t = self.database.t();
        let base2k = self.params.base2k();
        let baby_size = self.params.baby_size();
        let torus_bits = self.params.k();
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
        let bytes = self.scratch_for_pack();
        let nthreads = num_threads(nbatches).min(max_threads.max(1));
        // Spare cores tile each batch's mask-product contraction (balanced
        // nesting with the across-batch parallelism). `max_threads = 1` (online)
        // makes everything sequential — one scratch alloc, no spawn overhead.
        let mask_threads = (max_threads.max(1) / nthreads).max(1);
        let work = assign_panels(nbatches, 1, nthreads);

        // One batch per work item; batches are independent (own aggregate + scratch,
        // sequential per-batch mask product/prep ⇒ bit-identical). Output by index.
        type BatchOut<BE> = (
            Option<(Vec<PreparedF64<'static>>, PackingPrecomputations<BE>)>,
            [Duration; 4],
        );
        let mut outputs: Vec<BatchOut<BE>> = (0..nbatches)
            .map(|_| (None, [Duration::default(); 4]))
            .collect();

        let region = Instant::now();
        {
            let res_infos = &res_infos;
            let gemm: &dyn Gemm = &*self.gemm;
            let mut slabs: Vec<&mut [BatchOut<BE>]> = Vec::with_capacity(work.len());
            let mut rest = outputs.as_mut_slice();
            for group in &work {
                let (head, tail) = rest.split_at_mut(group.len());
                slabs.push(head);
                rest = tail;
            }
            scoped_workers::<BE, BatchOut<BE>, _>(slabs, &work, bytes, |slab, group, sc| {
                for (slot, w) in slab.iter_mut().zip(group.iter()) {
                    let (row_prep, precompute, d) = compute_pack_mask_batch(
                        module,
                        w.panel,
                        n,
                        t,
                        total,
                        base2k,
                        baby_size,
                        torus_bits,
                        mask_threads,
                        gamma,
                        key_stride,
                        res_infos,
                        size,
                        all_digits,
                        q_masks,
                        key_mask_source,
                        gemm,
                        &mut sc.borrow(),
                    );
                    *slot = (Some((row_prep, precompute)), d);
                }
            });
        }
        let region_wall = region.elapsed();

        let mut durations = PackMaskDurations::default();
        for (_, d) in &outputs {
            durations.prepare_db += d[0];
            durations.mask_product += d[1];
            durations.mask_prep += d[2];
            durations.pack_precompute += d[3];
        }
        if nthreads > 1 {
            let cpu = durations.prepare_db
                + durations.mask_product
                + durations.mask_prep
                + durations.pack_precompute;
            if !cpu.is_zero() {
                let scale = region_wall.as_secs_f64() / cpu.as_secs_f64();
                durations.prepare_db = durations.prepare_db.mul_f64(scale);
                durations.mask_product = durations.mask_product.mul_f64(scale);
                durations.mask_prep = durations.mask_prep.mul_f64(scale);
                durations.pack_precompute = durations.pack_precompute.mul_f64(scale);
            }
        }

        let mut prepared: Vec<Vec<PreparedF64<'static>>> = Vec::with_capacity(nbatches);
        let mut precomputes: Vec<PackingPrecomputations<BE>> = Vec::with_capacity(nbatches);
        for (slot, _) in outputs {
            let (row_prep, precompute) = slot.unwrap();
            prepared.push(row_prep);
            precomputes.push(precompute);
        }
        (prepared, precomputes, durations)
    }

    pub(super) fn precompute_pack_mask_online(
        &self,
        all_digits: &[Vec<i16>],
        q_masks: &[QueryMask],
        gamma: usize,
        key: &KeyBundle<'_, BE>,
        timings: &mut OnlineTimings,
    ) -> (
        Vec<Vec<PreparedF64<'static>>>,
        Vec<PackingPrecomputations<BE>>,
    ) {
        self.precompute_pack_mask_online_timed(
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

/// One resp-digit batch's mask-side precompute: build the `D1` matrices from the
/// digit DB, run the (sequential, `mask_threads = 1`) mask product, the partial
/// mask preprocessing, and the partial pack precompute. Pure w.r.t. shared state
/// (own `aggregate`, caller-supplied per-worker `scratch`) so it runs one batch
/// per worker thread — bit-identical to the sequential loop. Returns the prepared
/// matrices, the precompute, and `[prepare_db, mask_product, mask_prep,
/// pack_precompute]` sub-timings.
#[allow(clippy::too_many_arguments)]
fn compute_pack_mask_batch<BE>(
    module: &Module<BE>,
    m: usize,
    n: usize,
    t: usize,
    total: usize,
    base2k: usize,
    baby_size: usize,
    torus_bits: usize,
    mask_threads: usize,
    gamma: usize,
    key_stride: usize,
    res_infos: &LWEMatrixLayout,
    size: usize,
    all_digits: &[Vec<i16>],
    q_masks: &[QueryMask],
    key_mask_source: &GLWEAutomorphismKeyCompressed<BE::OwnedBuf>,
    gemm: &dyn Gemm,
    scratch: &mut ScratchArena<'_, BE>,
) -> (
    Vec<PreparedF64<'static>>,
    PackingPrecomputations<BE>,
    [Duration; 4],
)
where
    BE: Backend<OwnedBuf = Vec<u8>> + poulpy_cpu_ref::reference::fft64::reim::ReimArith,
    Module<BE>: RecursionServerModule<BE> + ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
    VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    let started = Instant::now();
    let mut row_prep: Vec<PreparedF64<'static>> = Vec::with_capacity(q_masks.len());
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
    let d_prepare = started.elapsed();

    let started = Instant::now();
    let res_mask = mask_product_to_pack(
        module,
        res_infos,
        &row_prep,
        q_masks,
        torus_bits,
        mask_threads,
        gemm,
    );
    let d_mask_product = started.elapsed();

    let started = Instant::now();
    let mut aggregate = module.vec_znx_alloc(gamma, size);
    if mask_threads > 1 {
        module.packing_partial_mask_preprocessing_threaded(
            &mut aggregate,
            base2k,
            gamma,
            res_mask.mask(),
            mask_threads,
            scratch,
        );
    } else {
        module.packing_partial_mask_preprocessing(
            &mut aggregate,
            base2k,
            gamma,
            res_mask.mask(),
            scratch,
        );
    }
    let d_mask_prep = started.elapsed();

    let started = Instant::now();
    let mut precompute =
        module.pack_partial_precompute_alloc(gamma - 1, size, base2k, baby_size, key_stride);
    module.pack_partial_precompute(&mut precompute, &aggregate, key_mask_source, scratch);
    let d_pack_precompute = started.elapsed();

    (
        row_prep,
        precompute,
        [d_prepare, d_mask_product, d_mask_prep, d_pack_precompute],
    )
}

/// Online body-side pack for a digit/precompute batch: per-row `D·b` GEMV, then
/// the pooled parallel partial pack. Returns the packed GLWEs and the
/// `(body_product, pack)` timings. Pure (own buffers, caller-supplied `pool`) so
/// it runs from `respond_recursion` with `&mut self.scratch_pool`.
#[allow(clippy::too_many_arguments)]
pub(super) fn pack_bodies_pooled<BE>(
    module: &Module<BE>,
    src_infos: &EncryptionLayout<GLWELayout>,
    qtilde_bits: usize,
    base2k: usize,
    torus_bits: usize,
    gamma: usize,
    prepared: &[Vec<PreparedF64<'_>>],
    precomputes: &[PackingPrecomputations<BE>],
    q_bodies: &[GLWECompressed<BE::OwnedBuf>],
    key_precomp: &PackingKeys<BE>,
    gemm: &dyn Gemm,
    pool: &mut [ScratchOwned<BE>],
) -> (Vec<GLWE<BE::OwnedBuf>>, Duration, Duration)
where
    BE: Backend<OwnedBuf = Vec<u8>> + poulpy_cpu_ref::reference::fft64::reim::ReimArith,
    Module<BE>: RecursionServerModule<BE> + ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
{
    let n = module.n();
    let res_infos = LWEMatrixLayout {
        rows: gamma,
        n: Degree(n as u32),
        base2k: src_infos.base2k(),
        k: src_infos.max_k(),
    };
    let size = res_infos.size();
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
            torus_bits,
            gemm,
        );
        body_product += started.elapsed();
        bodies.push(res_body);
    }
    let started = Instant::now();
    let inputs: Vec<_> = precomputes.iter().zip(bodies.iter()).collect();
    let out = partial_pack_batch_pooled(module, src_infos, qtilde_bits, &inputs, key_precomp, pool);
    (out, body_product, started.elapsed())
}
