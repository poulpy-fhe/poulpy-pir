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
        recursion::{partial_pack_batch_pooled, switch_final_mask_to_qtilde},
    },
    parallel::{
        assign_panels, num_threads_offline, scoped_workers_profiled,
        scoped_workers_pooled_profiled,
    },
    payload::Payload,
    server::{
        Gemm, OfflineTimings, OnlineTimings, Server,
        api::RecursionServerModule,
        common::{PreparedF64, QueryMask, full_torus_f64_body_product, mask_product_to_pack},
    },
};

use super::{KeyBundle, PackMaskDurations, PackMaskPhaseNames, qtilde_bits};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Resp2ScratchMode {
    Pooled,
    Fresh,
}

fn resp2_scratch_mode() -> Resp2ScratchMode {
    match std::env::var("PIR_RESP2_SCRATCH") {
        Ok(value) if value.eq_ignore_ascii_case("pooled") => Resp2ScratchMode::Pooled,
        Ok(value) if value.eq_ignore_ascii_case("fresh") => Resp2ScratchMode::Fresh,
        Ok(value) => panic!(
            "invalid PIR_RESP2_SCRATCH={value:?}; expected \"pooled\" or \"fresh\""
        ),
        Err(std::env::VarError::NotPresent) => Resp2ScratchMode::Pooled,
        Err(err) => panic!("cannot read PIR_RESP2_SCRATCH: {err}"),
    }
}

fn positive_env(name: &str) -> Option<usize> {
    let value = std::env::var(name).ok()?;
    Some(
        value
            .parse::<usize>()
            .ok()
            .filter(|&n| n >= 1)
            .unwrap_or_else(|| panic!("{name} must be a positive integer, got {value:?}")),
    )
}

/// Resolve the tunable nested `resp2` schedule. The defaults consume the full
/// per-query online budget; benchmark runs can select `(outer, inner)` with
/// `PIR_RESP2_OUTER_THREADS` and `PIR_RESP2_INNER_THREADS`. Both requested
/// values are clamped to the natural work/pool ceilings, and the product can
/// never exceed `max_threads`.
fn resp2_schedule(
    nbatches: usize,
    max_threads: usize,
    pool_len: usize,
    pooled: bool,
) -> (usize, usize) {
    assert!(nbatches >= 1, "resp2 requires at least one batch");
    let mut outer_cap = nbatches.min(max_threads.max(1));
    if pooled {
        assert!(pool_len != 0, "pooled resp2 requires a non-empty scratch pool");
        outer_cap = outer_cap.min(pool_len);
    }
    let outer = positive_env("PIR_RESP2_OUTER_THREADS")
        .unwrap_or(outer_cap)
        .clamp(1, outer_cap);
    let inner_cap = (max_threads.max(1) / outer).max(1);
    let inner = positive_env("PIR_RESP2_INNER_THREADS")
        .unwrap_or(inner_cap)
        .clamp(1, inner_cap);
    debug_assert!(outer * inner <= max_threads.max(1));
    (outer, inner)
}

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
            num_threads_offline(usize::MAX),
            None,
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
        pool: &mut [ScratchOwned<BE>],
        max_threads: usize,
        timings: &mut OnlineTimings,
        phase_names: PackMaskPhaseNames,
    ) -> (
        Vec<Vec<PreparedF64<'static>>>,
        Vec<PackingPrecomputations<BE>>,
    ) {
        // ONLINE (resp2): `nbatches` is small, but when a query is given several
        // cores (few queries, many cores) `max_threads > 1` lets the mask-product
        // contraction use them; at `max_threads = 1` it stays sequential.
        let (prepared, precomputes, durations) = self.precompute_pack_mask_inner(
            all_digits,
            q_masks,
            gamma,
            key_mask_source,
            key_stride,
            max_threads,
            Some(pool),
        );
        // This is a parallel/nested region, so the arithmetic counters overlap.
        // Record its complete wall time once as the exclusive phase and expose
        // worker maxima as diagnostics instead of scaling them to fill the gap.
        timings.add_worker_region(
            "recursion.resp2.worker_region",
            durations.worker_region.region_wall,
        );
        // Preserve the typed category counters as overlapping worker maxima;
        // `total()` uses the exclusive phase list and therefore does not add
        // these observations a second time.
        timings.prepare_db += durations.prepare_db;
        timings.mask_product += durations.mask_product;
        timings.mask_prep += durations.mask_prep;
        timings.pack_precompute += durations.pack_precompute;
        timings.record_diagnostic(phase_names.prepare_db, durations.prepare_db);
        timings.record_diagnostic(phase_names.mask_product, durations.mask_product);
        timings.record_diagnostic(phase_names.mask_prep, durations.mask_prep);
        timings.record_diagnostic(phase_names.pack_precompute, durations.pack_precompute);
        timings.record_diagnostic(
            "recursion.resp2.worker_allocation_max",
            durations.worker_region.allocation_max,
        );
        timings.record_diagnostic(
            "recursion.resp2.worker_callback_max",
            durations.worker_region.callback_max,
        );
        timings.record_diagnostic(
            "recursion.resp2.worker_deallocation_max",
            durations.worker_region.deallocation_max,
        );
        timings.record_diagnostic(
            "recursion.resp2.worker_critical",
            durations.worker_region.worker_critical,
        );
        timings.record_diagnostic(
            "recursion.resp2.scheduling_overhead",
            durations.worker_region.scheduling,
        );
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
        mut pool: Option<&mut [ScratchOwned<BE>]>,
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
        let qtilde = qtilde_bits(&self.params);
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
        let max_threads = max_threads.max(1);
        let online = pool.is_some();
        let use_pool = online && resp2_scratch_mode() == Resp2ScratchMode::Pooled;
        let (nthreads, mask_threads) = if online {
            let pool_len = pool.as_ref().map_or(0, |p| p.len());
            resp2_schedule(nbatches, max_threads, pool_len, use_pool)
        } else {
            let outer = num_threads_offline(nbatches).min(max_threads);
            (outer, (max_threads / outer).max(1))
        };
        assert!(
            nthreads * mask_threads <= max_threads,
            "resp2 outer × inner workers exceed the online thread budget"
        );
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

        let worker_region = {
            let res_infos = &res_infos;
            let gemm: &dyn Gemm = &*self.gemm;
            let mut slabs: Vec<&mut [BatchOut<BE>]> = Vec::with_capacity(work.len());
            let mut rest = outputs.as_mut_slice();
            for group in &work {
                let (head, tail) = rest.split_at_mut(group.len());
                slabs.push(head);
                rest = tail;
            }
            let run_batch = |slab: &mut [BatchOut<BE>],
                             group: &[crate::parallel::BlockWork],
                             sc: &mut ScratchOwned<BE>| {
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
                        qtilde,
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
            };
            if use_pool {
                let scratch_pool = pool
                    .take()
                    .expect("pooled resp2 requires a scratch pool");
                let scratch_slabs: Vec<&mut ScratchOwned<BE>> = scratch_pool[..work.len()]
                    .iter_mut()
                    .collect();
                scoped_workers_pooled_profiled::<BE, BatchOut<BE>, _>(
                    slabs,
                    scratch_slabs,
                    &work,
                    run_batch,
                )
            } else {
                scoped_workers_profiled::<BE, BatchOut<BE>, _>(
                    slabs,
                    &work,
                    bytes,
                    run_batch,
                )
            }
        };

        let mut durations = PackMaskDurations {
            worker_region,
            ..Default::default()
        };
        if online {
            // Diagnostics use maxima rather than sums: batches run in parallel,
            // so summing would be CPU work rather than elapsed time.
            for (_, d) in &outputs {
                durations.prepare_db = durations.prepare_db.max(d[0]);
                durations.mask_product = durations.mask_product.max(d[1]);
                durations.mask_prep = durations.mask_prep.max(d[2]);
                durations.pack_precompute = durations.pack_precompute.max(d[3]);
            }
        } else {
            for (_, d) in &outputs {
                durations.prepare_db += d[0];
                durations.mask_product += d[1];
                durations.mask_prep += d[2];
                durations.pack_precompute += d[3];
            }
        }
        if !online && nthreads > 1 {
            let cpu = durations.prepare_db
                + durations.mask_product
                + durations.mask_prep
                + durations.pack_precompute;
            if !cpu.is_zero() {
                let scale = worker_region.region_wall.as_secs_f64() / cpu.as_secs_f64();
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
        pool: &mut [ScratchOwned<BE>],
        max_threads: usize,
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
            pool,
            max_threads,
            timings,
            PackMaskPhaseNames {
                prepare_db: "recursion.resp2.prepare_db",
                mask_product: "recursion.resp2.mask_product",
                mask_prep: "recursion.resp2.mask_prep",
                pack_precompute: "recursion.resp2.pack_precompute",
            },
        )
    }

    /// Pack-scratch arena size, cached on first use: a pure function of the
    /// fixed parameters, but expensive to compute (`pack_precompute_tmp_bytes`
    /// runs the backend's sizing planner over a probe layout, ~30-60 ms) — and
    /// it is on the per-query online path twice (pool top-up + resp2 mask
    /// precompute), where recomputing it dominated the untimed wall-clock gap.
    pub(super) fn scratch_for_pack(&self) -> usize {
        *self
            .pack_scratch_bytes
            .get_or_init(|| self.compute_scratch_for_pack())
    }

    fn compute_scratch_for_pack(&self) -> usize {
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
                &module.vec_znx_alloc(1, size),
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
    qtilde_bits: usize,
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
    switch_final_mask_to_qtilde(module, &mut precompute, qtilde_bits, &mut scratch.borrow());
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
