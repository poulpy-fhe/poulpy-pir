//! PIR server: owns the plaintext database and its interpolated matrix form,
//! materializes the query mask `A` from its public [`ServerSeed`], runs the
//! query-independent OFFLINE pre-processing, and answers a client [`Query`].
//!
//! Phases:
//! - SETUP — [`Server::generate_query_mask`]: materialize the `block_cols` query
//!   masks from `server_seed.mask()`. Depends only on the public seed + DB shape,
//!   so it is reused across both DB updates and queries.
//! - OFFLINE — [`Server::offline`]: interpolate the plaintext DB into the matrix
//!   DB, then per interpolation panel compute `U·A`, `packing_mask_preprocessing`
//!   and `pack_precompute`. Depends on DB content + masks, query-independent.
//! - ONLINE — [`Server::respond`]: per panel `U·b`, `pack`, then the Horner
//!   reduction at the query's GGSW root.
//!
//! Host backends only (`BE::OwnedBuf = Vec<u8>`).

#![allow(clippy::too_many_arguments)]

use std::time::{Duration, Instant};

use poulpy_core::{
    CoeffMatrixPrepare, GLWEExpandLWEMatrix, GLWEMaskFill, LWEMatrixMul,
    layouts::{
        Base2K, CoeffMatrixPreparedOwned, Degree, GGSWPreparedFactory, GLWE, GLWECompressed,
        GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, LWEMatrix, LWEMatrixInfos,
        LWEMatrixLayout, LWEMatrixToBackendMut, ModuleCoreAlloc, TorusPrecision,
    },
};
use poulpy_hal::{
    api::{
        CoeffGemmPrepare, ModuleN, ModuleNew, ScratchOwnedAlloc, ScratchOwnedBorrow,
        VecZnxAddAssignBackend, VecZnxCopyBackend, VecZnxMatMulPrepared, VecZnxMatMulTmpBytes,
        VecZnxNormalize, VecZnxNormalizeTmpBytes, VecZnxZeroBackend, VmpPrepare, VmpPrepareTmpBytes,
        VmpZero,
    },
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScratchArena, ScratchOwned, VecZnx,
        VecZnxToBackendMut, VecZnxToBackendRef,
    },
    source::Source,
};

use crate::{
    client::{MaskSeeds, Response, ServerSeed},
    database::{Database, DatabaseInfos, DatabaseLayout},
    interpolation::{HornerEvaluation, Interpolation, InterpolationQuery, MonomialInterpolation},
    packing::{Packing, PackingKeysGenerate, PackingMaskAggregation, PackingPrecomputations},
    parameters::Parameters,
    payload::Payload,
};

/// PIR server, generic over backend `BE` and payload encoding `P`.
pub struct Server<BE: Backend, P: Payload<[u8; 32]>> {
    params: Parameters<BE>,
    layout: DatabaseLayout<P>,
    interpolation: Interpolation,
    server_seed: ServerSeed,
    /// Plaintext payload matrices (source of truth; `update`-able).
    plain: Database<BE, P>,
    /// Interpolated `U` matrices (`interpolation_t` block-rows), rebuilt by `offline`.
    matrix: Database<BE, P>,
    /// Query masks `A`, one per block-column (from `generate_query_mask`).
    masks: Vec<LWEMatrix<BE::OwnedBuf>>,
    /// Prepared `U` panels (`interpolation_t × block_cols`), from `offline`.
    prepared_u: Vec<Vec<CoeffMatrixPreparedOwned<BE, i16>>>,
    /// Fixed mask-side packing precomputations (one per panel), from `offline`.
    precomputations: Vec<PackingPrecomputations<BE>>,
    scratch: ScratchOwned<BE>,
}

impl<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload<[u8; 32]>> Server<BE, P>
where
    Module<BE>: ModuleNew<BE>
        + ModuleN
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
        + MonomialInterpolation<BE>
        + HornerEvaluation<BE>
        + CoeffMatrixPrepare<BE>
        + LWEMatrixMul<BE>
        + PackingMaskAggregation<BE>
        + Packing<BE>
        + PackingKeysGenerate<BE>
        + GGSWPreparedFactory<BE>
        + GLWEExpandLWEMatrix<BE>
        + GLWEMaskFill<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxAddAssignBackend<BE>
        + VecZnxCopyBackend<BE>
        + VecZnxZeroBackend<BE>
        + VecZnxMatMulTmpBytes
        + VecZnxMatMulPrepared<BE>
        + CoeffGemmPrepare<BE>
        + VmpPrepare<BE>
        + VmpPrepareTmpBytes
        + VmpZero<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + poulpy_hal::layouts::ZnxInfos,
    LWEMatrix<Vec<u8>>: LWEMatrixToBackendMut<BE>,
    GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    /// Build a server for `layout`. Allocates the plaintext DB and the (empty)
    /// matrix DB, and generates a random public [`ServerSeed`].
    pub fn new(layout: DatabaseLayout<P>) -> Self {
        let params = Parameters::<BE>::default();
        assert_eq!(
            layout.n(),
            params.n(),
            "database n must match the cryptosystem ring degree"
        );
        let module = params.module();
        let base2k = params.base2k();

        let mut root = [0u8; 32];
        getrandom::fill(&mut root).expect("OS entropy");
        let server_seed = ServerSeed::new(root);

        let plain = layout.instantiate(module, base2k);
        let matrix_layout =
            DatabaseLayout::<P>::new(layout.n(), layout.interpolation_t(), layout.block_cols());
        let matrix = matrix_layout.instantiate(module, base2k);

        let interpolation = Interpolation::new(&layout, &params);
        let scratch = ScratchOwned::<BE>::alloc(server_scratch_bytes(&params));

        Self {
            params,
            layout,
            interpolation,
            server_seed,
            plain,
            matrix,
            masks: Vec::new(),
            prepared_u: Vec::new(),
            precomputations: Vec::new(),
            scratch,
        }
    }

    /// The public seed the client needs to build its query.
    pub fn server_seed(&self) -> ServerSeed {
        self.server_seed
    }

    /// The database layout (the client needs it to resolve a payload `address`).
    pub fn layout(&self) -> &DatabaseLayout<P> {
        &self.layout
    }

    /// Overwrite the single payload at index `i`. Reflected in answers only after
    /// the next [`offline`](Self::offline).
    pub fn update(&mut self, i: usize, value: [u8; 32]) {
        self.plain.encode_shard(i, &[value]);
    }

    /// Plaintext lookup of the payload at index `i` from the server's own DB (the
    /// ground truth a PIR query should return). The server owns its plaintext, so
    /// this is a legitimate server-side read — distinct from the encrypted query.
    pub fn get(&self, i: usize) -> [u8; 32] {
        self.plain.payload(i)
    }

    /// Bulk-write `values` starting at payload index `start`.
    pub fn update_shard(&mut self, start: usize, values: &[[u8; 32]]) {
        self.plain.encode_shard(start, values);
    }

    /// SETUP: materialize the per-block-column query masks `A` from the public
    /// `server_seed.mask()`. Reused across DB updates and queries.
    pub fn generate_query_mask(&mut self) {
        let glwe_query = self.params.glwe_query();
        let mask_seeds = MaskSeeds::new(self.server_seed.mask());
        let block_cols = self.layout.block_cols();
        let mut masks = Vec::with_capacity(block_cols);
        for bc in 0..block_cols {
            let mut mask = self
                .params
                .module()
                .lwe_matrix_alloc_from_infos(&mask_regime_infos(&self.params));
            fill_default_query_mask(
                self.params.module(),
                &mut mask,
                mask_seeds.seed(bc),
                &glwe_query,
                &self.params.glwe_mask(),
                &mut self.scratch.borrow(),
            );
            masks.push(mask);
        }
        self.masks = masks;
    }

    /// OFFLINE: interpolate the plaintext DB into the matrix DB, then per
    /// interpolation panel compute `U·A`, `packing_mask_preprocessing` and
    /// `pack_precompute`. Query-independent; re-run after DB updates. Returns a
    /// per-step timing breakdown.
    pub fn offline(&mut self) -> OfflineTimings {
        if self.masks.is_empty() {
            self.generate_query_mask();
        }
        let mut timings = OfflineTimings::default();

        let encoder = self.params.encoder();
        let t = Instant::now();
        self.interpolation.interpolate_into(
            self.params.module(),
            &self.plain,
            &mut self.matrix,
            &encoder,
            &mut self.scratch,
        );
        timings.interpolation = t.elapsed();

        let module = self.params.module();
        let panels = self.interpolation.num_panels();
        let block_cols = self.layout.block_cols();

        // Prepare the matrix DB panels (the matmul operands).
        let t = Instant::now();
        let mut prepared_u: Vec<Vec<CoeffMatrixPreparedOwned<BE, i16>>> = Vec::with_capacity(panels);
        for panel in 0..panels {
            let row: Vec<_> = (0..block_cols)
                .map(|bc| module.coeff_matrix_prepare(&self.matrix.matrices()[panel * block_cols + bc]))
                .collect();
            prepared_u.push(row);
        }
        self.prepared_u = prepared_u;
        timings.prepare_u = t.elapsed();

        // A throwaway key whose mask is seeded by `server_seed.keys()` — its body
        // is ignored by `pack_precompute`, which consumes only the mask seed.
        let key_mask_src = {
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
        };

        let lwe_infos = self.params.lwe_matrix_infos();
        let precompute_metadata = self.params.packing_precompute_infos();
        let mut aggregate = module.vec_znx_alloc(self.params.n(), lwe_infos.size());

        let mut precomputations = Vec::with_capacity(panels);
        for panel in 0..panels {
            let mut product = module.lwe_matrix_alloc_from_infos(&lwe_infos);
            let t = Instant::now();
            accumulate_mask_product(
                module,
                &self.params,
                &mut product,
                &self.prepared_u[panel],
                &self.masks,
                &mut self.scratch,
            );
            timings.ua_mask += t.elapsed();
            let t = Instant::now();
            module.packing_mask_preprocessing(
                &mut aggregate,
                self.params.base2k(),
                product.mask(),
                &mut self.scratch.borrow(),
            );
            timings.mask_prep += t.elapsed();
            let mut precompute = module.pack_precompute_alloc(
                precompute_metadata.steps(),
                precompute_metadata.size(),
                precompute_metadata.base2k(),
                precompute_metadata.baby_size(),
            );
            let t = Instant::now();
            module.pack_precompute(
                &mut precompute,
                &aggregate,
                &key_mask_src,
                &mut self.scratch.borrow(),
            );
            timings.pack_precompute += t.elapsed();
            precomputations.push(precompute);
        }
        self.precomputations = precomputations;
        timings
    }

    /// ONLINE: answer `query`. Per panel computes `U·b`, packs it, then evaluates
    /// the interpolated polynomial at the query's GGSW root (Horner). Returns the
    /// [`Response`] and a per-step timing breakdown.
    pub fn respond(&mut self, query: &InterpolationQuery<BE>) -> (Response<BE>, OnlineTimings) {
        assert!(
            !self.precomputations.is_empty(),
            "call offline() before respond()"
        );
        let mut timings = OnlineTimings::default();
        let module = self.params.module();
        let glwe_pack = self.params.glwe_pack();
        let panels = self.interpolation.num_panels();

        let t = Instant::now();
        let key_precomputations = module.pack_keys_precompute(
            &query.common.key_g,
            &query.common.key_h,
            self.params.baby_size(),
            &mut self.scratch.borrow(),
        );
        timings.pack_keys_precompute = t.elapsed();

        let body_size = self.params.size_at(self.params.base2k());
        let mut packed_coeffs = Vec::with_capacity(panels);
        for panel in 0..panels {
            let mut body = module.vec_znx_alloc(1, body_size);
            let t = Instant::now();
            accumulate_body_product(
                module,
                &self.params,
                &mut body,
                &self.prepared_u[panel],
                &query.common.blocks,
                &mut self.scratch,
            );
            timings.ub_body += t.elapsed();
            let mut packed = module.glwe_alloc_from_infos(&glwe_pack);
            let t = Instant::now();
            module.pack(
                &mut packed,
                &body,
                &self.precomputations[panel],
                &key_precomputations,
                1,
                &mut self.scratch.borrow(),
            );
            timings.pack += t.elapsed();
            packed_coeffs.push(packed);
        }

        let t = Instant::now();
        let root_prepared = self.interpolation.prepare_root(module, &query.root, &mut self.scratch);
        timings.prepare_root = t.elapsed();
        let mut selected = module.glwe_alloc_from_infos(&glwe_pack);
        let t = Instant::now();
        self.interpolation
            .reduce(module, &packed_coeffs, &root_prepared, &mut selected, &mut self.scratch);
        timings.reduce = t.elapsed();
        (Response { selected }, timings)
    }
}

/// Per-step OFFLINE timing breakdown (query-independent pre-processing).
#[derive(Default, Clone, Copy, Debug)]
pub struct OfflineTimings {
    pub interpolation: Duration,
    pub prepare_u: Duration,
    pub ua_mask: Duration,
    pub mask_prep: Duration,
    pub pack_precompute: Duration,
}

impl OfflineTimings {
    pub fn total(&self) -> Duration {
        self.interpolation + self.prepare_u + self.ua_mask + self.mask_prep + self.pack_precompute
    }
}

/// Per-step ONLINE timing breakdown (per query).
#[derive(Default, Clone, Copy, Debug)]
pub struct OnlineTimings {
    pub pack_keys_precompute: Duration,
    pub ub_body: Duration,
    pub pack: Duration,
    pub prepare_root: Duration,
    pub reduce: Duration,
}

impl OnlineTimings {
    pub fn total(&self) -> Duration {
        self.pack_keys_precompute + self.ub_body + self.pack + self.prepare_root + self.reduce
    }
}

/// Scratch large enough for every server operation.
fn server_scratch_bytes<BE: Backend>(params: &Parameters<BE>) -> usize
where
    Module<BE>: ModuleN
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
        + GLWEExpandLWEMatrix<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxMatMulTmpBytes
        + MonomialInterpolation<BE>
        + PackingMaskAggregation<BE>
        + Packing<BE>
        + PackingKeysGenerate<BE>
        + GGSWPreparedFactory<BE>
        + HornerEvaluation<BE>
        + VmpPrepare<BE>
        + VmpPrepareTmpBytes
        + VmpZero<BE>,
{
    let module = params.module();
    let glwe_query = params.glwe_query();
    let glwe_pack = params.glwe_pack();
    let glwe_mask = params.glwe_mask();
    let lwe_infos = params.lwe_matrix_infos();
    let key_infos = params.key_layout();
    let ggsw_infos = params.ggsw_layout();
    let precompute_metadata = params.packing_precompute_infos();
    let aggregate = module.vec_znx_alloc(params.n(), lwe_infos.size());
    0usize
        .max(module.monomial_interpolate_tmp_bytes(1))
        .max(module.pack_keys_generate_tmp_bytes(&key_infos))
        .max(module.pack_keys_precompute_tmp_bytes(&key_infos, &key_infos, params.baby_size()))
        .max(module.ggsw_prepare_tmp_bytes(&ggsw_infos))
        .max(module.horner_evaluate_tmp_bytes(&glwe_pack, &ggsw_infos))
        .max(default_query_mask_tmp_bytes(
            module,
            &mask_regime_infos(params),
            &glwe_mask,
        ))
        .max(module.vec_znx_matmul_tmp_bytes(
            params.n(),
            params.n(),
            1,
            glwe_query.size(),
            1,
            glwe_query.size(),
        ))
        .max(module.vec_znx_normalize_tmp_bytes())
        .max(module.packing_mask_preprocessing_tmp_bytes(lwe_infos.size()))
        .max(module.pack_precompute_tmp_bytes(precompute_metadata, &aggregate, &key_infos))
}

// =============================================================================
// Query-mask / matmul helpers (moved from the example, generalized over `BE`).
// =============================================================================

fn default_query_mask_tmp_bytes<BE, R, GM>(module: &Module<BE>, dst_infos: &R, glwe_mask: &GM) -> usize
where
    BE: Backend,
    Module<BE>: GLWEExpandLWEMatrix<BE> + VecZnxNormalizeTmpBytes,
    R: LWEMatrixInfos,
    GM: GLWEInfos,
{
    module
        .vec_znx_normalize_tmp_bytes()
        .max(module.glwe_expand_lwe_matrix_tmp_bytes(dst_infos, glwe_mask))
}

/// Internal 2x32 mask-regime layout (`size * mask_base2k` precision).
fn mask_regime_infos<BE: Backend>(params: &Parameters<BE>) -> LWEMatrixLayout {
    let n = params.n();
    LWEMatrixLayout {
        rows: n,
        n: Degree(n as u32),
        base2k: Base2K(params.mask_base2k() as u32),
        k: TorusPrecision((params.size_at(params.mask_base2k()) * params.mask_base2k()) as u32),
    }
}

/// Fills the seed-derived query mask `A` into `dst` (in the mask regime).
fn fill_default_query_mask<BE, R, GF, GM>(
    module: &Module<BE>,
    dst: &mut R,
    seed: [u8; 32],
    glwe_fill: &GF,
    glwe_mask: &GM,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: GLWEExpandLWEMatrix<BE> + GLWEMaskFill<BE> + VecZnxZeroBackend<BE> + VecZnxNormalize<BE>,
    R: LWEMatrixToBackendMut<BE> + LWEMatrixInfos,
    GF: GLWEInfos,
    GM: GLWEInfos,
{
    assert_eq!(glwe_fill.n().as_usize(), module.n());
    assert_eq!(dst.n().as_usize(), glwe_fill.rank().as_usize() * module.n());
    assert!(dst.rows() <= module.n());
    assert_eq!(dst.base2k(), glwe_mask.base2k());

    let rank = glwe_fill.rank().as_usize();
    let mut fill_glwe = module.glwe_alloc_from_infos(glwe_fill);
    let mut coarse_glwe = module.glwe_alloc_from_infos(glwe_mask);

    {
        let mut fill_mut = GLWEToBackendMut::<BE>::to_backend_mut(&mut fill_glwe);
        module.vec_znx_zero_backend(fill_mut.data_mut(), 0);
    }
    module.fill_glwe_mask_from_seed(glwe_fill.base2k().as_usize(), &mut fill_glwe, 1, rank, seed);

    {
        let fill_ref = GLWEToBackendRef::<BE>::to_backend_ref(&fill_glwe);
        let mut coarse_mut = GLWEToBackendMut::<BE>::to_backend_mut(&mut coarse_glwe);
        for col in 0..rank + 1 {
            module.vec_znx_normalize(
                coarse_mut.data_mut(),
                glwe_mask.base2k().as_usize(),
                0,
                col,
                fill_ref.data(),
                glwe_fill.base2k().as_usize(),
                col,
                &mut scratch.borrow(),
            );
        }
    }

    module.glwe_expand_lwe_matrix(dst, &coarse_glwe, &mut scratch.borrow());
}

/// `product.mask = sum_bc U[bc] · A[bc]` for one interpolation panel.
fn accumulate_mask_product<BE>(
    module: &Module<BE>,
    params: &Parameters<BE>,
    product: &mut LWEMatrix<BE::OwnedBuf>,
    u_panel: &[CoeffMatrixPreparedOwned<BE, i16>],
    masks: &[LWEMatrix<BE::OwnedBuf>],
    scratch: &mut ScratchOwned<BE>,
) where
    BE: Backend<OwnedBuf = Vec<u8>>,
    Module<BE>: ModuleN
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
        + LWEMatrixMul<BE>
        + CoeffGemmPrepare<BE>
        + VecZnxMatMulPrepared<BE>
        + VecZnxMatMulTmpBytes
        + VecZnxCopyBackend<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxNormalize<BE>
        + VecZnxZeroBackend<BE>
        + VmpPrepare<BE>
        + VmpPrepareTmpBytes
        + VmpZero<BE>,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
    LWEMatrix<Vec<u8>>: LWEMatrixToBackendMut<BE>,
    VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    let n = params.n();
    let acc_infos = LWEMatrixLayout {
        rows: n,
        n: Degree(n as u32),
        base2k: Base2K(params.mask_base2k() as u32),
        k: TorusPrecision(params.k() as u32),
    };
    let mut tmp = module.lwe_matrix_alloc_from_infos(&acc_infos);
    let mut acc = module.lwe_matrix_alloc_from_infos(&acc_infos);
    for (block_col, mask) in masks.iter().enumerate() {
        module.lwe_matrix_mul_mask_prepared(
            &mut tmp,
            params.mask_base2k(),
            &u_panel[block_col],
            mask,
            params.mask_base2k(),
            &mut scratch.borrow(),
        );
        if block_col == 0 {
            copy_lwe_matrix_mask(module, &mut acc, &tmp);
        } else {
            add_assign_lwe_matrix_mask(module, &mut acc, &tmp);
        }
    }
    convert_lwe_matrix_mask(module, product, params.base2k(), &acc, params.mask_base2k(), &mut scratch.borrow());
}

/// `out_body = sum_bc U[bc] · b[bc]` for one interpolation panel. Writes a small
/// body `VecZnx` (no full `LWEMatrix`): online only needs the body, not the mask.
fn accumulate_body_product<BE>(
    module: &Module<BE>,
    params: &Parameters<BE>,
    out_body: &mut VecZnx<BE::OwnedBuf>,
    u_panel: &[CoeffMatrixPreparedOwned<BE, i16>],
    bodies: &[GLWECompressed<BE::OwnedBuf>],
    scratch: &mut ScratchOwned<BE>,
) where
    BE: Backend<OwnedBuf = Vec<u8>>,
    Module<BE>: ModuleN
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
        + LWEMatrixMul<BE>
        + CoeffGemmPrepare<BE>
        + VecZnxMatMulPrepared<BE>
        + VecZnxMatMulTmpBytes
        + VecZnxCopyBackend<BE>
        + VecZnxAddAssignBackend<BE>
        + VecZnxNormalize<BE>
        + VecZnxZeroBackend<BE>
        + VmpPrepare<BE>
        + VmpPrepareTmpBytes
        + VmpZero<BE>,
    ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
    VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    let size = params.size_at(params.matmul_base2k());
    let mut tmp = module.vec_znx_alloc(1, size);
    let mut acc = module.vec_znx_alloc(1, size);
    for (block_col, body) in bodies.iter().enumerate() {
        module.lwe_matrix_mul_body_prepared(
            &mut tmp,
            params.matmul_base2k(),
            &u_panel[block_col],
            body.data(),
            params.matmul_base2k(),
            &mut scratch.borrow(),
        );
        if block_col == 0 {
            copy_vec_znx(module, &mut acc, &tmp);
        } else {
            add_assign_vec_znx(module, &mut acc, &tmp);
        }
    }
    convert_vec_znx(module, out_body, params.base2k(), &acc, params.matmul_base2k(), &mut scratch.borrow());
}

fn copy_lwe_matrix_mask<BE>(module: &Module<BE>, res: &mut LWEMatrix<BE::OwnedBuf>, src: &LWEMatrix<BE::OwnedBuf>)
where
    BE: Backend,
    Module<BE>: VecZnxCopyBackend<BE>,
    LWEMatrix<BE::OwnedBuf>: LWEMatrixToBackendMut<BE>,
{
    let src_mask = VecZnxToBackendRef::<BE>::to_backend_ref(src.mask());
    let mut res_mask = VecZnxToBackendMut::<BE>::to_backend_mut(res.mask_mut());
    for col in 0..src.mask().cols() {
        module.vec_znx_copy_backend(&mut res_mask, col, &src_mask, col);
    }
}

fn add_assign_lwe_matrix_mask<BE>(module: &Module<BE>, res: &mut LWEMatrix<BE::OwnedBuf>, rhs: &LWEMatrix<BE::OwnedBuf>)
where
    BE: Backend,
    Module<BE>: VecZnxAddAssignBackend<BE>,
    LWEMatrix<BE::OwnedBuf>: LWEMatrixToBackendMut<BE>,
{
    let rhs_mask = VecZnxToBackendRef::<BE>::to_backend_ref(rhs.mask());
    let mut res_mask = VecZnxToBackendMut::<BE>::to_backend_mut(res.mask_mut());
    for col in 0..rhs.mask().cols() {
        module.vec_znx_add_assign_backend(&mut res_mask, col, &rhs_mask, col);
    }
}

fn copy_vec_znx<BE>(module: &Module<BE>, res: &mut VecZnx<BE::OwnedBuf>, src: &VecZnx<BE::OwnedBuf>)
where
    BE: Backend,
    Module<BE>: VecZnxCopyBackend<BE>,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
{
    let src_ref = VecZnxToBackendRef::<BE>::to_backend_ref(src);
    let mut res_mut = VecZnxToBackendMut::<BE>::to_backend_mut(res);
    module.vec_znx_copy_backend(&mut res_mut, 0, &src_ref, 0);
}

fn add_assign_vec_znx<BE>(module: &Module<BE>, res: &mut VecZnx<BE::OwnedBuf>, rhs: &VecZnx<BE::OwnedBuf>)
where
    BE: Backend,
    Module<BE>: VecZnxAddAssignBackend<BE>,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
{
    let rhs_ref = VecZnxToBackendRef::<BE>::to_backend_ref(rhs);
    let mut res_mut = VecZnxToBackendMut::<BE>::to_backend_mut(res);
    module.vec_znx_add_assign_backend(&mut res_mut, 0, &rhs_ref, 0);
}

fn convert_vec_znx<BE>(
    module: &Module<BE>,
    res: &mut VecZnx<BE::OwnedBuf>,
    res_base2k: usize,
    src: &VecZnx<BE::OwnedBuf>,
    src_base2k: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VecZnxNormalize<BE>,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
{
    let src_ref = VecZnxToBackendRef::<BE>::to_backend_ref(src);
    let mut res_mut = VecZnxToBackendMut::<BE>::to_backend_mut(res);
    module.vec_znx_normalize(&mut res_mut, res_base2k, 0, 0, &src_ref, src_base2k, 0, scratch);
}

fn convert_lwe_matrix_mask<BE>(
    module: &Module<BE>,
    res: &mut LWEMatrix<BE::OwnedBuf>,
    res_base2k: usize,
    src: &LWEMatrix<BE::OwnedBuf>,
    src_base2k: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend,
    Module<BE>: VecZnxNormalize<BE>,
    LWEMatrix<BE::OwnedBuf>: LWEMatrixToBackendMut<BE>,
    VecZnx<BE::OwnedBuf>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
{
    let src_mask = VecZnxToBackendRef::<BE>::to_backend_ref(src.mask());
    let mut res_mask = VecZnxToBackendMut::<BE>::to_backend_mut(res.mask_mut());
    for col in 0..src.mask().cols() {
        module.vec_znx_normalize(&mut res_mask, res_base2k, 0, col, &src_mask, src_base2k, col, scratch);
    }
}
