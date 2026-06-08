//! Interpolation: the concrete second-dimension reduction unit.
//!
//! The PIR second dimension (which of the `nb_matrices` matrices) is reduced by
//! interpolating the matrices into a degree-`interpolation_t` polynomial and
//! evaluating it, by encrypted Horner, at the GGSW-encrypted root `X^i` the
//! client ships. This unit owns that whole step:
//! - [`Interpolation::prepare`] interpolates the database **in place**;
//! - [`Interpolation::build_query`] builds the GGSW root selector;
//! - [`Interpolation::reduce`] collapses the per-panel packed GLWEs by Horner.
//!
//! A future Inspire² reduction would be a *parallel* concrete unit with its own
//! query type — there is deliberately no shared trait.

use poulpy_core::{
    GGSWEncryptSk,
    layouts::{
        GGSW, GGSWInfos, GGSWPrepared, GGSWPreparedFactory, GGSWPreparedToBackendRef, GLWE,
        GLWEAutomorphismKeyCompressed, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef,
        ModuleCoreAlloc,
    },
};
use poulpy_hal::{
    api::{ModuleN, ScratchOwnedBorrow},
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ScalarZnx, ScratchOwned, VecZnx,
        VecZnxToBackendMut, VecZnxToBackendRef, ZnxInfos, ZnxView, ZnxViewMut,
    },
};

use crate::{
    client::{QueryCommon, QueryContext},
    database::{CoeffMatrix, Database, DatabaseLayout, PayloadAddress},
    encoding::ModPEncoder,
    interpolation::{HornerCoeffs, HornerEvaluation, MonomialInterpolation},
    parameters::Parameters,
    payload::Payload,
};

/// Full-packing keys used by the interpolation collapse.
pub struct InterpolationKeys<BE: Backend> {
    key_g: GLWEAutomorphismKeyCompressed<BE::OwnedBuf>,
    key_h: GLWEAutomorphismKeyCompressed<BE::OwnedBuf>,
}

impl<BE: Backend> InterpolationKeys<BE> {
    pub(crate) fn new(
        key_g: GLWEAutomorphismKeyCompressed<BE::OwnedBuf>,
        key_h: GLWEAutomorphismKeyCompressed<BE::OwnedBuf>,
    ) -> Self {
        Self { key_g, key_h }
    }

    pub(crate) fn key_g(&self) -> &GLWEAutomorphismKeyCompressed<BE::OwnedBuf> {
        &self.key_g
    }

    pub(crate) fn key_h(&self) -> &GLWEAutomorphismKeyCompressed<BE::OwnedBuf> {
        &self.key_h
    }
}

/// The interpolation reduction's query: the common first-dim material, the
/// full-packing keys (interpolation uses full pack), and the GGSW root `Enc(X^i)`
/// selecting the target matrix.
pub struct InterpolationQuery<BE: Backend> {
    pub common: QueryCommon<BE>,
    pub(crate) keys: InterpolationKeys<BE>,
    pub root: GGSW<BE::OwnedBuf>,
}

/// The interpolation server response: one packed GLWE holding the selected
/// column.
pub struct InterpolationResponse<BE: Backend> {
    selected: GLWE<BE::OwnedBuf>,
}

impl<BE: Backend> InterpolationResponse<BE> {
    pub(crate) fn new(selected: GLWE<BE::OwnedBuf>) -> Self {
        Self { selected }
    }

    pub fn selected(&self) -> &GLWE<BE::OwnedBuf> {
        &self.selected
    }
}

/// A database interpolated **in place**: panels `0..nb_matrices` live in `db`'s
/// own matrices (overwritten), the `interpolation_t − nb_matrices` tail panels
/// in [`tail`](Self::tail). [`panel`](Self::panel) presents all `interpolation_t`
/// panels uniformly as `k_blocks` sub-matrices each.
pub struct Interpolated<'a> {
    db_matrices: &'a [CoeffMatrix],
    tail: Vec<CoeffMatrix>,
    nb_matrices: usize,
    k_blocks: usize,
}

impl Interpolated<'_> {
    /// Panel `k`'s `k_blocks` `U` sub-matrices (`db` for `k < nb_matrices`,
    /// otherwise the tail).
    pub fn panel(&self, k: usize) -> &[CoeffMatrix] {
        let kb = self.k_blocks;
        if k < self.nb_matrices {
            &self.db_matrices[k * kb..k * kb + kb]
        } else {
            let base = (k - self.nb_matrices) * kb;
            &self.tail[base..base + kb]
        }
    }
}

/// The interpolation reduction unit, parameterized by the database layout (for
/// the interpolation degree / grid) and the cryptosystem (for the GGSW layout).
pub struct Interpolation {
    interpolation_t: usize,
    nb_matrices: usize,
    k_blocks: usize,
    n: usize,
    ggsw_layout: poulpy_core::EncryptionLayout<poulpy_core::layouts::GGSWLayout>,
}

impl Interpolation {
    /// Build from a [`DatabaseLayout`] and the cryptosystem [`Parameters`] (only
    /// the GGSW layout is taken).
    pub fn new<BE: Backend, P: Payload<[u8; 32]>>(
        layout: &DatabaseLayout<P>,
        params: &Parameters<BE, [u8; 32], P>,
    ) -> Self {
        let n = params.n();
        Self {
            interpolation_t: layout.interpolation_t(n),
            nb_matrices: layout.block_rows(n),
            k_blocks: layout.block_cols(n),
            n,
            ggsw_layout: params.ggsw_layout(),
        }
    }

    /// Number of packed panels the first-dim loop must produce (= `interpolation_t`).
    pub fn num_panels(&self) -> usize {
        self.interpolation_t
    }

    /// Interpolate the database **in place** along the matrix axis. Each panel
    /// `U[j]` column-block `bc` becomes `inv_t · IDFT_m(db[m, bc])` at the same
    /// coefficient position — no transpose, because the stored orientation is
    /// already the matmul `U` orientation. Panels `0..nb_matrices` overwrite the
    /// database's matrices; the tail panels are returned in [`Interpolated`].
    pub fn prepare<'a, BE: Backend<OwnedBuf = Vec<u8>>, P: Payload<[u8; 32]>>(
        &self,
        module: &Module<BE>,
        db: &'a mut Database<BE, P>,
        encoder: &ModPEncoder,
        scratch: &mut ScratchOwned<BE>,
    ) -> Interpolated<'a>
    where
        Module<BE>: ModuleN + ModuleCoreAlloc<OwnedBuf = Vec<u8>> + MonomialInterpolation<BE>,
        ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
        VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
        for<'b> BE::BufRef<'b>: HostDataRef,
        for<'b> BE::BufMut<'b>: HostDataMut,
    {
        let n = self.n;
        let kb = self.k_blocks;
        let nb = self.nb_matrices;
        let t = self.interpolation_t;
        let inv_t = encoder.inv(t as i64);

        // Tail storage, panel-major: tail[(j - nb) * kb + bc].
        let mut tail: Vec<CoeffMatrix> = (0..(t - nb) * kb)
            .map(|_| CoeffMatrix::zeros(n, n))
            .collect();

        // Working set of `t` polynomials (n cols, one limb), reused across blocks.
        let mut working: Vec<VecZnx<Vec<u8>>> =
            (0..t).map(|_| module.vec_znx_alloc(n, 1)).collect();

        for bc in 0..kb {
            // Load the nb_matrices evaluation panels; zero-pad up to interpolation_t.
            for (m, w) in working.iter_mut().enumerate() {
                if m < nb {
                    let src = &db.matrices()[m * kb + bc];
                    for col in 0..n {
                        let w_col = w.at_mut(col, 0);
                        for (wv, &sv) in w_col.iter_mut().zip(src.row(col).iter()) {
                            *wv = sv as i64;
                        }
                    }
                } else {
                    for col in 0..n {
                        w.at_mut(col, 0).fill(0);
                    }
                }
            }

            // In-place radix-2 IDFT over the matrix axis, per column.
            for col in 0..n {
                module.monomial_interpolate(&mut working, col, &mut scratch.borrow());
            }

            // Scale by inv_t and write each panel back (db for j < nb, else tail).
            for (j, w) in working.iter().enumerate() {
                let dst = if j < nb {
                    &mut db.matrices_mut()[j * kb + bc]
                } else {
                    &mut tail[(j - nb) * kb + bc]
                };
                for col in 0..n {
                    let s = w.at(col, 0);
                    let d = dst.row_mut(col);
                    for (out, &raw) in d.iter_mut().zip(s.iter()) {
                        *out = encoder.mul(raw, inv_t) as i16;
                    }
                }
            }
        }

        Interpolated {
            db_matrices: db.matrices(),
            tail,
            nb_matrices: nb,
            k_blocks: kb,
        }
    }

    /// Interpolate `plain` (`block_rows × block_cols`) **out of place** into the
    /// `matrix` database (`interpolation_t × block_cols`), leaving `plain`
    /// untouched (so a server can re-run it after a payload `update`). `matrix`
    /// must already be sized to `interpolation_t` block-rows.
    pub fn interpolate_into<BE: Backend<OwnedBuf = Vec<u8>>, P: Payload<[u8; 32]>>(
        &self,
        module: &Module<BE>,
        plain: &Database<BE, P>,
        matrix: &mut Database<BE, P>,
        encoder: &ModPEncoder,
        scratch: &mut ScratchOwned<BE>,
    ) where
        Module<BE>: ModuleN + ModuleCoreAlloc<OwnedBuf = Vec<u8>> + MonomialInterpolation<BE>,
        ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
        VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE> + ZnxInfos,
        for<'b> BE::BufRef<'b>: HostDataRef,
        for<'b> BE::BufMut<'b>: HostDataMut,
    {
        let n = self.n;
        let kb = self.k_blocks;
        let nb = self.nb_matrices;
        let t = self.interpolation_t;
        let inv_t = encoder.inv(t as i64);
        assert_eq!(
            matrix.matrices().len(),
            t * kb,
            "matrix DB must hold interpolation_t block-rows"
        );

        let mut working: Vec<VecZnx<Vec<u8>>> =
            (0..t).map(|_| module.vec_znx_alloc(n, 1)).collect();
        for bc in 0..kb {
            for (m, w) in working.iter_mut().enumerate() {
                if m < nb {
                    let src = &plain.matrices()[m * kb + bc];
                    for col in 0..n {
                        let w_col = w.at_mut(col, 0);
                        for (wv, &sv) in w_col.iter_mut().zip(src.row(col).iter()) {
                            *wv = sv as i64;
                        }
                    }
                } else {
                    for col in 0..n {
                        w.at_mut(col, 0).fill(0);
                    }
                }
            }
            for col in 0..n {
                module.monomial_interpolate(&mut working, col, &mut scratch.borrow());
            }
            for (j, w) in working.iter().enumerate() {
                let dst = &mut matrix.matrices_mut()[j * kb + bc];
                for col in 0..n {
                    let s = w.at(col, 0);
                    let d = dst.row_mut(col);
                    for (out, &raw) in d.iter_mut().zip(s.iter()) {
                        *out = encoder.mul(raw, inv_t) as i16;
                    }
                }
            }
        }
    }

    /// Build the interpolation query: wrap the common material and the GGSW root
    /// `Enc(X^i)` for `addr.matrix`, encrypted under the client's secret handles.
    pub fn build_query<BE: Backend<OwnedBuf = Vec<u8>>>(
        &self,
        module: &Module<BE>,
        common: QueryCommon<BE>,
        ctx: &mut QueryContext<BE>,
        addr: &PayloadAddress,
        scratch: &mut ScratchOwned<BE>,
    ) -> InterpolationQuery<BE>
    where
        Module<BE>: ModuleN + ModuleCoreAlloc<OwnedBuf = Vec<u8>> + GGSWEncryptSk<BE>,
        ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
        for<'b> BE::BufRef<'b>: HostDataRef,
        for<'b> BE::BufMut<'b>: HostDataMut,
    {
        let exponent = interpolation_root_exponent(module.n(), addr.matrix, self.interpolation_t);
        let root_pt = root_monomial(module, exponent);
        let mut root = module.ggsw_alloc_from_infos(&self.ggsw_layout);
        module.ggsw_encrypt_sk(
            &mut root,
            &root_pt,
            &ctx.sk_pack_prep,
            &self.ggsw_layout,
            &mut ctx.source_xe,
            &mut ctx.source_xa,
            &mut scratch.borrow(),
        );
        // Take the full-packing keys the client generated and forwarded.
        let keys = ctx.take_interpolation_keys();
        InterpolationQuery { common, keys, root }
    }

    /// Prepare the received GGSW root for the Horner evaluation.
    pub fn prepare_root<BE: Backend<OwnedBuf = Vec<u8>>>(
        &self,
        module: &Module<BE>,
        root: &GGSW<BE::OwnedBuf>,
        scratch: &mut ScratchOwned<BE>,
    ) -> GGSWPrepared<BE::OwnedBuf, BE>
    where
        Module<BE>: GGSWPreparedFactory<BE>,
        ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
    {
        let mut prepared = module.ggsw_prepared_alloc_from_infos(root);
        module.ggsw_prepare(&mut prepared, root, &mut scratch.borrow());
        prepared
    }

    /// Reduce the per-panel packed GLWEs into the answer by encrypted Horner at
    /// the GGSW root.
    pub fn reduce<BE: Backend<OwnedBuf = Vec<u8>>>(
        &self,
        module: &Module<BE>,
        packed: &[GLWE<BE::OwnedBuf>],
        root_prepared: &GGSWPrepared<BE::OwnedBuf, BE>,
        res: &mut GLWE<BE::OwnedBuf>,
        scratch: &mut ScratchOwned<BE>,
    ) where
        Module<BE>: HornerEvaluation<BE>,
        ScratchOwned<BE>: ScratchOwnedBorrow<BE>,
        GLWE<BE::OwnedBuf>: GLWEToBackendRef<BE> + GLWEToBackendMut<BE> + GLWEInfos,
        GGSWPrepared<BE::OwnedBuf, BE>: GGSWPreparedToBackendRef<BE> + GGSWInfos,
    {
        let coeffs = HornerCoeffs(packed);
        module.horner_evaluate(res, &coeffs, root_prepared, &mut scratch.borrow());
    }
}

/// Interpolation-root exponent for a matrix index: `point · (2n / interpolation_t)`.
pub fn interpolation_root_exponent(n: usize, point: usize, interpolation_t: usize) -> usize {
    point * (2 * n / interpolation_t)
}

/// The `X^i` / `-X^{i-n}` monomial plaintext selecting an interpolation point.
fn root_monomial<BE: Backend<OwnedBuf = Vec<u8>>>(
    module: &Module<BE>,
    exponent: usize,
) -> ScalarZnx<BE::OwnedBuf>
where
    Module<BE>: ModuleN + ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
{
    let n = module.n();
    let exponent = exponent % (2 * n);
    let mut root = module.scalar_znx_alloc(1);
    if exponent < n {
        root.at_mut(0, 0)[exponent] = 1;
    } else {
        root.at_mut(0, 0)[exponent - n] = -1;
    }
    root
}
