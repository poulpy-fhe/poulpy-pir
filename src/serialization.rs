//! Wire (de)serialization for the transmitted PIR messages: the client
//! [`Query`] and the server [`Response`].
//!
//! ## Size optimization: base2k repacking
//!
//! Ciphertext coefficients are limb-decomposed in base `2^base2k`, stored one
//! `i64` per limb with `size = ceil(k / base2k)` limbs. The homomorphic
//! pipeline runs at a small working `base2k` (e.g. 18), but for *transmission*
//! the limbs only need to be a lossless re-encoding of the same torus value. So
//! before serializing each **non-gadget** ciphertext we renormalize it to the
//! densest `i64`-limb packing, `base2k = 63` (1 sign bit), which collapses the
//! limb count to `ceil(k / 63)` (e.g. 3 limbs → 1) and shrinks the payload
//! ~`63/base2k`×. The receiver renormalizes back to the working `base2k` (taken
//! from its [`Parameters`]) right after reading, so the rest of the system is
//! unaffected. The round-trip is exact: `size·base2k ≥ k` holds at both ends.
//!
//! Seed-compressed ciphertexts only ship their body; the mask is regenerated
//! from the seed *at the stored base2k*, so the wire body is restored to the
//! working base2k before the (later) decompression — the stored base2k/seed
//! stay at the working value.
//!
//! ### Gadget types (GGSW root, compressed automorphism keys)
//!
//! These are gadget ciphertexts: their `base2k` sets the gadget decomposition
//! spacing and they carry the structural invariant `dnum·dsize ≤ size`, which a
//! base2k=63 *typed* re-limbing would violate (the typed allocator would
//! reject it). But each individual matrix entry is an ordinary polynomial whose
//! limbs can be re-encoded losslessly, so we repack **per entry** to base2k=63
//! and restore on read; the gadget shape, Galois element, and PRNG seeds are
//! recovered from `Parameters` / explicit metadata. The gadget structure is
//! never used in the base2k=63 form — it only exists in flight.
//!
//! ## Wire size vs. theoretical minimum
//!
//! After repacking, every ciphertext is stored at `size = ceil(k / 63)` limbs of
//! the true torus precision `k` (not the working `max_k`, which can spill an
//! extra limb when `base2k` doesn't divide `k` — e.g. the 4×16 query regime,
//! `k=54` but `max_k=64`). This lands within **~1.15×** of the information-
//! theoretic minimum: each limb is a full `i64` (8 bytes) holding 63 bits, where
//! a tight bit-packing would store those bits in 7 bytes. That 8/7 ≈ 1.15× is the
//! only remaining overhead, left for a future optimization (sub-byte limb
//! packing).
//!
//! ## API
//!
//! Repacking needs `vec_znx_normalize`, so (de)serialization is module-aware
//! rather than the plain [`WriterTo`] trait: write with the owner's
//! [`Module`], read with the shared [`Parameters`]. The server reads a query
//! with `server.params()`; the client reads a response with `client.params()`.
//! The leading byte is a collapse tag; `Vec<_>` parts carry a `u64` length
//! prefix; an unknown tag is rejected as [`std::io::ErrorKind::InvalidData`].

use std::io::{Read, Result, Write};

use poulpy_core::GLWENormalize;
use poulpy_core::layouts::{
    Base2K, Degree, GGLWECompressedSeed, GGLWECompressedSeedMut, GGLWECompressedToBackendMut,
    GGLWECompressedToBackendRef, GGLWEInfos, GGSWCompressed, GGSWCompressedSeed,
    GGSWCompressedSeedMut, GGSWCompressedToBackendMut, GGSWCompressedToBackendRef, GGSWInfos, GLWE,
    GLWEAutomorphismKeyCompressed, GLWECompressed, GLWECompressedSeed, GLWECompressedSeedMut,
    GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, GetGaloisElement, LWEInfos,
    ModuleCoreAlloc, ModuleCoreCompressedAlloc, Rank, SetGaloisElement,
};
use poulpy_hal::{
    api::{
        ModuleN, ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxNormalize, VecZnxNormalizeTmpBytes,
    },
    layouts::{
        Backend, HostDataMut, HostDataRef, Module, ReaderFrom, ScratchOwned, VecZnx,
        VecZnxToBackendMut, VecZnxToBackendRef, WriterTo, ZnxView, ZnxViewMut,
    },
};

use crate::{
    client::{QueryCommon, RecursionResponse, Response},
    interpolation::{InterpolationKeys, InterpolationQuery, InterpolationResponse},
    packing::recursion::qtilde_glwe_layout,
    parameters::Parameters,
    payload::Payload,
    server::{CompressedKey, Query, RecursionKeys, RecursionQuery, qtilde_bits},
};

/// Densest signed-`i64` limb packing: 63 magnitude bits + 1 sign bit.
const TRANSMIT_BASE2K: usize = 63;

/// Collapse tag for the InsPIRe (interpolation) variant.
const TAG_INTERPOLATION: u8 = 0;
/// Collapse tag for the InsPIRe² (recursion) variant.
const TAG_RECURSION: u8 = 1;

/// Bundle of everything the (de)serializers need from the cryptosystem: the
/// module (for `vec_znx_normalize`) and a scratch arena sized for it.
struct Ctx<'a, BE: Backend> {
    module: &'a Module<BE>,
    scratch: ScratchOwned<BE>,
}

// --- trait bounds shared by every (de)serialization entry point -------------
//
// Spelled once as a helper bound so the public methods stay readable.
trait SerBackend: Backend<OwnedBuf = Vec<u8>> {}
impl<BE: Backend<OwnedBuf = Vec<u8>>> SerBackend for BE {}

fn invalid_tag(kind: &str, tag: u8) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("unknown {kind} serialization tag: {tag}"),
    )
}

// --- little-endian scalar helpers (no extra dependency) ---------------------

fn write_u8<W: Write>(writer: &mut W, v: u8) -> Result<()> {
    writer.write_all(&[v])
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8> {
    let mut b = [0u8; 1];
    reader.read_exact(&mut b)?;
    Ok(b[0])
}

fn write_u64<W: Write>(writer: &mut W, v: u64) -> Result<()> {
    writer.write_all(&v.to_le_bytes())
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    reader.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn write_len<W: Write>(writer: &mut W, len: usize) -> Result<()> {
    write_u64(writer, len as u64)
}

fn read_len<R: Read>(reader: &mut R) -> Result<usize> {
    Ok(read_u64(reader)? as usize)
}

// --- base2k repacking helpers (non-gadget VecZnx-backed ciphertexts) --------

/// The `base2k = 63` GLWE layout holding the ciphertext's true torus precision
/// `k` (`size = ceil(k / 63)`), used for the dense transmission form. Using `k`
/// rather than `max_k` matters when the working `base2k` doesn't divide `k`
/// (e.g. the 4×16 query regime: `k = 54` but `max_k = 64`) — `max_k = 64` would
/// spill to 2 base2k=63 limbs, while the real 54-bit value fits in 1. The
/// padding bits above `k` are sign extension, so dropping them is lossless.
fn transmit_layout<I: GLWEInfos>(src: &I) -> GLWELayout {
    GLWELayout {
        n: src.n(),
        base2k: Base2K(TRANSMIT_BASE2K as u32),
        k: src.k(),
        rank: src.rank(),
    }
}

/// Renormalizes every column of a `VecZnx` from `src_base2k` to `dst_base2k`
/// (lossless re-limbing of the same value).
fn renorm_vec_znx<BE>(
    ctx: &mut Ctx<BE>,
    dst: &mut VecZnx<Vec<u8>>,
    src: &VecZnx<Vec<u8>>,
    dst_base2k: usize,
    src_base2k: usize,
) where
    BE: SerBackend,
    Module<BE>: VecZnxNormalize<BE>,
    VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
{
    let src_ref = src.to_backend_ref();
    let mut dst_mut = dst.to_backend_mut();
    for col in 0..src.cols() {
        ctx.module.vec_znx_normalize(
            &mut dst_mut,
            dst_base2k,
            0,
            col,
            &src_ref,
            src_base2k,
            col,
            &mut ctx.scratch.borrow(),
        );
    }
}

/// Copies every limb of every column from a host `VecZnx` into another
/// (same shape), as plain `i64` slices — used to move borrowed GGSW/GGLWE
/// matrix entries to/from an owned scratch buffer without backend-view
/// conversion.
fn copy_vec_znx_host<DS, DD>(dst: &mut VecZnx<DD>, src: &VecZnx<DS>)
where
    DS: HostDataRef,
    DD: HostDataMut,
{
    for col in 0..src.cols() {
        for limb in 0..src.size() {
            dst.at_mut(col, limb).copy_from_slice(src.at(col, limb));
        }
    }
}

// --- per-ciphertext (de)serializers -----------------------------------------

/// Writes a full `GLWE` in the dense base2k=63 form.
fn write_glwe<BE, W: Write>(ctx: &mut Ctx<BE>, writer: &mut W, ct: &GLWE<Vec<u8>>) -> Result<()>
where
    BE: SerBackend,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = Vec<u8>> + GLWENormalize<BE>,
    GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
{
    let mut dense = ctx.module.glwe_alloc_from_infos(&transmit_layout(ct));
    ctx.module
        .glwe_normalize(&mut dense, ct, &mut ctx.scratch.borrow());
    dense.write_to(writer)
}

/// Reads a full `GLWE`: read the dense base2k=63 form, renormalize back to the
/// working layout `infos`.
fn read_glwe<BE, R: Read, I: GLWEInfos>(
    ctx: &mut Ctx<BE>,
    reader: &mut R,
    infos: &I,
) -> Result<GLWE<Vec<u8>>>
where
    BE: SerBackend,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = Vec<u8>> + GLWENormalize<BE>,
    GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
{
    let mut working = ctx.module.glwe_alloc_from_infos(infos);
    let mut dense = ctx.module.glwe_alloc_from_infos(&transmit_layout(&working));
    dense.read_from(reader)?;
    ctx.module
        .glwe_normalize(&mut working, &dense, &mut ctx.scratch.borrow());
    Ok(working)
}

/// Writes a seed-compressed `GGSW` (body-only): the seeds, then each of the
/// `dnum·(rank+1)` one-column body entries in the dense base2k=63 form. A
/// gadget's base2k is structural, so the typed ciphertext can't be re-based;
/// each body entry is an ordinary polynomial repacked independently and
/// losslessly. The gadget shape is recovered from `Parameters` on read.
fn write_ggsw_compressed<BE, W: Write>(
    ctx: &mut Ctx<BE>,
    writer: &mut W,
    ggsw: &GGSWCompressed<Vec<u8>>,
) -> Result<()>
where
    BE: SerBackend,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = Vec<u8>> + GLWENormalize<BE>,
    GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
{
    let seeds = ggsw.seed();
    write_len(writer, seeds.len())?;
    for s in seeds {
        writer.write_all(s)?;
    }
    let dnum = ggsw.dnum().as_usize();
    let cols = ggsw.rank().as_usize() + 1;
    let entry_working = GLWELayout {
        n: ggsw.n(),
        base2k: ggsw.base2k(),
        k: ggsw.max_k(),
        rank: Rank(0),
    };
    let body = GGSWCompressedToBackendRef::<BE>::to_backend_ref(ggsw);
    for row in 0..dnum {
        for col in 0..cols {
            let mut work = ctx.module.glwe_alloc_from_infos(&entry_working);
            copy_vec_znx_host(work.data_mut(), body.at_view(row, col).data());
            let mut dense = ctx.module.glwe_alloc_from_infos(&transmit_layout(&work));
            ctx.module
                .glwe_normalize(&mut dense, &work, &mut ctx.scratch.borrow());
            dense.write_to(writer)?;
        }
    }
    Ok(())
}

/// Reads a seed-compressed `GGSW` written by [`write_ggsw_compressed`],
/// allocating the receiver at the working layout `infos`.
fn read_ggsw_compressed<BE, R: Read, A: GGSWInfos>(
    ctx: &mut Ctx<BE>,
    reader: &mut R,
    infos: &A,
) -> Result<GGSWCompressed<Vec<u8>>>
where
    BE: SerBackend,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = Vec<u8>> + ModuleCoreCompressedAlloc + GLWENormalize<BE>,
    GLWE<Vec<u8>>: GLWEToBackendRef<BE> + GLWEToBackendMut<BE>,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    let seed_len = read_len(reader)?;
    let mut seeds = vec![[0u8; 32]; seed_len];
    for s in &mut seeds {
        reader.read_exact(s)?;
    }
    let mut ggsw = ctx.module.ggsw_compressed_alloc_from_infos(infos);
    *ggsw.seed_mut() = seeds;

    let dnum = ggsw.dnum().as_usize();
    let cols = ggsw.rank().as_usize() + 1;
    let entry_dense = GLWELayout {
        n: ggsw.n(),
        base2k: Base2K(TRANSMIT_BASE2K as u32),
        k: ggsw.max_k(),
        rank: Rank(0),
    };
    let entry_working = GLWELayout {
        n: ggsw.n(),
        base2k: ggsw.base2k(),
        k: ggsw.max_k(),
        rank: Rank(0),
    };
    {
        let mut body = GGSWCompressedToBackendMut::<BE>::to_backend_mut(&mut ggsw);
        for row in 0..dnum {
            for col in 0..cols {
                let mut dense = ctx.module.glwe_alloc_from_infos(&entry_dense);
                dense.read_from(reader)?;
                let mut work = ctx.module.glwe_alloc_from_infos(&entry_working);
                ctx.module
                    .glwe_normalize(&mut work, &dense, &mut ctx.scratch.borrow());
                let mut view = body.at_view_mut(row, col);
                copy_vec_znx_host(view.data_mut(), work.data());
            }
        }
    }
    Ok(ggsw)
}

/// Writes a seed-compressed `GLWE`: the body in the dense base2k=63 form, plus
/// its (working-base2k) seed. The stored base2k stays at 63 for the body, but
/// the seed/working-base2k pairing is reconstructed by the reader.
fn write_glwe_compressed<BE, W: Write>(
    ctx: &mut Ctx<BE>,
    writer: &mut W,
    ct: &GLWECompressed<Vec<u8>>,
) -> Result<()>
where
    BE: SerBackend,
    Module<BE>: ModuleCoreCompressedAlloc + VecZnxNormalize<BE>,
    VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
{
    let mut dense = ctx
        .module
        .glwe_compressed_alloc_from_infos(&transmit_layout(ct));
    *dense.seed_mut() = *ct.seed();
    let src_base2k = ct.base2k().as_usize();
    let (dst_data, src_data) = (dense.data_mut(), ct.data());
    renorm_vec_znx(ctx, dst_data, src_data, TRANSMIT_BASE2K, src_base2k);
    dense.write_to(writer)
}

/// Reads a seed-compressed `GLWE`: dense base2k=63 body, restored to the working
/// layout `infos`, seed copied across.
fn read_glwe_compressed<BE, R: Read, I: GLWEInfos>(
    ctx: &mut Ctx<BE>,
    reader: &mut R,
    infos: &I,
) -> Result<GLWECompressed<Vec<u8>>>
where
    BE: SerBackend,
    Module<BE>: ModuleCoreCompressedAlloc + VecZnxNormalize<BE>,
    VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
{
    let mut working = ctx.module.glwe_compressed_alloc_from_infos(infos);
    let mut dense = ctx
        .module
        .glwe_compressed_alloc_from_infos(&transmit_layout(&working));
    dense.read_from(reader)?;
    *working.seed_mut() = *dense.seed();
    let dst_base2k = working.base2k().as_usize();
    let (dst_data, src_data) = (working.data_mut(), dense.data());
    renorm_vec_znx(ctx, dst_data, src_data, dst_base2k, TRANSMIT_BASE2K);
    Ok(working)
}

// --- top-level Query / Response ---------------------------------------------

impl<BE: Backend<OwnedBuf = Vec<u8>>> Query<BE> {
    /// Serialize the query, repacking non-gadget ciphertexts to base2k=63.
    pub fn write_to<W: Write>(&self, module: &Module<BE>, writer: &mut W) -> Result<()>
    where
        Module<BE>: ModuleN
            + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
            + ModuleCoreCompressedAlloc
            + GLWENormalize<BE>
            + VecZnxNormalize<BE>
            + VecZnxNormalizeTmpBytes,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
        for<'b> BE::BufRef<'b>: HostDataRef,
        for<'b> BE::BufMut<'b>: HostDataMut,
    {
        let mut ctx = Ctx {
            module,
            scratch: ScratchOwned::<BE>::alloc(module.vec_znx_normalize_tmp_bytes()),
        };
        match self {
            Query::Interpolation(q) => {
                write_u8(writer, TAG_INTERPOLATION)?;
                write_len(writer, q.common.blocks.len())?;
                for b in &q.common.blocks {
                    write_glwe_compressed(&mut ctx, writer, b)?;
                }
                write_auto_key(&mut ctx, writer, q.keys.key_g())?;
                write_auto_key(&mut ctx, writer, q.keys.key_h())?;
                write_ggsw_compressed(&mut ctx, writer, &q.root)
            }
            Query::Recursion(q) => {
                write_u8(writer, TAG_RECURSION)?;
                write_len(writer, q.src0.len())?;
                for b in &q.src0 {
                    write_glwe_compressed(&mut ctx, writer, b)?;
                }
                write_len(writer, q.src1.len())?;
                for b in &q.src1 {
                    write_glwe_compressed(&mut ctx, writer, b)?;
                }
                write_compressed_key(&mut ctx, writer, &q.keys.gamma0)?;
                write_compressed_key(&mut ctx, writer, &q.keys.gamma1)?;
                write_compressed_key(&mut ctx, writer, &q.keys.gamma2)
            }
        }
    }

    /// Deserialize a query written by [`Query::write_to`], sizing every buffer
    /// from `params`.
    pub fn read_from<R: Read, P: Payload<[u8; 32]>>(
        reader: &mut R,
        params: &Parameters<BE, [u8; 32], P>,
    ) -> Result<Self>
    where
        Module<BE>: ModuleN
            + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
            + ModuleCoreCompressedAlloc
            + GLWENormalize<BE>
            + VecZnxNormalize<BE>
            + VecZnxNormalizeTmpBytes,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
        for<'b> BE::BufRef<'b>: HostDataRef,
        for<'b> BE::BufMut<'b>: HostDataMut,
    {
        let module = params.module();
        let mut ctx = Ctx {
            module,
            scratch: ScratchOwned::<BE>::alloc(module.vec_znx_normalize_tmp_bytes()),
        };
        let tag = read_u8(reader)?;
        match tag {
            TAG_INTERPOLATION => {
                let glwe_query = params.glwe_query();
                let key_layout = params.key_layout();
                let ggsw_layout = params.ggsw_layout();
                let n_blocks = read_len(reader)?;
                let mut blocks = Vec::with_capacity(n_blocks);
                for _ in 0..n_blocks {
                    blocks.push(read_glwe_compressed(&mut ctx, reader, &glwe_query)?);
                }
                let key_g = read_auto_key(&mut ctx, reader, &key_layout)?;
                let key_h = read_auto_key(&mut ctx, reader, &key_layout)?;
                let root = read_ggsw_compressed(&mut ctx, reader, &ggsw_layout)?;
                Ok(Query::Interpolation(InterpolationQuery {
                    common: QueryCommon { blocks },
                    keys: InterpolationKeys::new(key_g, key_h),
                    root,
                }))
            }
            TAG_RECURSION => {
                let glwe_pack = params.glwe_pack();
                let key_layout = params.key_layout();
                let n0 = read_len(reader)?;
                let mut src0 = Vec::with_capacity(n0);
                for _ in 0..n0 {
                    src0.push(read_glwe_compressed(&mut ctx, reader, &glwe_pack)?);
                }
                let n1 = read_len(reader)?;
                let mut src1 = Vec::with_capacity(n1);
                for _ in 0..n1 {
                    src1.push(read_glwe_compressed(&mut ctx, reader, &glwe_pack)?);
                }
                let gamma0 = read_compressed_key(&mut ctx, reader, &key_layout)?;
                let gamma1 = read_compressed_key(&mut ctx, reader, &key_layout)?;
                let gamma2 = read_compressed_key(&mut ctx, reader, &key_layout)?;
                Ok(Query::Recursion(RecursionQuery {
                    src0,
                    src1,
                    keys: RecursionKeys {
                        gamma0,
                        gamma1,
                        gamma2,
                    },
                }))
            }
            other => Err(invalid_tag("Query", other)),
        }
    }
}

impl<BE: Backend<OwnedBuf = Vec<u8>>> Response<BE> {
    /// Serialize the response, repacking the (non-gadget) GLWE ciphertexts to
    /// base2k=63.
    pub fn write_to<W: Write>(&self, module: &Module<BE>, writer: &mut W) -> Result<()>
    where
        Module<BE>: ModuleN
            + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
            + GLWENormalize<BE>
            + VecZnxNormalizeTmpBytes,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        for<'b> BE::BufRef<'b>: HostDataRef,
        for<'b> BE::BufMut<'b>: HostDataMut,
    {
        let mut ctx = Ctx {
            module,
            scratch: ScratchOwned::<BE>::alloc(module.vec_znx_normalize_tmp_bytes()),
        };
        match self {
            Response::Interpolation(r) => {
                write_u8(writer, TAG_INTERPOLATION)?;
                write_glwe(&mut ctx, writer, r.selected())
            }
            Response::Recursion(r) => {
                write_u8(writer, TAG_RECURSION)?;
                write_len(writer, r.resp1().len())?;
                for g in r.resp1() {
                    write_glwe(&mut ctx, writer, g)?;
                }
                write_len(writer, r.resp2().len())?;
                for g in r.resp2() {
                    write_glwe(&mut ctx, writer, g)?;
                }
                Ok(())
            }
        }
    }

    /// Deserialize a response written by [`Response::write_to`], sizing every
    /// buffer from `params`.
    pub fn read_from<R: Read, P: Payload<[u8; 32]>>(
        reader: &mut R,
        params: &Parameters<BE, [u8; 32], P>,
    ) -> Result<Self>
    where
        Module<BE>: ModuleN
            + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
            + GLWENormalize<BE>
            + VecZnxNormalizeTmpBytes,
        ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
        GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
        for<'b> BE::BufRef<'b>: HostDataRef,
        for<'b> BE::BufMut<'b>: HostDataMut,
    {
        let module = params.module();
        let mut ctx = Ctx {
            module,
            scratch: ScratchOwned::<BE>::alloc(module.vec_znx_normalize_tmp_bytes()),
        };
        let tag = read_u8(reader)?;
        match tag {
            TAG_INTERPOLATION => {
                let glwe_pack = params.glwe_pack();
                let selected = read_glwe(&mut ctx, reader, &glwe_pack)?;
                Ok(Response::Interpolation(InterpolationResponse::new(
                    selected,
                )))
            }
            TAG_RECURSION => {
                let qtilde_infos =
                    qtilde_glwe_layout(Degree(params.n() as u32), qtilde_bits(params));
                let n1 = read_len(reader)?;
                let mut resp1 = Vec::with_capacity(n1);
                for _ in 0..n1 {
                    resp1.push(read_glwe(&mut ctx, reader, &qtilde_infos)?);
                }
                let n2 = read_len(reader)?;
                let mut resp2 = Vec::with_capacity(n2);
                for _ in 0..n2 {
                    resp2.push(read_glwe(&mut ctx, reader, &qtilde_infos)?);
                }
                Ok(Response::Recursion(RecursionResponse::new(resp1, resp2)))
            }
            other => Err(invalid_tag("Response", other)),
        }
    }
}

// --- gadget compressed automorphism keys ------------------------------------
//
// A compressed automorphism key is a seed-compressed GGLWE: it ships only the
// body matrix (`dnum × rank_in` entries of one column each; the mask is seed-
// derived). Like the GGSW root, its base2k is structural, so we repack each body
// entry to base2k=63 for transmission and restore it on read. The Galois element
// `p` and PRNG seeds travel as explicit metadata (the `dnum·dsize ≤ size`
// invariant rules out re-deriving them from a base2k=63 typed key).

/// Writes a compressed automorphism key: `p`, the seeds, then each body entry in
/// the dense base2k=63 form.
fn write_auto_key<BE, W: Write>(
    ctx: &mut Ctx<BE>,
    writer: &mut W,
    key: &GLWEAutomorphismKeyCompressed<Vec<u8>>,
) -> Result<()>
where
    BE: SerBackend,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = Vec<u8>> + GLWENormalize<BE>,
    GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
{
    write_u64(writer, key.p() as u64)?;
    let seeds = key.seed();
    write_len(writer, seeds.len())?;
    for s in seeds {
        writer.write_all(s)?;
    }
    let dnum = key.dnum().as_usize();
    let rank_in = key.rank_in().as_usize();
    let entry_working = GLWELayout {
        n: key.n(),
        base2k: key.base2k(),
        k: key.max_k(),
        rank: Rank(0),
    };
    let body_ref = GGLWECompressedToBackendRef::<BE>::to_backend_ref(key);
    let body = body_ref.body_as_gglwe();
    for row in 0..dnum {
        for col in 0..rank_in {
            let mut work = ctx.module.glwe_alloc_from_infos(&entry_working);
            copy_vec_znx_host(work.data_mut(), body.at(row, col).data());
            let mut dense = ctx.module.glwe_alloc_from_infos(&transmit_layout(&work));
            ctx.module
                .glwe_normalize(&mut dense, &work, &mut ctx.scratch.borrow());
            dense.write_to(writer)?;
        }
    }
    Ok(())
}

/// Reads a compressed automorphism key written by [`write_auto_key`], allocating
/// the receiver at the working layout `key_infos`.
fn read_auto_key<BE, R: Read, A>(
    ctx: &mut Ctx<BE>,
    reader: &mut R,
    key_infos: &A,
) -> Result<GLWEAutomorphismKeyCompressed<Vec<u8>>>
where
    BE: SerBackend,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = Vec<u8>> + ModuleCoreCompressedAlloc + GLWENormalize<BE>,
    GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    for<'b> BE::BufMut<'b>: HostDataMut,
    A: GGLWEInfos,
{
    let p = read_u64(reader)? as i64;
    let seed_len = read_len(reader)?;
    let mut seeds = vec![[0u8; 32]; seed_len];
    for s in &mut seeds {
        reader.read_exact(s)?;
    }
    let mut key = ctx
        .module
        .glwe_automorphism_key_compressed_alloc_from_infos(key_infos);
    key.set_p(p);
    *key.seed_mut() = seeds;

    let dnum = key.dnum().as_usize();
    let rank_in = key.rank_in().as_usize();
    let entry_dense = GLWELayout {
        n: key.n(),
        base2k: Base2K(TRANSMIT_BASE2K as u32),
        k: key.max_k(),
        rank: Rank(0),
    };
    let entry_working = GLWELayout {
        n: key.n(),
        base2k: key.base2k(),
        k: key.max_k(),
        rank: Rank(0),
    };
    {
        let mut body = GGLWECompressedToBackendMut::<BE>::to_backend_mut(&mut key);
        for row in 0..dnum {
            for col in 0..rank_in {
                let mut dense = ctx.module.glwe_alloc_from_infos(&entry_dense);
                dense.read_from(reader)?;
                let mut work = ctx.module.glwe_alloc_from_infos(&entry_working);
                ctx.module
                    .glwe_normalize(&mut work, &dense, &mut ctx.scratch.borrow());
                let mut view = body.at_view_mut(row, col);
                copy_vec_znx_host(view.data_mut(), work.data());
            }
        }
    }
    Ok(key)
}

fn write_compressed_key<BE, W: Write>(
    ctx: &mut Ctx<BE>,
    writer: &mut W,
    ck: &CompressedKey<BE>,
) -> Result<()>
where
    BE: SerBackend,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = Vec<u8>> + GLWENormalize<BE>,
    GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
{
    write_u64(writer, ck.stride as u64)?;
    write_auto_key(ctx, writer, &ck.key)
}

fn read_compressed_key<BE, R: Read, A>(
    ctx: &mut Ctx<BE>,
    reader: &mut R,
    key_infos: &A,
) -> Result<CompressedKey<BE>>
where
    BE: SerBackend,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = Vec<u8>> + ModuleCoreCompressedAlloc + GLWENormalize<BE>,
    GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    for<'b> BE::BufMut<'b>: HostDataMut,
    A: GGLWEInfos,
{
    let stride = read_u64(reader)? as usize;
    let key = read_auto_key(ctx, reader, key_infos)?;
    Ok(CompressedKey { key, stride })
}

// --- size measurement (test-only) -------------------------------------------

/// Per-component serialized byte sizes of a [`Query`] (each part written to its
/// own buffer), for reporting/regression of wire sizes.
#[cfg(test)]
#[allow(private_interfaces)]
pub(crate) fn query_component_sizes<BE: Backend<OwnedBuf = Vec<u8>>>(
    query: &Query<BE>,
    module: &Module<BE>,
) -> Vec<(&'static str, usize)>
where
    Module<BE>: ModuleN
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>>
        + ModuleCoreCompressedAlloc
        + GLWENormalize<BE>
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    VecZnx<Vec<u8>>: VecZnxToBackendMut<BE> + VecZnxToBackendRef<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    let mut ctx = Ctx {
        module,
        scratch: ScratchOwned::<BE>::alloc(module.vec_znx_normalize_tmp_bytes()),
    };
    let mut out = Vec::new();
    match query {
        Query::Interpolation(q) => {
            let mut b = Vec::new();
            for blk in &q.common.blocks {
                write_glwe_compressed(&mut ctx, &mut b, blk).unwrap();
            }
            out.push(("blocks (one-hot)", b.len()));
            let mut b = Vec::new();
            write_auto_key(&mut ctx, &mut b, q.keys.key_g()).unwrap();
            out.push(("key_g", b.len()));
            let mut b = Vec::new();
            write_auto_key(&mut ctx, &mut b, q.keys.key_h()).unwrap();
            out.push(("key_h", b.len()));
            let mut b = Vec::new();
            write_ggsw_compressed(&mut ctx, &mut b, &q.root).unwrap();
            out.push(("root (GGSW)", b.len()));
        }
        Query::Recursion(q) => {
            let mut b = Vec::new();
            for s in &q.src0 {
                write_glwe_compressed(&mut ctx, &mut b, s).unwrap();
            }
            out.push(("src0 (one-hot)", b.len()));
            let mut b = Vec::new();
            for s in &q.src1 {
                write_glwe_compressed(&mut ctx, &mut b, s).unwrap();
            }
            out.push(("src1 (one-hot)", b.len()));
            for (label, ck) in [
                ("key gamma0", &q.keys.gamma0),
                ("key gamma1", &q.keys.gamma1),
                ("key gamma2", &q.keys.gamma2),
            ] {
                let mut b = Vec::new();
                write_compressed_key(&mut ctx, &mut b, ck).unwrap();
                out.push((label, b.len()));
            }
        }
    }
    out
}

/// Per-component serialized byte sizes of a [`Response`].
#[cfg(test)]
pub(crate) fn response_component_sizes<BE: Backend<OwnedBuf = Vec<u8>>>(
    response: &Response<BE>,
    module: &Module<BE>,
) -> Vec<(&'static str, usize)>
where
    Module<BE>:
        ModuleN + ModuleCoreAlloc<OwnedBuf = Vec<u8>> + GLWENormalize<BE> + VecZnxNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
    GLWE<Vec<u8>>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE>,
    for<'b> BE::BufRef<'b>: HostDataRef,
    for<'b> BE::BufMut<'b>: HostDataMut,
{
    let mut ctx = Ctx {
        module,
        scratch: ScratchOwned::<BE>::alloc(module.vec_znx_normalize_tmp_bytes()),
    };
    let mut out = Vec::new();
    match response {
        Response::Interpolation(r) => {
            let mut b = Vec::new();
            write_glwe(&mut ctx, &mut b, r.selected()).unwrap();
            out.push(("selected (GLWE)", b.len()));
        }
        Response::Recursion(r) => {
            let mut b = Vec::new();
            for g in r.resp1() {
                write_glwe(&mut ctx, &mut b, g).unwrap();
            }
            out.push(("resp1", b.len()));
            let mut b = Vec::new();
            for g in r.resp2() {
                write_glwe(&mut ctx, &mut b, g).unwrap();
            }
            out.push(("resp2", b.len()));
        }
    }
    out
}
