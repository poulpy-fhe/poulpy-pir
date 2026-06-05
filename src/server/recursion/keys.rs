//! Query and partial-packing key types carried by an InsPIRe² query, plus
//! server-side key generation.

use poulpy_core::{
    EncryptionInfos,
    layouts::{GGLWEInfos, GLWEAutomorphismKeyCompressed, GLWECompressed, LWESecretToBackendRef},
};
use poulpy_hal::{
    layouts::{Backend, Module, ScratchArena},
    source::Source,
};

use crate::packing::PackingKeysGenerate;

/// Client query: two seeded one-hot LWE queries (level-1 column, level-2 batch).
pub struct RecursionQuery<BE: Backend> {
    pub(crate) src0: Vec<GLWECompressed<BE::OwnedBuf>>,
    pub(crate) src1: Vec<GLWECompressed<BE::OwnedBuf>>,
    pub(crate) keys: RecursionKeys<BE>,
}

/// A client-generated compressed partial-packing key plus its order-`gamma`
/// subgroup stride.
pub(crate) struct CompressedKey<BE: Backend> {
    pub(crate) key: GLWEAutomorphismKeyCompressed<BE::OwnedBuf>,
    pub(crate) stride: usize,
}

/// The three partial-packing keys (`gamma0`/`gamma1`/`gamma2`) carried by an
/// InsPIRe² query.
pub(crate) struct RecursionKeys<BE: Backend> {
    pub(crate) gamma0: CompressedKey<BE>,
    pub(crate) gamma1: CompressedKey<BE>,
    pub(crate) gamma2: CompressedKey<BE>,
}

pub(crate) fn generate_recursion_key<BE, E, S>(
    module: &Module<BE>,
    key_infos: &E,
    sk_lwe: &S,
    n: usize,
    gamma: usize,
    seed: [u8; 32],
    source_xe: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) -> CompressedKey<BE>
where
    BE: Backend<OwnedBuf = Vec<u8>>,
    Module<BE>: PackingKeysGenerate<BE>,
    E: EncryptionInfos + GGLWEInfos,
    S: LWESecretToBackendRef<BE>,
{
    let stride = n / 2 / gamma;
    let key = module.pack_partial_key_generate(key_infos, sk_lwe, seed, stride, source_xe, scratch);
    CompressedKey { key, stride }
}
