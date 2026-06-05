use poulpy_core::layouts::{BackendGLWESecretPrepared, GLWECompressed, LWESecret};
use poulpy_hal::{layouts::Backend, source::Source};

use crate::{
    database::Address,
    interpolation::{InterpolationKeys, InterpolationResponse},
};

use super::recursion::RecursionResponse;

/// The reduction-independent half of the query: only the per block-column one-hot
/// selector bodies (the first PIR dimension). The packing keys are
/// reduction-specific (full `(key_g, key_h)` for interpolation, partial keys for
/// InsPIRe²), so they live on each reduction's own query struct, not here.
pub struct QueryCommon<BE: Backend> {
    /// One-hot query bodies, one per block-column (compressed; seed-derived `a`).
    pub blocks: Vec<GLWECompressed<BE::OwnedBuf>>,
}

/// Ephemeral secret/key handles the client hands to a reduction unit so it can
/// build its query: the prepared pack secret + the continued error/`a` streams,
/// plus the freshly generated full-packing keys (a handoff — the reduction moves
/// the keys it needs into its own query struct). Holds nothing the server ever
/// sees beyond the keys it forwards; drop it once the reduction has built its
/// query.
pub struct QueryContext<BE: Backend> {
    pub sk_pack_prep: BackendGLWESecretPrepared<BE>,
    pub source_xe: Source,
    pub source_xa: Source,
    /// Full-packing keys generated in [`Client::begin_query`](super::Client::begin_query),
    /// forwarded to the reduction's query (`take`n by the reduction unit).
    pub(crate) interpolation_keys: Option<InterpolationKeys<BE>>,
}

impl<BE: Backend> QueryContext<BE> {
    pub(crate) fn take_interpolation_keys(&mut self) -> InterpolationKeys<BE> {
        self.interpolation_keys
            .take()
            .expect("interpolation keys forwarded by begin_query")
    }
}

/// The client's secret state, returned alongside the query and kept locally to
/// decrypt the [`Response`]. Never transmitted.
pub struct Sk<BE: Backend> {
    sk_lwe: LWESecret<BE::OwnedBuf>,
}

impl<BE: Backend> Sk<BE> {
    pub(crate) fn new(sk_lwe: LWESecret<BE::OwnedBuf>) -> Self {
        Self { sk_lwe }
    }

    pub fn sk_lwe(&self) -> &LWESecret<BE::OwnedBuf> {
        &self.sk_lwe
    }
}

/// Client-local query state needed to decrypt and decode a response.
pub struct QueryState<BE: Backend> {
    sk: Sk<BE>,
    address: Address,
}

impl<BE: Backend> QueryState<BE> {
    pub(crate) fn new(sk: Sk<BE>, address: Address) -> Self {
        Self { sk, address }
    }

    pub(crate) fn sk(&self) -> &Sk<BE> {
        &self.sk
    }

    pub fn address(&self) -> Address {
        self.address
    }
}

/// Final response noise, reported as log2 of the absolute max coefficient and
/// log2 of the coefficient standard deviation.
#[derive(Clone, Copy, Debug)]
pub struct ResponseNoise {
    max_log2: f64,
    std_log2: f64,
}

impl ResponseNoise {
    pub(crate) fn new(max: f64, std: f64) -> Self {
        Self {
            max_log2: max.log2(),
            std_log2: std.log2(),
        }
    }

    pub fn max_log2(&self) -> f64 {
        self.max_log2
    }

    pub fn std_log2(&self) -> f64 {
        self.std_log2
    }
}

/// The server's answer, one variant per second-dimension collapse.
pub enum Response<BE: Backend> {
    /// InsPIRe (interpolation) response.
    Interpolation(InterpolationResponse<BE>),
    /// InsPIRe² (recursive packing) response.
    Recursion(RecursionResponse<BE>),
}
