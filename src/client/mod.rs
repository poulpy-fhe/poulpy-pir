//! PIR client: builds the common first-dimension query material (one-hot
//! selector bodies + packing keys) and decrypts the server's [`Response`]. The
//! reduction-specific part of the query (e.g. the interpolation GGSW root) is
//! built by the chosen reduction unit from the ephemeral [`QueryContext`] this
//! returns — see [`crate::interpolation::Interpolation`].
//!
//! Seeds. The server hands the client one public root seed, wrapped in
//! [`ServerSeed`], from which three domain-separated sub-seeds are derived:
//! - [`ServerSeed::mask`]: the query mask `A`. Wrapped in [`MaskSeeds`], each
//!   block-column derives its own seed (`mask[0..28] ‖ i`) so the per-block masks
//!   are independent. The server re-derives the identical seeds to materialize
//!   `A`, so the `a·s` terms cancel.
//! - [`ServerSeed::keys`]: the packing keys' public `a` part.
//! - [`ServerSeed::root_a`]: the (full, self-contained) GGSW root's public `a`.
//!
//! The client's own secret (the LWE secret and all encryption error) is sampled
//! locally from OS entropy on each [`Client::begin_query`] — never an input.
//!
//! Generic over the backend `BE`, but assumes a **host** backend
//! (`BE::OwnedBuf = Vec<u8>`) so the query/answer buffers are plain `Vec<u8>`.

mod core;
mod recursion;
mod seed;
mod state;

pub use core::Client;
pub use recursion::RecursionResponse;
pub use seed::{MaskSeeds, ServerSeed};
pub use state::{QueryCommon, QueryContext, QueryState, Response, ResponseNoise, Sk};
