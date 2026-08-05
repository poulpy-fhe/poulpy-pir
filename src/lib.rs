//! Single-server, communication-efficient Private Information Retrieval (PIR)
//! with server-side preprocessing.
//!
//! This crate implements the InsPIRe family of PIR protocols on top of the
//! Poulpy FHE backends. A client retrieves one payload from a server-held
//! database without revealing which payload was requested. The server can do the
//! expensive query-independent work ahead of time from a public CRS seed and the
//! plaintext database.
//!
//! ## Protocol Families
//!
//! - [`config::Collapse::Interpolation`] selects InsPIRe, which collapses the
//!   second database dimension by polynomial interpolation.
//! - [`config::Collapse::Recursion`] selects InsPIRe2, which uses recursive
//!   packing to reduce response size and online work for large databases.
//!
//! Ready-made 32-byte parameter sets are available through
//! [`config::DefaultPirParameters32B`].
//!
//! ## Main API Layers
//!
//! - [`client::Client`] builds queries and decrypts/decodes responses.
//! - [`server::Server`] owns the plaintext database, runs offline preprocessing,
//!   and answers queries.
//! - [`database::DatabaseLayout`] describes the coefficient-matrix shape shared
//!   by both constructions.
//! - [`keyword`] provides a minimal-perfect-hash layer for key-addressed PIR,
//!   with client-side record verification left to the application.
//!
//! A typical index-PIR flow is:
//!
//! ```text
//! let mut server = Server::new(config, layout);
//! server.update_shard(start, payloads);
//! server.offline();
//!
//! let mut client = Client::new(config, layout);
//! let (query, state) = client.query(payload_index);
//! let response = server.respond(&query);
//! let payload = client.decode(&response, &state);
//! ```
//!
//! Service boundaries should prefer the fallible `try_*` variants, such as
//! [`client::Client::try_query`], [`server::Server::try_update_shard`],
//! [`server::Server::try_respond`], and [`client::Client::try_decode`].
//!
//! ## Security Status
//!
//! This crate has not yet had an independent third-party security audit. See
//! `SECURITY.md` for reporting and deployment notes.

pub mod client;
pub mod config;
pub mod database;
pub mod encoding;
pub mod error;
pub mod interpolation;
pub mod keyword;
pub(crate) mod numa;
pub mod packing;
pub(crate) mod parallel;
pub mod parameters;
pub mod payload;
mod serialization;
pub mod server;

pub use error::{PirError, Result};

#[cfg(all(test, feature = "avx2-fhe"))]
pub(crate) mod test_oracle;

#[cfg(test)]
mod tests;
