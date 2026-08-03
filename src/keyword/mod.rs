//! Keyword indexing: a `key ↔ index` mapping for retrieval by application key
//! instead of by position.
//!
//! The index-based core ([`crate::client::Client::query`]) addresses a record by
//! its position `i ∈ [0, num_payloads)`. This module closes the gap between an
//! application-level key — any fixed-size byte string `[u8; N]`, e.g. a 20-byte
//! ETH address or a 32-byte hash — and that position with a **minimal perfect
//! hash function** (MPHF, built with `ptr_hash`):
//!
//! ```text
//!   server (once, offline)          client (once, on MPHF download)
//!   ────────────────────────        ────────────────────────────────
//!   KeywordIndex::build(&keys)  ──▶  KeywordIndex::read_from(bytes)
//!   DB[index(&key)] = record         i = index.index(&key)
//!                                    resp = pir(i)
//! ```
//!
//! The client only ever downloads the MPHF *parameters* (~2.1 bits/key, so
//! ~4 MB for 16 M keys), never the key set itself.
//!
//! That mapping is **all** this module provides. What a record contains, how it
//! is laid out in payload slots, and how a retrieved record is verified are the
//! application's concern — see `examples/key_word_pir.rs` for a complete
//! deployment (64-byte records holding the full key and value).
//!
//! ## The index is total
//!
//! An MPHF is only defined on the key set it was built from. Querying it with an
//! **out-of-set** key still returns a perfectly valid index in `[0, n)` —
//! silently, pointing at some unrelated entry. There is no way to detect this
//! from the index alone, so any deployment must store enough of the key (or a
//! commitment to it) inside the record for the client to check the retrieved
//! record against the key it *asked for*. Because that check happens
//! client-side on plaintext, it costs the server nothing and leaks nothing.
//!
//! ## Freshly added keys
//!
//! Re-deriving the MPHF is an offline batch job, so keys added since the last
//! one are not in it. [`KeywordDirectory`] closes that gap: a flat append-only
//! `key → index` map, consulted *before* the MPHF, whose entries take the
//! indices just past the MPHF's range. Transports should ship its validated
//! delta envelope (`KeywordDirectory::write_delta_envelope_from`) rather than
//! the raw tail helpers. Once the delta passes [`DEFAULT_REBUILD_THRESHOLD`] the
//! MPHF is re-derived over the whole key set and the delta resets.
//!
//! Re-derivation permutes every index — a minimal perfect hash over a different
//! key set is a different permutation — so it comes with a full rewrite of
//! whatever the indices point at. See [`KeywordDirectory::rebuilt`].

mod directory;
mod mphf;

pub use directory::{DEFAULT_REBUILD_THRESHOLD, KeywordDirectory};
pub use mphf::KeywordIndex;

/// Failure modes of the keyword layer, carrying the `N`-byte key type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeywordError<const N: usize> {
    /// The key set contains the same key twice. An MPHF is only defined over
    /// distinct keys, so this must be resolved by the caller (it usually means
    /// two updates for one key were not merged).
    DuplicateKey([u8; N]),
    /// `ptr_hash` could not find a pilot assignment after its internal retries.
    /// Rare, and generally means the key set is degenerate rather than unlucky.
    MphfConstruction,
    /// The key set is larger than the index range the caller can back.
    CapacityExceeded { keys: usize, capacity: usize },
}

impl<const N: usize> std::fmt::Display for KeywordError<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateKey(key) => write!(f, "duplicate key in the key set: 0x{}", hex(key)),
            Self::MphfConstruction => write!(f, "MPHF construction failed"),
            Self::CapacityExceeded { keys, capacity } => {
                write!(f, "{keys} keys exceed the capacity of {capacity} indices")
            }
        }
    }
}

impl<const N: usize> std::error::Error for KeywordError<N> {}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests;
