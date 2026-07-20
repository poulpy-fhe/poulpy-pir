//! Keyword PIR: retrieval by ETH address instead of by payload index.
//!
//! The index-based core ([`crate::client::Client::query`]) addresses a record by
//! its position `i ∈ [0, num_payloads)`. Keyword PIR closes the gap between an
//! application-level key (a 20-byte ETH address) and that position with a
//! **minimal perfect hash function** (MPHF, built with `ptr_hash`):
//!
//! ```text
//!   server (once, offline)          client (once, on MPHF download)
//!   ────────────────────────        ────────────────────────────────
//!   KeywordIndex::build(&keys)  ──▶  KeywordIndex::read_from(bytes)
//!   fill_payloads(&index, ..)        i = index.index(&key)
//!   server.update_shard(0, ..)       resp = pir(i)
//!                                    Record::decode(&key, resp)?  ← verifies
//! ```
//!
//! The client only ever downloads the MPHF *parameters* (~2.4 bits/key, so
//! ~3 MB for 10 M addresses), never the key set itself.
//!
//! ## Why every record carries its key
//!
//! An MPHF is only defined on the key set it was built from. Querying it with an
//! **out-of-set** address still returns a perfectly valid index in `[0, n)` —
//! silently, pointing at some unrelated account's record. There is no way to
//! detect this from the index alone.
//!
//! So each 32-byte payload stores a [`Record`]: a tag derived from the key
//! alongside the value. After decrypting, the client re-derives the tag from the
//! key it *asked for* and compares. A mismatch means the address was never in the
//! set, and [`Record::decode`] returns [`KeywordError::KeyNotInSet`] rather than
//! a wrong balance. Because the whole check happens client-side on plaintext, it
//! costs the server nothing and leaks nothing.
//!
//! ## The 32-byte budget
//!
//! A payload is exactly 32 bytes (see [`crate::payload`]), but a key (20 B) plus
//! a full `u256` value (32 B) is 52 B. The record therefore stores an
//! [`TAG_BYTES`]-byte *hash* of the key rather than the key itself, leaving
//! [`VALUE_BYTES`] = 24 bytes for the value:
//!
//! ```text
//!   payload[0..8]   key tag  = xxh3-128(DOMAIN ‖ key).low64
//!   payload[8..32]  value    = balance, little-endian, < 2^192
//! ```
//!
//! Two consequences, both deliberate:
//!
//! - **Balances must fit in 192 bits.** `2^192 ≈ 6.3·10^57`, i.e. `6.3·10^39`
//!   whole tokens at 18 decimals — orders of magnitude past any real supply.
//!   [`Record::encode`] rejects anything larger with
//!   [`KeywordError::ValueTooLarge`] instead of truncating.
//! - **A false accept has probability `2^-64`.** An out-of-set address is caught
//!   unless its tag collides with the tag of whichever record it landed on. If
//!   that is ever not enough, raise [`TAG_BYTES`] and shrink the value — the
//!   codec is written against both constants.
//!
//! If a genuine full `u256` is ever required, the alternative is two payload
//! slots per record (tag in one, value in the other), which doubles both the
//! database and the per-lookup query cost. That trade is not taken here.
//!
//! ## Freshly added keys
//!
//! Re-deriving the MPHF is an offline batch job, so keys added since the last
//! one are not in it. [`KeywordDirectory`] closes that gap: a flat append-only
//! `key → index` map, consulted *before* the MPHF, whose entries take the payload
//! slots just past the MPHF's range. Clients fetch its tail daily. Once it
//! passes [`DEFAULT_REBUILD_THRESHOLD`] the MPHF is re-derived over the whole
//! key set and the delta resets.
//!
//! Re-derivation permutes every index — a minimal perfect hash over a different
//! key set is a different permutation — so it comes with a full database
//! rewrite. See [`KeywordDirectory::rebuilt`].

mod db;
mod directory;
mod mphf;
mod record;

pub use db::{fill_payloads, payloads_fit};
pub use directory::{DEFAULT_REBUILD_THRESHOLD, KeywordDirectory};
pub use mphf::{EthAddress, KeywordIndex};
pub use record::{Record, TAG_BYTES, VALUE_BYTES, key_tag};

/// Failure modes of the keyword layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeywordError {
    /// The decoded record's key tag does not match the queried key: the address
    /// was not in the set the MPHF was built over, so the index it produced
    /// pointed at an unrelated record. The accompanying value is meaningless and
    /// is deliberately not returned.
    KeyNotInSet,
    /// The key set contains the same address twice. An MPHF is only defined over
    /// distinct keys, so this must be resolved by the caller (it usually means
    /// two balance updates for one account were not merged).
    DuplicateKey(EthAddress),
    /// The value does not fit the [`VALUE_BYTES`]-byte field, i.e. it is
    /// `≥ 2^192`. See the module docs on the 32-byte budget.
    ValueTooLarge,
    /// `ptr_hash` could not find a pilot assignment after its internal retries.
    /// Rare, and generally means the key set is degenerate rather than unlucky.
    MphfConstruction,
    /// The key set is larger than the database can hold.
    CapacityExceeded { keys: usize, capacity: usize },
}

impl std::fmt::Display for KeywordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KeyNotInSet => write!(f, "key not in set: record tag does not match the queried key"),
            Self::DuplicateKey(key) => write!(f, "duplicate key in the key set: 0x{}", hex(key)),
            Self::ValueTooLarge => write!(f, "value does not fit {VALUE_BYTES} bytes (must be < 2^{})", VALUE_BYTES * 8),
            Self::MphfConstruction => write!(f, "MPHF construction failed"),
            Self::CapacityExceeded { keys, capacity } => {
                write!(f, "{keys} keys exceed the database capacity of {capacity} payloads")
            }
        }
    }
}

impl std::error::Error for KeywordError {}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests;
