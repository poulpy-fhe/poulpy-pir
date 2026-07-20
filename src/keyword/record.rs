//! The 32-byte `(key tag, value)` record stored in every payload slot.
//!
//! See the module docs on [the 32-byte budget](super#the-32-byte-budget) for why
//! the key is stored as a tag and the value is capped at [`VALUE_BYTES`].

use super::{EthAddress, KeywordError};

/// Bytes of the key tag at the front of a record.
///
/// Sets the false-accept probability for an out-of-set key at `2^-64`.
pub const TAG_BYTES: usize = 8;

/// Bytes left for the value: the rest of the 32-byte payload.
pub const VALUE_BYTES: usize = 32 - TAG_BYTES;

/// Domain separator mixed into the tag hash.
///
/// The tag must be statistically independent of the hash `ptr_hash` uses to
/// place the key, otherwise the tag carries no information the index did not
/// already determine and the out-of-set check weakens. Both use XXH3, so they
/// are separated by this constant.
const TAG_DOMAIN: &[u8] = b"poulpy-pir/keyword/tag/v1";

/// The `TAG_BYTES`-byte fingerprint of an address.
///
/// Non-cryptographic and intentionally so: the tag defends against an *accidental*
/// mismatch (an address that is simply not in the set), not against a malicious
/// server. A server that wants to return a wrong balance can already do so by
/// serving a doctored database — PIR hides the query, it does not authenticate
/// the response.
pub fn key_tag(key: &EthAddress) -> [u8; TAG_BYTES] {
    let mut buf = [0u8; TAG_DOMAIN.len() + 20];
    buf[..TAG_DOMAIN.len()].copy_from_slice(TAG_DOMAIN);
    buf[TAG_DOMAIN.len()..].copy_from_slice(key);
    let h = xxhash_rust::xxh3::xxh3_128(&buf);
    (h as u64).to_le_bytes()
}

/// The plaintext contents of one payload slot.
///
/// On the wire it is exactly the 32 bytes a payload holds:
///
/// ```text
///   [0 .. 8)   key tag, little-endian
///   [8 .. 32)  value, little-endian, < 2^192
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Record;

impl Record {
    /// Packs `(key, value)` into a payload.
    ///
    /// `value` is a little-endian `u256` for API convenience, but only
    /// [`VALUE_BYTES`] of it are storable: anything with a non-zero byte above
    /// that is rejected as [`KeywordError::ValueTooLarge`] rather than silently
    /// truncated, since a truncated balance would round-trip cleanly and be
    /// indistinguishable from a correct one.
    pub fn encode(key: &EthAddress, value: &[u8; 32]) -> Result<[u8; 32], KeywordError> {
        if value[VALUE_BYTES..].iter().any(|&b| b != 0) {
            return Err(KeywordError::ValueTooLarge);
        }
        let mut payload = [0u8; 32];
        payload[..TAG_BYTES].copy_from_slice(&key_tag(key));
        payload[TAG_BYTES..].copy_from_slice(&value[..VALUE_BYTES]);
        Ok(payload)
    }

    /// Unpacks a payload, checking it really belongs to `key`.
    ///
    /// This is the whole point of storing the tag: `key` is the address the
    /// caller asked for, and the payload is whatever the MPHF index led to. If
    /// they disagree the address was never in the key set, and the caller gets
    /// [`KeywordError::KeyNotInSet`] instead of another account's balance.
    ///
    /// An empty (all-zero) slot fails the check too, so unfilled capacity beyond
    /// the key count needs no special handling.
    ///
    /// Returns the value as a little-endian `u256` (the top
    /// `32 - VALUE_BYTES` bytes are always zero).
    pub fn decode(key: &EthAddress, payload: &[u8; 32]) -> Result<[u8; 32], KeywordError> {
        if payload[..TAG_BYTES] != key_tag(key) {
            return Err(KeywordError::KeyNotInSet);
        }
        let mut value = [0u8; 32];
        value[..VALUE_BYTES].copy_from_slice(&payload[TAG_BYTES..]);
        Ok(value)
    }
}
