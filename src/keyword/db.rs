//! Turning a `(key, value)` set into the payload vector the database stores.

use crate::database::DatabaseLayout;
use crate::payload::Payload;

use super::{EthAddress, KeywordError, KeywordIndex, Record};

/// Checks that a key set fits the database.
///
/// The MPHF addresses `[0, index.len())`, so the database must hold at least
/// that many payloads at the given `column_height` (`n` for InsPIRe, `γ0` for
/// InsPIRe²; take it from `Config::column_height`).
pub fn payloads_fit<P: Payload<[u8; 32]>>(
    index: &KeywordIndex,
    layout: &DatabaseLayout<P>,
    column_height: usize,
) -> Result<(), KeywordError> {
    let capacity = layout.num_payloads(column_height);
    if index.len() > capacity {
        return Err(KeywordError::CapacityExceeded { keys: index.len(), capacity });
    }
    Ok(())
}

/// Scatters `entries` into a payload vector indexed by the MPHF.
///
/// The result has length `index.len()` and is written to the database in one
/// shot with `server.update_shard(0, &payloads)`, which keeps the MPHF's index
/// and the payload index the same number — no offset to carry around.
///
/// Every entry's key must be in the set `index` was built over, and every slot
/// must be written exactly once; because the hash is *minimal* and *perfect*,
/// both hold automatically iff `entries` covers exactly that set. Slots left
/// unwritten stay all-zero, which [`Record::decode`] rejects as
/// [`KeywordError::KeyNotInSet`] — so a partial fill degrades to "key not
/// found" rather than to a wrong balance.
///
/// Values are little-endian `u256` but must fit [`super::VALUE_BYTES`] bytes;
/// see [`Record::encode`].
pub fn fill_payloads(
    index: &KeywordIndex,
    entries: &[(EthAddress, [u8; 32])],
) -> Result<Vec<[u8; 32]>, KeywordError> {
    let mut payloads = vec![[0u8; 32]; index.len()];
    for (key, value) in entries {
        let slot = index.index(key);
        debug_assert!(slot < payloads.len(), "MPHF index out of range");
        payloads[slot] = Record::encode(key, value)?;
    }
    Ok(payloads)
}
