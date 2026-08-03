//! The MPHF plus the overlay of keys added since it was last derived.

use std::collections::HashMap;
use std::io::{Read, Result as IoResult, Write};

use super::{KeywordError, KeywordIndex};

const DELTA_ENVELOPE_MAGIC: &[u8; 8] = b"PIRDLT1\0";

/// Delta size at which a full MPHF re-derivation pays for itself.
///
/// A delta entry costs the key's bytes on the wire (its index is implied by
/// its position), against ~2.375 bits/key for the MPHF itself — so with
/// 20-byte keys a delta of `d` keys over a base of `n` costs `20·d` bytes
/// versus `0.3·n` bytes to just re-download the rebuilt MPHF. At `n = 32 M`
/// (9.5 MB) the two cross at `d ≈ 400 K`; past that the overlay is strictly
/// more expensive than the thing it exists to avoid.
///
/// Chosen against the 32 M ceiling rather than the 16 M floor so the threshold
/// does not move as the set grows; at 16 M keys it rebuilds somewhat earlier
/// than strictly optimal, which is the safe direction.
pub const DEFAULT_REBUILD_THRESHOLD: usize = 400_000;

/// A [`KeywordIndex`] extended with a flat map of recently added keys.
///
/// Resolution order is delta first, MPHF second. This matters: a key added after
/// the last re-derivation is *not* in the MPHF, and asking the MPHF about it
/// returns a valid-looking index belonging to some other key. The delta must
/// therefore shadow the MPHF, never the other way round.
///
/// Delta indices continue straight on from the MPHF's range: the MPHF owns
/// `[0, mphf.len())` and `delta_keys[j]` owns `mphf.len() + j`. Because the
/// index is implied by position, the raw tail carries only keys. Transport
/// integrations should use [`write_delta_envelope_from`](Self::write_delta_envelope_from)
/// so clients can validate version and offset before appending.
///
/// # Lifecycle
///
/// ```text
///   build ──▶ push, push, … ──▶ delta_len() > threshold ──▶ re-derive
///     │                                                         │
///     └──────────────── version += 1, delta cleared ◀───────────┘
/// ```
///
/// Re-derivation permutes every index, so it comes with a full rewrite of
/// whatever the indices point at — see [`rebuilt`](Self::rebuilt).
pub struct KeywordDirectory<const N: usize> {
    mphf: KeywordIndex<N>,
    /// `delta_keys[j]` has index `mphf.len() + j`. Ordering is append-only and
    /// is the wire order, so it must never be sorted or compacted in place.
    delta_keys: Vec<[u8; N]>,
    /// Reverse lookup into `delta_keys`. Derived state, rebuilt on read.
    delta_lookup: HashMap<[u8; N], u32>,
    /// Bumped on every re-derivation. A client whose version differs from the
    /// server's must re-download the MPHF; a client whose version matches needs
    /// only the delta tail.
    version: u64,
    /// Indices available to hand out; growth is refused past it.
    capacity: usize,
}

impl<const N: usize> KeywordDirectory<N> {
    /// Wraps a freshly derived MPHF, with an empty delta.
    ///
    /// `capacity` is the number of indices the caller can back with storage;
    /// growth is refused past it. `version` identifies this MPHF generation to
    /// clients — pass `0` for the first, or use [`rebuilt`](Self::rebuilt) to
    /// have it advance for you.
    pub fn new(
        mphf: KeywordIndex<N>,
        capacity: usize,
        version: u64,
    ) -> Result<Self, KeywordError<N>> {
        if mphf.len() > capacity {
            return Err(KeywordError::CapacityExceeded {
                keys: mphf.len(),
                capacity,
            });
        }
        Ok(Self {
            mphf,
            delta_keys: Vec::new(),
            delta_lookup: HashMap::new(),
            version,
            capacity,
        })
    }

    /// The index of `key`, checking the delta before the MPHF.
    ///
    /// Total, like [`KeywordIndex::index`]: a key in neither the delta nor the
    /// MPHF's build set still gets a valid index pointing at an unrelated
    /// entry. The caller must verify whatever the index resolves to against
    /// `key` before trusting it.
    #[inline]
    pub fn index(&self, key: &[u8; N]) -> usize {
        match self.delta_lookup.get(key) {
            Some(&j) => self.mphf.len() + j as usize,
            None => self.mphf.index(key),
        }
    }

    /// Appends a new key, returning the index it now owns.
    ///
    /// The caller writes the corresponding record there directly — one entry,
    /// no refill. Only keys genuinely absent from the set may be pushed.
    /// Re-pushing a key already in the delta is caught as
    /// [`KeywordError::DuplicateKey`], but a key already covered by the *MPHF*
    /// cannot be detected here — the MPHF stores no key set to test against.
    /// Callers that cannot guarantee novelty should check their storage first:
    /// resolve `self.index(&key)` and see whether the stored record actually
    /// belongs to `key`.
    pub fn push(&mut self, key: &[u8; N]) -> Result<usize, KeywordError<N>> {
        if self.delta_lookup.contains_key(key) {
            return Err(KeywordError::DuplicateKey(*key));
        }
        let len = self.len();
        if len >= self.capacity {
            return Err(KeywordError::CapacityExceeded {
                keys: len + 1,
                capacity: self.capacity,
            });
        }
        self.delta_lookup.insert(*key, self.delta_keys.len() as u32);
        self.delta_keys.push(*key);
        Ok(len)
    }

    /// Total keys addressed: MPHF base plus delta.
    pub fn len(&self) -> usize {
        self.mphf.len() + self.delta_keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Keys added since the last re-derivation.
    pub fn delta_len(&self) -> usize {
        self.delta_keys.len()
    }

    /// The MPHF generation. Clients compare this to decide between a full
    /// refresh and an incremental delta fetch.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Indices still free.
    pub fn remaining_capacity(&self) -> usize {
        self.capacity - self.len()
    }

    /// Whether the delta has grown past [`DEFAULT_REBUILD_THRESHOLD`].
    pub fn needs_rebuild(&self) -> bool {
        self.needs_rebuild_at(DEFAULT_REBUILD_THRESHOLD)
    }

    /// [`needs_rebuild`](Self::needs_rebuild) against a caller-chosen threshold.
    pub fn needs_rebuild_at(&self, threshold: usize) -> bool {
        self.delta_keys.len() >= threshold
    }

    /// Re-derives the MPHF over `all_keys`, returning a fresh directory at the
    /// next version with an empty delta.
    ///
    /// `all_keys` must be the complete current key set — both the old MPHF's
    /// base and every delta key. This does not read the old MPHF, because
    /// nothing in it survives: a minimal perfect hash over a different key set
    /// is a different permutation, so **every** index moves.
    ///
    /// Whatever the indices point at must therefore be rewritten wholesale
    /// against the returned directory before it serves another lookup.
    /// Publishing the new `version()` before that rewrite completes would hand
    /// clients indices into storage still laid out the old way.
    pub fn rebuilt(&self, all_keys: &[[u8; N]]) -> Result<Self, KeywordError<N>> {
        Self::new(
            KeywordIndex::build(all_keys)?,
            self.capacity,
            self.version + 1,
        )
    }

    /// The underlying MPHF, for bulk fill paths that only need the base set.
    pub fn mphf(&self) -> &KeywordIndex<N> {
        &self.mphf
    }

    /// Writes the whole directory: version, MPHF parameters, and full delta.
    ///
    /// This is the blob a client fetches when its version is stale.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> IoResult<()> {
        writer.write_all(&self.version.to_le_bytes())?;
        writer.write_all(&(self.capacity as u64).to_le_bytes())?;
        self.mphf.write_to(writer)?;
        self.write_delta_from(writer, 0)
    }

    /// Reads a directory written by [`write_to`](Self::write_to).
    pub fn read_from<R: Read>(reader: &mut R) -> IoResult<Self> {
        let version = read_u64(reader)?;
        let capacity = read_u64(reader)? as usize;
        let mphf = KeywordIndex::read_from(reader)?;
        if mphf.len() > capacity {
            return Err(invalid_data("MPHF length exceeds directory capacity"));
        }
        let mut directory = Self {
            mphf,
            delta_keys: Vec::new(),
            delta_lookup: HashMap::new(),
            version,
            capacity,
        };
        directory.read_delta(reader)?;
        Ok(directory)
    }

    /// Writes the raw delta entries from position `have` onward.
    ///
    /// This low-level format is keys only — each one's index is `mphf.len() +
    /// its position`. Prefer
    /// [`write_delta_envelope_from`](Self::write_delta_envelope_from) for any
    /// transport-facing incremental sync, because the raw form cannot validate
    /// version or offset.
    pub fn write_delta_from<W: Write>(&self, writer: &mut W, have: usize) -> IoResult<()> {
        let tail = &self.delta_keys[have.min(self.delta_keys.len())..];
        writer.write_all(&(tail.len() as u64).to_le_bytes())?;
        for key in tail {
            writer.write_all(key)?;
        }
        Ok(())
    }

    /// Writes a self-describing delta tail with the directory version and start
    /// offset needed to validate client-side application.
    ///
    /// Unlike [`write_delta_from`](Self::write_delta_from), this rejects a
    /// `have` value beyond the current delta length instead of silently clamping
    /// it. Use this for transport-facing incremental sync.
    pub fn write_delta_envelope_from<W: Write>(&self, writer: &mut W, have: usize) -> IoResult<()> {
        let Some(tail) = self.delta_keys.get(have..) else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "delta start is beyond the server delta length",
            ));
        };
        writer.write_all(DELTA_ENVELOPE_MAGIC)?;
        write_u64(writer, self.version)?;
        write_u64(writer, self.mphf.len() as u64)?;
        write_u64(writer, have as u64)?;
        write_u64(writer, tail.len() as u64)?;
        write_u64(writer, self.capacity as u64)?;
        for key in tail {
            writer.write_all(key)?;
        }
        Ok(())
    }

    /// Appends raw delta entries produced by [`write_delta_from`](Self::write_delta_from).
    ///
    /// The caller is responsible for having passed its own [`delta_len`](Self::delta_len)
    /// as `have`; appending a tail taken from a different offset would shift
    /// every subsequent index. Prefer
    /// [`apply_delta_envelope`](Self::apply_delta_envelope) for transport-facing
    /// sync.
    pub fn apply_delta<R: Read>(&mut self, reader: &mut R) -> IoResult<()> {
        self.read_delta(reader)
    }

    /// Appends a delta envelope produced by
    /// [`write_delta_envelope_from`](Self::write_delta_envelope_from).
    ///
    /// The envelope must match this client's current MPHF generation and exact
    /// delta offset. A stale post-rebuild tail or a tail fetched from the wrong
    /// offset is rejected instead of shifting every appended index.
    pub fn apply_delta_envelope<R: Read>(&mut self, reader: &mut R) -> IoResult<()> {
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != DELTA_ENVELOPE_MAGIC {
            return Err(invalid_data("invalid keyword delta envelope magic"));
        }

        let version = read_u64(reader)?;
        let base_len = read_u64(reader)? as usize;
        let start = read_u64(reader)? as usize;
        let count = read_u64(reader)? as usize;
        let capacity = read_u64(reader)? as usize;

        if version != self.version {
            return Err(invalid_data(
                "keyword delta version does not match client directory",
            ));
        }
        if base_len != self.mphf.len() {
            return Err(invalid_data(
                "keyword delta base length does not match client directory",
            ));
        }
        if start != self.delta_keys.len() {
            return Err(invalid_data(
                "keyword delta start offset does not match client directory",
            ));
        }
        if capacity != self.capacity {
            return Err(invalid_data(
                "keyword delta capacity does not match client directory",
            ));
        }
        self.read_delta_entries(reader, count)
    }

    fn read_delta<R: Read>(&mut self, reader: &mut R) -> IoResult<()> {
        let count = read_u64(reader)? as usize;
        self.read_delta_entries(reader, count)
    }

    fn read_delta_entries<R: Read>(&mut self, reader: &mut R, count: usize) -> IoResult<()> {
        if self.len().saturating_add(count) > self.capacity {
            return Err(invalid_data("keyword delta exceeds directory capacity"));
        }
        self.delta_keys.reserve(count);
        for _ in 0..count {
            let mut key = [0u8; N];
            reader.read_exact(&mut key)?;
            if self.delta_lookup.contains_key(&key) {
                return Err(invalid_data("duplicate key in keyword delta"));
            }
            self.delta_lookup.insert(key, self.delta_keys.len() as u32);
            self.delta_keys.push(key);
        }
        Ok(())
    }
}

fn read_u64<R: Read>(reader: &mut R) -> IoResult<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> IoResult<()> {
    writer.write_all(&value.to_le_bytes())
}

fn invalid_data(message: &'static str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message)
}
