//! The minimal perfect hash function mapping fixed-size byte keys to indices.

use std::io::{Read, Result as IoResult, Write};

use ptr_hash::{PtrHash, PtrHashParams, bucket_fn::CubicEps, hash::Xxh3_128};

use super::KeywordError;

/// The concrete `ptr_hash` instantiation, pinned so that the bytes a server
/// serializes are exactly what a client deserializes. Generic over the key's
/// byte width `N` only (a 20-byte ETH address, a 32-byte hash, …).
///
/// - `CubicEps` + `Vec<u32>` remap: the `default_compact()` regime, ~2.4
///   bits/key on the wire.
/// - `Xxh3_128`: a 128-bit hash. `ptr_hash`'s default `FxHash` is 64-bit, which
///   starts producing hash collisions (an unconditional construction failure, not
///   a slowdown) as the key count approaches `2^32`; 128 bits removes that
///   ceiling. It is also the hasher family that implements the `epserde` traits.
/// - `SINGLE_PART = false`: splits construction into parts so it can run
///   multi-threaded, which is what makes a full rebuild over tens of millions of
///   keys practical.
/// - `REMAP = true`: makes the PHF *minimal*, i.e. indices land in `[0, n)` with
///   no holes. Without it the database would have to be sized to the (larger)
///   slot count instead of the key count.
type Phf<const N: usize> = PtrHash<[u8; N], CubicEps, Vec<u32>, Xxh3_128, Vec<u8>, false, true>;

/// Construction parameters, tuned for wire size at the 32 M-key ceiling.
///
/// Serialized size splits cleanly into two terms, which is what makes this
/// tunable rather than guesswork:
///
/// ```text
///   pilots =  8 / lambda        bits/key   (one u8 per bucket, n/lambda buckets)
///   remap  = 32 · (1 - alpha)   bits/key   (one u32 per spare slot)
/// ```
///
/// `default_compact()` is `alpha = 0.99, lambda = 3.9` → `2.05 + 0.32 = 2.375`
/// bits/key, i.e. the remap is 13% of the blob purely because it is stored
/// uncompressed. `ptr_hash`'s compressed packings (`CachelineEfVec`, `EliasFano`)
/// would erase that term, but neither can be used here: `cacheline-ef 1.1` pins
/// `epserde 0.8` while `ptr_hash 2.0.2` builds against `epserde 0.13`, so the two
/// `Epserde` impls never unify and the derive's bounds go unsatisfied; `EliasFano`
/// (via `sucds`) has no `Epserde` impl at all. With serialization required,
/// `Vec<u32>` is the only remap that works.
///
/// So the remap is shrunk by raising `alpha` instead. Measured over 32 M random
/// 20-byte keys (`lambda = 3.9` throughout):
///
/// ```text
///   alpha    bits/key   size     build
///   0.99     2.375      9.50 MB   1.5 s
///   0.995    2.212      8.85 MB   2.1 s
///   0.998    2.116      8.46 MB  10.0 s   ← chosen
///   0.999    2.083      8.33 MB  22.5 s
/// ```
///
/// `0.998` takes 11% off the client download. The build cost is irrelevant
/// against what it accompanies: re-derivation always rides along with a full
/// rewrite and re-preprocessing of the 1 GiB database, which is minutes.
/// `0.999` was not worth a further 12 s for 130 KB.
///
/// Raising `alpha` leaves fewer spare slots and so makes construction harder;
/// [`KeywordIndex::build`] goes through `try_new`, so exhausting the retries
/// surfaces as [`KeywordError::MphfConstruction`] rather than a panic. If that
/// ever fires, step `alpha` back down — it costs only wire size.
fn params() -> PtrHashParams<CubicEps> {
    PtrHashParams {
        alpha: 0.998,
        ..PtrHashParams::default_compact()
    }
}

/// A minimal perfect hash from a fixed set of `N`-byte keys onto `[0, len())`.
///
/// Built once by the server over the whole key set, then shipped to clients as
/// opaque parameters via [`KeywordIndex::write_to`]. Both sides then agree on
/// where each key's record lives without the client ever seeing the key set.
///
/// Remember that [`index`](Self::index) is total: it answers for keys that were
/// never in the set too — validating whatever the index leads to against the
/// queried key is the caller's job (see the module docs).
pub struct KeywordIndex<const N: usize> {
    phf: Phf<N>,
    n: usize,
}

impl<const N: usize> KeywordIndex<N> {
    /// Builds the MPHF over `keys`.
    ///
    /// The keys must be pairwise distinct — an MPHF is not defined otherwise —
    /// so this checks first and reports the offender as
    /// [`KeywordError::DuplicateKey`] rather than letting construction spin.
    /// Construction itself is multi-threaded (rayon) and roughly linear in the
    /// key count.
    pub fn build(keys: &[[u8; N]]) -> Result<Self, KeywordError<N>> {
        if let Some(dup) = first_duplicate(keys) {
            return Err(KeywordError::DuplicateKey(dup));
        }
        let phf = Phf::try_new(keys, params()).ok_or(KeywordError::MphfConstruction)?;
        Ok(Self { phf, n: keys.len() })
    }

    /// The index of `key`, always in `[0, len())`.
    ///
    /// For a key in the build set this is its unique slot. For any other key it
    /// is an arbitrary but valid slot belonging to some *other* key — the
    /// caller must verify whatever the index resolves to against `key` before
    /// trusting it.
    #[inline]
    pub fn index(&self, key: &[u8; N]) -> usize {
        self.phf.index(key)
    }

    /// The number of keys the MPHF was built over, and hence the size of the
    /// index range it addresses.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Serializes the MPHF parameters — this is the blob the client downloads.
    ///
    /// A `u64` key count precedes the `ptr_hash` payload, matching the length
    /// prefixing used elsewhere on the wire and letting the reader restore
    /// `len()` without walking the structure.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> IoResult<()> {
        writer.write_all(&(self.n as u64).to_le_bytes())?;
        self.phf.write_portable(writer)
    }

    /// Reads back a [`KeywordIndex`] written by [`write_to`](Self::write_to).
    ///
    /// This is a full (copying) deserialization, so the result owns its buffers
    /// and carries no alignment requirement on `reader`.
    pub fn read_from<R: Read>(reader: &mut R) -> IoResult<Self> {
        let mut n = [0u8; 8];
        reader.read_exact(&mut n)?;
        let phf = Phf::read_portable(reader)?;
        Ok(Self {
            phf,
            n: u64::from_le_bytes(n) as usize,
        })
    }
}

/// Returns a key that occurs more than once, if any.
///
/// Sorts a copy rather than building a hash set: at the tens-of-millions scale
/// this module targets, the sort is both faster and far lighter on memory than
/// a `HashSet<[u8; N]>`, and it is a rounding error next to MPHF construction.
fn first_duplicate<const N: usize>(keys: &[[u8; N]]) -> Option<[u8; N]> {
    let mut sorted = keys.to_vec();
    sorted.sort_unstable();
    sorted.windows(2).find(|w| w[0] == w[1]).map(|w| w[0])
}
