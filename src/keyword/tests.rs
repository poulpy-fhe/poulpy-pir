use super::*;

/// Deterministic pseudo-random `N`-byte keys (splitmix64, as in `examples/pir.rs`).
fn keys<const N: usize>(count: usize) -> Vec<[u8; N]> {
    (0..count as u64)
        .map(|i| {
            let mut key = [0u8; N];
            let mut z = i.wrapping_add(0x9e3779b97f4a7c15);
            for chunk in key.chunks_mut(8) {
                z = z.wrapping_add(0x9e3779b97f4a7c15);
                let mut x = z;
                x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
                x ^= x >> 31;
                let bytes = x.to_le_bytes();
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            }
            key
        })
        .collect()
}

#[test]
fn mphf_is_minimal_and_perfect() {
    let keys = keys::<20>(50_000);
    let index = KeywordIndex::build(&keys).unwrap();
    assert_eq!(index.len(), keys.len());

    let mut taken = vec![false; keys.len()];
    for key in &keys {
        let i = index.index(key);
        assert!(i < keys.len(), "index {i} out of range");
        assert!(!taken[i], "index {i} assigned twice");
        taken[i] = true;
    }
    assert!(taken.iter().all(|&t| t), "not every slot was hit");
}

/// The module is generic over the key width: the same instantiation works for
/// 32-byte keys (e.g. hashes) as for 20-byte addresses.
#[test]
fn mphf_supports_other_key_widths() {
    let keys = keys::<32>(10_000);
    let index = KeywordIndex::build(&keys).unwrap();
    let mut taken = vec![false; keys.len()];
    for key in &keys {
        let i = index.index(key);
        assert!(!taken[i], "index {i} assigned twice");
        taken[i] = true;
    }
    assert!(taken.iter().all(|&t| t));
}

#[test]
fn mphf_rejects_duplicate_keys() {
    let mut keys = keys::<20>(1_000);
    keys.push(keys[7]);
    assert_eq!(
        KeywordIndex::build(&keys).err(),
        Some(KeywordError::DuplicateKey(keys[7]))
    );
}

#[test]
fn mphf_serialization_preserves_every_index() {
    let keys = keys::<20>(20_000);
    let index = KeywordIndex::build(&keys).unwrap();

    let mut bytes = Vec::new();
    index.write_to(&mut bytes).unwrap();
    let restored = KeywordIndex::read_from(&mut bytes.as_slice()).unwrap();

    assert_eq!(restored.len(), index.len());
    for key in &keys {
        assert_eq!(restored.index(key), index.index(key));
    }
}

// ---------------------------------------------------------------- directory

/// Record capacity of the 1 GiB presets at 64 B/record.
const CAPACITY: usize = 16_777_216;

#[test]
fn delta_shadows_the_mphf_and_extends_its_range() {
    let base = keys::<20>(1_000);
    let index = KeywordIndex::build(&base).unwrap();
    let mut dir = KeywordDirectory::new(index, CAPACITY, 0).unwrap();

    // Keys added after derivation take the indices just past the MPHF's range.
    let added: Vec<_> = keys::<20>(1_010).into_iter().skip(1_000).collect();
    for (j, key) in added.iter().enumerate() {
        assert_eq!(dir.push(key).unwrap(), 1_000 + j);
    }
    assert_eq!(dir.len(), 1_010);
    assert_eq!(dir.delta_len(), 10);

    // Base keys still resolve through the MPHF, into its own range.
    for key in &base {
        assert!(dir.index(key) < 1_000);
    }
    // Added keys resolve through the delta — never through the MPHF, which
    // would hand back a base key's index.
    for (j, key) in added.iter().enumerate() {
        assert_eq!(dir.index(key), 1_000 + j);
    }

    // Every index is still uniquely owned across both ranges.
    let mut taken = vec![false; dir.len()];
    for key in base.iter().chain(added.iter()) {
        let i = dir.index(key);
        assert!(!taken[i], "index {i} assigned twice");
        taken[i] = true;
    }
    assert!(taken.iter().all(|&t| t));
}

#[test]
fn directory_rejects_duplicate_and_overfull_pushes() {
    let index = KeywordIndex::build(&keys::<20>(4)).unwrap();
    let mut dir = KeywordDirectory::new(index, 6, 0).unwrap();
    let extra = keys::<20>(8);

    assert_eq!(dir.push(&extra[4]).unwrap(), 4);
    assert_eq!(
        dir.push(&extra[4]).err(),
        Some(KeywordError::DuplicateKey(extra[4]))
    );
    assert_eq!(dir.push(&extra[5]).unwrap(), 5);
    assert_eq!(dir.remaining_capacity(), 0);
    assert_eq!(
        dir.push(&extra[6]).err(),
        Some(KeywordError::CapacityExceeded {
            keys: 7,
            capacity: 6
        })
    );
}

#[test]
fn full_and_incremental_downloads_agree() {
    let base = keys::<20>(5_000);
    let index = KeywordIndex::build(&base).unwrap();
    let mut server = KeywordDirectory::new(index, CAPACITY, 7).unwrap();

    let added: Vec<_> = keys::<20>(5_200).into_iter().skip(5_000).collect();
    for key in added.iter().take(120) {
        server.push(key).unwrap();
    }

    // A cold client takes the whole blob.
    let mut blob = Vec::new();
    server.write_to(&mut blob).unwrap();
    let mut client = KeywordDirectory::read_from(&mut blob.as_slice()).unwrap();
    assert_eq!(client.version(), 7);
    assert_eq!(client.delta_len(), 120);

    // More keys arrive; the client fetches only the tail past what it holds.
    for key in added.iter().skip(120) {
        server.push(key).unwrap();
    }
    let mut tail = Vec::new();
    server
        .write_delta_from(&mut tail, client.delta_len())
        .unwrap();
    assert_eq!(
        tail.len(),
        8 + 80 * 20,
        "tail should carry 80 keys and no indices"
    );
    client.apply_delta(&mut tail.as_slice()).unwrap();

    assert_eq!(client.len(), server.len());
    for key in base.iter().chain(added.iter()) {
        assert_eq!(client.index(key), server.index(key));
    }
}

#[test]
fn delta_envelope_validates_version_and_offset() {
    let base = keys::<20>(5_000);
    let mut server =
        KeywordDirectory::new(KeywordIndex::build(&base).unwrap(), CAPACITY, 7).unwrap();
    let added: Vec<_> = keys::<20>(5_020).into_iter().skip(5_000).collect();
    for key in added.iter().take(10) {
        server.push(key).unwrap();
    }

    let mut full = Vec::new();
    server.write_to(&mut full).unwrap();
    let mut client = KeywordDirectory::<20>::read_from(&mut full.as_slice()).unwrap();

    for key in added.iter().skip(10) {
        server.push(key).unwrap();
    }

    let mut tail = Vec::new();
    server
        .write_delta_envelope_from(&mut tail, client.delta_len())
        .unwrap();
    client.apply_delta_envelope(&mut tail.as_slice()).unwrap();
    assert_eq!(client.len(), server.len());

    let mut shifted = Vec::new();
    server.write_delta_envelope_from(&mut shifted, 11).unwrap();
    let mut stale_offset_client = KeywordDirectory::<20>::read_from(&mut full.as_slice()).unwrap();
    assert_eq!(
        stale_offset_client
            .apply_delta_envelope(&mut shifted.as_slice())
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::InvalidData
    );

    let rebuilt = server.rebuilt(&keys::<20>(5_020)).unwrap();
    let mut stale_version = Vec::new();
    rebuilt
        .write_delta_envelope_from(&mut stale_version, 0)
        .unwrap();
    assert_eq!(
        client
            .apply_delta_envelope(&mut stale_version.as_slice())
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::InvalidData
    );
}

#[test]
fn rebuild_bumps_version_clears_delta_and_permutes() {
    let base = keys::<20>(2_000);
    let mut dir = KeywordDirectory::new(KeywordIndex::build(&base).unwrap(), CAPACITY, 3).unwrap();
    let added: Vec<_> = keys::<20>(2_050).into_iter().skip(2_000).collect();
    for key in &added {
        dir.push(key).unwrap();
    }

    let all = keys::<20>(2_050);
    let before: Vec<_> = all.iter().map(|k| dir.index(k)).collect();
    let next = dir.rebuilt(&all).unwrap();

    assert_eq!(next.version(), 4);
    assert_eq!(next.delta_len(), 0);
    assert_eq!(next.len(), 2_050);

    // The whole index space is reassigned, which is why a rebuild forces a full
    // rewrite of whatever the indices point at, not an incremental patch.
    let after: Vec<_> = all.iter().map(|k| next.index(k)).collect();
    assert_ne!(before, after, "rebuild should permute indices");
    let mut taken = vec![false; 2_050];
    for i in after {
        assert!(!taken[i]);
        taken[i] = true;
    }
    assert!(taken.iter().all(|&t| t));
}

#[test]
fn rebuild_threshold_tracks_the_delta() {
    let mut dir =
        KeywordDirectory::new(KeywordIndex::build(&keys::<20>(100)).unwrap(), CAPACITY, 0).unwrap();
    assert!(!dir.needs_rebuild());
    for key in keys::<20>(110).into_iter().skip(100) {
        dir.push(&key).unwrap();
    }
    assert!(!dir.needs_rebuild_at(11));
    assert!(dir.needs_rebuild_at(10));
    assert!(
        !dir.needs_rebuild(),
        "10 keys is far below the default threshold"
    );
}

/// The MPHF is built on a 64-bit server and evaluated on 32-bit wasm clients, so
/// the key hash must not depend on the host's word size. It did: `Hash for
/// [T]` prefixes the length via `write_usize`, which writes 8 bytes on x86-64
/// and 4 on wasm32, so the same key hashed to different values and every index
/// disagreed.
///
/// Pin the hash rather than an MPHF index — construction is parallel and not
/// reproducible across runs, but this is, and it is the part that has to be
/// portable.
#[test]
fn the_key_hash_does_not_depend_on_word_size() {
    use ptr_hash::hash::{KeyHasher, Xxh3_128};

    const GOLDEN: [u128; 3] = [
        329742151411625726951160329755639952067,
        56427818990099317979833982576654605952,
        56427818990099317278857707775691644586,
    ];

    let got = [
        <Xxh3_128 as KeyHasher<[u8; 20]>>::hash(&[0u8; 20], 0),
        <Xxh3_128 as KeyHasher<[u8; 20]>>::hash(&[7u8; 20], 0),
        <Xxh3_128 as KeyHasher<[u8; 20]>>::hash(&[7u8; 20], 42),
    ];
    assert_eq!(got, GOLDEN, "the key hash changed; every client must resync");
}
