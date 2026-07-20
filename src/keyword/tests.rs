use super::*;

/// Deterministic pseudo-random addresses (splitmix64, as in `examples/pir.rs`).
fn addresses(count: usize) -> Vec<EthAddress> {
    (0..count as u64)
        .map(|i| {
            let mut key = [0u8; 20];
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

fn balance(n: u128) -> [u8; 32] {
    let mut v = [0u8; 32];
    v[..16].copy_from_slice(&n.to_le_bytes());
    v
}

#[test]
fn mphf_is_minimal_and_perfect() {
    let keys = addresses(50_000);
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

#[test]
fn mphf_rejects_duplicate_keys() {
    let mut keys = addresses(1_000);
    keys.push(keys[7]);
    assert_eq!(
        KeywordIndex::build(&keys).err(),
        Some(KeywordError::DuplicateKey(keys[7]))
    );
}

#[test]
fn mphf_serialization_preserves_every_index() {
    let keys = addresses(20_000);
    let index = KeywordIndex::build(&keys).unwrap();

    let mut bytes = Vec::new();
    index.write_to(&mut bytes).unwrap();
    let restored = KeywordIndex::read_from(&mut bytes.as_slice()).unwrap();

    assert_eq!(restored.len(), index.len());
    for key in &keys {
        assert_eq!(restored.index(key), index.index(key));
    }
}

#[test]
fn record_roundtrips_value_for_the_right_key() {
    let keys = addresses(4);
    for (i, key) in keys.iter().enumerate() {
        let value = balance(1_000_000_000_000_000_000u128 * (i as u128 + 1));
        let payload = Record::encode(key, &value).unwrap();
        assert_eq!(Record::decode(key, &payload), Ok(value));
    }
}

#[test]
fn record_rejects_a_wrong_or_empty_slot() {
    let keys = addresses(2);
    let payload = Record::encode(&keys[0], &balance(42)).unwrap();
    assert_eq!(Record::decode(&keys[1], &payload), Err(KeywordError::KeyNotInSet));
    assert_eq!(Record::decode(&keys[0], &[0u8; 32]), Err(KeywordError::KeyNotInSet));
}

#[test]
fn record_rejects_a_value_over_192_bits() {
    let key = addresses(1)[0];
    let mut value = [0u8; 32];
    value[VALUE_BYTES] = 1; // exactly 2^192
    assert_eq!(Record::encode(&key, &value), Err(KeywordError::ValueTooLarge));

    let mut largest = [0u8; 32];
    largest[..VALUE_BYTES].fill(0xff); // 2^192 - 1
    assert_eq!(Record::decode(&key, &Record::encode(&key, &largest).unwrap()), Ok(largest));
}

/// The end-to-end property the module exists for: an in-set address recovers its
/// own balance, an out-of-set one is reported rather than served someone else's.
#[test]
fn fill_then_lookup_by_keyword() {
    let keys = addresses(10_000);
    let entries: Vec<_> = keys.iter().map(|&k| (k, balance(u128::from(k[0]) + 1))).collect();

    let index = KeywordIndex::build(&keys).unwrap();
    let payloads = fill_payloads(&index, &entries).unwrap();
    assert_eq!(payloads.len(), keys.len());

    // Every stored key resolves to its own value.
    for (key, value) in &entries {
        let payload = payloads[index.index(key)];
        assert_eq!(Record::decode(key, &payload).as_ref(), Ok(value));
    }

    // Addresses the MPHF never saw land on valid slots but are always caught.
    for key in addresses(20_000).into_iter().skip(10_000) {
        let payload = payloads[index.index(&key)];
        assert_eq!(Record::decode(&key, &payload), Err(KeywordError::KeyNotInSet));
    }
}

#[test]
fn capacity_check_matches_the_layout() {
    use crate::database::DatabaseLayout;
    use crate::payload::U256P65535;

    // 4 grid rows of 8 columns at column_height = 32. `U256P65535::EXPONENT` is
    // 17, so a 32-row column holds 32/17 = 1 payload (the truncation is exactly
    // the waste that makes the interpolation scheme the less dense of the two).
    let layout = DatabaseLayout::<U256P65535>::new(4 * 32, 8);
    let capacity = layout.num_payloads(32);

    let index = KeywordIndex::build(&addresses(capacity)).unwrap();
    assert_eq!(payloads_fit(&index, &layout, 32), Ok(()));

    let index = KeywordIndex::build(&addresses(capacity + 1)).unwrap();
    assert_eq!(
        payloads_fit(&index, &layout, 32),
        Err(KeywordError::CapacityExceeded { keys: capacity + 1, capacity })
    );
}

// ---------------------------------------------------------------- directory

/// Capacity of `InsPIRe2-g64-1GiB-c8192`, the confirmed target preset.
const CAPACITY: usize = 33_554_432;

#[test]
fn delta_shadows_the_mphf_and_extends_its_range() {
    let base = addresses(1_000);
    let index = KeywordIndex::build(&base).unwrap();
    let mut dir = KeywordDirectory::new(index, CAPACITY, 0).unwrap();

    // Keys added after derivation take the slots just past the MPHF's range.
    let added: Vec<_> = addresses(1_010).into_iter().skip(1_000).collect();
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
    // would hand back a base key's slot.
    for (j, key) in added.iter().enumerate() {
        assert_eq!(dir.index(key), 1_000 + j);
    }

    // Every slot is still uniquely owned across both ranges.
    let mut taken = vec![false; dir.len()];
    for key in base.iter().chain(added.iter()) {
        let i = dir.index(key);
        assert!(!taken[i], "slot {i} assigned twice");
        taken[i] = true;
    }
    assert!(taken.iter().all(|&t| t));
}

#[test]
fn directory_rejects_duplicate_and_overfull_pushes() {
    let index = KeywordIndex::build(&addresses(4)).unwrap();
    let mut dir = KeywordDirectory::new(index, 6, 0).unwrap();
    let extra = addresses(8);

    assert_eq!(dir.push(&extra[4]).unwrap(), 4);
    assert_eq!(dir.push(&extra[4]).err(), Some(KeywordError::DuplicateKey(extra[4])));
    assert_eq!(dir.push(&extra[5]).unwrap(), 5);
    assert_eq!(dir.remaining_capacity(), 0);
    assert_eq!(
        dir.push(&extra[6]).err(),
        Some(KeywordError::CapacityExceeded { keys: 7, capacity: 6 })
    );
}

#[test]
fn full_and_incremental_downloads_agree() {
    let base = addresses(5_000);
    let index = KeywordIndex::build(&base).unwrap();
    let mut server = KeywordDirectory::new(index, CAPACITY, 7).unwrap();

    let added: Vec<_> = addresses(5_200).into_iter().skip(5_000).collect();
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
    server.write_delta_from(&mut tail, client.delta_len()).unwrap();
    assert_eq!(tail.len(), 8 + 80 * 20, "tail should carry 80 keys and no indices");
    client.apply_delta(&mut tail.as_slice()).unwrap();

    assert_eq!(client.len(), server.len());
    for key in base.iter().chain(added.iter()) {
        assert_eq!(client.index(key), server.index(key));
    }
}

#[test]
fn rebuild_bumps_version_clears_delta_and_permutes() {
    let base = addresses(2_000);
    let mut dir = KeywordDirectory::new(KeywordIndex::build(&base).unwrap(), CAPACITY, 3).unwrap();
    let added: Vec<_> = addresses(2_050).into_iter().skip(2_000).collect();
    for key in &added {
        dir.push(key).unwrap();
    }

    let all = addresses(2_050);
    let before: Vec<_> = all.iter().map(|k| dir.index(k)).collect();
    let next = dir.rebuilt(&all).unwrap();

    assert_eq!(next.version(), 4);
    assert_eq!(next.delta_len(), 0);
    assert_eq!(next.len(), 2_050);

    // The whole index space is reassigned, which is why a rebuild forces a full
    // database rewrite rather than an incremental patch.
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
    let mut dir = KeywordDirectory::new(KeywordIndex::build(&addresses(100)).unwrap(), CAPACITY, 0).unwrap();
    assert!(!dir.needs_rebuild());
    for key in addresses(110).into_iter().skip(100) {
        dir.push(&key).unwrap();
    }
    assert!(!dir.needs_rebuild_at(11));
    assert!(dir.needs_rebuild_at(10));
    assert!(!dir.needs_rebuild(), "10 keys is far below the default threshold");
}

/// The end-to-end keyword lookup once the delta is in play: base keys, added
/// keys, and strangers all behave.
#[test]
fn lookup_through_the_directory() {
    let base = addresses(3_000);
    let mut dir = KeywordDirectory::new(KeywordIndex::build(&base).unwrap(), CAPACITY, 0).unwrap();

    // The database is sized to the directory, not just the MPHF.
    let added: Vec<_> = addresses(3_100).into_iter().skip(3_000).collect();
    let mut payloads = vec![[0u8; 32]; 3_100];
    for key in &base {
        payloads[dir.index(key)] = Record::encode(key, &balance(u128::from(key[0]) + 1)).unwrap();
    }
    for key in &added {
        let i = dir.push(key).unwrap();
        payloads[i] = Record::encode(key, &balance(u128::from(key[0]) + 1)).unwrap();
    }

    for key in base.iter().chain(added.iter()) {
        let got = Record::decode(key, &payloads[dir.index(key)]).unwrap();
        assert_eq!(got, balance(u128::from(key[0]) + 1));
    }
    for key in addresses(3_200).into_iter().skip(3_100) {
        assert_eq!(
            Record::decode(&key, &payloads[dir.index(&key)]),
            Err(KeywordError::KeyNotInSet)
        );
    }
}
