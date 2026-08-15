//! Architecture-independent serialization for the concrete `PtrHash`
//! instantiation used by keyword PIR.
//!
//! Not upstream. `epserde` encodes `usize` at the host's width and refuses to
//! read a blob written at another, so an x86-64 server cannot ship an MPHF to a
//! wasm32 client. Every field here is written at a fixed width instead.
//!
//! The derived fields (`parts`, `slots`, …) are written rather than recomputed
//! from `n` on read. `PtrHash::init` derives them with floating-point
//! `ln`/`floor`, and libm's last ulp is not guaranteed identical across targets;
//! a single ulp either side of a `floor` boundary would change the geometry and
//! silently produce a different hash function.

use std::io::{Error, ErrorKind, Read, Result, Write};
use std::marker::PhantomData;

use crate::PtrHash;
use crate::PtrHashParams;
use crate::bucket_fn::BucketFn;
use crate::hash::KeyHasher;
use crate::reduce::Reduce;
use crate::shard::Sharding;
use crate::KeyT;

const MAGIC: &[u8; 8] = b"PTRHPRT1";

impl<Key, BF, Hx, const SINGLE_PART: bool, const REMAP: bool>
    PtrHash<Key, BF, Vec<u32>, Hx, Vec<u8>, SINGLE_PART, REMAP>
where
    Key: KeyT + ?Sized,
    BF: BucketFn + Default,
    Hx: KeyHasher<Key>,
{
    pub fn write_portable<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(MAGIC)?;
        w.write_all(&self.params.alpha.to_bits().to_le_bytes())?;
        w.write_all(&self.params.lambda.to_bits().to_le_bytes())?;
        u64s(
            w,
            &[
                self.params.keys_per_shard as u64,
                self.n as u64,
                self.parts as u64,
                self.shards as u64,
                self.parts_per_shard as u64,
                self.slots_total as u64,
                self.buckets_total as u64,
                self.slots as u64,
                self.buckets as u64,
                self.seed,
            ],
        )?;
        write_sharding(w, self.params.sharding)?;

        w.write_all(&(self.pilots.len() as u64).to_le_bytes())?;
        w.write_all(&self.pilots)?;

        w.write_all(&(self.remap.len() as u64).to_le_bytes())?;
        for v in &self.remap {
            w.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn read_portable<R: Read>(r: &mut R) -> Result<Self> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "not a portable PtrHash blob",
            ));
        }

        let alpha = f64::from_bits(u64::from_le_bytes(read_8(r)?));
        let lambda = f64::from_bits(u64::from_le_bytes(read_8(r)?));
        let mut f = [0u64; 10];
        for slot in &mut f {
            *slot = u64::from_le_bytes(read_8(r)?);
        }
        let sharding = read_sharding(r)?;

        let [keys_per_shard, n, parts, shards, parts_per_shard, slots_total, buckets_total, slots, buckets, seed] =
            f;

        let pilots_len = usize_from(u64::from_le_bytes(read_8(r)?))?;
        let mut pilots = vec![0u8; pilots_len];
        r.read_exact(&mut pilots)?;

        let remap_len = usize_from(u64::from_le_bytes(read_8(r)?))?;
        let mut remap = Vec::with_capacity(remap_len);
        for _ in 0..remap_len {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            remap.push(u32::from_le_bytes(b));
        }

        let buckets = usize_from(buckets)?;
        let mut bucket_fn = BF::default();
        bucket_fn.set_buckets_per_part(buckets as u64);

        let parts = usize_from(parts)?;
        let shards = usize_from(shards)?;
        let slots = usize_from(slots)?;
        let buckets_total = usize_from(buckets_total)?;

        Ok(Self {
            params: PtrHashParams {
                alpha,
                lambda,
                bucket_fn,
                keys_per_shard: usize_from(keys_per_shard)?,
                sharding,
            },
            n: usize_from(n)?,
            parts,
            shards,
            parts_per_shard: usize_from(parts_per_shard)?,
            slots_total: usize_from(slots_total)?,
            buckets_total,
            slots,
            buckets,
            rem_shards: Reduce::new(shards),
            rem_parts: Reduce::new(parts),
            rem_buckets: Reduce::new(buckets),
            rem_buckets_total: Reduce::new(buckets_total),
            rem_slots: Reduce::new(slots.max(1)),
            seed,
            pilots,
            remap,
            _key: PhantomData,
            _hx: PhantomData,
        })
    }
}

fn u64s<W: Write>(w: &mut W, values: &[u64]) -> Result<()> {
    for v in values {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_8<R: Read>(r: &mut R) -> Result<[u8; 8]> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(b)
}

/// On a 32-bit target a field written by a 64-bit server can exceed `usize`.
/// That is a real inability to represent the MPHF, so it must be an error and
/// not a truncation.
fn usize_from(v: u64) -> Result<usize> {
    usize::try_from(v).map_err(|_| {
        Error::new(
            ErrorKind::InvalidData,
            format!("{v} does not fit this target's usize"),
        )
    })
}

fn write_sharding<W: Write>(w: &mut W, sharding: Sharding) -> Result<()> {
    let (tag, value) = match sharding {
        Sharding::None => (0u8, 0u64),
        Sharding::Memory => (1, 0),
        Sharding::Disk => (2, 0),
        Sharding::Hybrid(bytes) => (3, bytes as u64),
    };
    w.write_all(&[tag])?;
    w.write_all(&value.to_le_bytes())
}

fn read_sharding<R: Read>(r: &mut R) -> Result<Sharding> {
    let mut tag = [0u8; 1];
    r.read_exact(&mut tag)?;
    let value = u64::from_le_bytes(read_8(r)?);
    Ok(match tag[0] {
        0 => Sharding::None,
        1 => Sharding::Memory,
        2 => Sharding::Disk,
        3 => Sharding::Hybrid(usize_from(value)?),
        other => {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("unknown sharding tag {other}"),
            ));
        }
    })
}
