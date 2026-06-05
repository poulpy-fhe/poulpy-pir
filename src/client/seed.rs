use poulpy_hal::source::Source;

/// The single public root seed the server hands the client. All public
/// randomness a query needs — the mask `A`, the packing keys' `a`, and the GGSW
/// root's `a` — is derived from it as domain-separated sub-seeds (independent
/// ChaCha streams). The server holds the same root and re-derives [`mask`](Self::mask)
/// and [`keys`](Self::keys) to reconstruct its half of the protocol.
#[derive(Copy, Clone, Debug)]
pub struct ServerSeed {
    root: [u8; 32],
}

impl ServerSeed {
    pub fn new(root: [u8; 32]) -> Self {
        Self { root }
    }

    /// The `n`-th sub-seed: the `(n+1)`-th draw of a PRNG seeded by the root.
    /// Each is a full 32-byte ChaCha output, so distinct sub-seeds share no
    /// prefix (unlike a stamped index) and [`MaskSeeds`] can safely re-stamp the
    /// low bytes of [`mask`](Self::mask) without colliding with the others.
    fn sub_seed(&self, n: usize) -> [u8; 32] {
        let mut source = Source::new(self.root);
        let mut seed = [0u8; 32];
        for _ in 0..=n {
            seed = source.new_seed();
        }
        seed
    }

    /// Sub-seed for the query mask `A` (feed to [`MaskSeeds`]).
    pub fn mask(&self) -> [u8; 32] {
        self.sub_seed(0)
    }

    /// Sub-seed for the packing keys' public `a`.
    pub fn keys(&self) -> [u8; 32] {
        self.sub_seed(1)
    }

    /// Sub-seed for the GGSW root's public `a`.
    pub fn root_a(&self) -> [u8; 32] {
        self.sub_seed(2)
    }

    /// Sub-seed for the InsPIRe² level-1 query mask `A0`.
    pub fn recursion_a0(&self) -> [u8; 32] {
        self.sub_seed(3)
    }

    /// Sub-seed for the InsPIRe² level-2 query mask `A1`.
    pub fn recursion_a1(&self) -> [u8; 32] {
        self.sub_seed(4)
    }

    /// Sub-seed for the `idx`-th InsPIRe² partial-packing key mask.
    pub fn recursion_key(&self, idx: usize) -> [u8; 32] {
        assert!(idx < 3, "InsPIRe² has exactly three partial-packing keys");
        self.sub_seed(5 + idx)
    }
}

impl Default for ServerSeed {
    fn default() -> Self {
        Self { root: [0x51; 32] }
    }
}

/// Per-block-column mask seeds, addressable in O(1). The seed for block `i` is
/// the public root with its low 4 bytes overwritten by `i` (`root[0..28] ‖ i`),
/// so any block's seed is computed directly — no sequential PRNG walk, no
/// materialized table — which scales to thousands (up to `2^32`) of blocks.
/// Distinct indices yield distinct ChaCha keys, so the per-block masks are
/// independent. The root is [`ServerSeed::mask`]; the client seeds its query
/// bodies and the server its masks from it, so the `a·s` terms cancel.
#[derive(Copy, Clone, Debug)]
pub struct MaskSeeds {
    root: [u8; 32],
}

impl MaskSeeds {
    pub fn new(root: [u8; 32]) -> Self {
        Self { root }
    }

    /// Seed for block-column `block`. Requires `block <= u32::MAX`.
    pub fn seed(&self, block: usize) -> [u8; 32] {
        assert!(
            block <= u32::MAX as usize,
            "block index {block} exceeds 2^32"
        );
        let mut seed = self.root;
        seed[28..].copy_from_slice(&(block as u32).to_le_bytes());
        seed
    }
}
