//! Explicit NUMA placement for the server's large long-lived buffers.
//!
//! Two properties motivate binding placement instead of relying on
//! first-touch:
//! - the memory-bound online phases (body GEMV, packing) want their operands
//!   spread across every node's memory controllers rather than resident on
//!   whichever thread allocated them;
//! - pages under an explicit VMA policy are exempt from automatic NUMA
//!   balancing, whose hint faults and migrations otherwise churn read-shared
//!   data that scoped worker threads (freshly spawned per request, placed
//!   arbitrarily by the scheduler) stream from both sockets.

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod imp {
    use std::sync::OnceLock;

    /// Bitmask of online NUMA nodes (bit `i` = node `i`); 0 when unknown,
    /// single-node, or beyond 64 nodes (all "leave placement alone" cases).
    fn online_nodes() -> u64 {
        static MASK: OnceLock<u64> = OnceLock::new();
        *MASK.get_or_init(|| {
            let Ok(s) = std::fs::read_to_string("/sys/devices/system/node/online") else {
                return 0;
            };
            let mut mask = 0u64;
            for part in s.trim().split(',') {
                let mut ends = part.splitn(2, '-');
                let Some(Ok(lo)) = ends.next().map(str::parse::<u32>) else {
                    return 0;
                };
                let hi = match ends.next().map(str::parse::<u32>) {
                    Some(Ok(hi)) => hi,
                    Some(Err(_)) => return 0,
                    None => lo,
                };
                if hi >= 64 {
                    return 0;
                }
                for n in lo..=hi {
                    mask |= 1 << n;
                }
            }
            if mask.count_ones() >= 2 { mask } else { 0 }
        })
    }

    /// Apply `MPOL_INTERLEAVE` over all online nodes to the whole pages inside
    /// `[ptr, ptr + len)`. Unfaulted pages are placed round-robin across the
    /// nodes when later touched; already-resident pages keep their current
    /// placement (no `MPOL_MF_MOVE` — for buffers written by parallel workers
    /// the existing spread is already fine, and migrating them costs a bulk
    /// copy). Either way the explicit VMA policy exempts the pages from
    /// automatic NUMA balancing. Best-effort: on any failure the pages simply
    /// keep their current placement and policy.
    pub(crate) fn interleave(ptr: *const u8, len: usize) {
        let nodes = online_nodes();
        if nodes == 0 {
            return;
        }
        const PAGE: usize = 4096;
        // Round inward: partial edge pages may be shared with neighboring heap
        // allocations, so leave their policy untouched.
        let start = (ptr as usize).next_multiple_of(PAGE);
        let end = (ptr as usize + len) & !(PAGE - 1);
        if end <= start {
            return;
        }
        const SYS_MBIND: core::ffi::c_long = 237;
        const MPOL_INTERLEAVE: usize = 3;
        unsafe extern "C" {
            fn syscall(num: core::ffi::c_long, ...) -> core::ffi::c_long;
        }
        let mask = [nodes];
        // SAFETY: mbind only sets the placement policy of pages in our own
        // address range; it reads `mask` (valid for the 64 node bits declared
        // by `maxnode`) and never alters the memory contents.
        unsafe {
            syscall(
                SYS_MBIND,
                start,
                end - start,
                MPOL_INTERLEAVE,
                mask.as_ptr(),
                64usize,
                0usize,
            );
        }
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
mod imp {
    /// No-op off Linux/x86-64: pages keep first-touch placement.
    pub(crate) fn interleave(_ptr: *const u8, _len: usize) {}
}

pub(crate) use imp::interleave;
