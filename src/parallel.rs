//! Highest-level work partitioning and a scoped-thread runner for the server's
//! embarrassingly-parallel block loops.
//!
//! The unit of work is a 2048×2048 `CoeffMatrix` block. The interpolated DB is a
//! `P × K` grid of such blocks (`P = interpolation_t` panels, `K = block_cols`
//! contraction blocks per panel). A worker is assigned a set of blocks expressed
//! as `(panel, col-range)` [`BlockWork`] tiles. See `CONCURRENCY_PLAN.md`.
//!
//! Two assignment strategies:
//! - [`assign_panels`]: panel-major — each worker owns whole panels. The `K`-way
//!   contraction of each panel stays sequential, preserving f64 determinism.
//!   Use when `P >= threads`.
//! - [`assign_blocks`]: block-tiled — balance to `ceil(P*K/threads)` blocks per
//!   worker, panel-contiguous. A panel may be split across workers, so callers
//!   must reduce per-panel partials in a fixed order. Use when `P < threads`.
//!
//! Both produce a deterministic, panel-major ordering: with `threads == 1` the
//! result is a single group equal to the sequential block order.
//!
//! NOTE: the partitioners and runner are scaffolding consumed starting in
//! concurrency task M1.2; the module-level `dead_code` allow is removed once they
//! are wired into the offline/online drivers.
#![allow(dead_code)]

use poulpy_hal::{
    api::ScratchOwnedAlloc,
    layouts::{Backend, ScratchOwned},
};

/// A contiguous slice `col_start..col_start + col_len` of one panel's `K`
/// contraction blocks, assigned to a single worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockWork {
    pub panel: usize,
    pub col_start: usize,
    pub col_len: usize,
}

/// Resolve the worker count: `PIR_THREADS` (if set and `>= 1`) overrides the
/// detected core count, then the result is clamped to `1..=cap`. `cap` is the
/// natural parallelism ceiling at the call site (e.g. the number of panels).
pub(crate) fn num_threads(cap: usize) -> usize {
    if cap <= 1 {
        return 1;
    }
    let detected = std::env::var("PIR_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&t| t >= 1)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(1)
        });
    detected.clamp(1, cap)
}

/// Panel-major assignment: distribute the `p` panels across `threads` workers as
/// evenly as possible (counts differ by at most one), each panel handed out as a
/// full-width [`BlockWork`] (`col_start = 0`, `col_len = k`). No empty groups.
pub(crate) fn assign_panels(p: usize, k: usize, threads: usize) -> Vec<Vec<BlockWork>> {
    if p == 0 {
        return Vec::new();
    }
    let t = threads.clamp(1, p);
    let base = p / t;
    let rem = p % t;
    let mut out = Vec::with_capacity(t);
    let mut panel = 0;
    for i in 0..t {
        let len = base + usize::from(i < rem);
        let mut group = Vec::with_capacity(len);
        for _ in 0..len {
            group.push(BlockWork {
                panel,
                col_start: 0,
                col_len: k,
            });
            panel += 1;
        }
        out.push(group);
    }
    out
}

/// Block-tiled assignment: flatten the `P × K` grid in panel-major order and cut
/// it into `threads` contiguous runs whose block counts differ by at most one,
/// coalescing each run into per-panel [`BlockWork`] tiles. Every `(panel, col)`
/// is covered exactly once.
pub(crate) fn assign_blocks(p: usize, k: usize, threads: usize) -> Vec<Vec<BlockWork>> {
    let total = p * k;
    if total == 0 {
        return Vec::new();
    }
    let t = threads.clamp(1, total);
    let base = total / t;
    let rem = total % t;
    let mut out = Vec::with_capacity(t);
    let mut global = 0usize; // flattened block index in [0, total)
    for i in 0..t {
        let mut remaining = base + usize::from(i < rem);
        let mut group = Vec::new();
        while remaining > 0 {
            let panel = global / k;
            let col = global % k;
            let take = remaining.min(k - col);
            group.push(BlockWork {
                panel,
                col_start: col,
                col_len: take,
            });
            global += take;
            remaining -= take;
        }
        out.push(group);
    }
    out
}

/// Run `f` once per work group, each invocation on its own scoped thread with a
/// freshly allocated per-worker [`ScratchOwned`] of `bytes`. `outputs` is aligned
/// 1:1 with `work` (caller splits the result buffer into disjoint `&mut` slabs,
/// e.g. via `split_at_mut`), so writes never alias across threads.
///
/// `Module<BE>` and any read-only inputs are shared by `&` (captured in `f`);
/// only the disjoint `outputs` slabs are mutated.
#[allow(dead_code)] // exercised by M1.2 / M2.2
pub(crate) fn scoped_workers<'a, BE, T, F>(
    outputs: Vec<&'a mut [T]>,
    work: &'a [Vec<BlockWork>],
    bytes: usize,
    f: F,
) where
    BE: Backend,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    T: Send,
    F: Fn(&mut [T], &[BlockWork], &mut ScratchOwned<BE>) + Sync,
{
    assert_eq!(
        outputs.len(),
        work.len(),
        "output slabs must align 1:1 with work groups"
    );
    if work.len() <= 1 {
        let f = &f;
        for (slab, group) in outputs.into_iter().zip(work.iter()) {
            let mut sc = ScratchOwned::<BE>::alloc(bytes);
            f(slab, group, &mut sc);
        }
        return;
    }
    let f = &f;
    std::thread::scope(|scope| {
        for (slab, group) in outputs.into_iter().zip(work.iter()) {
            scope.spawn(move || {
                let mut sc = ScratchOwned::<BE>::alloc(bytes);
                f(slab, group, &mut sc);
            });
        }
    });
}

/// Like [`scoped_workers`], but each worker borrows a caller-owned
/// [`ScratchOwned`] from a persistent pool instead of allocating a fresh one.
/// `outputs` and `scratch` are both aligned 1:1 with `work` groups (disjoint
/// `&mut` slabs, so nothing aliases across threads). This avoids the per-call
/// arena allocation + first-touch fault that dominates short latency-critical
/// regions (plan M2′).
#[allow(dead_code)] // wired by M2 / M4
pub(crate) fn scoped_workers_pooled<'a, BE, T, F>(
    outputs: Vec<&'a mut [T]>,
    scratch: Vec<&'a mut ScratchOwned<BE>>,
    work: &'a [Vec<BlockWork>],
    f: F,
) where
    BE: Backend,
    ScratchOwned<BE>: Send,
    T: Send,
    F: Fn(&mut [T], &[BlockWork], &mut ScratchOwned<BE>) + Sync,
{
    assert_eq!(
        outputs.len(),
        work.len(),
        "output slabs must align 1:1 with work groups"
    );
    assert_eq!(
        scratch.len(),
        work.len(),
        "scratch slabs must align 1:1 with work groups"
    );
    if work.len() <= 1 {
        let f = &f;
        for ((slab, sc), group) in outputs.into_iter().zip(scratch).zip(work.iter()) {
            f(slab, group, sc);
        }
        return;
    }
    let f = &f;
    std::thread::scope(|scope| {
        for ((slab, sc), group) in outputs.into_iter().zip(scratch).zip(work.iter()) {
            scope.spawn(move || {
                f(slab, group, sc);
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flatten an assignment back to its `(panel, col)` block sequence.
    fn flatten(groups: &[Vec<BlockWork>]) -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        for g in groups {
            for w in g {
                for c in w.col_start..w.col_start + w.col_len {
                    v.push((w.panel, c));
                }
            }
        }
        v
    }

    /// The canonical panel-major sequential order of every block.
    fn sequential(p: usize, k: usize) -> Vec<(usize, usize)> {
        (0..p)
            .flat_map(|panel| (0..k).map(move |c| (panel, c)))
            .collect()
    }

    #[test]
    fn num_threads_caps_and_floors() {
        assert_eq!(num_threads(0), 1);
        assert_eq!(num_threads(1), 1);
        for cap in [2, 4, 16, 64] {
            let t = num_threads(cap);
            assert!((1..=cap).contains(&t), "cap={cap} -> {t}");
        }
    }

    #[test]
    fn assign_panels_covers_each_block_once() {
        for &(p, k) in &[(1, 1), (16, 32), (16, 1), (3, 5), (7, 4)] {
            for threads in 1..=2 * p {
                let g = assign_panels(p, k, threads);
                // No empty groups; at most `p` (or `threads`) groups.
                assert!(g.len() <= p.min(threads.max(1)).max(1));
                assert!(g.iter().all(|grp| !grp.is_empty()));
                // Each tile is a full panel.
                assert!(
                    g.iter()
                        .flatten()
                        .all(|w| w.col_start == 0 && w.col_len == k)
                );
                // Exact cover, panel-major order preserved.
                assert_eq!(flatten(&g), sequential(p, k), "p={p} k={k} t={threads}");
                // Panel-count balance within 1.
                let counts: Vec<usize> = g.iter().map(|grp| grp.len()).collect();
                let (mn, mx) = (counts.iter().min().unwrap(), counts.iter().max().unwrap());
                assert!(
                    mx - mn <= 1,
                    "unbalanced panels p={p} t={threads}: {counts:?}"
                );
            }
        }
    }

    #[test]
    fn assign_blocks_covers_each_block_once_and_balances() {
        for &(p, k) in &[(1, 1), (16, 32), (16, 1), (3, 5), (7, 4), (2, 9)] {
            let total = p * k;
            for threads in 1..=2 * total {
                let g = assign_blocks(p, k, threads);
                // Exact cover, panel-major order preserved.
                assert_eq!(flatten(&g), sequential(p, k), "p={p} k={k} t={threads}");
                // Tiles never straddle a panel boundary.
                assert!(g.iter().flatten().all(|w| w.col_start + w.col_len <= k));
                // Block-count balance within 1 across non-empty groups.
                let counts: Vec<usize> = g
                    .iter()
                    .map(|grp| grp.iter().map(|w| w.col_len).sum::<usize>())
                    .filter(|&n| n > 0)
                    .collect();
                let (mn, mx) = (counts.iter().min().unwrap(), counts.iter().max().unwrap());
                assert!(
                    mx - mn <= 1,
                    "unbalanced blocks p={p} k={k} t={threads}: {counts:?}"
                );
            }
        }
    }

    #[test]
    fn single_thread_is_sequential_order() {
        let (p, k) = (16, 32);
        let panels = assign_panels(p, k, 1);
        let blocks = assign_blocks(p, k, 1);
        assert_eq!(panels.len(), 1);
        assert_eq!(blocks.len(), 1);
        assert_eq!(flatten(&panels), sequential(p, k));
        assert_eq!(flatten(&blocks), sequential(p, k));
    }
}
