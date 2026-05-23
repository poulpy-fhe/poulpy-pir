# DFT-Hot Collapse: Automorphism Investigation and BSGS Strategy

## Purpose

This note summarizes the investigation into the DFT-domain automorphism cost in
the PIR DFT-hot collapse path, the AVX2 strategies that were tested, the issues
encountered, and the BSGS strategy implemented to reduce the hot-path
automorphism count. It is written for external review of the approach and the
current implementation direction.

## Context

The current DFT-hot collapse keeps the GLWE body accumulator in DFT domain
through the sequential collapse and only performs one final IDFT and
normalization. The online loop is roughly:

```text
body_acc_dft = DFT(body)

for step in 0..kg_steps:
    product = VMP(mask_dft[step], Kg_body)
    product_auto = tau_alpha_step(product)
    body_acc_dft += product_auto

product_h = VMP(mask_dft[final], Kh_body)
body_acc_dft += product_h

body = IDFT(body_acc_dft)
mask = precomputed_final_mask
```

On `n = 1024`, the strict collapse benchmark reported the following phased
breakdown before BSGS:

```text
online DFT-hot body-only collapse:   3.520 ms
DFT-hot operation breakdown:         3.572 ms
  Kg body 1x1 VMPs (1022 steps):     1.368 ms   38.3%
  DFT automorphisms (1022 steps):    2.044 ms   57.2%
  DFT additions (1023 steps):        0.151 ms    4.2%
  other:                             tiny
```

This made DFT-domain automorphism the dominant online cost.

## FFT64 DFT Automorphism Layout

For `FFT64`, the DFT buffer is split into real and imaginary halves:

```text
[0 .. n/2)      real slots
[n/2 .. n)      imaginary slots
```

The automorphism applies the same slot permutation to both halves:

```text
res_re[i] =  a_re[perm[i]]
res_im[i] =  a_im[perm[i]]
```

or, for the conjugating case:

```text
res_re[i] =  a_re[perm[i]]
res_im[i] = -a_im[perm[i]]
```

The symmetry reuses the same `perm` for both halves, but it does not halve the
number of data loads with AVX2. For four complex slots, AVX2 still needs eight
`f64` values. With the current split layout, that means one real gather and one
imaginary gather, or scalar indexed loads from both halves.

## AVX2 Strategies Tested

### 1. Original AVX2 Gather Kernel

The initial `FFT64Avx` implementation used `_mm256_i64gather_pd`:

```text
idx = perm[i..i+4]
re = gather(a_re, idx)
im = gather(a_im, idx)
store re
store im
```

This was intended to exploit the common permutation for the two halves. In
practice, AVX2 gather is slow for this workload. The source indices are
bit-reversed FFT slots, so memory access is scattered and not a simple
arithmetic progression.

### 2. Porting the Coefficient-Domain AVX Trick

The coefficient-domain AVX automorphism is much faster than the reference
implementation because output indices are contiguous and source indices can be
computed as:

```text
j * p^{-1} mod 2N
```

The AVX implementation computes four source indices at a time and gathers the
coefficient data with a SIMD sign mask.

This does not transfer cleanly to `FFT64` DFT automorphism. In FFT slot order,
the source index is:

```text
source = bitrev((p_eff * (2 * bitrev(dst) + 1) - 1) / 2)
```

The bit reversal around the multiplication destroys the simple SIMD-computable
progression used by coefficient-domain automorphism.

### 3. AVX-Owned Scalar Indexed Kernel

An AVX-owned scalar kernel was tested in `poulpy-cpu-avx`:

```text
for i in 0..n/2:
    s = perm[i]
    res_re[i] = a_re[s]
    res_im[i] = +/- a_im[s]
```

Variants tested:

- unchecked scalar loads/stores;
- unrolled-by-four scalar loads/stores;
- `#[inline(always)]` on the wrapper and inner loops;
- diagnostic delegation from the AVX wrapper directly to the reference
  automorphism implementation.

Outcome: the AVX-owned scalar path was only a modest improvement over the
gather path and still did not beat the reference backend in the strict collapse
microbenchmark. In one comparison, `FFT64Ref` spent about `0.8 ms` on the same
1022 DFT automorphisms while the AVX path was around `2.0 ms`.

This suggested that standalone `FFT64` DFT automorphism is not a good target for
AVX2 acceleration under the current layout. The operation is dominated by
scattered indexed loads, and AVX2 gather is not competitive here.

### Correction (2026-05-22): the kernel is not the problem; the pipeline is

The "`0.8 ms` ref vs `2.0 ms` AVX" figure above conflated two things. The
current AVX kernel is the unchecked scalar gather/store loop in
`poulpy-cpu-avx/src/fft64/automorphism.rs`, and `FFT64Avx` dispatches
`vec_znx_dft_automorphism_with_plan` to it. A direct, same-process A/B of that
kernel against the reference kernel (`benches/automorphism_compare.rs`,
i9-12900K, n = 1024, 2 limbs) shows the **opposite** of the report:

```text
isolated, hot source:
  FFT64Avx  1022 automorphisms:  0.26 ms  (264 ns/call)
  FFT64Ref  1022 automorphisms:  0.59 ms  (586 ns/call)   AVX ~2x FASTER
```

The unchecked indexing wins because the loop is a bounds-check-heavy scattered
gather. So the *kernel* is not slower than the reference.

However, **in the full collapse pipeline the AVX automorphism is still ~3x
slower than the reference automorphism**, and this reproduces both in the phased
breakdown and in the end-to-end online numbers (back-to-back runs):

```text
in-context, 1022 DFT automorphisms:   FFT64Ref 0.81 ms   FFT64Avx 2.49 ms
end-to-end online DFT-hot collapse:   FFT64Ref 2.30 ms   FFT64Avx 4.08 ms
end-to-end online BSGS DFT-hot B=16:  FFT64Ref 1.61 ms   FFT64Avx 2.77 ms
```

`FFT64Avx` wins everywhere else (baseline collapse, both precomputes, and core
GLWE packing), but the *automorphism-dominated online collapse is faster on the
all-scalar `FFT64Ref` backend*.

### Root cause (pinned): per-transition cost in the AVX <-> scalar alternation

The cause is **not** the automorphism kernel, the data, or cache residency. It
is the cost of switching between the AVX-vectorized VMP and the scalar
automorphism, paid on the `FFT64Avx` backend at every boundary of the hot loop.
Two experiments pin it.

1. *Second back-to-back automorphism.* In the phased breakdown, after each VMP,
   a first automorphism is timed, then a **second identical automorphism on the
   same buffers with no VMP between** is timed:

   ```text
   1st automorphism (right after VMP):       1.90 us/call
   2nd automorphism (back-to-back, no VMP):   0.27 us/call   <- 7x faster
   ```

   The second matches the isolated micro-bench (264 ns) exactly. So the penalty
   is entirely a function of *temporal proximity to the VMP*, not the op.

2. *Isolated VMP, both backends* (`bench_vmp` in `automorphism_compare.rs`):

   ```text
   isolated 1x1 VMP (dnum=2, size=2):  FFT64Avx 596 ns   FFT64Ref 1264 ns
   ```

   The AVX VMP is ~2x faster than ref in isolation, yet in-context the breakdown
   shows them nearly equal (AVX ~1366 ns vs ref ~1460 ns). The AVX VMP pays a
   ~770 ns in-context penalty; the ref VMP pays ~196 ns.

Putting the two ops together (per call):

```text
op            isolated   in-context   AVX in-context penalty
VMP (AVX)       596 ns     ~1366 ns    +770 ns
VMP (Ref)      1264 ns     ~1460 ns    +196 ns
auto (AVX)      264 ns     ~1900 ns    +1640 ns
auto (Ref)      586 ns      ~810 ns    +224 ns
```

Both AVX ops are individually ~2x faster than ref, but **each pays a large
penalty at the AVX<->scalar boundary, in both directions**, while the all-scalar
ref pipeline has no such boundaries. The collapse alternates VMP (AVX) ->
automorphism (scalar) -> VMP -> ... ~1022 times, so AVX pays ~2x1022 transitions
per query. That is why `FFT64Ref` wins the online collapse end-to-end (2.30 ms
vs 4.08 ms) despite being ~2x slower per isolated op.

Falsified along the way (none reproduced the in-context blow-up): the kernel
itself (AVX faster isolated), AVX2 frequency throttle (heavy FMA burst before
each automorphism only ~1.5x; Alder Lake offset is small), cache eviction (16 MB
stream <=1.8x, AVX still beats ref), RFO on the automorphism output (redundant
pre-zero made it *slower*), and a wide-store -> narrow-load hazard on the source
(AVX-writing the source first left the automorphism at 269 ns). The exact
microarchitectural mechanism of the per-transition cost (store-buffer / memory-
ordering drain after the VMP's tail, vs an AVX->scalar state-transition flush)
would need hardware perf counters to separate; WSL2 does not expose them. The
behavioral signature -- one-shot, per transition, both directions -- is what
matters for the fix.

### Update (2026-05-22, second round): mechanism is **producer->consumer through memory**, not AVX<->scalar

Following the earlier "AVX<->scalar transition" framing, I implemented an
AVX2-gather DFT automorphism (`avx_auto::automorphism` in
`automorphism_compare.rs`, 256-bit YMM loads/stores) to test whether keeping
both ops in AVX would remove the penalty. It **does not**:

```text
isolated:   scalar auto 264 ns,   AVX2-gather auto 310 ns,   AVX seq_copy 58 ns
in-context (per step, VMP + op, FFT64Avx):
  vmp_vmp           (2nd VMP reads UNRELATED buffer)        1200 ns  <- fast
  alt_avx_seq       (AVX seq copy of VMP output)            2435 ns  <- slow
  alt_avx_gather    (AVX gather of VMP output)              3180 ns  <- slow
  alt_scalar_auto   (scalar gather of VMP output)           3200 ns  <- slow
  vmp_vmp_chain     (2nd VMP READS 1st VMP's output)        3786 ns  <- slow
```

The decisive control is `vmp_vmp_chain`: two AVX VMPs alternating, where the
second reads the first's just-written output. Same op type, same encoding -- it
is the **slowest** of all schedules. That rules out the AVX<->scalar mode
switch as the cause.

The actual cause: **any consumer that reads memory the VMP just wrote pays
~2-2.5 us per step**, regardless of whether the consumer is scalar or AVX,
gather or sequential. The op type does not matter; the read-after-write through
memory does. This also explains why software pipelining did not help (one
iteration's gap is shorter than the dependency latency), and why the earlier
wide-store->narrow-load probe missed it (a short FMA-write tail drains fast; a
full gemm VMP leaves a much deeper memory tail).

### Decisive schedule experiment (`bench_schedule` in `automorphism_compare.rs`)

Driving the real op pair (one 1x1 VMP + one DFT automorphism, FFT64Avx) under
four schedules makes the cause unambiguous:

```text
per step = 1 VMP + 1 automorphism                ns/step
  alternating (VMP -> auto, the real loop)        ~3200
  pipelined   (VMP one step ahead of its auto)    ~3100   <- pipelining does NOT help
  batched     (all VMPs, then all automorphisms)   ~830   <- 4x faster
  vmp_vmp     (two AVX ops, no scalar)            ~1200   <- == 2x isolated VMP, no penalty
isolated:  VMP 596 ns,  automorphism 264 ns  (sum 860 ns)
```

- `batched` (830 ns) == isolated VMP + isolated automorphism (860 ns): batching
  to one transition removes the entire penalty.
- `vmp_vmp` (1200 ns) == 2x isolated VMP: alternating two **AVX** ops costs
  nothing extra. So the penalty is **specifically the AVX<->scalar mode switch**,
  not alternation, and not a data dependency.
- `pipelined` does not help: the automorphism still runs right after an AVX VMP
  every step, so the transition count is unchanged. The cost is the boundary
  itself, not waiting on a specific buffer to drain.

Per-transition cost is therefore ~1.15 us (the ~2.3 us/step penalty split over
the two boundaries per step).

### Data-layout sizing (per step, n = 1024, 2 limbs)

```text
VmpPMat (key)        : n * (cols_in*rows) * (cols_out*size) f64 = 1024*2*2*8 = 32 KB
                       same key every step -> stays hot (L1/L2)
input mask (a)       : 16 KB, but a DIFFERENT one each step (from the 16 MB
                       precompute_dft cache) -> cold streaming read
automorphism source  : the VMP's 16 KB output -> hot (just written)
```

The VMP's own in-context inflation (596 -> ~1366 ns) is mostly the cold-mask
stream, measured at ~270 ns/step (batched 830 -> 1090 with distinct masks); the
16 MB/query is memory-bandwidth bound. This is a *separate, smaller* effect from
the automorphism's transition penalty -- the earlier draft over-unified them.

Practical takeaways:

- Do not "fix" the AVX kernels; both the VMP and the automorphism are ~2x faster
  than ref in isolation. The loss is structural (transition count), not kernel
  quality.
- **Break the per-step VMP -> read-of-VMP-output dependency.** With the
  mechanism reframed (it is the RAW through memory, not an AVX<->scalar
  switch), what BSGS actually buys is *partly* fewer scalar consumers of the
  VMP output: it replaces the scalar gather automorphism on every step with an
  AVX `dft_add_assign` on every step (~800 ns/step cheaper consumer, see
  `alt_avx_seq` vs `alt_scalar_auto`) plus one scalar auto per group. The fully
  batched schedule shows the floor: ~830 ns/step (= isolated VMP + isolated
  auto), achieved by doing *all* VMPs before *any* consumer reads their output.
- **Deep-batched BSGS (now the default).**
  `sequential_keyswitch_collapse_aggregate_mask_precomputed_bsgs_dft_hot` in
  `src/circuit/collapse.rs` was unified to the deep-batched schedule: Phase 1
  -- all `B` baby VMPs into `B` distinct buffers; Phase 2 -- all `B` adds into
  the group accumulator (each reading a buffer written ~B VMPs ago, well past
  the dependency window); then one giant automorphism. Numerically identical
  to the previous single-buffer schedule (same f64 ops in the same order; the
  bench's `dft_hot` vs `bsgs(B=1)` raw-buffer assert and the
  `dft_hot_body_collapse_decrypts` test still pass). Extra memory:
  `B * 16 KB` of heap scratch held across the inner loop (B=16 -> 256 KB,
  fits L2). Measured win on `strict_collapse` (FFT64Avx, n=1024):

  ```text
                  before (single-buffer)   after (deep-batched, unified)
  B=8             2.930 ms                  1.914 ms       -35%
  B=16            2.850 ms                  1.899 ms       -33%
  B=32            3.014 ms                  1.913 ms       -37%
  B=64            2.907 ms                  1.920 ms       -34%
  ```

  Flat across B (1.90-1.92 ms) -- the per-baby VMP -> add penalty that
  previously made small B optimal is gone, so B can be chosen by baby-key
  cache budget alone. This is the only previously unimplemented recommendation
  that the mechanism analysis predicted would work; it did.
- ~~**Reconsider an AVX-native DFT automorphism.**~~ **Retracted.** Tested with
  a real AVX2-gather kernel (`avx_auto::automorphism`): it does *not* help
  (3300 ns/step alternating, essentially identical to the scalar kernel). The
  cause is the read-after-write dependency through memory on the VMP's output,
  not the consumer's encoding -- so an AVX or NTT permutation automorphism would
  hit the same penalty. The transition framing in the previous round was wrong.
  This entry stays to record the dead end.
- Software pipelining the VMP ahead of its automorphism does **not** help
  (measured); do not pursue it.
- The cold-mask stream (~270 ns/step, 16 MB/query) is a separate, secondary,
  bandwidth-bound cost; attack it (if at all) with lower-precision mask storage
  or reuse, not with scheduling.
- Short term, for the online collapse specifically, the all-scalar `FFT64Ref`
  backend is ~1.8x faster end-to-end; a split (AVX offline/packing, scalar
  online) is worth evaluating.

## Strategies Considered and Deferred

### Fused Automorphism Add

A fused operation:

```text
body_acc_dft += tau(product)
```

would remove:

```text
product_auto = tau(product)
body_acc_dft += product_auto
```

This saves the temporary write/read and the standalone DFT add. However, the
DFT add bucket was only about `0.15 ms`, while automorphism was around
`2.0 ms`. Even deleting the whole add pass would only save around four percent
of the DFT-hot total. It is useful cleanup, but not the main performance fix.

### Fused VMP accumulate for the BSGS group sum (tested, no benefit)

The BSGS group loop does one `1 x 1` VMP per baby into a temporary, then a DFT
add into the group accumulator. The HAL exposes
`vmp_apply_dft_to_dft_accumulate` (`res += a . pmat`), which looks like a way to
fuse those two and drop the temporary. For `FFT64` it is **not** kernel-fused:
the default impl (`poulpy-cpu-ref/.../hal_defaults/vmp_pmat.rs`) zeroes an
internal temporary, runs the same matvec kernel into it, then DFT-adds into
`res` -- i.e. exactly the existing pair plus an extra zeroing per call. It
allocates an equivalent temporary internally, so there is no memory-traffic win
either. Prototyped in the BSGS loop and reverted: at best neutral, and the DFT
add bucket it targets is only ~4% and L1-resident. Do not re-attempt for the
FFT64 backend without a genuinely kernel-fused accumulate.

### Changing the FFT64 DFT Layout

Changing the DFT layout could make automorphisms more SIMD-friendly, but it
would also affect DFT, IDFT, VMP, and the rest of the FFT path. Since
DFT-domain automorphisms are rare outside this specific collapse, a global
layout change is unlikely to be worth the cost.

### NTT120 or NTT60

`NTT120` DFT automorphism is more AVX-friendly because each slot is a 32-byte
block and the operation is a pure permutation. However, `NTT120` is generally
slower than `FFT64` for the broader workload. A future `NTT60` variant with two
30-bit primes may be worth benchmarking, but it is a separate backend
experiment rather than a direct fix for `FFT64`.

### Special Plans for Specific Automorphisms

Special-casing particular automorphism permutations was considered brittle.
The collapse uses many automorphisms, and maintaining a collection of
permutation-specific kernels is not attractive.

## BSGS Strategy

The successful direction is to reduce the number of DFT automorphisms in the
hot path algorithmically.

For the current schedule:

```text
sum_step tau_alpha_step(VMP(mask_step, Kg_body))
```

Within each half of the collapse, the automorphism sequence can be grouped. For
a baby size `B`, decompose:

```text
alpha_{q,r} = giant_q * baby_r
```

with:

```text
baby_r = g^r
```

Using automorphism linearity:

```text
sum_r tau_{giant_q * baby_r}(VMP(mask_{q,r}, Kg_body))
=
tau_{giant_q}(
    sum_r VMP(tau_{baby_r}(mask_{q,r}), tau_{baby_r}(Kg_body))
)
```

The online loop becomes:

```text
body_acc_dft = DFT(body)

for group q:
    group_acc = 0
    for baby r in group:
        product = VMP(baby_mask[q,r], baby_key[r])
        group_acc += product
    body_acc_dft += tau_giant_q(group_acc)

product_h = VMP(mask_final, Kh_body)
body_acc_dft += product_h

body = IDFT(body_acc_dft)
mask = precomputed_final_mask
```

This preserves the `Kg` VMP count but reduces the automorphism count from:

```text
n - 2
```

to roughly:

```text
ceil((n/2 - 1) / B) * 2
```

For `n = 1024`:

```text
B = 8   => 1022 autos -> 128 autos
B = 16  => 1022 autos -> 64 autos
B = 32  => 1022 autos -> 32 autos
B = 64  => 1022 autos -> 16 autos
```

Groups are built separately for the two collapse halves so no group crosses
the sign boundary between the `tau_g` and `tau_h` views.

## Current Implementation

The implementation is PIR-local and side-by-side with the existing DFT-hot
path.

Added precompute/state:

- `SequentialCollapseBsgsGroup`
- `SequentialCollapseBsgsDft`
- `sequential_collapse_bsgs_dft_build`

Added online collapse:

- `sequential_keyswitch_collapse_aggregate_mask_precomputed_bsgs_dft_hot`

Added benchmark support:

- baby-key preparation for `B = 8, 16, 32, 64`;
- strict collapse timings for each BSGS baby size.

Added correctness coverage:

- `dft_hot_body_collapse_decrypts` now decrypt-checks both the original
  DFT-hot path and the BSGS DFT-hot path for `B = 8`.

## Benchmark Results

Latest strict collapse benchmark, `FFT64Avx`, `n = 1024`, five iterations:

```text
baseline online full 1x2 collapse:      22.773 ms
offline fixed-mask precompute:           9.933 ms
offline DFT-accumulated precompute:      9.513 ms
online precomputed body-only collapse:   8.215 ms
online DFT-hot body-only collapse:       3.520 ms
online BSGS DFT-hot (B=8):               2.843 ms
online BSGS DFT-hot (B=16):              2.642 ms
online BSGS DFT-hot (B=32):              2.805 ms
online BSGS DFT-hot (B=64):              2.760 ms
core GLWE pack of n GLWEs:              26.919 ms
```

`B = 16` was the best value in this run, reducing the online DFT-hot collapse
from `3.520 ms` to `2.642 ms`, about a `25%` improvement.

This gain is consistent with the original breakdown. BSGS removes most of the
DFT automorphism count, but it does not reduce the `1022` `Kg` `1 x 1` VMPs,
which were already around `36-38%` of the DFT-hot work before BSGS.

## Review Notes

The BSGS approach is expected to be beneficial in the intended PIR workload.
The application is expected to perform roughly `8k` to `16k` packing/collapse
calls for databases in the `16 GB` to `32 GB` range. Because the baby keys are
small and precomputed, extra key storage is not a practical concern.

The current performance tradeoff is:

- lower DFT automorphism count;
- unchanged VMP count;
- extra baby-key variants;
- extra per-group accumulation;
- additional DB-side BSGS mask DFT cache.

The next optimization target, if needed, is the repeated `Kg` `1 x 1` VMP
bucket rather than standalone FFT64 DFT automorphism.

## Validation Commands

Commands run during this investigation:

```text
cargo check --manifest-path Cargo.toml --benches
cargo test --manifest-path Cargo.toml dft_hot_body_collapse_decrypts
RUSTFLAGS='-C target-feature=+avx2,+fma' cargo check -p poulpy-cpu-avx --features enable-avx
RUSTFLAGS='-C target-feature=+avx2,+fma' cargo bench --manifest-path Cargo.toml --bench strict_collapse
```

