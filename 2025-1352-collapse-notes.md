# 2025-1352 Collapse Notes

Source: `2025-1352.pdf`, extracted locally with `pdftotext -layout`.

## [P1] Parameters And Secrets

- Ring: `Rq = Zq[X] / (X^d + 1)`, with power-of-two `d`.
- Base LWE secret as polynomial: `s~`.
- Galois generators:
  - `g = 5`
  - `h = 2d - 1`
- Intermediate secret vector `s^ in Rq^d`:
  - For `0 <= j < d/2`: `s^[j] = tau_g^j(s~)`.
  - For `0 <= j < d/2`: `s^[j + d/2] = tau_h tau_g^j(s~)`.

## [P2] Stage 1 Output

For an LWE ciphertext `(a, b)`, Stage 1 constructs an intermediate ciphertext
`IRCtx(m) = (a^, b~) in Rq^d x Rq` such that:

```text
b~ = - <a^, s^> + m  (mod q)
```

The mask components are:

```text
a^[j]       = tau_g^j(a~)       for 0 <= j < d/2
a^[j+d/2]   = tau_h tau_g^j(a~) for 0 <= j < d/2
```

up to the global `d^{-1}` scaling in the trace construction.

## [P3] Stage 2 Aggregate

Given `d` intermediate ciphertexts, Stage 2 computes:

```text
(a^agg, b~agg) = sum_{k=0}^{d-1} IRCtx(m_k) * X^k
```

so the plaintext becomes:

```text
m^agg(X) = sum_{k=0}^{d-1} m_k X^k
```

The aggregate is still an intermediate ciphertext with `d` mask components:

```text
b~agg = - sum_{j=0}^{d/2-1} a^agg[j]     * tau_g^j(s~)
        - sum_{j=0}^{d/2-1} a^agg[j+d/2] * tau_h tau_g^j(s~)
        + m^agg
```

## [P4] Required Keys

The paper requires two elementary key-switching matrices:

```text
Kg = KS.Setup(tau_g(s~), s~)
Kh = KS.Setup(tau_h(s~), s~)
```

Automorphic images of `Kg` are used to switch:

```text
tau_g^{j+1}(s~)       -> tau_g^j(s~)
tau_h tau_g^{j+1}(s~) -> tau_h tau_g^j(s~)
```

`Kh` is used only after the two halves have been reduced to:

```text
b2 = -a1 * s~ - a2 * tau_h(s~) + m^agg + error
```

## [P5] COLLAPSE_ONE

Input:

```text
(a, b) in Rq^k x Rq
K = KS.Setup(source_secret, target_secret)
```

The eliminated component is the last mask component `a[k-1]`.

Key-switch:

```text
(a_ks, b_ks) = KS.Switch((a[k-1], b), K)
```

If `K` switches from `source_secret` to `target_secret`, then the returned mask
`a_ks` is added into the previous component that is already associated with the
target secret:

```text
a' = [a[0], ..., a[k-3], a[k-2] + a_ks]
b' = b_ks
```

The rank drops from `k` to `k-1`.

## [P6] COLLAPSE_HALF

Input half:

```text
(a^half, b^half) in Rq^{d/2} x Rq
rho in {I, tau_h}
```

For `k = d/2 - 1` down to `1`, collapse component `k` into component `k-1`.

The key for step `k` is:

```text
K_{k-1} = rho tau_g^{k-1}(Kg)
```

This switches:

```text
rho tau_g^k(s~) -> rho tau_g^{k-1}(s~)
```

After `d/2 - 1` steps, the half has one mask component:

```text
rho = I:     under s~
rho = tau_h: under tau_h(s~)
```

## [P7] Full Collapse Count

Full `COLLAPSE`:

1. Collapse first half with `rho = I`: `d/2 - 1` key-switches using automorphic images of `Kg`.
2. Collapse second half with `rho = tau_h`: `d/2 - 1` key-switches using automorphic images of `Kg`.
3. Collapse `[a1, a2]` with `Kh`: `1` key-switch.

Total:

```text
(d/2 - 1) + (d/2 - 1) + 1 = d - 1
```

Thus, the paper uses two base key-switching matrices, not two key-switch calls.

## [P8] Preprocessing Observation

For server-side preprocessing, the mask-side updates depend only on:

- the fixed random masks of the LWE/RLWE ciphertexts,
- the fixed random masks of `Kg` and `Kh`,
- the deterministic automorphisms applied to `Kg`,
- the deterministic collapse schedule.

The online body update still needs the body part of each key-switch contribution,
because the body carries the message-dependent `b`.

## [P9] Avoiding Key Automorphisms In Implementation

The paper writes the step key as an automorphic image of `Kg`, but an implementation
does not need to automorph the prepared/DFT key. For an automorphism `alpha`, using
`alpha(Kg)` is equivalent to:

```text
ct'  = alpha^{-1}(ct)
out' = KS.Switch(ct', Kg)
out  = alpha(out')
```

Thus, the key remains the base prepared key `Kg`; the automorphisms are applied to
the temporary rank-1 GLWE term and to the switched output. In this codebase the
aggregate secret view for column parameter `p` is represented with the inverse
Galois action, so the conjugating automorphism is `alpha = p^{-1}`.
