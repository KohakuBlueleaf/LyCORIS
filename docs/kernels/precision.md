# Precision

## Dtype policy

The input (`x`, or the original weight) and the LyCORIS module weights carry
their dtypes independently. Both may be fp16, bf16 or fp32, in any
combination — except fp16 together with bf16, which is a caller bug rather
than a case to resolve: the two have no common half, so either would have to
widen to fp32 to meet, and silently doing that hides a mistake.

For each call, `lycoris/kernels/precision.py` resolves two dtypes:

* **compute** — what the matmul operands are in. With any 16-bit operand
  present, that 16-bit type (policy `mma16`); otherwise fp32.
* **output** — the promotion of every operand's dtype. An fp32 operand
  anywhere yields an fp32 result whatever the matmul ran in.

Accumulation is **fp32 under every policy**. A 16-bit matmul with fp32
accumulation is the arrangement the tensor cores implement natively, so this
costs nothing and is what the eager path gets from cuBLAS too.

Gradients come back in each leaf's own dtype: an fp32 master weight beside an
fp16 activation gets an fp32 gradient, with no cast in your training loop.

fp16 carries 11 mantissa bits against bf16's 8, so an fp16 run is the more
accurate of the two 16-bit options here by roughly three bits — bf16 buys
range, not precision.

## Measured accuracy

Accuracy is reported in **ULP of the working dtype** against an fp64 oracle:
1.0 ULP is as exact as the dtype can represent, so the number is comparable
across fp16, bf16 and fp32 rows. Reductions and GEMMs are scored against the
reference's RMS (a near-zero output there is cancellation, not a small true
value); pointwise kernels are scored elementwise.

Across the benchmark sweep the fused kernels land at or below the eager
error — e.g. `lora` merge at 9.6 ULP against eager's 24.6, `oft` at 5.0
against 4.0 — because the whole chain accumulates in fp32 registers instead
of round-tripping intermediates through 16-bit memory. TileLang's BOFT path
is the one place where the fused error is materially larger than eager's
(31.7 vs 2.3 ULP on the butterfly sweep); the Triton path is 10.8.

Where an intermediate is deliberately rounded — `h = x @ downᵀ` in the LoRA
bypass, for instance — it is rounded to the storage dtype before its second
dot, which is exactly the rounding the eager two-`Linear` chain applies. The
fused path reproduces the reference's rounding rather than quietly improving
on it, so a fused and an unfused run stay comparable.

## Checking it yourself

```bash
python scripts/bench/kernels/quick.py          # all paths, fp16/bf16/fp32, ULP + device time
python -m pytest test/kernels -q               # correctness against the eager references
```

The bench harness computes ULP in the same row as the timing, so no figure
reports a speed without the accuracy it was obtained at.

## VRAM

The kernels are written so that the large intermediates never exist:

* ΔW and its gradient are never materialised on the bypass paths — the token
  tile is contracted straight through both factors.
* The Kronecker halves `w1` and `w2` are generated per output tile in
  registers, not built and then read back.
* BOFT's backward replays forward prefixes instead of caching the `m` stage
  inputs (`m · O · I` bytes).

What *is* cached is small and expensive to recompute: the DoRA row norms, one
fp32 value per row. Everything else — `h`, `q`, the Cayley transform, the
hadamard tiles — is recomputed in the backward, because recomputing them is
cheaper than the memory traffic of storing them.
