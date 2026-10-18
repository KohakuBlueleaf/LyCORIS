# Kernel designs

How each fused op computes its result, per family. Every op lives behind the
model-ranked tuner ([tuning.md](tuning.md)), which times the op's kernel
configs against whole-op alternatives (the torch composition, and the
single-launch kernels of each family's `apply.py` where they apply) and keeps
the fastest. Operand and accumulator dtypes follow
[precision.md](precision.md): matmul operands in the 16-bit dtype present,
accumulation in fp32, the result rounded once at the store.

## The addend fold

A bypass layer computes `base + delta(x)`, a merge `W + delta`. The fused ops
take the addend (`base=` on the bypass and merge entry points of
`lycoris.kernels.autograd`, `delta=` on `apply_dora`) and add it in the
kernel's store, so the base output (or the original weight) is read once and
the result written once, rounded once from the fp32 accumulator. Without the
fold the delta is stored, read back and added: one extra read and write of a
`(tokens, out)` or `(out, in)` tensor. The gradient of the addend is the
incoming gradient itself. The fold applies when `base + delta` keeps the
base's dtype (`lycoris.kernels.precision.folds`); otherwise the add runs in
torch with the usual promotion.

## LoRA / LoCon (`triton/lora/stream.py`, `rebuild.py`)

Bypass forward: one launch per token tile and output slice. `h = x @ down^T`
(tokens x rank) stays in fp32 registers, is rounded to x's dtype (as the
two-Linear chain rounds it), and the output axis streams in chunks with the
base added in the store. `SPLIT_O` spreads the output axis over more programs
when there are few token tiles. Bypass backward: one launch computes `gx` and
stores `h` and `q = g @ up`; the factor gradients are two GEMMs over tokens
(no atomics). The single-launch backward that adds the factor gradients
atomically is a timed alternative.

The planner-ranked bypass forward (`triton/lora/bypass.py`, ranked by
`plans/lora.py`) is 1-D over token tiles: `h` is computed once per tile and
reused across the whole output axis, so x is read once. Splitting the output
axis would re-read x once per output slice; the token tile is the only
parallelism axis.

Merge forward: a 2-D grid over the weight reduced over rank, the base add in
the epilogue. Merge backward: "roles" (g_up and g_down programs, each split S
ways along its reduction) or "tile" (each grad tile read once for both
gradients); fp32 partials summed in torch. GLoRA's `u @ a2 + b1 @ b2` is the
rank-(r1 + r2) LoRA merge of the stacked factors (`triton/merge/modes.py`).

## LoHa (`triton/loha/hada.py`, `regen.py`)

Merge forward: both rank products accumulate per output tile in one pass; the
hadamard, scale and base add run in the epilogue. Merge backward: "roles"
rebuilds both products in each role, "tile" builds them once per grad tile and
feeds all four gradients. Bypass: regenerate the DeltaW tile inside the GEMM's
K loop, or materialize DeltaW once (the merge kernel) and run one cuBLAS GEMM
with the base as its addend. The bypass backward materializes DeltaW:
`gx = g @ DeltaW`, `g_W = g^T x`, then the merge backward; the single-launch
backward (ranks up to `MAX_HADA_RANK`) is a timed alternative.

## LoKr (`triton/lokr/apply.py`, `grad.py`, `kron.py`)

Bypass forward, the Kronecker product never built: a program takes a tile of
tokens, of w1's rows and of w2's rows, with `M = X @ w2^T` in registers, in
two layouts. "batch" makes tokens the batch dimension of 3-D dots; "diag"
contracts w1 as one 2-D dot against a block-diagonal (token, u) x (token, v)
tile, so stacked tokens fill the 16-wide dot minimum for small factors. Every
factor axis is tiled, so there is no factor-size bound.

Bypass backward, contracting w1 first (eager's order): per token tile and gx
column tile, `N = blockdiag(w1^T) @ G` is one dot against the token-diagonal
tile, rounded to the compute dtype and stored once, then `gx` accumulates
`N @ w2`. `N` is the operand of `g_w2 = N^T X`, one GEMM over (token, b) rows.
`g_w1 = sum_t G_t mid_t^T` with `mid = X @ w2^T` is reduced per token tile as
the token-diagonal blocks of one (BT*a, BT*b) tile product; fp32 partials are
summed in torch.

Full merge: one program per output tile gathers w1 and w2 per element (an
elementwise product, no matmul) and adds the base. Its backward gives each
w1 element a program over its (C, D) block of g, and sums w1 x g over S splits
for each w2 tile; fp32 partials, no atomics.

The planner-ranked full merge (`triton/lokr/merge.py`) is output-tiled: a
program recovers the factor indices (u, c, v, d) once per row and once per
column vector, where a flat form pays four integer div / mods per element.
Every tile of the rebuild planner (`plans/lokr.py`) moves the same bytes, so
its order comes from wave efficiency, the unused share of a 128 B line for
short rows, and a slight preference for fewer warps and smaller tiles; its
space reaches 16-row tiles.

## Diag-OFT (`triton/oft/apply.py`)

One launch per direction. Each program rebuilds its block's `q = b - b^T`,
inverts `I - q` by an in-register Gauss-Jordan (no pivoting: `I - q` has a
positive-definite symmetric part), forms the Cayley rotation, folds the
rescale rows and the identity shift, and streams its column slice. The
backward chains the partial rotation gradient through the fixed rotation and
inverse in-kernel, with atomic `g_blocks` / `g_rescale`. Blocks up to 32.

## BOFT (`triton/boft/apply.py`)

One launch per butterfly stage; each stage kernel rebuilds its blocks'
rotations in-kernel (the diag-OFT Gauss-Jordan), folds the multiplier and, on
the last stage, the rescale rows and the diff. The per-stage gradient kernel
chains the rotation gradient to `g_blocks` in-kernel. The backward recomputes
the stage inputs by replaying forward prefixes instead of caching the `m`
stage inputs.

When the butterfly's cone fits one program (up to `CONE_MAX` = 32 channels;
no transpose, reverse or offset), the `m` stages compose to one PC x PC map,
built once per cone rather than per column block, and the cone kernel
(`SENTINEL_CONE`) is timed against the stage chain.

## DoRA (`triton/dora/decompose.py`)

`y = M * (mult * (d / ||M_row|| - 1) + 1)` with `M = w [+ delta]`, rowwise or
columnwise by transposed strides. One program per row tile reduces the row's
norm, then scales it; "resident" configs hold whole rows of M in registers
between the two, so w and delta are read once and M is never stored. The
backward reuses the forward's fp32 norms: `gM = g * s + coef * M` with
`coef = -mult * d * rowdot / n^3`, `gd = mult * rowdot / n`; the gradient of M
is the gradient of both w and delta.

## IA3 (`triton/ia3/channel.py`)

`y = x * (alpha + gamma * w[ch])` over x viewed as (rows, cols) with the
channel on the columns (channels-last) or as `row % C` (channels-first), so
the scale is loaded once per tile and no element computes its channel. x and w
may differ in dtype: both are read into fp32 and y is stored in their promoted
dtype. Backward: gx and the per-channel sums of `g * x` in one launch, from
fp32 per-program partials summed in a fixed order.

## TileLang (sm_70 / sm_75)

The TileLang backend follows the designs above, on the model-ranked tuner,
with the constraints of its lowering:

* A `T.gemm` tile splits over the program's warps, each owning a multiple of
  16 rows and of 16 (sm_70) or 8 (sm_75) columns; `common.partitions` drops a
  config that does not split before it is compiled. A fragment's elements
  split evenly over the program's threads, so whole-row tiles pad to a
  multiple of 128 lanes.
* `T.copy` between global and shared / fragment memory converts dtypes and
  zero-fills rows past a symbolic extent, so token tails need no masks and
  the partial sums of a tail tile are exact.
* An fp32 `T.gemm` lowers to FMAs on sm_70 / sm_75 and to TF32 MMA from
  sm_80, so the fp32-operand Diag-OFT and BOFT configs below exist on sm_70 /
  sm_75 only; elsewhere those ops take the torch composition or the
  single-launch kernels (`common.fp32_gemm_exact`).

LoKr full merge backward (`tilelang/lokr/kron.py`): one program per (w2 tile,
split of the a*b blocks) reads each grad block tile once, accumulating
`w1[u, v] * G_uv` (the split's g_w2 partial) and storing the row sums of
`G_uv * w2` (g_w1 partials); both partial sets are summed in torch, no
atomics. Tiles divide (c, d) exactly.

DoRA (`tilelang/dora/decompose.py`): the design above on the contiguous row
form; the column form (`wd_on_out=False`) runs on a transposed copy.

Diag-OFT (`tilelang/oft/solve.py`, `rotate.py`): the rotation in the
`R = I + D` form, `D = 2 (I - q)^-1 q` solved once per block by a
shared-memory Gauss-Jordan on `[I - q | 2q]` (`solve.deltas`), stored as
`Dh = D * rescale`. The apply streams x viewed as (outer, C, inner):
`out_i = keep_i x_i + sum_j x_j Dh[j, i]` with `keep = rescale - shift`, in
two layouts: "rows" (channels-last: a row tile against the block-diagonal Dh
of a channel tile) and "cols" (one block over a slice of the columns:
weights, channels-first activations). Dh and the x tile are fp32. Backward:
the same layouts give `gx = keep * g + Dh g` and fp32 partials of
`P = sum x g^T`; one program per block then chains `G_D = P * rescale` to
`g_blocks` through a second solve (`solve.chain`), and
`g_rescale = diag(P) + colsum(D * P)`. A solve whose (ps, ps) tiles exceed
the device's shared memory runs as `torch.linalg.solve` in fp32 instead. Blocks
up to 64; the single-launch
kernels of `tilelang/oft/apply.py` (blocks up to 32) are a timed alternative.

BOFT (`tilelang/boft/stages.py`, `stage_kernels.py`): the m x num rotations
are solved once (the diag-OFT solve, the multiplier folded into D, D in x's
dtype). Stage i applies `v = u + D u` on blocks whose b channels sit at
stride `k = 2^i * b / 2` inside groups of `2k`; the running activation is fp32
and only the D product is an MMA. One launch per stage on channels-last rows,
with the stage geometry (k, blocks per group) a runtime argument so one
compiled kernel serves every stage; the last stage folds the rescale and the
diff into its store. The backward recomputes the stage inputs once (x's
dtype, the MMA operands the partials need), then walks the stages in reverse:
`g_u = g + D^T g` and fp32 partials of `G_D = sum g u^T`, chained to the
blocks by `solve.chain`. Its transient memory is (m - 1) copies of the
activation, freed when the backward returns. Weights and channels-first
activations take the torch chain or the single-launch kernels of
`tilelang/boft/apply.py`.

IA3 (`tilelang/ia3/channel.py`): the design above; a channels-first program
owns a channel tile and walks (outer index, inner tile) pairs.
