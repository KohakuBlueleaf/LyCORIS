# Config selection (`lycoris/kernels/tuner/`)

The fused ops pick their tile configuration per call key (kernel, dtype class
and flags, shape bucket) with an analytical model that ranks and a short timed
shortlist that decides. Nothing is swept and no block size is chosen by hand.

## Per key, once per process

1. The winner stored in the table for this card, when there is one and it is
   still a legal config of the op's planner.
2. Otherwise the planner enumerates its legal configs and the model orders
   them. `TILE_SHORTLIST` (4) tile shapes are timed: the model's best, then the
   model's best with a new value on each tile axis in turn. The whole-op
   alternatives (`{"impl": ...}`: the torch composition, other formulations,
   the single-launch kernels of the family's `apply.py`) are timed with them.
   Then `LAUNCH_SHORTLIST` (3) launch settings (warps, stages) of the fastest
   tile. The fastest of all is stored.
3. Inside CUDA-graph capture or `torch.compile` tracing, and with
   `LYCORIS_KERNEL_TUNE=off`, the model's top config runs untimed and is not
   stored.

Before a candidate is timed, its result is compared with the torch
composition's: a candidate off by more than a loose relative Frobenius
distance (1e-2 fp16, 4e-2 bf16) or not finite times as infinity. This guards
against a miscompiled config; the precision rule itself is a test
([precision.md](precision.md)). Candidates compile one at a time.

## The model (`tuner/model.py`)

Each planner describes a config as `Work`: grid size, warps, the fp32
accumulator and operand registers and the shared memory one program holds, its
streaming loops (DRAM bytes and the bytes one program keeps in flight in
each), the bytes all programs load including L2-served re-reads of shared
operands, and its MMA and vector FLOPs. The model scores that with no
per-product constants:

- occupancy by the CUDA occupancy calculator's rules, on the limits the driver
  reports (SMs, registers, shared memory and its per-block reserve, threads and
  blocks per SM);
- each loop streams at the bandwidth this card reaches with that many bytes in
  flight per SM (pipeline stages count only where `cp.async` exists, sm_80+);
- L2 bytes over the L2 bandwidth;
- compute over the matmul rate (16-bit MMA where the kernel's dots reach the
  matrix units, the fp32 vector FMA rate otherwise), wave-quantized;
- a launch per wave.

The bandwidth-against-bytes-in-flight curve, the L2 bandwidth, the 16-bit MMA
rate and the fp32 vector FMA rate are measured on the card once per process
(a dependent-load chain kernel, an L2-resident re-read, a 4096^3 fp16 GEMM and
an FMA chain; no fp32 matmul). Triton's dots reach the matrix units from sm_80;
TileLang's `T.gemm` from sm_70, so TileLang planners rank with the MMA rate on
every card.

## Timing

Candidates are timed as CUDA graphs over enough independent copies of the
inputs to cycle L2 out (4x its size), interleaved round-robin, best of four
rounds; a candidate over twice the leader after the first round stops early.

## The table

`<LYCORIS_KERNEL_CACHE_DIR>/kernels/<arch>-<card>.json` (default
`~/.cache/lycoris_kernels`), one file per card name, stamped with the torch /
Triton / TileLang / LyCORIS versions; a stamp mismatch starts the table empty.
A save merges with rows another process wrote since this one loaded the
table and replaces the file atomically (retried on Windows while another
handle holds it).
