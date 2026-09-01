# Benchmarks

## Running them

```bash
# seconds: every algorithm, every path, small shapes, precision + device time
python scripts/bench/kernels/quick.py --dtype fp16

# the full sweep: writes one JSON per family, then draws the figures
python scripts/bench/kernels/run_all.py --out out/bench/kernels

# roll every family's JSON into one verdict table
python scripts/bench/kernels/verdict.py --dir out/bench/kernels
```

`quick.py` is the working loop — a kernel that is wrong or slow shows it at
1280x1280 as clearly as at 11008x4096. `run_all.py` is the expensive run:
denominators first (the machine's own DRAM bandwidth and matmul peak, measured
on the spot, so a figure drawn later still says what it was scored against),
then each family, then the figures. Measure scripts write JSON; plot scripts
read that JSON and never re-measure, so a finished run redraws without a GPU.

## How the numbers are taken

* **Device time** is the primary metric: per-kernel time summed from the
  profiler, best of three windows of 50 calls. At these sizes the wall clock
  is mostly Python dispatch, so a wall-based ratio compares launchers rather
  than kernels. Every figure draws the wall faintly behind the device line;
  the gap between them is the dispatch cost.
* **Arms are interleaved** round-robin, each keeping its best round, so
  thermal drift cannot land on one arm.
* **L2 is flushed** between iterations (256 MiB). Rows whose working set is
  L2-resident are annotated as such — a bandwidth figure above 100% of DRAM
  is a cache hit, not a fast kernel.
* **Accuracy is in the same row as the time**, in ULP against an fp64 oracle,
  so no speedup is reported without the precision it was obtained at.
* The `compile` arm calls `torch._dynamo.reset()` per case: without it the
  sweep hits the recompile limit and silently runs eager, which reads as a
  suspiciously fast baseline.

## Current results

RTX 4090, fp16, geometric mean over each family's shape sweep. `dev` is
device (kernel) time, `fb` is the fwd+bwd wall, `vram` is peak fwd+bwd memory
(all ratios >1 mean ours is better).

| family | backend | n | dev/eager | dev/compile | fb/eager | fb/compile | vram | ULP | eager ULP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora | triton | 24 | 4.87 | 3.10 | 1.46 | 1.43 | 1.18 | 9.6 | 24.6 |
| lora | tilelang | 24 | 2.54 | 1.62 | 1.25 | 1.23 | 1.13 | 9.6 | 24.6 |
| lora_bypass | triton | 4 | 1.37 | 1.19 | 0.91 | 0.95 | 1.10 | 3.8 | 5.5 |
| lora_bypass | tilelang | 4 | 1.29 | 1.13 | 0.95 | 1.00 | 1.07 | 3.6 | 5.5 |
| lokr | triton | 16 | 2.98 | 0.93 | 0.73 | 0.89 | 1.13 | 6.1 | 6.1 |
| lokr | tilelang | 16 | 1.70 | 0.53 | 0.69 | 0.84 | 1.08 | 6.1 | 6.1 |
| lokr_bypass | triton | 4 | 4.26 | 2.60 | 1.47 | 1.58 | 1.06 | 3.1 | 3.1 |
| lokr_bypass | tilelang | 4 | 1.38 | 0.84 | 1.49 | 1.61 | 1.06 | 3.1 | 3.1 |
| oft | triton | 8 | 5.70 | 5.05 | 2.23 | 1.82 | 1.10 | 5.0 | 4.0 |
| oft | tilelang | 8 | 4.35 | 3.86 | 2.16 | 1.76 | 1.10 | 8.4 | 4.0 |
| oft_bypass | triton | 3 | 7.26 | 6.15 | 2.15 | 1.89 | 1.06 | 2.4 | 2.2 |
| oft_bypass | tilelang | 3 | 4.59 | 3.89 | 1.93 | 1.70 | 1.06 | 4.9 | 2.2 |

Reading the rows that are below 1.0:

* **`fb/*` below `dev/*`** — the fwd+bwd wall includes the Python dispatch of
  the whole autograd graph, which the kernels cannot shrink. On the small
  end, dispatch is the run.
* **`lokr` vs compile (0.93 / 0.53)** — inductor fuses the both-full Kronecker
  rebuild about as well as a hand-written kernel does; that shape's win is
  memory, not time.
* **TileLang under Triton throughout** — expected on Windows, where TileLang
  compiles through `cl`; the ordering on Linux is the other way around for
  several families.

## Figures

`run_all.py` writes six line panels per family into `out/bench/kernels/`:
forward latency, fwd+bwd latency, bandwidth efficiency against the measured
DRAM ceiling, device-time speedup against eager, peak VRAM, and ULP. Bold is
device time, faint is wall, crimson dashes are the measured ceilings of the
card the run happened on.
