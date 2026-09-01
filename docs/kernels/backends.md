# Backend Selection

## Order

Per call, in this order, first one that applies wins:

1. **triton** — the reference implementation of every fused op.
2. **tilelang** — the same op set, TileLang-authored.
3. **compile** — `torch.compile` applied to the individual op (not to your
   module), cached on the function object so each op compiles once per
   process.
4. **torch** — the stock eager implementation.

A call steps one tier down when the operands or the layout are outside a
fused kernel's scope: CPU tensors, fp16 mixed with bf16 in the same call, a
conv layout where the kernel is written for the linear one, a rank past the
kernel's register budget, or a scale that carries a gradient. The tier below
covers everything the tier above does, so the fallback is always defined —
there is no configuration in which a call fails for lack of a kernel.

The compile tier is used **on CUDA only** unless you name it explicitly: the
CPU callers here are weight merges, where an inductor warmup costs more than
the operation it replaces.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `LYCORIS_KERNEL_BACKEND` | `auto` | `auto`, `triton`, `tilelang`, `compile` or `torch`. A named backend is still narrowed per call by the scope rules above. |
| `LYCORIS_KERNEL_TUNE` | `on` | `off` skips the measured tile pick and takes the planner's first candidate. |
| `LYCORIS_KERNEL_CACHE_DIR` | `~/.cache/lycoris_kernels` | Where the tuning table (`tuning.json`) lives. |

## From Python

```python
from lycoris.kernels import available_backends, fused_backends, resolve_backend

available_backends()   # ('triton', 'tilelang', 'compile', 'torch')
fused_backends()       # ('triton', 'tilelang')  -- the ones with an op set
resolve_backend()      # 'triton'  -- process-wide preference
```

The per-call decision is `lycoris.kernels.select.choose(tensors, supported)`;
`supported` is the caller's own layout test. Every functional entry point and
every module method that has a fused path goes through it, so there is one
place where the choice is made.

## Tile selection

The kernels are not `triton.autotune`d. Each op has an analytic cost model in
`lycoris/kernels/plans/` that scores the candidate tile shapes for the current
device — SM count, L2 size, measured DRAM bandwidth and matmul peak, whether
`cp.async` is available (sm_80+, which sets the usable pipeline depths) — and
returns a shortlist of the best five to seven. The shortlist is timed once,
best of three rounds, and the winner is cached in memory and in
`tuning.json`, keyed by device name, op, bucketed shape and dtype. A later
process reuses the table and pays nothing.

`SENTINEL_EAGER` is a member of most shortlists: if the fused kernel loses to
the eager formulation at some shape, the tuner picks eager for that shape and
records it. Being slower than PyTorch at a shape is therefore a measurement,
not a regression you have to hit in production.

## Adding a backend

An op set is a module exposing the 23 op functions (`lycoris/kernels/triton/ops.py`
lists them). Register the module name in `lycoris/kernels/dispatch.py`:
`_PROBE` maps a backend name to the import that must succeed, `ORDER` is the
preference order, and `FUSED` names the tiers that carry an op set. Nothing
else in the library needs to know.
