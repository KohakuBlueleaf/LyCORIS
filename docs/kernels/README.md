# Fused Kernels (experimental)

> **Status: early experimental.** The kernels are new. They are selected
> automatically when their dependencies are installed, they fall back to the
> stock PyTorch path whenever anything is out of scope, and every path is
> checked against the eager implementation. If you hit a discrepancy, set
> `LYCORIS_KERNEL_BACKEND=torch` and open an issue.

LyCORIS ships hand-written [Triton](https://github.com/triton-lang/triton) and
[TileLang](https://github.com/tile-ai/tilelang) kernels for the hot paths of
every algorithm. Nothing in your code changes: the module and functional APIs
keep the same signatures and pick a backend per call.

```python
# unchanged code, fused underneath
from lycoris import create_lycoris
net = create_lycoris(model, 1.0, linear_dim=16, linear_alpha=8.0, algo="lokr")
net.apply_to()
```

## What is fused

Each algorithm has up to four fused kernels — merge forward (ΔW), merge
backward, bypass forward (ΔWx), bypass backward — one kernel per direction,
never a chain of small launches.

| Algorithm | Merge (ΔW) | Bypass (ΔWx) | Out of scope, falls back |
| --- | --- | --- | --- |
| `lora` / `locon` | fused fwd + bwd | fused fwd + bwd, linear | conv bypass |
| `lora` tucker | fused fwd | — | rank > 64; backward is an einsum chain |
| `loha` | fused fwd + bwd | fused fwd + bwd, linear | rank > 128, tucker backward, conv bypass |
| `lokr` | fused fwd + bwd, factored and full | fused fwd + bwd, linear | conv or tucker `w2`, factor > 128 |
| `oft` (diag) | fused fwd + bwd | fused fwd + bwd | block size > 32 |
| `boft` | fused fwd + bwd | fused fwd + bwd | — |
| `dora` weight decompose | fused fwd + bwd | — | `wd_on_out=False` on a conv |
| `ia3` | fused fwd + bwd | fused fwd + bwd | — |
| `glora` | fused fwd + bwd, linear | — | conv, tucker |
| `dylora` | fused fwd + bwd on the active ranks | — | — |
| `full`, `norm` | fused scaled add | — | — |

The weight-decompose (DoRA) epilogue is one kernel shared by every algorithm
that has a decomposed variant — dora, doha and dokr all reach the same code.

Anything not in the table runs the same PyTorch code it always did. Rank
dropout, module dropout and a multiplier that interpolates per stage are all
honoured; where a mask has to sit between two factors, that call takes the
unfused path, since fusing the two factors is exactly what removes the place
to put the mask.

## What it buys

Measured on an RTX 4090, geometric mean over each family's shape sweep, device
(kernel) time against stock eager and against `torch.compile`:

| Family | vs eager | vs compile |
| --- | --- | --- |
| `oft` bypass | 7.3x | 6.2x |
| `oft` merge | 5.7x | 5.1x |
| `lora` merge | 4.9x | 3.1x |
| `lokr` bypass | 4.3x | 2.6x |
| `boft` merge | 3.4x | 2.3x |
| `lokr` merge | 3.0x | 0.9x |
| `boft` bypass | 1.5x | 1.2x |
| `lora` bypass | 1.4x | 1.2x |

Peak memory drops with it (1.06x–1.6x smaller on the fwd+bwd path): ΔW, its
gradient and the reconstructed Kronecker halves are never materialised —
the kernel generates each tile in registers as it goes. See
[benchmarks.md](benchmarks.md) for the full table, the per-case figures and
how to reproduce them.

## Installing the backends

```bash
pip install triton        # Linux, and Windows via triton-windows
pip install tilelang      # optional, second choice after triton
```

Neither is a hard dependency. With neither installed, LyCORIS uses
`torch.compile` on CUDA and plain eager elsewhere, exactly as before.

## Turning it off

```bash
export LYCORIS_KERNEL_BACKEND=torch   # stock PyTorch everywhere
```

See [backends.md](backends.md) for the full selection order and the rest of
the environment variables, and [precision.md](precision.md) for the dtype
rules and the measured accuracy.
