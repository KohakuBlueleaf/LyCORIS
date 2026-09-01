"""F1/F2 bench: LoHa rebuild and weight-free bypass, all arms.

Question answered: how far are the fused low-rank kernels from the measured
bandwidth ceiling, and what do they save over eager/compiled LyCORIS in time
and peak VRAM, forward and fwd+bwd? Sibling: ``kron.py`` (same protocol,
LoKr family). Plot with ``plot_all.py``.

Usage:
    .venv/Scripts/python scripts/bench/kernels/lowrank.py --out out/bench/kernels
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lycoris.functional import loha as f_loha
from lycoris.kernels.autograd.loha import loha_bypass_diff, loha_diff_weight
from lycoris.kernels.dispatch import fused_backends
from lycoris.kernels.plans.device import resolve_device
from scripts.bench.kernels import shapes
from scripts.bench.kernels.family_common import compiled, make_arm, measure_case
from scripts.bench.kernels.harness import device_meta, save_rows
from test.kernels import refs

BACKENDS = list(fused_backends())


def _mk(dtype, *shape):
    return torch.randn(*shape, device="cuda", dtype=dtype) * 0.1


_arm = make_arm


def _rebuild_case(dtype, dram, cname, o, i, r):
    w1d, w1u = _mk(dtype, r, i), _mk(dtype, o, r)
    w2d, w2u = _mk(dtype, r, i), _mk(dtype, o, r)
    g = _mk(dtype, o, i)
    ref = refs.lowrank_rebuild(w1u, w1d, w2u, w2d, gamma=0.5, mode="hada")
    gt = torch.tensor(0.5, device="cuda", dtype=dtype)

    # The eager arm pins the reference body: the functional API itself now
    # dispatches, and would measure our own kernel as "eager".
    def eager(a, b, c, d):
        return f_loha.diff_weight(a, b, c, d, None, None, gamma=gt, backend="torch")

    tensors = (w1d, w1u, w2d, w2u)
    arms = {
        "eager": _arm(eager, tensors, g),
        "compile": _arm(compiled(eager), tensors, g),
    }
    for be in BACKENDS:

        def ours(a, b, c, d, be=be):
            return loha_diff_weight(a, b, c, d, gamma=0.5, backend=be)

        arms[be] = _arm(ours, tensors, g)
    return measure_case(
        "lora",
        f"{cname}_r{r}",
        {"o": o, "i": i, "r": r, "dtype": str(dtype)},
        arms,
        ref,
        logical_bytes=o * i * dtype.itemsize + 2 * (o + i) * r * dtype.itemsize,
        logical_flops=4.0 * o * i * r,
        dram_gbps=dram,
    )


def _bypass_case(dtype, dram, o, i, r, t):
    x = _mk(dtype, t, i)
    w1d, w1u = _mk(dtype, r, i), _mk(dtype, o, r)
    w2d, w2u = _mk(dtype, r, i), _mk(dtype, o, r)
    g = _mk(dtype, t, o)
    ref = refs.hada_delta(x, w1u, w1d, w2u, w2d, 0.5)
    gt = torch.tensor(0.5, device="cuda", dtype=dtype)

    def eager(xx, a, b, c, d):
        dw = f_loha.diff_weight(a, b, c, d, None, None, gamma=gt, backend="torch")
        return xx @ dw.T

    tensors = (x, w1d, w1u, w2d, w2u)
    arms = {
        "eager": _arm(eager, tensors, g),
        "compile": _arm(compiled(eager), tensors, g),
    }
    for be in BACKENDS:

        def ours(xx, a, b, c, d, be=be):
            return loha_bypass_diff(xx, a, b, c, d, gamma=0.5, backend=be)

        arms[be] = _arm(ours, tensors, g)
    return measure_case(
        "lora_bypass",
        f"dit_qkv_t{t}",
        {"o": o, "i": i, "r": r, "t": t, "dtype": str(dtype)},
        arms,
        ref,
        logical_bytes=(t * i + t * o) * dtype.itemsize,
        logical_flops=2.0 * t * o * i,
        dram_gbps=dram,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/bench/kernels")
    args = ap.parse_args()
    dev = resolve_device()
    rows = []
    for dtype in (torch.float16,):
        for cname, (o, i) in shapes.LINEAR.items():
            for r in shapes.RANKS:
                rows += _rebuild_case(dtype, dev.dram_bw, cname, o, i, r)
        o, i = shapes.LINEAR["dit_qkv"]
        for t in shapes.TOKENS:
            rows += _bypass_case(dtype, dev.dram_bw, o, i, 16, t)
    save_rows(f"{args.out}/lora.json", device_meta(), rows)


if __name__ == "__main__":
    main()
