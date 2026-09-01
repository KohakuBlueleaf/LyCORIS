"""F7 bench: channel scale, DoRA epilogue, fused merge add.

Question answered: fraction of measured bandwidth for the pointwise family
and the DoRA epilogue's cost against the eager apply_weight_decompose chain.
Plot with ``plot_all.py``.

Usage:
    .venv/Scripts/python scripts/bench/kernels/pointwise.py --out out/bench/kernels
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lycoris.kernels.autograd.dora import apply_dora
from lycoris.kernels.autograd.ia3 import ChannelScaleFn
from lycoris.kernels.dispatch import fused_backends
from lycoris.kernels.ops import get_ops
from lycoris.kernels.plans.device import resolve_device
from scripts.bench.kernels.family_common import compiled, make_arm, measure_case
from scripts.bench.kernels.harness import device_meta, save_rows
from test.kernels import refs

BACKENDS = list(fused_backends())


def _arm2(fn, t1, t2, g):
    return make_arm(fn, (t1, t2), g)


def _dora_case(dtype, dram, cname, o, i):
    w = torch.randn(o, i, device="cuda", dtype=dtype) * 0.1
    dsc = torch.rand(o, 1, device="cuda", dtype=dtype) + 0.5
    g = torch.randn_like(w)
    eps = torch.finfo(dtype).eps
    ref = refs.dora_scale(w, dsc, 0.8, 0, eps)

    def eager(ww, dd):
        n = ww.reshape(ww.shape[0], -1).norm(dim=1, keepdim=True) + eps
        return ww * (0.8 * (dd / n - 1) + 1)

    arms = {
        "eager": _arm2(eager, w, dsc, g),
        "compile": _arm2(compiled(eager), w, dsc, g),
    }
    for be in BACKENDS:

        def ours(ww, dd, be=be):
            return apply_dora(ww, dd, 0.8, True, backend=be)

        arms[be] = _arm2(ours, w, dsc, g)
    return measure_case(
        "dora",
        cname,
        {"o": o, "i": i, "dtype": str(dtype)},
        arms,
        ref,
        logical_bytes=3.0 * o * i * dtype.itemsize,
        logical_flops=4.0 * o * i,
        dram_gbps=dram,
    )


def _scale_case(dtype, dram):
    x = torch.randn(8, 320, 64, 64, device="cuda", dtype=dtype)
    wch = torch.randn(320, device="cuda", dtype=dtype) * 0.3
    g = torch.randn_like(x)
    ref = refs.channel_scale(x, wch, 1, 1.0, 0.9)

    def eager(xx, ww):
        return xx * (1.0 + 0.9 * ww.view(1, -1, 1, 1))

    arms = {
        "eager": _arm2(eager, x, wch, g),
        "compile": _arm2(compiled(eager), x, wch, g),
    }
    for be in BACKENDS:

        def ours(xx, ww, be=be):
            return ChannelScaleFn.apply(xx, ww, 1, 1.0, 0.9, be)

        arms[be] = _arm2(ours, x, wch, g)
    return measure_case(
        "ia3",
        "conv320_8x64",
        {"n": x.numel(), "dtype": str(dtype)},
        arms,
        ref,
        logical_bytes=2.0 * x.numel() * dtype.itemsize,
        logical_flops=2.0 * x.numel(),
        dram_gbps=dram,
    )


def _merge_case(dtype, dram):
    base = torch.randn(4096, 4096, device="cuda", dtype=dtype)
    delta = torch.randn_like(base)
    ref = base.double() + 0.7 * delta.double()

    def eager():
        return base + 0.7 * delta

    arms = {
        "eager": {"fwd": eager, "fwdbwd": None, "out": eager},
        "compile": {"fwd": compiled(eager), "fwdbwd": None, "out": eager},
    }
    for be in BACKENDS:
        ops = get_ops(be)

        def ours(ops=ops):
            return ops.add_scaled(base, delta, 0.7)

        arms[be] = {"fwd": ours, "fwdbwd": None, "out": ours}
    return measure_case(
        "merge",
        "4096sq",
        {"n": base.numel(), "dtype": str(dtype)},
        arms,
        ref,
        logical_bytes=3.0 * base.numel() * dtype.itemsize,
        logical_flops=2.0 * base.numel(),
        dram_gbps=dram,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/bench/kernels")
    args = ap.parse_args()
    dev = resolve_device()
    rows = []
    for dtype in (torch.float16,):
        for cname, o, i in (("sdxl_ff_in", 5120, 1280), ("llm_mlp", 11008, 4096)):
            rows += _dora_case(dtype, dev.dram_bw, cname, o, i)
        rows += _scale_case(dtype, dev.dram_bw)
        rows += _merge_case(dtype, dev.dram_bw)
    save_rows(f"{args.out}/dora.json", device_meta(), rows)


if __name__ == "__main__":
    main()
