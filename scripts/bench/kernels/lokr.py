"""F3/F4 bench: LoKr kron rebuild and grouped bypass, all arms.

Question answered: what the gather-FMA rebuild and the SMEM-chain bypass
reach against measured bandwidth, and the transpose tax the eager bypass
pays (visible as its logical-GB/s gap). Sibling: ``lowrank.py``.
Plot with ``plot_all.py``.

Usage:
    .venv/Scripts/python scripts/bench/kernels/kron.py --out out/bench/kernels
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lycoris.functional import lokr as f_lokr
from lycoris.functional.general import factorization
from lycoris.kernels.autograd.lokr import lokr_bypass_diff, lokr_diff_weight
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


def _rebuild_case(dtype, dram, cname, o, i, factor):
    a, c = factorization(o, factor)
    b, d = factorization(i, factor)
    w1, w2 = _mk(dtype, a, b), _mk(dtype, c, d)
    g = _mk(dtype, o, i)
    ref = refs.kron_rebuild(w1, w2, gamma=1.0)

    # The eager arm pins the reference body: the functional API itself now
    # dispatches, and would measure our own kernel as "eager".
    def eager(x1, x2):
        return f_lokr.diff_weight(
            x1, None, None, x2, None, None, None, gamma=1.0, backend="torch"
        )

    tensors = (w1, w2)
    arms = {
        "eager": _arm(eager, tensors, g),
        "compile": _arm(compiled(eager), tensors, g),
    }
    for be in BACKENDS:

        def ours(x1, x2, be=be):
            return lokr_diff_weight(
                x1, None, None, x2, None, None, gamma=1.0, backend=be
            )

        arms[be] = _arm(ours, tensors, g)
    return measure_case(
        "lokr",
        f"{cname}_f{factor}",
        {"o": o, "i": i, "factor": factor, "dtype": str(dtype)},
        arms,
        ref,
        logical_bytes=o * i * dtype.itemsize,
        logical_flops=float(o * i),
        dram_gbps=dram,
    )


def _bypass_case(dtype, dram, o, i, t):
    a, c = factorization(o, -1)
    b, d = factorization(i, -1)
    x = _mk(dtype, t, i)
    w1, w2 = _mk(dtype, a, b), _mk(dtype, c, d)
    g = _mk(dtype, t, o)
    ref = refs.kron_apply(x, w1, w2, 1.0)

    def eager(xx, x1, x2):
        uq = x1.shape[1]
        h = xx.reshape(*xx.shape[:-1], uq, -1)
        hb = torch.nn.functional.linear(h, x2)
        hc = torch.nn.functional.linear(hb.transpose(-1, -2), x1)
        return hc.transpose(-1, -2).reshape(*xx.shape[:-1], -1)

    tensors = (x, w1, w2)
    arms = {
        "eager": _arm(eager, tensors, g),
        "compile": _arm(compiled(eager), tensors, g),
    }
    for be in BACKENDS:

        def ours(xx, x1, x2, be=be):
            return lokr_bypass_diff(
                xx, x1, None, None, x2, None, None, gamma=1.0, backend=be
            )

        arms[be] = _arm(ours, tensors, g)
    return measure_case(
        "lokr_bypass",
        f"llm_qkv_t{t}",
        {"o": o, "i": i, "t": t, "dtype": str(dtype)},
        arms,
        ref,
        logical_bytes=(t * i + t * o) * dtype.itemsize,
        logical_flops=2.0 * t * (b * d * c + c * b * a),
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
            for factor in shapes.LOKR_FACTORS:
                rows += _rebuild_case(dtype, dev.dram_bw, cname, o, i, factor)
        o, i = shapes.LINEAR["llm_qkv"]
        for t in shapes.TOKENS:
            rows += _bypass_case(dtype, dev.dram_bw, o, i, t)
    save_rows(f"{args.out}/lokr.json", device_meta(), rows)


if __name__ == "__main__":
    main()
