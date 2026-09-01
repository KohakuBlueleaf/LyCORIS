"""F6 bench: BOFT butterfly chains, weight and activation paths.

Question answered: the traffic cut from index-math permutes (one read one
write per stage vs eager's ~3 materializations), and the backward's
recompute cost against eager autograd's cached stages. Sibling:
``blockdiag.py``. Plot with ``plot_all.py``.

Usage:
    .venv/Scripts/python scripts/bench/kernels/butterfly.py --out out/bench/kernels
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lycoris.functional import boft as f_boft
from lycoris.kernels.autograd.boft import boft_bypass_diff, boft_diff_weight
from lycoris.kernels.dispatch import fused_backends
from lycoris.kernels.plans.device import resolve_device
from scripts.bench.kernels import shapes
from scripts.bench.kernels.family_common import compiled, make_arm, measure_case
from scripts.bench.kernels.harness import device_meta, save_rows

BACKENDS = list(fused_backends())


def _stages(o: int, b: int) -> int:
    nb = 1
    while o % (b * nb * 2) == 0:
        nb *= 2
    return nb.bit_length()


def _arm(fn, blocks, g):
    return make_arm(fn, (blocks,), g)


def _weight_case(dtype, dram, cname, o, i, b):
    m = _stages(o, b)
    blocks = torch.randn(m, o // b, b, b, device="cuda", dtype=dtype) * 0.05
    w = torch.randn(o, i, device="cuda", dtype=dtype) * 0.1
    g = torch.randn_like(w)
    ref = f_boft.diff_weight(w.float(), blocks.float(), None, backend="torch").double()

    # The eager arm and the oracle pin the reference body: the functional API
    # itself now dispatches, and would measure our own kernel as "eager".
    def eager(bl):
        return f_boft.diff_weight(w, bl, None, backend="torch")

    arms = {
        "eager": _arm(eager, blocks, g),
        "compile": _arm(compiled(eager), blocks, g),
    }
    for be in BACKENDS:

        def ours(bl, be=be):
            return boft_diff_weight(w, bl, backend=be)

        arms[be] = _arm(ours, blocks, g)
    return measure_case(
        "boft",
        f"{cname}_b{b}m{m}",
        {"o": o, "i": i, "b": b, "m": m, "dtype": str(dtype)},
        arms,
        ref,
        logical_bytes=2.0 * m * o * i * dtype.itemsize,
        logical_flops=2.0 * m * o * i * b,
        dram_gbps=dram,
    )


def _act_case(dtype, dram, o, b, t):
    m = _stages(o, b)
    blocks = torch.randn(m, o // b, b, b, device="cuda", dtype=dtype) * 0.05
    y = torch.randn(t, o, device="cuda", dtype=dtype) * 0.1
    g = torch.randn_like(y)
    ref = f_boft.bypass_forward_diff(
        y.float(), blocks.float(), None, backend="torch"
    ).double()

    def eager(bl):
        return f_boft.bypass_forward_diff(y, bl, None, backend="torch")

    arms = {
        "eager": _arm(eager, blocks, g),
        "compile": _arm(compiled(eager), blocks, g),
    }
    for be in BACKENDS:

        def ours(bl, be=be):
            return boft_bypass_diff(y, bl, backend=be)

        arms[be] = _arm(ours, blocks, g)
    return measure_case(
        "boft_bypass",
        f"c{o}_t{t}",
        {"o": o, "b": b, "m": m, "t": t, "dtype": str(dtype)},
        arms,
        ref,
        logical_bytes=2.0 * m * t * o * dtype.itemsize,
        logical_flops=2.0 * m * t * o * b,
        dram_gbps=dram,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/bench/kernels")
    args = ap.parse_args()
    dev = resolve_device()
    rows = []
    for dtype in (torch.float16,):
        for cname, o, i in (("sdxl_attn_xl", 1280, 1280), ("llm_qkv", 4096, 4096)):
            for b in (4, 8):
                rows += _weight_case(dtype, dev.dram_bw, cname, o, i, b)
        for t in shapes.TOKENS[:3]:
            rows += _act_case(dtype, dev.dram_bw, 4096, 4, t)
    save_rows(f"{args.out}/boft.json", device_meta(), rows)


if __name__ == "__main__":
    main()
