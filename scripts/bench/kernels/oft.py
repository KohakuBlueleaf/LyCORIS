"""F5 bench: Diag-OFT block-diagonal apply, weight and activation paths.

Question answered: what the folded single-pass apply reaches against
bandwidth, and what the eager path pays in einsum temporaries and conv
transposes. Sibling: ``butterfly.py``. Plot with ``plot_all.py``.

Usage:
    .venv/Scripts/python scripts/bench/kernels/blockdiag.py --out out/bench/kernels
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lycoris.functional import diag_oft as f_oft
from lycoris.functional.general import factorization
from lycoris.kernels.autograd.diag_oft import (
    diag_oft_bypass_diff,
    diag_oft_diff_weight,
)
from lycoris.kernels.dispatch import fused_backends
from lycoris.kernels.plans.device import resolve_device
from scripts.bench.kernels import shapes
from scripts.bench.kernels.family_common import compiled, make_arm, measure_case
from scripts.bench.kernels.harness import device_meta, save_rows

BACKENDS = list(fused_backends())


def _arm(fn, blocks, g):
    return make_arm(fn, (blocks,), g)


def _weight_case(dtype, dram, cname, o, i, lora_dim):
    s, k = factorization(o, lora_dim)
    w = torch.randn(o, i, device="cuda", dtype=dtype) * 0.1
    blocks = torch.randn(k, s, s, device="cuda", dtype=dtype) * 0.05
    g = torch.randn_like(w)
    ref = f_oft.diff_weight(w.float(), blocks.float(), None, backend="torch").double()

    # The eager arm and the oracle pin the reference body: the functional API
    # itself now dispatches, and would measure our own kernel as "eager".
    def eager(bl):
        return f_oft.diff_weight(w, bl, None, backend="torch")

    arms = {
        "eager": _arm(eager, blocks, g),
        "compile": _arm(compiled(eager), blocks, g),
    }
    for be in BACKENDS:

        def ours(bl, be=be):
            return diag_oft_diff_weight(w, bl, backend=be)

        arms[be] = _arm(ours, blocks, g)
    return measure_case(
        "oft",
        f"{cname}_d{lora_dim}",
        {"o": o, "i": i, "s": s, "k": k, "dtype": str(dtype)},
        arms,
        ref,
        logical_bytes=2.0 * o * i * dtype.itemsize,
        logical_flops=2.0 * o * i * s,
        dram_gbps=dram,
    )


def _act_case(dtype, dram, o, lora_dim, t):
    s, k = factorization(o, lora_dim)
    blocks = torch.randn(k, s, s, device="cuda", dtype=dtype) * 0.05
    y = torch.randn(t, o, device="cuda", dtype=dtype) * 0.1
    g = torch.randn_like(y)
    ref = f_oft.bypass_forward_diff(
        None, y.float(), blocks.float(), None, backend="torch"
    ).double()

    def eager(bl):
        return f_oft.bypass_forward_diff(None, y, bl, None, backend="torch")

    arms = {
        "eager": _arm(eager, blocks, g),
        "compile": _arm(compiled(eager), blocks, g),
    }
    for be in BACKENDS:

        def ours(bl, be=be):
            return diag_oft_bypass_diff(y, bl, backend=be)

        arms[be] = _arm(ours, blocks, g)
    return measure_case(
        "oft_bypass",
        f"c{o}_t{t}",
        {"o": o, "s": s, "k": k, "t": t, "dtype": str(dtype)},
        arms,
        ref,
        logical_bytes=2.0 * t * o * dtype.itemsize,
        logical_flops=2.0 * t * o * s,
        dram_gbps=dram,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/bench/kernels")
    args = ap.parse_args()
    dev = resolve_device()
    rows = []
    for dtype in (torch.float16,):
        for cname, (o, i) in list(shapes.LINEAR.items())[:4]:
            for lora_dim in shapes.OFT_DIMS:
                rows += _weight_case(dtype, dev.dram_bw, cname, o, i, lora_dim)
        for t in shapes.TOKENS[:3]:
            rows += _act_case(dtype, dev.dram_bw, 1280, 16, t)
    save_rows(f"{args.out}/oft.json", device_meta(), rows)


if __name__ == "__main__":
    main()
