"""Measured ceilings every other bench divides by.

Bandwidth: copy / read / triad over a size sweep past L2, best-of per size,
peak reported per pattern. Matmul: cuBLAS fp16/bf16 rates at two shapes —
reported as the library-achieved rate for MFU context, NOT written into
Device.mma_peak (that stays the stated mma-issue figure until a raw mma.sync
microbench exists; a library rate is not a ceiling). dram_bw IS written back
into the device JSON the planners load.

Usage:
    .venv/Scripts/python scripts/bench/kernels/denominators.py --out out/bench/kernels
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lycoris.kernels.plans.device import _json_path, resolve_device
from scripts.bench.kernels.harness import bench_ms, device_meta, gpu_busy, save_rows

SIZES_MB = (64, 256, 1024)


def bandwidth_rows() -> tuple[list[dict], float]:
    rows = []
    peak_triad = 0.0
    for mb in SIZES_MB:
        n = mb * 2**20 // 4
        a = torch.empty(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        c = torch.randn(n, device="cuda", dtype=torch.float32)
        ms_copy = bench_ms(lambda a=a, b=b: a.copy_(b), warmup=5, iters=20)
        ms_read = bench_ms(lambda b=b: b.sum(), warmup=5, iters=20)
        ms_triad = bench_ms(
            lambda a=a, b=b, c=c: torch.add(b, c, alpha=1.5, out=a),
            warmup=5,
            iters=20,
        )

        def gbs(byt, ms):
            return byt / ms * 1e3 / 1e9

        rows += [
            {"pattern": "copy", "mb": mb, "gbps": gbs(8 * n, ms_copy)},
            {"pattern": "read", "mb": mb, "gbps": gbs(4 * n, ms_read)},
            {"pattern": "triad", "mb": mb, "gbps": gbs(12 * n, ms_triad)},
        ]
        peak_triad = max(peak_triad, gbs(12 * n, ms_triad))
        del a, b, c
    torch.cuda.empty_cache()
    return rows, peak_triad


def gemm_rows() -> list[dict]:
    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        for m, n, k in ((4096, 4096, 4096), (8192, 4096, 4096)):
            a = torch.randn(m, k, device="cuda", dtype=dtype)
            b = torch.randn(k, n, device="cuda", dtype=dtype)
            ms = bench_ms(lambda a=a, b=b: a @ b, warmup=10, iters=30)
            rows.append(
                {
                    "dtype": str(dtype).split(".")[-1],
                    "shape": [m, n, k],
                    "tflops": 2.0 * m * n * k / (ms * 1e-3) / 1e12,
                }
            )
            del a, b
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/bench/kernels")
    ap.add_argument("--allow-busy", action="store_true")
    args = ap.parse_args()

    # A desktop-session card idles at ~10-15% util from compositing (DWM,
    # browsers); only sustained compute load above that means contention.
    util, mem = gpu_busy()
    if util > 30 and not args.allow_busy:
        raise SystemExit(f"GPU busy (util={util}%, mem={mem}MiB); rerun when idle")
    print(f"gpu baseline: util={util}%, mem={mem}MiB (desktop session)")

    bw_rows, peak_triad = bandwidth_rows()
    gm_rows = gemm_rows()
    for r in bw_rows:
        print(f"{r['pattern']:>5} {r['mb']:>5} MB  {r['gbps']:8.1f} GB/s")
    for r in gm_rows:
        print(f"cublas {r['dtype']:>8} {r['shape']}  {r['tflops']:7.1f} TF/s")

    dev = resolve_device()
    dev = dataclasses.replace(dev, dram_bw=round(peak_triad, 1))
    path = _json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    dev.to_json(str(path))
    print(f"device json updated: {path} (dram_bw={dev.dram_bw} GB/s)")

    save_rows(
        f"{args.out}/denominators.json",
        device_meta(),
        [{"kind": "bw", **r} for r in bw_rows]
        + [{"kind": "gemm", **r} for r in gm_rows],
    )


if __name__ == "__main__":
    main()
