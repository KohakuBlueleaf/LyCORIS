"""Triton kernels the cost model measures the device with."""

import triton
import triton.language as tl


@triton.jit
def chain_read_kernel(src, out, per_prog, block: tl.constexpr):
    """Each program reads its chunk one block at a time, every address depending on the last load."""
    pid = tl.program_id(0)
    lanes = tl.arange(0, block)
    acc = tl.zeros((block,), tl.float32)
    dep = pid * 0
    for off in range(0, per_prog, block):
        x = tl.load(src + pid * per_prog + off + dep + lanes)
        acc += x
        total = tl.sum(x, axis=0)
        dep = (total != total).to(tl.int32)
    tl.store(out + pid * block + lanes, acc)


@triton.jit
def l2_read_kernel(src, out, n, reps, block: tl.constexpr):
    """Each program sums reps blocks of src, cycling through it."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lanes = tl.arange(0, block)
    acc = tl.zeros((block,), tl.float32)
    for it in range(reps):
        acc += tl.load(src + ((pid + it * nprog) * block + lanes) % n)
    tl.store(out + pid * block + lanes, acc)


@triton.jit
def fma_kernel(out, iters, block: tl.constexpr):
    """iters dependent fp32 FMAs per lane: the vector FMA rate."""
    pid = tl.program_id(0)
    lanes = tl.arange(0, block)
    a = (lanes + pid).to(tl.float32) * 1e-7
    acc = a
    for _ in range(iters):
        acc = acc * 0.999 + a
    tl.store(out + pid * block + lanes, acc)
