"""F7 (TileLang): fused scaled add."""

import tilelang
import tilelang.language as T

from ...plans import dora as plan
from ...plans import tune
from ...plans.device import resolve_device


@tilelang.jit(out_idx=[2])
def _add_scaled(N, dtype, blk=1024, threads=128):
    @T.prim_func
    def main(
        base: T.Tensor((N,), dtype),
        delta: T.Tensor((N,), dtype),
        out: T.Tensor((N,), dtype),
        gamma: T.float32,
    ):
        with T.Kernel(T.ceildiv(N, blk), threads=threads) as (pid,):
            for i in T.Parallel(blk):
                idx = pid * blk + i
                if idx < N:
                    out[idx] = T.cast(
                        T.cast(base[idx], "float32")
                        + gamma * T.cast(delta[idx], "float32"),
                        dtype,
                    )

    return main


def add_scaled(base, delta, gamma=1.0):
    bc = base.contiguous().reshape(-1)
    dc = delta.contiguous().reshape(-1)
    n = bc.numel()

    def build(p):
        fn = _add_scaled(
            n, str(base.dtype).split(".")[-1], blk=p.bm, threads=32 * p.warps
        )
        return lambda: fn(bc, dc, float(gamma))

    best = tune.tuned(
        "tilelang.merge.add_scaled",
        (tune.bucket_tokens(n), str(base.dtype)),
        lambda: plan.topk_elementwise(n, 3.0, resolve_device(), base.element_size()),
        build,
    )
    return build(best)().view_as(base)
