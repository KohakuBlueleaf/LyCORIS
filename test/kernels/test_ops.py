"""Backend op correctness against fp64 references, forward and backward.

Every op x backend x dtype: outputs and every gradient compared to an fp64
oracle from ``refs`` that shares no kernel assumptions. Gradients of the
oracle come from torch autograd on the fp64 graph.

Usage:
    .venv/Scripts/python -m unittest test.kernels.test_ops -v
"""

import unittest
from itertools import product

import torch
from parameterized import parameterized

from lycoris.kernels.dispatch import available_backends
from lycoris.kernels.ops import get_ops
from . import refs

EPS = {torch.float16: 2e-2, torch.bfloat16: 5e-2, torch.float32: 5e-3}
BACKENDS = [b for b in available_backends() if b != "torch"]
DTYPES = [torch.float16, torch.float32]
CASES = list(product(BACKENDS, DTYPES))
CUDA = torch.cuda.is_available()


def _rel(got, ref):
    err = (got.double() - ref.double()).abs().max().item()
    return err / (ref.double().abs().max().item() + 1e-8)


class OpsVsFp64(unittest.TestCase):
    def setUp(self):
        if not CUDA:
            self.skipTest("needs CUDA")
        torch.manual_seed(0)

    def check(self, got, ref, dtype, what):
        self.assertLess(_rel(got, ref), EPS[dtype], f"{what}: exceeds eps")

    @parameterized.expand(CASES)
    def test_lowrank_rebuild_all_modes(self, backend, dtype):
        ops = get_ops(backend)
        o, i, r = 192, 176, 12
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.1)
        a1, b1, a2, b2 = mk(o, r), mk(r, i), mk(o, r), mk(r, i)
        base = mk(o, i)
        for mode in ("plain", "hada", "sum"):
            got = ops.lora_merge_fwd(a1, b1, a2, b2, base=base, gamma=1.7, mode=mode)
            ref = refs.lowrank_rebuild(a1, b1, a2, b2, base=base, gamma=1.7, mode=mode)
            self.check(got, ref, dtype, f"rebuild {mode}")

    @parameterized.expand(CASES)
    def test_lowrank_hada_bwd(self, backend, dtype):
        ops = get_ops(backend)
        o, i, r = 160, 144, 8
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.1)
        a1, b1, a2, b2 = mk(o, r), mk(r, i), mk(o, r), mk(r, i)
        g = mk(o, i)
        leaves = [t.double().requires_grad_(True) for t in (a1, b1, a2, b2)]
        ((leaves[0] @ leaves[1]) * (leaves[2] @ leaves[3]) * 1.3).backward(g.double())
        got = ops.loha_merge_bwd(g, a1, b1, a2, b2, gamma=1.3)
        for gv, leaf, name in zip(got, leaves, ("ga1", "gb1", "ga2", "gb2")):
            self.check(gv, leaf.grad, dtype, name)

    @parameterized.expand(CASES)
    def test_lowrank_merge_bwd(self, backend, dtype):
        ops = get_ops(backend)
        o, i, r = 160, 144, 8
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.1)
        up, down, g = mk(o, r), mk(r, i), mk(o, i)
        lu = up.double().requires_grad_(True)
        ld = down.double().requires_grad_(True)
        (1.3 * lu @ ld).backward(g.double())
        gu, gd = ops.lora_merge_bwd(g, up, down, gamma=1.3)
        self.check(gu, lu.grad, dtype, "merge g_up")
        self.check(gd, ld.grad, dtype, "merge g_down")

    @parameterized.expand(CASES)
    def test_lowrank_bypass_fwd_bwd(self, backend, dtype):
        ops = get_ops(backend)
        t, o, i, r = 130, 96, 112, 16
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.1)
        x, up, down, gy = mk(t, i), mk(o, r), mk(r, i), mk(t, o)
        lx = x.double().requires_grad_(True)
        lu = up.double().requires_grad_(True)
        ld = down.double().requires_grad_(True)
        y = 0.5 * (lx @ ld.T) @ lu.T
        self.check(ops.lora_bypass_fwd(x, up, down, gamma=0.5), y, dtype, "bypass fwd")
        y.backward(gy.double())
        gx, gu, gd = ops.lora_bypass_bwd(x, up, down, gy, gamma=0.5)
        self.check(gx, lx.grad, dtype, "bypass gx")
        self.check(gu, lu.grad, dtype, "bypass g_up")
        self.check(gd, ld.grad, dtype, "bypass g_down")

    @parameterized.expand(CASES)
    def test_kron_generated_fwd_bwd(self, backend, dtype):
        ops = get_ops(backend)
        a, b, c, d, r1, r2 = 20, 24, 16, 12, 6, 8
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.1)
        w1a, w1b, w2a, w2b = mk(a, r1), mk(r1, b), mk(c, r2), mk(r2, d)
        g = mk(a * c, b * d)
        leaves = [v.double().requires_grad_(True) for v in (w1a, w1b, w2a, w2b)]
        out = torch.kron(leaves[0] @ leaves[1], leaves[2] @ leaves[3]) * 3.0
        self.check(
            ops.lokr_merge_fwd(w1a, w1b, w2a, w2b, (a, b, c, d), gamma=3.0),
            out,
            dtype,
            "kron gen fwd",
        )
        out.backward(g.double())
        got = ops.lokr_merge_bwd(g, w1a, w1b, w2a, w2b, (a, b, c, d), gamma=3.0)
        for gv, leaf, name in zip(got, leaves, ("g1a", "g1b", "g2a", "g2b")):
            self.check(gv, leaf.grad, dtype, f"kron gen {name}")

    @parameterized.expand(CASES)
    def test_lowrank_tucker(self, backend, dtype):
        ops = get_ops(backend)
        o, i, r, k = 96, 80, 8, 9
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.1)
        a, t, b = mk(o, r), mk(r, r, k), mk(r, i)
        got = ops.lora_tucker_fwd(a, t, b, gamma=0.7)
        self.check(got, refs.tucker_rebuild(a, t, b, 0.7), dtype, "tucker")

    @parameterized.expand(CASES)
    def test_loha_bypass_fwd_bwd(self, backend, dtype):
        ops = get_ops(backend)
        t, o, i, r = 130, 96, 112, 16
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.1)
        x, a1, b1, a2, b2 = mk(t, i), mk(o, r), mk(r, i), mk(o, r), mk(r, i)
        gy = mk(t, o)
        self.check(
            ops.loha_bypass_fwd(x, a1, b1, a2, b2, gamma=0.5),
            refs.hada_delta(x, a1, b1, a2, b2, 0.5),
            dtype,
            "delta fwd",
        )
        leaves = [v.double().requires_grad_(True) for v in (x, a1, b1, a2, b2)]
        y = 0.5 * leaves[0] @ ((leaves[1] @ leaves[2]) * (leaves[3] @ leaves[4])).T
        y.backward(gy.double())
        got = ops.loha_bypass_bwd(gy, x, a1, b1, a2, b2, gamma=0.5)
        names = ("dx", "ga1", "gb1", "ga2", "gb2")
        for gv, leaf, name in zip(got, leaves, names):
            self.check(gv, leaf.grad, dtype, f"delta {name}")

    @parameterized.expand(CASES)
    def test_kron_lora_merge_fwd_bwd(self, backend, dtype):
        ops = get_ops(backend)
        a, b, c, d = 20, 24, 16, 12
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.1)
        w1, w2, base = mk(a, b), mk(c, d), mk(a * c, b * d)
        self.check(
            ops.lokr_full_merge_fwd(w1, w2, base=base, gamma=3.0),
            refs.kron_rebuild(w1, w2, base, 3.0),
            dtype,
            "kron fwd",
        )
        g = mk(a * c, b * d)
        l1 = w1.double().requires_grad_(True)
        l2 = w2.double().requires_grad_(True)
        (torch.kron(l1, l2) * 3.0).backward(g.double())
        gw1, gw2 = ops.lokr_full_merge_bwd(g, w1, w2, gamma=3.0)
        self.check(gw1, l1.grad, dtype, "kron gw1")
        self.check(gw2, l2.grad, dtype, "kron gw2")

    @parameterized.expand(CASES)
    def test_lokr_bypass_fwd_bwd(self, backend, dtype):
        ops = get_ops(backend)
        t, a, b, c, d = 70, 20, 24, 16, 12
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.1)
        x, w1, w2, gy = mk(t, b * d), mk(a, b), mk(c, d), mk(t, a * c)
        self.check(
            ops.lokr_bypass_fwd(x, w1, w2, gamma=0.5),
            refs.kron_apply(x, w1, w2, 0.5),
            dtype,
            "kron apply fwd",
        )
        lx = x.double().requires_grad_(True)
        l1 = w1.double().requires_grad_(True)
        l2 = w2.double().requires_grad_(True)
        (0.5 * lx @ torch.kron(l1, l2).T).backward(gy.double())
        gx, gw1, gw2 = ops.lokr_bypass_bwd(gy, x, w1, w2, gamma=0.5)
        self.check(gx, lx.grad, dtype, "kron apply gx")
        self.check(gw1, l1.grad, dtype, "kron apply gw1")
        self.check(gw2, l2.grad, dtype, "kron apply gw2")

    @parameterized.expand(CASES)
    def test_blockdiag_fused_fwd(self, backend, dtype):
        ops = get_ops(backend)
        k, s, cols = 24, 8, 88
        mk = lambda *s_: (torch.randn(*s_, device="cuda", dtype=dtype) * 0.1)
        blocks = torch.randn(k, s, s, device="cuda", dtype=dtype) * 0.2
        w = mk(k * s, cols)
        res = torch.rand(k * s, device="cuda", dtype=dtype) + 0.5
        for rescale, shift in ((None, True), (res, True), (None, False)):
            got = ops.oft_fwd(blocks, w, rescale, 1.0, shift, True)
            ref = refs.bd_fused(blocks, w, rescale, 1.0, shift, True)
            self.check(got, ref, dtype, f"bd w rescale={rescale is not None}")
        xa = mk(42, k * s)
        self.check(
            ops.oft_fwd(blocks, xa, None, 1.0, True, False),
            refs.bd_fused(blocks, xa, None, 1.0, True, False),
            dtype,
            "bd act2d",
        )
        x3 = mk(3, k * s, 21)
        self.check(
            ops.oft_fwd(blocks, x3, None, 1.0, True, False),
            refs.bd_fused(blocks, x3, None, 1.0, True, False),
            dtype,
            "bd act3d",
        )

    @parameterized.expand(CASES)
    def test_blockdiag_fused_bwd(self, backend, dtype):
        ops = get_ops(backend)
        k, s, cols = 16, 8, 64
        mk = lambda *s_: (torch.randn(*s_, device="cuda", dtype=dtype) * 0.1)
        blocks = torch.randn(k, s, s, device="cuda", dtype=dtype) * 0.2
        w, g = mk(k * s, cols), mk(k * s, cols)
        res = torch.rand(k * s, device="cuda", dtype=dtype) + 0.5
        lb = blocks.double().requires_grad_(True)
        lw = w.double().requires_grad_(True)
        lr = res.double().requires_grad_(True)
        refs.bd_fused(lb, lw, lr, 1.0, True, True).backward(g.double())
        gx, gb, gres = ops.oft_bwd(blocks, w, g, res, 1.0, True, True)
        self.check(gx, lw.grad, dtype, "bd gx")
        self.check(gb, lb.grad, dtype, "bd gblocks")
        self.check(gres, lr.grad, dtype, "bd grescale")

    @parameterized.expand(CASES)
    def test_boft_cone(self, backend, dtype):
        """s=8, m=3 puts the whole butterfly in one cone (b*2^(m-1) = 32)."""
        ops = get_ops(backend)
        n, cols, s, m = 128, 96, 8, 3
        nb = n // s
        blocks = torch.randn(m, nb, s, s, device="cuda", dtype=dtype) * 0.3
        w = torch.randn(n, cols, device="cuda", dtype=dtype) * 0.1
        self.check(
            ops.boft_fwd(blocks, w, axis=0),
            refs.butterfly_blocks(blocks, w, axis=0),
            dtype,
            "boft cone rows",
        )
        xa = torch.randn(5, 7, n, device="cuda", dtype=dtype) * 0.1
        self.check(
            ops.boft_fwd(blocks, xa, axis=-1),
            refs.butterfly_blocks(blocks, xa, axis=-1),
            dtype,
            "boft cone cols",
        )

    @parameterized.expand(CASES)
    def test_butterfly_fused_fwd_bwd(self, backend, dtype):
        ops = get_ops(backend)
        n, cols, s, m = 64, 72, 4, 5
        nb = n // s
        blocks = torch.randn(m, nb, s, s, device="cuda", dtype=dtype) * 0.3
        w = torch.randn(n, cols, device="cuda", dtype=dtype) * 0.1
        self.check(
            ops.boft_fwd(blocks, w, axis=0),
            refs.butterfly_blocks(blocks, w, axis=0),
            dtype,
            "bf rows",
        )
        xa = torch.randn(5, 9, n, device="cuda", dtype=dtype) * 0.1
        self.check(
            ops.boft_fwd(blocks, xa, axis=-1),
            refs.butterfly_blocks(blocks, xa, axis=-1),
            dtype,
            "bf cols",
        )
        g = torch.randn_like(w)
        lb = blocks.double().requires_grad_(True)
        lw = w.double().requires_grad_(True)
        refs.butterfly_blocks(lb, lw, axis=0).backward(g.double())
        gx, gb = ops.boft_bwd(blocks, w, g, axis=0)
        self.check(gx, lw.grad, dtype, "bf gx")
        self.check(gb, lb.grad, dtype, "bf gblocks")

    @parameterized.expand(CASES)
    def test_pointwise_all(self, backend, dtype):
        ops = get_ops(backend)
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.3)
        x, wch = mk(4, 24, 5, 5), mk(24)
        self.check(
            ops.ia3_fwd(x, wch, 1, 1.0, 0.7),
            refs.channel_scale(x, wch, 1, 1.0, 0.7),
            dtype,
            "scale fwd",
        )
        g = mk(4, 24, 5, 5)
        lx = x.double().requires_grad_(True)
        lw = wch.double().requires_grad_(True)
        refs.channel_scale(lx, lw, 1, 1.0, 0.7).backward(g.double())
        gx, gw = ops.ia3_bwd(g, x, wch, 1, 1.0, 0.7)
        self.check(gx, lx.grad, dtype, "scale gx")
        self.check(gw, lw.grad, dtype, "scale gw")

        wt, dsc = mk(48, 36), torch.rand(48, device="cuda", dtype=dtype) + 0.5
        eps = torch.finfo(dtype).eps
        y, norms = ops.dora_fwd(wt, dsc, 0.8, 0)
        self.check(norms, wt.double().norm(dim=1) + eps, dtype, "dora norm")
        self.check(y, refs.dora_scale(wt, dsc, 0.8, 0, eps), dtype, "dora fwd")
        gd = mk(48, 36)
        lw = wt.double().requires_grad_(True)
        ld = dsc.double().requires_grad_(True)
        nrm = lw.norm(dim=1, keepdim=True) + eps
        (lw * (0.8 * (ld[:, None] / nrm - 1.0) + 1.0)).backward(gd.double())
        gw, gdsc = ops.dora_bwd(gd, wt, dsc, norms, 0.8, 0)
        self.check(gw, lw.grad, dtype, "dora gw")
        self.check(gdsc, ld.grad, dtype, "dora gd")

        b_, d_ = mk(33, 45), mk(33, 45)
        self.check(
            ops.add_scaled(b_, d_, 0.3),
            b_.double() + 0.3 * d_.double(),
            dtype,
            "add_scaled",
        )


if __name__ == "__main__":
    unittest.main()
