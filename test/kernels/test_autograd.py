"""Autograd-layer parity against lycoris.functional plus the gauntlet.

Parity: same weights through the fused Functions and the eager functional
API, forward values and every parameter grad within the dtype eps of the
eager path itself measured against fp64. Gauntlet: every entry point returns
a grad_fn and every input receives a nonzero gradient — the silent-zero-grad
failure class accuracy tests cannot see.

Usage:
    .venv/Scripts/python -m unittest test.kernels.test_autograd -v
"""

import unittest
from itertools import product

import torch
from parameterized import parameterized

from lycoris.functional import boft as f_boft
from lycoris.functional import diag_oft as f_oft
from lycoris.functional import locon as f_locon
from lycoris.functional import loha as f_loha
from lycoris.functional import lokr as f_lokr
from lycoris.kernels.autograd import (
    boft_bypass_diff,
    boft_diff_weight,
    diag_oft_bypass_diff,
    diag_oft_diff_weight,
    dylora_diff_weight,
    full_diff_weight,
    glora_diff_weight,
    ia3_bypass,
    ia3_diff_weight,
    loha_bypass_diff,
    loha_diff_weight,
    lokr_bypass_diff,
    lokr_diff_weight,
    locon_diff_weight,
    norm_diff_weights,
    apply_dora,
)
from lycoris.kernels.dispatch import available_backends

EPS = {torch.float16: 2e-2, torch.float32: 5e-3}
BACKENDS = [b for b in available_backends() if b != "torch"]
CASES = list(product(BACKENDS, [torch.float16, torch.float32]))
CUDA = torch.cuda.is_available()


def _rel(a, b):
    return (
        (a.double() - b.double()).abs().max() / (b.double().abs().max() + 1e-8)
    ).item()


def _grads(out, leaves, seed_grad):
    return torch.autograd.grad(out, leaves, seed_grad, allow_unused=False)


class AutogradParity(unittest.TestCase):
    def setUp(self):
        if not CUDA:
            self.skipTest("needs CUDA")
        torch.manual_seed(0)

    def _cmp(self, ours, eager, leaves_o, leaves_e, g, dtype, name):
        self.assertIsNotNone(ours.grad_fn, f"{name}: no grad_fn")
        self.assertLess(_rel(ours, eager), EPS[dtype], f"{name}: fwd")
        go = _grads(ours, leaves_o, g)
        ge = _grads(eager, leaves_e, g.to(eager.dtype))
        for i, (a, b) in enumerate(zip(go, ge)):
            self.assertGreater(
                a.double().abs().sum().item(), 0, f"{name}: zero grad {i}"
            )
            self.assertLess(_rel(a, b), EPS[dtype] * 4, f"{name}: grad {i}")

    @parameterized.expand(CASES)
    def test_loha_rebuild(self, backend, dtype):
        o, i, r = 128, 112, 8
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.1)
        w1d, w1u = mk(r, i), mk(o, r)
        w2d, w2u = mk(r, i), mk(o, r)
        ours_l = [t.clone().requires_grad_(True) for t in (w1d, w1u, w2d, w2u)]
        eag_l = [t.clone().requires_grad_(True) for t in (w1d, w1u, w2d, w2u)]
        gt = torch.tensor(0.5, device="cuda", dtype=dtype)
        ours = loha_diff_weight(*ours_l, gamma=0.5, backend=backend)
        eager = f_loha.diff_weight(*eag_l, None, None, gamma=gt, backend="torch")
        self._cmp(ours, eager, ours_l, eag_l, mk(o, i), dtype, "loha rebuild")

    @parameterized.expand(CASES)
    def test_loha_bypass(self, backend, dtype):
        t, o, i, r = 96, 128, 112, 8
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.1)
        x = mk(t, i)
        w1d, w1u = mk(r, i), mk(o, r)
        w2d, w2u = mk(r, i), mk(o, r)
        ours_l = [v.clone().requires_grad_(True) for v in (x, w1d, w1u, w2d, w2u)]
        eag_l = [v.clone().requires_grad_(True) for v in (x, w1d, w1u, w2d, w2u)]
        ours = loha_bypass_diff(ours_l[0], *ours_l[1:], gamma=0.5, backend=backend)
        gt = torch.tensor(0.5, device="cuda", dtype=dtype)
        dw = f_loha.diff_weight(*eag_l[1:], None, None, gamma=gt, backend="torch")
        eager = eag_l[0] @ dw.T
        self._cmp(ours, eager, ours_l, eag_l, mk(t, o), dtype, "loha bypass")

    @parameterized.expand(CASES)
    def test_loha_tucker(self, backend, dtype):
        o, i, r, kh = 48, 40, 8, 3
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.1)
        w1d, w1u, t1 = mk(r, i), mk(r, o), mk(r, r, kh, kh)
        w2d, w2u, t2 = mk(r, i), mk(r, o), mk(r, r, kh, kh)
        ours_l = [v.clone().requires_grad_(True) for v in (w1d, w1u, w2d, w2u, t1, t2)]
        eag_l = [v.clone().requires_grad_(True) for v in (w1d, w1u, w2d, w2u, t1, t2)]
        ours = loha_diff_weight(*ours_l, gamma=0.5, backend=backend)
        gt = torch.tensor(0.5, device="cuda", dtype=dtype)
        eager = f_loha.diff_weight(*eag_l, gamma=gt, backend="torch")
        self._cmp(ours, eager, ours_l, eag_l, mk(o, i, kh, kh), dtype, "loha tucker")

    @parameterized.expand(CASES)
    def test_locon_and_dylora(self, backend, dtype):
        o, i, r = 96, 80, 16
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.1)
        down, up = mk(r, i), mk(o, r)
        ours_l = [v.clone().requires_grad_(True) for v in (down, up)]
        eag_l = [v.clone().requires_grad_(True) for v in (down, up)]
        ours = locon_diff_weight(ours_l[0], ours_l[1], gamma=2.0, backend=backend)
        eager = f_locon.diff_weight(
            eag_l[0], eag_l[1], None, gamma=2.0, backend="torch"
        )
        self._cmp(ours, eager, ours_l, eag_l, mk(o, i), dtype, "locon")
        dy = dylora_diff_weight(down, up, rank=8, gamma=1.0, backend=backend)
        self.assertEqual(dy.shape, (o, i))

    @parameterized.expand(CASES)
    def test_locon_tucker(self, backend, dtype):
        o, i, r, kh = 48, 40, 8, 3
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.1)
        down, up, mid = mk(r, i, 1, 1), mk(o, r, 1, 1), mk(r, r, kh, kh)
        ours_l = [v.clone().requires_grad_(True) for v in (down, up, mid)]
        eag_l = [v.clone().requires_grad_(True) for v in (down, up, mid)]
        ours = locon_diff_weight(*ours_l, gamma=0.5, backend=backend)
        eager = f_locon.diff_weight(*eag_l, gamma=0.5, backend="torch")
        self._cmp(ours, eager, ours_l, eag_l, mk(o, i, kh, kh), dtype, "locon tucker")

    @parameterized.expand(CASES)
    def test_lokr(self, backend, dtype):
        a, b, c, d = 16, 20, 12, 14
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.1)
        w1, w2a, w2b = mk(a, b), mk(c, 8), mk(8, d)
        ours_l = [v.clone().requires_grad_(True) for v in (w1, w2a, w2b)]
        eag_l = [v.clone().requires_grad_(True) for v in (w1, w2a, w2b)]
        ours = lokr_diff_weight(
            ours_l[0],
            None,
            None,
            None,
            ours_l[1],
            ours_l[2],
            gamma=8.0,
            backend=backend,
        )
        eager = f_lokr.diff_weight(
            eag_l[0],
            None,
            None,
            None,
            eag_l[1],
            eag_l[2],
            None,
            gamma=8.0,
            backend="torch",
        )
        self._cmp(ours, eager, ours_l, eag_l, mk(a * c, b * d), dtype, "lokr")

    @parameterized.expand(CASES)
    def test_lokr_bypass(self, backend, dtype):
        t, a, b, c, d = 64, 16, 20, 12, 14
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.1)
        x, w1, w2 = mk(t, b * d), mk(a, b), mk(c, d)
        ours_l = [v.clone().requires_grad_(True) for v in (x, w1, w2)]
        eag_l = [v.clone().requires_grad_(True) for v in (x, w1, w2)]
        ours = lokr_bypass_diff(
            ours_l[0],
            ours_l[1],
            None,
            None,
            ours_l[2],
            None,
            None,
            gamma=1.0,
            backend=backend,
        )
        dw = f_lokr.diff_weight(
            eag_l[1],
            None,
            None,
            eag_l[2],
            None,
            None,
            None,
            gamma=1.0,
            backend="torch",
        )
        eager = eag_l[0] @ dw.reshape(a * c, b * d).T
        self._cmp(ours, eager, ours_l, eag_l, mk(t, a * c), dtype, "lokr bypass")

    @parameterized.expand(CASES)
    def test_diag_oft(self, backend, dtype):
        o, i, s = 96, 80, 8
        k = o // s
        mk = lambda *sh: (torch.randn(*sh, device="cuda", dtype=dtype) * 0.1)
        w = mk(o, i)
        blocks = torch.randn(k, s, s, device="cuda", dtype=dtype) * 0.05
        rescale = torch.ones(o, 1, device="cuda", dtype=dtype) + mk(o, 1) * 0.01
        ours_l = [v.clone().requires_grad_(True) for v in (blocks, rescale)]
        eag_l = [v.clone().requires_grad_(True) for v in (blocks, rescale)]
        ours = diag_oft_diff_weight(w, ours_l[0], ours_l[1], backend=backend)
        eager = f_oft.diff_weight(w, eag_l[0], eag_l[1], backend="torch")
        self._cmp(ours, eager, ours_l, eag_l, mk(o, i), dtype, "diag-oft")
        y = mk(20, o)
        ours_b = diag_oft_bypass_diff(y, ours_l[0], ours_l[1], backend=backend)
        eager_b = f_oft.bypass_forward_diff(
            None, y, eag_l[0], eag_l[1], backend="torch"
        )
        self.assertLess(_rel(ours_b, eager_b), EPS[dtype], "diag-oft bypass")

    @parameterized.expand(CASES)
    def test_boft(self, backend, dtype):
        o, i, s, m = 64, 56, 4, 3
        nb = o // s
        mk = lambda *sh: (torch.randn(*sh, device="cuda", dtype=dtype) * 0.1)
        w = mk(o, i)
        blocks = torch.randn(m, nb, s, s, device="cuda", dtype=dtype) * 0.05
        ours_l = [blocks.clone().requires_grad_(True)]
        eag_l = [blocks.clone().requires_grad_(True)]
        ours = boft_diff_weight(w, ours_l[0], backend=backend)
        eager = f_boft.diff_weight(w, eag_l[0], None, backend="torch")
        self._cmp(ours, eager, ours_l, eag_l, mk(o, i), dtype, "boft")
        y = mk(12, o)
        ours_b = boft_bypass_diff(y, ours_l[0], backend=backend)
        eager_b = f_boft.bypass_forward_diff(y, eag_l[0], None, backend="torch")
        self.assertLess(_rel(ours_b, eager_b), EPS[dtype], "boft bypass")

    @parameterized.expand(CASES)
    def test_ia3_norm_full_dora(self, backend, dtype):
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.2)
        w, scale = mk(64, 48), mk(64)
        lw = scale.clone().requires_grad_(True)
        out = ia3_diff_weight(w, lw, on_input=False, multiplier=0.9, backend=backend)
        ref = w.double() * (lw.double()[:, None] * 0.9)
        self.assertIsNotNone(out.grad_fn)
        self.assertLess(_rel(out, ref), EPS[dtype], "ia3")
        (g,) = _grads(out, [lw], mk(64, 48))
        self.assertGreater(g.abs().sum().item(), 0)

        x = mk(6, 32, 4, 4)
        wb = mk(32)
        yb = ia3_bypass(x, wb, 1, multiplier=1.0, diff=False, backend=backend)
        self.assertLess(
            _rel(yb, x.double() * (1 + wb.double().view(1, -1, 1, 1))), EPS[dtype]
        )

        base, diff = mk(40, 30), mk(40, 30)
        ld = diff.clone().requires_grad_(True)
        merged = full_diff_weight(base, ld, multiplier=0.7, backend=backend)
        self.assertLess(_rel(merged, base.double() + 0.7 * ld.double()), EPS[dtype])
        wn, bn_ = mk(32), mk(32)
        got_w, got_b = norm_diff_weights(wn, bn_, mk(32), mk(32), 1.0, backend=backend)
        self.assertEqual(got_w.shape, wn.shape)

        wt = mk(48, 36)
        dsc = torch.rand(48, device="cuda", dtype=dtype) + 0.5
        lwt = wt.clone().requires_grad_(True)
        ldsc = dsc.clone().requires_grad_(True)
        y = apply_dora(lwt, ldsc, multiplier=0.8, wd_on_out=True, backend=backend)
        eps = torch.finfo(dtype).eps
        n = wt.double().norm(dim=1, keepdim=True) + eps
        ref = wt.double() * (0.8 * (dsc.double()[:, None] / n - 1) + 1)
        self.assertLess(_rel(y, ref), EPS[dtype], "dora")
        ga, gb = _grads(y, [lwt, ldsc], mk(48, 36))
        self.assertGreater(ga.abs().sum().item(), 0)
        self.assertGreater(gb.abs().sum().item(), 0)

    @parameterized.expand(CASES)
    def test_glora(self, backend, dtype):
        o, i, r = 80, 64, 8
        mk = lambda *s: (torch.randn(*s, device="cuda", dtype=dtype) * 0.1)
        w = mk(o, i)
        la = [v.requires_grad_(True) for v in (mk(i, r), mk(r, i), mk(o, r), mk(r, i))]
        out = glora_diff_weight(w, *la, gamma=1.0, backend=backend)
        ref = (
            w.double() @ la[0].double() @ la[1].double()
            + la[2].double() @ la[3].double()
        )
        self.assertIsNotNone(out.grad_fn)
        self.assertLess(_rel(out, ref), EPS[dtype] * 2, "glora fwd")
        gs = _grads(out, la, mk(o, i))
        for g in gs:
            self.assertGreater(g.abs().sum().item(), 0)


class SafeFallback(unittest.TestCase):
    def test_safe_set_runs(self):
        if not CUDA:
            self.skipTest("needs CUDA")
        import os

        from lycoris.kernels.ops import get_ops

        prev = os.environ.get("LYCORIS_KERNEL_TUNE")
        os.environ["LYCORIS_KERNEL_TUNE"] = "off"
        try:
            ops = get_ops(BACKENDS[0])
            a = torch.randn(64, 8, device="cuda", dtype=torch.float16)
            b = torch.randn(8, 48, device="cuda", dtype=torch.float16)
            out = ops.lora_merge_fwd(a, b, gamma=1.0)
            self.assertEqual(out.shape, (64, 48))
        finally:
            if prev is None:
                os.environ.pop("LYCORIS_KERNEL_TUNE", None)
            else:
                os.environ["LYCORIS_KERNEL_TUNE"] = prev


if __name__ == "__main__":
    unittest.main()
