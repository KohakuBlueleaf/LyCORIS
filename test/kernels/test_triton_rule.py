"""The Triton kernels on the model-ranked tuner meet the precision bar, config by config.

Every kernel config the tuner can time (the top ``TOP`` of each plan, at odd
shapes and padded ranks) and every whole-op alternative is run directly and
judged with ``test.kernels.judge`` against an fp64 reference, the bar being
the eager composition in the same 16-bit dtype; a broken result is shown to be
flagged.

Usage:
    python -m unittest test.kernels.test_triton_rule -v
"""

import unittest

import torch

from . import judge

try:
    from lycoris.kernels.triton.dora import decompose
    from lycoris.kernels.triton.ia3 import channel
    from lycoris.kernels.triton.loha import hada, regen
    from lycoris.kernels.triton.lokr import apply as lokr
    from lycoris.kernels.triton.lokr import grad as lokr_grad
    from lycoris.kernels.triton.lokr import kron
    from lycoris.kernels.triton.lora import rebuild, stream
    from lycoris.kernels.tuner import TORCH, device_info
except ImportError:
    stream = None

CUDA = torch.cuda.is_available()
TOP = 3
DTYPES = (
    (torch.float16, torch.bfloat16)
    if CUDA and torch.cuda.get_device_capability()[0] >= 8
    else (torch.float16,)
)


def _kernels(plan):
    return [c for c in plan if "impl" not in c][:TOP]


@unittest.skipUnless(CUDA and stream is not None, "needs CUDA and Triton")
class TritonRule(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dev = device_info(torch.device("cuda"))

    def mk(self, *shape, std=1.0):
        return torch.randn(*shape, device="cuda", dtype=self.dtype) * std

    def judge(self, name, cand, eager, ref):
        errs = judge.errors(list(cand), list(ref))
        bar = judge.errors(list(eager), list(ref))
        self.assertTrue(judge.passes(errs, bar), f"{name}: {errs} vs eager {bar}")

    def each_dtype(self):
        for dtype in DTYPES:
            self.dtype = dtype
            with self.subTest(dtype=dtype):
                yield dtype

    def test_lora(self):
        for dt in self.each_dtype():
            t, o, i, r = 300, 192, 160, 24
            x, d, u = self.mk(t, i), self.mk(r, i, std=0.1), self.mk(o, r, std=0.1)
            base, g = self.mk(t, o), self.mk(t, o)
            ref = (x.double() @ d.double().t()) @ u.double().t() * 0.5 + base.double()
            eager = base + (x @ d.t()) @ u.t() * 0.5
            plan = stream._plan_bypass_fwd(self.dev, t, o, i, r, x.element_size(), True)
            self.assertTrue(_kernels(plan), "no legal bypass config")
            for cfg in [TORCH, stream.ATOMIC, *_kernels(plan)]:
                got = stream._bypass_fwd_run(cfg, x, u, d, 0.5, base, dt)
                self.judge(f"lora fwd {cfg}", [got], [eager], [ref])
            x64, d64, u64 = (v.double().requires_grad_() for v in (x, d, u))
            ref_b = torch.autograd.grad(
                (x64 @ d64.t()) @ u64.t() * 0.5, (x64, u64, d64), g.double()
            )
            h, q = x @ d.t(), g @ u
            eager_b = [q @ d * 0.5, g.t() @ h * 0.5, q.t() @ x * 0.5]
            plan = stream._plan_bypass_bwd(self.dev, t, o, i, r, x.element_size())
            for cfg in [stream.ATOMIC, *_kernels(plan)]:
                got = stream._bypass_bwd_run(cfg, g, x, u, d, 0.5)
                self.judge(f"lora bwd {cfg}", got, eager_b, ref_b)

    def test_lora_merge(self):
        for dt in self.each_dtype():
            o, i, r = 192, 176, 12
            d, u, w = (
                self.mk(r, i, std=0.1),
                self.mk(o, r, std=0.1),
                self.mk(o, i, std=0.02),
            )
            g = self.mk(o, i)
            ref = u.double() @ d.double() * 1.7 + w.double()
            eager = w + u @ d * 1.7
            plan = rebuild._plan_merge(
                self.dev, o, i, r, w.element_size(), u.element_size()
            )
            for cfg in [TORCH, *_kernels(plan)]:
                got = rebuild._merge_run(cfg, u, d, 1.7, w, dt)
                self.judge(f"lora merge {cfg}", [got], [eager], [ref])
            d64, u64 = d.double().requires_grad_(), u.double().requires_grad_()
            ref_b = torch.autograd.grad(u64 @ d64 * 1.3, (u64, d64), g.double())
            eager_b = rebuild._merge_bwd_run(TORCH, g, u, d, 1.3)
            plan = rebuild._plan_merge_bwd(
                self.dev, o, i, r, g.element_size(), u.element_size()
            )
            for cfg in [rebuild.WHOLE, *_kernels(plan)]:
                got = rebuild._merge_bwd_run(cfg, g, u, d, 1.3)
                self.judge(f"lora merge bwd {cfg}", got, eager_b, ref_b)

    def test_loha(self):
        for dt in self.each_dtype():
            o, i, r, t = 192, 176, 12, 300
            fs = tuple(self.mk(*s, std=0.3) for s in ((o, r), (r, i), (o, r), (r, i)))
            w, gw = self.mk(o, i, std=0.02), self.mk(o, i)
            f64 = [f.double().requires_grad_() for f in fs]
            p64 = (f64[0] @ f64[1]) * (f64[2] @ f64[3]) * 0.7
            eager = hada._merge_run(TORCH, fs, 0.7, w, dt)
            plan = hada._plan_merge(
                self.dev, o, i, r, w.element_size(), fs[0].element_size()
            )
            for cfg in _kernels(plan):
                got = hada._merge_run(cfg, fs, 0.7, w, dt)
                self.judge(
                    f"loha merge {cfg}", [got], [eager], [p64.detach() + w.double()]
                )
            ref_b = torch.autograd.grad(p64, f64, gw.double())
            eager_b = hada._merge_bwd_run(TORCH, gw, fs, 0.7)
            plan = hada._plan_merge_bwd(
                self.dev, o, i, r, gw.element_size(), fs[0].element_size()
            )
            for cfg in _kernels(plan):
                got = hada._merge_bwd_run(cfg, gw, fs, 0.7)
                self.judge(f"loha merge bwd {cfg}", got, eager_b, ref_b)
            x, base, g = self.mk(t, i), self.mk(t, o), self.mk(t, o)
            x64 = x.double().requires_grad_()
            y64 = x64 @ ((f64[0] @ f64[1]) * (f64[2] @ f64[3])).t() * 0.5
            ref = y64.detach() + base.double()
            eager_y = regen._bypass_run(TORCH, x, fs, 0.5, base, dt)
            plan = regen._plan_bypass(self.dev, t, o, i, r, x.element_size(), True)
            for cfg in plan[:TOP] + [c for c in plan if "impl" in c]:
                got = regen._bypass_run(cfg, x, fs, 0.5, base, dt)
                self.judge(f"loha bypass {cfg}", [got], [eager_y], [ref])
            ref_g = torch.autograd.grad(y64, [x64, *f64], g.double())
            xl = x.clone().requires_grad_()
            fl = [f.clone().requires_grad_() for f in fs]
            y = xl @ ((fl[0] @ fl[1]) * (fl[2] @ fl[3])).t() * 0.5
            eager_g = torch.autograd.grad(y, [xl, *fl], g)
            got = regen._bypass_bwd_materialized(g, x, *fs, 0.5)
            self.judge("loha bypass bwd", got, eager_g, ref_g)
            wrong = [got[0], *(gr * 2 for gr in got[1:])]
            self.assertFalse(
                judge.passes(
                    judge.errors(wrong, list(ref_g)),
                    judge.errors(list(eager_g), list(ref_g)),
                ),
                "a 2x factor gradient passes",
            )

    def test_lokr(self):
        for dt in self.each_dtype():
            t, a, b, c, d = 300, 4, 6, 48, 40
            x, w1, w2 = (
                self.mk(t, b * d),
                self.mk(a, b, std=0.3),
                self.mk(c, d, std=0.1),
            )
            base, g = self.mk(t, a * c), self.mk(t, a * c)
            x64, w164, w264 = (v.double().requires_grad_() for v in (x, w1, w2))
            y64 = (w164 @ (x64.reshape(t, b, d) @ w264.t())).reshape(t, a * c) * 0.5
            eager = lokr._apply_run(TORCH, x, w1, w2, 0.5, base, dt)
            plan = lokr._plan(self.dev, t, a, b, c, d, x.element_size(), True, [])
            self.assertTrue(_kernels(plan), "no legal apply config")
            for cfg in [lokr.ATOMIC, *_kernels(plan)]:
                got = lokr._apply_run(cfg, x, w1, w2, 0.5, base, dt)
                self.judge(
                    f"lokr fwd {cfg}", [got], [eager], [y64.detach() + base.double()]
                )
            ref = torch.autograd.grad(y64, (x64, w164, w264), g.double())
            xl, w1l, w2l = (v.clone().requires_grad_() for v in (x, w1, w2))
            y = (w1l @ (xl.reshape(t, b, d) @ w2l.t())).reshape(t, a * c) * 0.5
            eager_g = torch.autograd.grad(y, (xl, w1l, w2l), g)
            for name, bwd in (
                ("dxn", lokr_grad._bwd_dxn),
                ("apply", lokr_grad._bwd_apply),
            ):
                self.judge(f"lokr bwd {name}", bwd(g, x, w1, w2, 0.5), eager_g, ref)
            for cfg in _kernels(
                lokr_grad._plan_dxn(self.dev, t, a, b, c, d, x.element_size())
            ):
                gx, _ = lokr_grad._dxn_run(cfg, g, w1, w2, 0.5, dt)
                self.judge(f"lokr dxn {cfg}", [gx], eager_g[:1], ref[:1])
            mid = x.reshape(t * b, d) @ w2.t()
            plan = _kernels(lokr_grad._plan_gw1(self.dev, t, a, b, c, x.element_size()))
            self.assertTrue(plan, "no legal g_w1 config")
            for cfg in plan:
                gw1 = lokr_grad._gw1_run(cfg, g, mid, a, b, c).mul_(0.5)
                self.judge(f"lokr g_w1 {cfg}", [gw1], eager_g[1:2], ref[1:2])

    def test_lokr_merge(self):
        for dt in self.each_dtype():
            a, b, c, d = 6, 8, 40, 24
            w1, w2 = self.mk(a, b, std=0.3), self.mk(c, d, std=0.1)
            base, g = self.mk(a * c, b * d, std=0.02), self.mk(a * c, b * d)
            w164, w264 = w1.double().requires_grad_(), w2.double().requires_grad_()
            k64 = torch.kron(w164, w264) * 3.0
            eager = base + torch.kron(w1, w2) * 3.0
            plan = kron._plan(
                self.dev, a, b, c, d, base.element_size(), w2.element_size()
            )
            for cfg in [kron.TILED, *_kernels(plan)]:
                got = kron._run(cfg, w1, w2, 3.0, base, dt)
                self.judge(
                    f"kron fwd {cfg}", [got], [eager], [k64.detach() + base.double()]
                )
            ref = torch.autograd.grad(k64, (w164, w264), g.double())
            eager_b = kron._bwd_run(TORCH, g, w1, w2, 3.0)
            plan = kron._plan_bwd(
                self.dev, a, b, c, d, g.element_size(), w2.element_size()
            )
            for cfg in [kron.TILED, *_kernels(plan)]:
                got = kron._bwd_run(cfg, g, w1, w2, 3.0)
                self.judge(f"kron bwd {cfg}", got, eager_b, ref)

    def test_dora(self):
        for dt in self.each_dtype():
            o, i = 96, 200
            w, dw = self.mk(o, i, std=0.02), self.mk(o, i, std=0.002)
            ds = ((w + dw).float().norm(dim=1) * 1.2).to(dt)
            g = self.mk(o, i)
            m64 = (w.double() + dw.double()).requires_grad_()
            ds64 = ds.double().requires_grad_()
            n64 = m64.norm(dim=1, keepdim=True) + torch.finfo(dt).eps
            y64 = m64 * (0.8 * (ds64[:, None] / n64 - 1) + 1)
            ref_b = torch.autograd.grad(y64, (m64, ds64), g.double())
            eager_y, eager_n = decompose._fwd_run(TORCH, w, ds, 0.8, 0, dw)
            eager_b = decompose._bwd_run(TORCH, g, w, ds, eager_n, 0.8, 0, dw)
            plan = decompose._plan(
                self.dev, o, i, 2 * w.element_size(), w.element_size(), False
            )
            for cfg in _kernels(plan):
                y, n = decompose._fwd_run(cfg, w, ds, 0.8, 0, dw)
                self.judge(f"dora fwd {cfg}", [y], [eager_y], [y64.detach()])
            plan = decompose._plan(
                self.dev, o, i, 3 * w.element_size(), w.element_size(), True
            )
            for cfg in _kernels(plan):
                got = decompose._bwd_run(cfg, g, w, ds, eager_n, 0.8, 0, dw)
                self.judge(f"dora bwd {cfg}", got, eager_b, ref_b)

    def test_ia3(self):
        for dt in self.each_dtype():
            x, wv, g = self.mk(300, 96), self.mk(96, std=0.1), self.mk(300, 96)
            for w in (wv, wv.float()):
                x64, w64 = x.double().requires_grad_(), w.double().requires_grad_()
                y64 = x64 * (1.0 + 0.7 * w64)
                ref = [y64.detach(), *torch.autograd.grad(y64, (x64, w64), g.double())]
                xl, wl = x.clone().requires_grad_(), w.clone().requires_grad_()
                y = xl * (1.0 + 0.7 * wl)
                eager = [y.detach(), *torch.autograd.grad(y, (xl, wl), g)]
                got_y = channel.ia3_fwd(x, w, -1, 1.0, 0.7)
                self.assertEqual(got_y.dtype, y.dtype)
                got = [got_y, *channel.ia3_bwd(g, x, w, -1, 1.0, 0.7)]
                self.judge(f"ia3 {w.dtype}", got, eager, ref)

    def test_rule_catches_a_broken_kernel(self):
        self.dtype = torch.float16
        t, o, i, r = 300, 192, 160, 8
        x, d, u = self.mk(t, i), self.mk(r, i, std=0.1), self.mk(o, r, std=0.1)
        ref = (x.double() @ d.double().t()) @ u.double().t() * 0.5
        eager = judge.errors([(x @ d.t()) @ u.t() * 0.5], [ref])
        cfg = _kernels(stream._plan_bypass_fwd(self.dev, t, o, i, r, 2, False))[0]
        got = stream._bypass_fwd_run(cfg, x, u, d, 0.5, None, torch.float16)
        self.assertTrue(judge.passes(judge.errors([got], [ref]), eager))
        self.assertFalse(judge.passes(judge.errors([got.roll(1, 0)], [ref]), eager))
        broken = got.clone()
        broken[7, 3] = float("inf")
        self.assertFalse(judge.passes(judge.errors([broken], [ref]), eager))


if __name__ == "__main__":
    unittest.main()
