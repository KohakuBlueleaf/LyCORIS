"""The TileLang kernels meet the precision bar, config by config.

Every model-ranked kernel config the tuner can time (the top ``TOP`` of each
plan, odd shapes and padded ranks) is run directly and judged with
``test.kernels.judge`` against an fp64 reference, the bar being the torch
composition in the same dtype; a broken result is shown to be flagged.

Usage:
    python -m unittest test.kernels.test_tilelang_rule -v
"""

import unittest

import torch

from lycoris.functional import boft as f_boft
from lycoris.functional import diag_oft as f_oft

from . import judge, refs

try:
    from lycoris.kernels.tilelang.boft import stages as boft
    from lycoris.kernels.tilelang.common import fp32_gemm_exact
    from lycoris.kernels.tilelang.dora import decompose as dora
    from lycoris.kernels.tilelang.ia3 import channel as ia3
    from lycoris.kernels.tilelang.lokr import apply as lokr
    from lycoris.kernels.tilelang.lokr import grad as lokr_grad
    from lycoris.kernels.tilelang.lokr import kron as lokr_kron
    from lycoris.kernels.tilelang.loha import hada
    from lycoris.kernels.tilelang.lora import rebuild, stream
    from lycoris.kernels.tilelang.oft import rotate
    from lycoris.kernels.tuner import TORCH, device_info
except ImportError:
    lokr = None

CUDA = torch.cuda.is_available()
TOP = 3
F16 = torch.float16
F32 = torch.float32


def _mk(*shape, std=1.0):
    return torch.randn(*shape, device="cuda", dtype=F16) * std


def _kernels(plan):
    return [c for c in plan if "impl" not in c][:TOP]


def _factors(o, i, r):
    return (
        _mk(o, r, std=0.3),
        _mk(r, i, std=0.3),
        _mk(o, r, std=0.3),
        _mk(r, i, std=0.3),
    )


@unittest.skipUnless(CUDA and lokr is not None, "needs CUDA and TileLang")
class TileLangRule(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dev = device_info(torch.device("cuda"))

    def judge(self, name, cand, eager, ref):
        errs = judge.errors(cand, ref)
        bar = judge.errors(eager, ref)
        self.assertTrue(judge.passes(errs, bar), f"{name}: {errs} vs eager {bar}")

    def test_lora_bypass(self):
        t, o, i, r = 300, 192, 160, 8
        x, d, u = _mk(t, i), _mk(r, i, std=0.1), _mk(o, r, std=0.1)
        base = _mk(t, o)
        ref = (x.double() @ d.double().t()) @ u.double().t() * 0.5 + base.double()
        eager = base + (x @ d.t()) @ u.t() * 0.5
        plan = stream._plan(self.dev, t, o, i, r, True, [])
        self.assertTrue(_kernels(plan), "no legal config")
        for cfg in [TORCH, *_kernels(plan)]:
            got = stream._run(cfg, x, u, d, 0.5, base, F16)
            self.judge(f"lora fwd {cfg}", [got], [eager], [ref])
        out32 = stream._run(_kernels(plan)[0], x, u, d, 0.5, base.float(), F32)
        self.assertEqual(out32.dtype, F32)
        self.judge("lora fwd fp32 base", [out32], [eager], [ref])

    def test_lora_bypass_bwd(self):
        t, o, i, r = 300, 192, 160, 24
        x, d, u = _mk(t, i), _mk(r, i, std=0.1), _mk(o, r, std=0.1)
        g = _mk(t, o)
        x64, d64, u64 = (v.double().requires_grad_() for v in (x, d, u))
        y64 = (x64 @ d64.t()) @ u64.t() * 0.5
        ref = torch.autograd.grad(y64, (x64, u64, d64), g.double())
        h, q = x @ d.t(), g @ u
        eager = [q @ d * 0.5, g.t() @ h * 0.5, q.t() @ x * 0.5]
        got = stream._bwd_gemms(x, u, d, g, 0.5)
        self.judge("lora bwd", list(got), eager, list(ref))
        # the gx (+ q) kernel itself, config by config
        q_ref = g.double() @ u.double()
        for cfg in _kernels(stream._plan(self.dev, t, i, o, r, False, [])):
            gx, qk = stream._run_dxq(cfg, g, u, d, 0.5)
            self.judge(f"lora dxq {cfg}", [gx, qk], [eager[0], q], [ref[0], q_ref])

    def test_lora_merge(self):
        o, i, r = 192, 176, 12
        d, u, w = _mk(r, i, std=0.1), _mk(o, r, std=0.1), _mk(o, i, std=0.02)
        ref = u.double() @ d.double() * 1.7 + w.double()
        eager = w + u @ d * 1.7
        for cfg in [TORCH, *_kernels(rebuild._plan_merge(self.dev, o, i, r, 2))]:
            got = rebuild._merge_run(cfg, u, d, 1.7, w, F16)
            self.judge(f"lora merge {cfg}", [got], [eager], [ref])

    def test_lora_merge_bwd(self):
        o, i, r = 192, 176, 12
        d, u, g = _mk(r, i, std=0.1), _mk(o, r, std=0.1), _mk(o, i)
        d64, u64 = d.double().requires_grad_(), u.double().requires_grad_()
        ref = torch.autograd.grad(u64 @ d64 * 1.3, (u64, d64), g.double())
        eager = rebuild._merge_bwd_run(TORCH, g, u, d, 1.3)
        for cfg in _kernels(rebuild._plan_merge_bwd(self.dev, o, i, r, 2)):
            got = rebuild._merge_bwd_run(cfg, g, u, d, 1.3)
            self.judge(f"lora merge bwd {cfg}", list(got), list(eager), list(ref))

    def test_loha_merge_fwd_bwd(self):
        o, i, r = 192, 176, 12
        fs = _factors(o, i, r)
        w, g = _mk(o, i, std=0.02), _mk(o, i)
        f64 = [f.double().requires_grad_() for f in fs]
        p64 = (f64[0] @ f64[1]) * (f64[2] @ f64[3]) * 0.7
        ref_fwd = p64.detach() + w.double()
        ref_bwd = torch.autograd.grad(p64, f64, g.double())
        eager = hada._merge_run(TORCH, fs, 0.7, w, F16)
        for cfg in _kernels(hada._plan_merge(self.dev, o, i, r, 2)):
            got = hada._merge_run(cfg, fs, 0.7, w, F16)
            self.judge(f"loha merge {cfg}", [got], [eager], [ref_fwd])
        eager_b = hada._merge_bwd_run(TORCH, g, fs, 0.7)
        for cfg in _kernels(hada._plan_merge_bwd(self.dev, o, i, r, 2)):
            got = hada._merge_bwd_run(cfg, g, fs, 0.7)
            self.judge(f"loha merge bwd {cfg}", list(got), list(eager_b), list(ref_bwd))

    def test_loha_bypass(self):
        t, o, i, r = 300, 192, 160, 8
        x, g = _mk(t, i), _mk(t, o)
        fs = _factors(o, i, r)
        x64 = x.double().requires_grad_()
        f64 = [f.double().requires_grad_() for f in fs]
        y64 = x64 @ ((f64[0] @ f64[1]) * (f64[2] @ f64[3])).t() * 0.5
        ref = [y64.detach(), *torch.autograd.grad(y64, [x64, *f64], g.double())]
        xl = x.clone().requires_grad_()
        fl = [f.clone().requires_grad_() for f in fs]
        y = xl @ ((fl[0] @ fl[1]) * (fl[2] @ fl[3])).t() * 0.5
        eager = [y.detach(), *torch.autograd.grad(y, [xl, *fl], g)]
        out = hada._bypass_run(hada.MATERIALIZE, x, fs, 0.5, None, F16)
        self.judge("loha fwd", [out], eager[:1], ref[:1])
        got = hada._bypass_bwd_materialized(g, x, fs, 0.5)
        self.judge("loha bwd", list(got), eager[1:], ref[1:])
        # factor gradients missing the scale (2x) are flagged
        wrong = [got[0], *(gr * 2 for gr in got[1:])]
        errs = judge.errors(wrong, ref[1:])
        self.assertFalse(judge.passes(errs, judge.errors(eager[1:], ref[1:])))

    def test_lokr_apply(self):
        t, a, b, c, d = 300, 4, 6, 48, 40
        x, w1, w2 = _mk(t, b * d), _mk(a, b, std=0.3), _mk(c, d, std=0.1)
        base = _mk(t, a * c)
        x3 = x.double().reshape(t, b, d)
        ref = (w1.double() @ (x3 @ w2.double().t())).reshape(t, a * c) * 0.5
        ref = ref + base.double()
        eager = lokr._run(TORCH, x, w1, w2, 0.5, base, F16)
        plan = lokr._plan(self.dev, t, a, b, c, d, True, [])
        self.assertTrue(_kernels(plan), "no legal config")
        for cfg in _kernels(plan):
            got = lokr._run(cfg, x, w1, w2, 0.5, base, F16)
            self.judge(f"lokr fwd {cfg}", [got], [eager], [ref])

    def test_lokr_bwd(self):
        t, a, b, c, d = 300, 4, 6, 48, 40
        x, w1, w2 = _mk(t, b * d), _mk(a, b, std=0.3), _mk(c, d, std=0.1)
        g = _mk(t, a * c)
        x64, w164, w264 = (v.double().requires_grad_() for v in (x, w1, w2))
        y64 = (w164 @ (x64.reshape(t, b, d) @ w264.t())).reshape(t, a * c) * 0.5
        ref = torch.autograd.grad(y64, (x64, w164, w264), g.double())
        xl, w1l, w2l = (v.clone().requires_grad_() for v in (x, w1, w2))
        y = (w1l @ (xl.reshape(t, b, d) @ w2l.t())).reshape(t, a * c) * 0.5
        eager = torch.autograd.grad(y, (xl, w1l, w2l), g)
        for name, bwd in (("dxn", lokr_grad._bwd_dxn), ("apply", lokr_grad._bwd_apply)):
            got = bwd(g, x, w1, w2, 0.5)
            self.judge(f"lokr bwd {name}", list(got), list(eager), list(ref))
        # the gx kernels themselves, config by config
        w1t, w2t = w1.t().contiguous(), w2.t().contiguous()
        for cfg in _kernels(lokr._plan(self.dev, t, b, a, d, c, False, [])):
            gx = lokr._run(cfg, g, w1t, w2t, 0.5, None, F16)
            self.judge(f"lokr dx {cfg}", [gx], list(eager[:1]), list(ref[:1]))
        mid = x.reshape(t * b, d) @ w2.t()
        plan = _kernels(lokr_grad._plan_gw1(self.dev, t, a, b, c))
        self.assertTrue(plan, "no legal g_w1 config")
        for cfg in plan:
            gw1 = lokr_grad._gw1_run(cfg, g, mid, a, b, c).mul_(0.5)
            self.judge(f"lokr g_w1 {cfg}", [gw1], [eager[1]], [ref[1]])
        n_ref = torch.matmul(w1.double().t(), g.double().reshape(t, a, c))
        n_eager = torch.matmul(w1.t(), g.reshape(t, a, c))
        for cfg in _kernels(lokr_grad._plan(self.dev, t, a, b, c, d)):
            gx, n = lokr_grad._dxn_run(cfg, g, w1, w2, 0.5, F16)
            self.judge(
                f"lokr dxn {cfg}",
                [gx, n.reshape(t, b, c)],
                [eager[0], n_eager],
                [ref[0], n_ref],
            )

    def test_lokr_full_merge_bwd(self):
        a, b, c, d = 4, 6, 48, 40
        w1, w2, g = _mk(a, b, std=0.3), _mk(c, d, std=0.1), _mk(a * c, b * d)
        w164, w264 = w1.double().requires_grad_(), w2.double().requires_grad_()
        ref = torch.autograd.grad(
            torch.kron(w164, w264) * 0.7, (w164, w264), g.double()
        )
        w1l, w2l = w1.clone().requires_grad_(), w2.clone().requires_grad_()
        eager = torch.autograd.grad(torch.kron(w1l, w2l) * 0.7, (w1l, w2l), g)
        plan = _kernels(lokr_kron._plan(self.dev, a, b, c, d, 2, 2, []))
        self.assertTrue(plan, "no legal config")
        for cfg in plan + [{"BM": 16, "BN": 8, "SPLIT": 4, "threads": 128}]:
            got = lokr_kron._bwd_run(cfg, g, w1, w2, 0.7)
            self.judge(f"lokr merge bwd {cfg}", list(got), list(eager), list(ref))
        # g_w2 counted once per split (partials not divided out) is flagged
        got = lokr_kron._bwd_run(plan[0], g, w1, w2, 0.7)
        wrong = [got[0], got[1] * 4]
        self.assertFalse(
            judge.passes(judge.errors(wrong, ref), judge.errors(eager, ref))
        )

    def test_dora(self):
        rows, cols = 96, 200
        w, dl, g = (
            _mk(rows, cols, std=0.02),
            _mk(rows, cols, std=0.002),
            _mk(rows, cols),
        )
        ds = ((w + dl).float().norm(dim=1) * 1.2).to(F16)
        m64 = (w.double() + dl.double()).requires_grad_()
        d64 = ds.double().requires_grad_()
        n64 = m64.norm(dim=1) + torch.finfo(F16).eps
        y64 = m64 * (0.8 * (d64 / n64 - 1.0) + 1.0)[:, None]
        ref_b = torch.autograd.grad(y64, (m64, d64), g.double())
        y_e, n_e = dora._fwd_run(TORCH, w, ds, 0.8, dl)
        b_e = dora._bwd_run(TORCH, g, w, ds, n_e, 0.8, dl)
        plan = _kernels(dora._plan(self.dev, rows, cols, 4, 2, False))
        extra = [
            {"RESIDENT": True, "BM": 4, "BN": 256, "threads": 128},
            {"RESIDENT": False, "BM": 2, "BN": 64, "threads": 128},
        ]
        for cfg in plan + extra:
            y, n = dora._fwd_run(cfg, w, ds, 0.8, dl)
            self.judge(f"dora fwd {cfg}", [y], [y_e], [y64.detach()])
            got = dora._bwd_run(cfg, g, w, ds, n, 0.8, dl)
            self.judge(f"dora bwd {cfg}", list(got), list(b_e), list(ref_b))
        # gd with the norm left out (rowdot alone) is flagged
        wrong = [got[0], got[1] * n.to(F16)]
        self.assertFalse(
            judge.passes(judge.errors(wrong, ref_b), judge.errors(b_e, ref_b))
        )

    def test_oft(self):
        for weight, shape in ((False, (300, 192)), (True, (192, 72))):
            for s in (8, 16, 48, 64):
                k = 192 // s
                blocks = (torch.randn(k, s, s, device="cuda") * 0.05).to(F16)
                res = (1 + 0.1 * torch.randn(192, device="cuda")).to(F16)
                x, g = _mk(*shape), _mk(*shape)
                self._oft_case(blocks, res, x, g, weight, s)

    def _oft_case(self, blocks, res, x, g, weight, s):
        def body(b, xx, rr):
            if weight:
                return f_oft._diff_weight(xx, b, rr.reshape(-1, 1), None)
            return f_oft._bypass_diff(None, xx, b, rr, None, False)

        # eager's Cayley inverse runs in fp32, so the fp64 reference is refs'
        leaves64 = [v.double().requires_grad_() for v in (blocks, x, res)]
        b64, x64, r64 = leaves64
        y64 = refs.bd_fused(b64, x64, r64, 1.0, True, weight)
        ref = [y64.detach(), *torch.autograd.grad(y64, leaves64, g.double())]
        leaves = [v.clone().requires_grad_() for v in (blocks, x, res)]
        y = body(*leaves)
        eager = [y.detach(), *torch.autograd.grad(y, leaves, g.float())]
        dh = rotate.deltas(blocks, res, 1.0, F32)
        k = blocks.shape[0]
        _, outer, _, inner = rotate._layout(x, weight)
        tag = f"s{s} w{weight}"
        alts = rotate._alternatives(s, True)
        exact = fp32_gemm_exact(self.dev)
        plan = rotate._plan(self.dev, k, s, outer, inner, 2, False, True, alts)
        # fp32 kernel configs exist exactly where an fp32 T.gemm is IEEE
        self.assertEqual(bool(_kernels(plan)), exact, f"oft fwd configs {tag}")
        for cfg in _kernels(plan) + [rotate.TORCH]:
            got = rotate._fwd_run(
                cfg, self.dev, dh, x, res, True, weight, F32, blocks, 1.0
            )
            self.judge(f"oft fwd {tag} {cfg}", [got], eager[:1], ref[:1])
        plan = rotate._plan(self.dev, k, s, outer, inner, 2, True, True, alts)
        if not exact or s <= 32:
            self.assertEqual(bool(_kernels(plan)), exact, f"oft bwd configs {tag}")
        for cfg in _kernels(plan) + [rotate.TORCH]:
            gx, gb, gres = rotate._bwd_run(
                cfg, self.dev, dh, x, g, res, True, weight, blocks, 1.0
            )
            got = [gb, gx, gres.reshape(res.shape)]
            self.judge(f"oft bwd {tag} {cfg}", got, eager[1:], ref[1:])
        # the block gradient of a transposed P (x^T g taken as g^T x) is flagged
        first = (_kernels(plan) or [rotate.TORCH])[0]
        p = rotate._bwd_apply(first, self.dev, dh, x, g, res, True, weight)[1]
        gb_t, _ = rotate.chain(blocks, res, p.transpose(1, 2), F32, 1.0)
        self.assertFalse(
            judge.passes(
                judge.errors([gb_t], ref[1:2]),
                judge.errors(eager[1:2], ref[1:2]),
            ),
            f"transposed P not flagged {tag}",
        )

    def test_boft(self):
        c = 128
        for b in (4, 8, 16, 32):
            blocks, _ = f_boft.weight_gen(torch.empty(c, 1), b)
            blocks = (torch.randn_like(blocks) * 0.05).to("cuda", F16)
            x, g = _mk(300, c), _mk(300, c)
            self._boft_case(blocks, x, g, b)

    def _boft_case(self, blocks, x, g, b):
        def body(bb, xx):
            return f_boft._bypass_diff(xx, bb, None, None, False, 0.9)

        leaves64 = [v.double().requires_grad_() for v in (blocks, x)]
        b64, x64 = leaves64
        y64 = refs.butterfly_blocks(b64, x64, 1.0, 0.9, axis=-1) - x64
        ref = [y64.detach(), *torch.autograd.grad(y64, leaves64, g.double())]
        leaves = [v.clone().requires_grad_() for v in (blocks, x)]
        y = body(*leaves)
        eager = [y.detach(), *torch.autograd.grad(y, leaves, g.float())]
        # the stages run on the fp32-promoted operands, as the autograd layer passes them
        bk, xk = blocks.to(F32), x.to(F32)
        m, num = blocks.shape[:2]
        tag = f"b{b}"
        dh = boft._deltas(bk, 1.0, 0.9, F32)
        exact = fp32_gemm_exact(self.dev)
        plan = boft._plan(self.dev, m, num, b, 300, 1, 4, False)
        # fp32 stage configs exist exactly where an fp32 T.gemm is IEEE
        self.assertEqual(bool(_kernels(plan)), exact, f"boft fwd configs {tag}")
        for cfg in _kernels(plan) + [boft.TORCH]:
            got = boft._fwd_run(cfg, dh, bk, xk, -1, 1.0, 0.9, None, True)
            self.judge(f"boft fwd {tag} {cfg}", [got], eager[:1], ref[:1])
        plan = boft._plan(self.dev, m, num, b, 300, 1, 4, True)
        self.assertEqual(bool(_kernels(plan)), exact, f"boft bwd configs {tag}")
        for cfg in _kernels(plan) + [boft.TORCH]:
            gx, gb = boft._bwd_run(cfg, self.dev, dh, bk, xk, g.float(), -1, 1.0, 0.9)
            got = [gb, gx - g.float()]
            self.judge(f"boft bwd {tag} {cfg}", got, eager[1:], ref[1:])
        if not exact:
            return
        # stage gradients taken in forward order (the stages walked the wrong way) are flagged
        _, gd = boft._bwd_stages(_kernels(plan)[0], self.dev, dh, xk, g.float())
        gd = (gd.flip(0) * 0.9).reshape(m * num, b, b)
        gb, _ = boft.chain(bk.reshape(m * num, b, b), None, gd, F32, 1.0)
        if m > 1:
            errs = judge.errors([gb.view(blocks.shape)], ref[1:2])
            self.assertFalse(
                judge.passes(errs, judge.errors(eager[1:2], ref[1:2])), tag
            )

    def test_ia3(self):
        for shape, axis in (((300, 96), -1), ((6, 96, 50), 1)):
            x, g = _mk(*shape), _mk(*shape)
            wv = _mk(96, std=0.1)
            for w in (wv, wv.float()):
                wshape = [1] * x.dim()
                wshape[axis] = -1
                x64, w64 = x.double().requires_grad_(), w.double().requires_grad_()
                y64 = x64 * (1.0 + 0.7 * w64.reshape(wshape))
                ref = [y64.detach(), *torch.autograd.grad(y64, (x64, w64), g.double())]
                xl, wl = x.clone().requires_grad_(), w.clone().requires_grad_()
                y = xl * (1.0 + 0.7 * wl.reshape(wshape))
                eager = [y.detach(), *torch.autograd.grad(y, (xl, wl), g)]
                _, outer, c, inner = ia3._view3(x, axis)
                rows, cols = (outer, c) if inner == 1 else (outer * c, inner)
                tag = f"{shape} {w.dtype}"
                fwd = _kernels(ia3._plan_fwd(self.dev, rows, cols, 2))
                bwd = _kernels(ia3._plan_bwd(self.dev, outer, c, inner, 2))
                self.assertTrue(fwd and bwd, f"no legal ia3 config {tag}")
                for fc, bc in zip(fwd, bwd):
                    got_y = ia3._fwd_run(fc, x, w, axis, 1.0, 0.7)
                    self.assertEqual(got_y.dtype, y.dtype)
                    gx, gw = ia3._bwd_run(bc, self.dev, g, x, w, axis, 1.0, 0.7)
                    got = [got_y, gx, gw.to(w.dtype)]
                    self.judge(f"ia3 {tag} {fc} {bc}", got, eager, ref)
                # a channel sum missing half its partials is flagged
                errs = judge.errors([(gw * 0.5).to(w.dtype)], ref[2:])
                self.assertFalse(judge.passes(errs, judge.errors(eager[2:], ref[2:])))

    def test_rule_catches_a_broken_kernel(self):
        t, o, i, r = 300, 192, 160, 8
        x, d, u = _mk(t, i), _mk(r, i, std=0.1), _mk(o, r, std=0.1)
        ref = (x.double() @ d.double().t()) @ u.double().t() * 0.5
        eager = judge.errors([(x @ d.t()) @ u.t() * 0.5], [ref])
        cfg = _kernels(stream._plan(self.dev, t, o, i, r, False, []))[0]
        got = stream._run(cfg, x, u, d, 0.5, None, F16)
        self.assertTrue(judge.passes(judge.errors([got], [ref]), eager))
        # a row-offset store and a non-finite element are flagged
        shifted = judge.errors([got.roll(1, 0)], [ref])
        self.assertFalse(judge.passes(shifted, eager))
        broken = got.clone()
        broken[7, 3] = float("inf")
        self.assertFalse(judge.passes(judge.errors([broken], [ref]), eager))


if __name__ == "__main__":
    unittest.main()
