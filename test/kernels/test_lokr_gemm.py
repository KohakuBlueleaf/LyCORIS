"""The LoKr GEMM-layout kernels (triton.lokr.gemm) against fp64, beside eager torch.

Forward, gx with g_w1, and g_w2, one fixed config per mix ("batch" where both w1
sides pad to >= 16, "diag" always), fp16 and bf16, on the kron shapes of a 3072-wide
model at factors 16 (attention, fused up, down), 8 (attention, fused up, down), 4, a
non-pow2 factor (12), a rectangular w1 and a wide w1 (48), at an odd and a padded token
count; the public entry point through autograd with full and factorized halves. Every
kernel tensor and the eager one are scored against fp64 (``stats``); a test fails only
on explosion. On a cold table each op's tuner runs at least one kernel config, in fp16
and bf16, so a kernel that cannot build or that the torch-agreement guard rejects fails
a test instead of falling back to torch unseen; a key whose timing ran out of memory is
not stored. Each check is shown to fire on a broken variant.

Usage:
    python -m unittest test.kernels.test_lokr_gemm -v
"""

import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from parameterized import parameterized

from lycoris.functional import lokr as f_lokr
from lycoris.kernels import set_tuning
from lycoris.kernels.autograd.lokr import lokr_kron_bypass
from lycoris.kernels.tuner import tuner
from lycoris.kernels.tuner.measure import disagreement

from . import stats

CUDA = torch.cuda.is_available()
if CUDA:
    from lycoris.kernels.triton.lokr import gemm
    from lycoris.kernels.tuner.measure import pow2

SHAPES = {
    "f16_attn": (16, 16, 192, 192),
    "f16_up": (16, 16, 1024, 192),
    "f16_down": (16, 16, 192, 512),
    "f8": (8, 8, 384, 384),
    "f8_up": (8, 8, 2048, 384),
    "f8_down": (8, 8, 384, 1024),
    "f4": (4, 4, 768, 768),
    "f12": (12, 12, 256, 256),
    "rect": (16, 8, 192, 384),
    "wide_w1": (48, 48, 64, 64),
}
TOKENS = (7, 1027)
DTYPES = (torch.float16, torch.bfloat16)
CASES = [(name, t, dt) for name in SHAPES for t in TOKENS for dt in DTYPES]


def _bt(pa: int, pb: int) -> int:
    return max(1, 128 // max(pa, pb))


def configs(a: int, b: int) -> dict[str, dict]:
    """One fixed config per legal mix for every GEMM kernel."""
    pa, pb = pow2(a), pow2(b)
    out = {}
    for mix in ("batch", "diag"):
        if mix == "batch" and min(pa, pb) < 16:
            continue
        bt = _bt(pa, pb)
        base = {"MIX": mix, "BT": bt, "num_warps": 4, "num_stages": 2}
        out[mix] = {
            "fwd": {**base, "BN": 64, "BK": 32},
            "dx": {**base, "BD": 64, "BK": 32},
            "gw2": {**base, "BC": 64, "BD": 64},
        }
    return out


def _mk(*shape, dtype, scale=0.1):
    return (torch.randn(*shape, device="cuda", dtype=torch.float32) * scale).to(dtype)


def _eager(x, w1, w2, scale):
    """Eager LoKr at the operands' precision (lycoris.functional, backend torch)."""
    return f_lokr.kron_bypass(
        x, w1, None, None, w2, None, None, scale=scale, backend="torch"
    )


def _case(a, b, c, d, t, dtype):
    x = _mk(t, b * d, dtype=dtype, scale=1.0)
    w1 = _mk(a, b, dtype=dtype, scale=0.3)
    w2 = _mk(c, d, dtype=dtype, scale=0.1)
    gy = _mk(t, a * c, dtype=dtype, scale=1.0)
    return x, w1, w2, gy


def _fp64(x, w1, w2, gy, scale, base=None):
    """(y, gx, gw1, gw2) in fp64 by autograd."""
    lx, l1, l2 = (v.double().requires_grad_(True) for v in (x, w1, w2))
    y = scale * lx @ torch.kron(l1, l2).T
    if base is not None:
        y = y + base.double()
    y.backward(gy.double())
    return y.detach(), lx.grad, l1.grad, l2.grad


def _eager_grads(x, w1, w2, gy, scale):
    lx, l1, l2 = (v.clone().requires_grad_(True) for v in (x, w1, w2))
    y = _eager(lx, l1, l2, scale)
    y.backward(gy)
    return y.detach(), lx.grad, l1.grad, l2.grad


class GemmVsFp64(unittest.TestCase):
    def setUp(self):
        if not CUDA:
            self.skipTest("needs CUDA")
        if not torch.cuda.is_bf16_supported():
            self.skipTest("bf16 needs sm_80+")
        torch.manual_seed(0)

    def fail_on(self, problems):
        problems = [p for p in problems if p is not None]
        self.assertFalse(problems, "\n".join(problems))

    @parameterized.expand(CASES)
    def test_forward(self, name, t, dtype):
        a, b, c, d = SHAPES[name]
        x, w1, w2, _ = _case(a, b, c, d, t, dtype)
        base = _mk(t, a * c, dtype=dtype, scale=1.0)
        ref = _fp64(x, w1, w2, torch.zeros_like(base), 0.7, base)[0]
        eager = _eager(x, w1, w2, 0.7) + base
        for mix, cf in configs(a, b).items():
            got = gemm._fwd_run(
                cf["fwd"], x, w1[None], w2[None], (0.7,), base[None], dtype
            )[0]
            self.fail_on(
                [stats.judge(f"fwd {name} t{t} {dtype} {mix}", got, eager, ref)]
            )

    @parameterized.expand(CASES)
    def test_backward(self, name, t, dtype):
        a, b, c, d = SHAPES[name]
        x, w1, w2, gy = _case(a, b, c, d, t, dtype)
        _, rx, r1, r2 = _fp64(x, w1, w2, gy, 0.7)
        _, ex, e1, e2 = _eager_grads(x, w1, w2, gy, 0.7)
        for mix, cf in configs(a, b).items():
            gx, gw1 = gemm._dx_run(cf["dx"], gy[None], x, w1[None], w2[None], (0.7,))
            gw2 = gemm._gw2_run(cf["gw2"], gy[None], x, w1[None], c, d, (0.7,))
            tag = f"{name} t{t} {dtype} {mix}"
            self.fail_on(
                [
                    stats.judge(f"gx {tag}", gx, ex, rx),
                    stats.judge(f"gw1 {tag}", gw1[0], e1, r1),
                    stats.judge(f"gw2 {tag}", gw2[0], e2, r2),
                ]
            )

    @parameterized.expand([(dt, full) for dt in DTYPES for full in (True, False)])
    def test_entry_points_autograd(self, dtype, full):
        """lokr_kron_bypass with full or factorized halves, whatever the tuner picks,
        against fp64 autograd through the factors."""
        a, b, c, d, t = 16, 16, 192, 192, 517
        x = _mk(t, b * d, dtype=dtype, scale=1.0)
        if full:
            leaves = [_mk(a, b, dtype=dtype, scale=0.3), _mk(c, d, dtype=dtype)]
        else:
            leaves = [
                _mk(a, 4, dtype=dtype, scale=0.5),
                _mk(4, b, dtype=dtype, scale=0.5),
                _mk(c, 8, dtype=dtype, scale=0.3),
                _mk(8, d, dtype=dtype, scale=0.3),
            ]
        gy = _mk(t, a * c, dtype=dtype, scale=1.0)

        def factors(ls):
            if full:
                return (ls[0], None, None, ls[1], None, None, None)
            return (None, ls[0], ls[1], None, ls[2], ls[3], None)

        def run(ls, xx):
            return lokr_kron_bypass(xx, *factors(ls), scale=0.7, backend="triton")

        def eager_run(ls, xx):
            f = factors(ls)
            return f_lokr.kron_bypass(xx, *f[:6], scale=0.7, backend="torch")

        def grads(fn, dt):
            xx = x.to(dt).requires_grad_(True)
            ls = [v.to(dt).requires_grad_(True) for v in leaves]
            y = fn(ls, xx)
            y.backward(gy.to(dt))
            return [y.detach(), xx.grad, *(v.grad for v in ls)]

        ref = grads(eager_run, torch.float64)
        eager = grads(eager_run, dtype)
        names = ["y", "gx", *(f"g{i}" for i in range(len(leaves)))]
        got = grads(run, dtype)
        self.fail_on(
            [
                stats.judge(
                    f"entry {'full' if full else 'factored'} {dtype} {n}", g, e, r
                )
                for n, g, e, r in zip(names, got, eager, ref)
            ]
        )


GEMM_KERNELS = ("lokr.gemm_fwd", "lokr.gemm_dx", "lokr.gemm_gw2")


class _Unbuildable:
    """The broken variant: a kernel whose every launch fails as a build would."""

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            raise RuntimeError("constructed build failure")

        return launch


class TunedOpsRunTheKernel(unittest.TestCase):
    """On a cold table each GEMM-layout op times at least one kernel config that ran (a
    finite timing) and records no failure, whatever it then selects; a kernel that
    cannot build fails that check, is recorded in ``tuner.failed`` and logs one WARNING
    naming it."""

    def setUp(self):
        if not CUDA:
            self.skipTest("needs CUDA")
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        patches = [
            mock.patch.dict(os.environ, {"LYCORIS_KERNEL_CACHE_DIR": self.dir.name}),
            mock.patch.dict(tuner._chosen, clear=True),
            mock.patch.dict(tuner._untimed, clear=True),
            mock.patch.dict(tuner._disk, clear=True),
            mock.patch.dict(tuner.failed, clear=True),
            mock.patch.object(tuner, "record", []),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        prev = set_tuning(True)
        self.addCleanup(set_tuning, prev)
        torch.manual_seed(0)

    def _run_ops(self):
        for name in ("f16_attn", "f8"):
            a, b, c, d = SHAPES[name]
            for dtype in DTYPES:
                x, w1, w2, gy = _case(a, b, c, d, 1027, dtype)
                gemm.lokr_gemm_fwd(x, w1[None], w2[None], (0.7,))
                gemm.lokr_gemm_bwd(gy[None], x, w1[None], w2[None], (0.7,))
        torch.cuda.synchronize()

    def check_kernels_ran(self):
        ran = dict.fromkeys(GEMM_KERNELS, 0)
        for r in tuner.record:
            if r["key"][0] in ran:
                ran[r["key"][0]] += sum(
                    1 for c, t in r["timed"] if "impl" not in c and math.isfinite(t)
                )
        for name in GEMM_KERNELS:
            assert ran[name] > 0, f"{name}: no kernel config ran ({ran})"
            assert name not in tuner.failed, f"{name}: {tuner.failed.get(name)}"

    def test_each_op_runs_a_kernel_config(self):
        self._run_ops()
        self.check_kernels_ran()

    def test_unbuildable_kernel_is_caught_and_logged(self):
        with mock.patch.object(gemm, "_gemm_fwd_kernel", _Unbuildable()):
            with self.assertLogs("LyCORIS", "WARNING") as logs:
                self._run_ops()
        self.assertIn("constructed build failure", tuner.failed["lokr.gemm_fwd"])
        named = [m for m in logs.output if "lokr.gemm_fwd" in m]
        self.assertEqual(len(named), 1, logs.output)
        with self.assertRaises(AssertionError):
            self.check_kernels_ran()


class TuningGuards(unittest.TestCase):
    """The torch-agreement guard takes the compute dtype's tolerance, so an fp32
    reduction of bf16 products is judged as bf16 work; a key whose timing ran out of
    memory holds for the process and is not stored."""

    def test_agreement_uses_the_compute_dtype(self):
        ref = torch.randn(4096, dtype=torch.float32)
        got = ref + 3e-3 * ref.norm() / 64 * torch.randn(4096)
        self.assertIsNone(disagreement(got, ref, torch.bfloat16))
        self.assertIsNotNone(disagreement(got, ref))

    def test_out_of_memory_timing_is_not_stored(self):
        dev = mock.Mock(name="dev")
        dev.name = "probe-card"
        key = ("probe.kernel", ("bf16",), (1024,))
        plan = [{"BT": 1}, {"BT": 2}, {"impl": "torch"}]

        def measure(configs):
            for c in configs:
                if c.get("BT") == 2:
                    tuner.note_failure(c, "OutOfMemoryError: CUDA out of memory")
            return [1.0 if c.get("BT") == 1 else math.inf for c in configs]

        with tempfile.TemporaryDirectory() as root:
            with (
                mock.patch.dict(os.environ, {"LYCORIS_KERNEL_CACHE_DIR": root}),
                mock.patch.dict(tuner._chosen, clear=True),
                mock.patch.dict(tuner._disk, clear=True),
                mock.patch.dict(tuner.failed, clear=True),
                mock.patch("lycoris.kernels.tuner._table_path") as path,
            ):
                path.return_value = Path(root) / "t.json"
                prev = set_tuning(True)
                try:
                    picked = tuner._tune(dev, key, plan, measure)
                finally:
                    set_tuning(prev)
                self.assertEqual(picked, {"BT": 1})
                self.assertNotIn(key, tuner._disk.get(dev.name, {}))
                self.assertFalse(path.return_value.exists())
                ok = tuner._tune(dev, key, plan, lambda cs: [1.0] * len(cs))
                self.assertIn(key, tuner._disk[dev.name], "the negative stores")
                self.assertEqual(ok, {"BT": 1})


class ChecksFire(unittest.TestCase):
    """Each check above flags a constructed broken variant."""

    def setUp(self):
        if not CUDA:
            self.skipTest("needs CUDA")
        torch.manual_seed(0)

    def _fwd(self, scale=0.7, w1_t=False, drop_tail=False):
        a, b, c, d = SHAPES["rect"]
        dtype, t = torch.float16, 1027
        x, w1, w2, _ = _case(a, b, c, d, t, dtype)
        ref = _fp64(x, w1, w2, torch.zeros(t, a * c, device="cuda"), 0.7)[0]
        eager = _eager(x, w1, w2, 0.7)
        cf = configs(a, b)["diag"]["fwd"]
        w1k = w1.t().contiguous().reshape(a, b) if w1_t else w1
        got = gemm._fwd_run(cf, x, w1k[None], w2[None], (scale,), None, dtype)[
            0
        ].clone()
        if drop_tail:
            got[-1].zero_()
        return stats.judge("fwd negative", got, eager, ref)

    def test_correct_forward_passes(self):
        self.assertIsNone(self._fwd())

    def test_wrong_scale_fires(self):
        self.assertIsNotNone(self._fwd(scale=1.4))

    def test_wrong_w1_layout_fires(self):
        self.assertIsNotNone(self._fwd(w1_t=True))

    def test_dropped_token_fires(self):
        self.assertIsNotNone(self._fwd(drop_tail=True))

    def test_backward_checks_fire(self):
        a, b, c, d = SHAPES["f16_attn"]
        dtype, t = torch.bfloat16, 333
        x, w1, w2, gy = _case(a, b, c, d, t, dtype)
        _, rx, r1, r2 = _fp64(x, w1, w2, gy, 0.7)
        _, ex, e1, e2 = _eager_grads(x, w1, w2, gy, 0.7)
        cf = configs(a, b)["batch"]
        gx, gw1 = gemm._dx_run(cf["dx"], gy[None], x, w1[None], w2[None], (0.7,))
        gw2 = gemm._gw2_run(cf["gw2"], gy[None], x, w1[None], c, d, (0.7,))
        self.assertIsNone(stats.judge("gx ok", gx, ex, rx))
        self.assertIsNone(stats.judge("gw1 ok", gw1[0], e1, r1))
        self.assertIsNone(stats.judge("gw2 ok", gw2[0], e2, r2))
        self.assertIsNotNone(stats.judge("gx broken", gx * 1.3, ex, rx))
        self.assertIsNotNone(stats.judge("gw1 broken", gw1[0].t(), e1, r1))
        self.assertIsNotNone(stats.judge("gw2 broken", gw2[0] * 0.5, e2, r2))

    def test_stats_fire_on_nan(self):
        ref = torch.randn(64, dtype=torch.float64)
        bad = ref.float().clone()
        bad[3] = float("nan")
        self.assertIsNotNone(stats.judge("nan", bad, ref.float(), ref))


if __name__ == "__main__":
    unittest.main()
