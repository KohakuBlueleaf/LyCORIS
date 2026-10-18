"""Kernel tuning leaves the caller's random generators untouched.

A cold-table call that times its configs (the model-ranked tuner, its device
probe and the planner tuner) must leave the CPU and CUDA generator states
bit-identical; a draw from either generator is shown to be caught.

Usage:
    python -m unittest test.kernels.test_rng -v
"""

import os
import tempfile
import unittest
from unittest import mock

import torch

from lycoris.kernels import set_tuning
from lycoris.kernels.dispatch import fused_backends
from lycoris.kernels.ops import get_ops
from lycoris.kernels.plans import tune
from lycoris.kernels.tuner import device_info, model, tuner

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda", 0) if CUDA else None
TRITON = CUDA and "triton" in fused_backends(DEVICE)


def _states() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.get_rng_state(), torch.cuda.get_rng_state(DEVICE)


def _kept(fn) -> bool:
    """Whether fn leaves the CPU and CUDA generator states unchanged."""
    before = _states()
    fn()
    torch.cuda.synchronize(DEVICE)
    after = _states()
    return all(torch.equal(b, a) for b, a in zip(before, after))


class RngKept(unittest.TestCase):
    def setUp(self):
        if not TRITON:
            self.skipTest("needs CUDA and Triton")
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        patches = [
            mock.patch.dict(os.environ, {"LYCORIS_KERNEL_CACHE_DIR": self.dir.name}),
            mock.patch.dict(tune._CACHE, clear=True),
            mock.patch.dict(tune._UNTIMED, clear=True),
            mock.patch.object(tune, "_LOADED", False),
            mock.patch.dict(tuner._chosen, clear=True),
            mock.patch.dict(tuner._untimed, clear=True),
            mock.patch.dict(tuner._disk, clear=True),
            mock.patch.object(tuner, "record", []),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        prev = set_tuning(True)
        self.addCleanup(set_tuning, prev)
        # The device probe runs again, inside the checked call.
        model._measure.cache_clear()
        self.ops = get_ops("triton")

    def test_measuring_tune_keeps_generators(self):
        dtype = torch.float16
        mk = lambda *s: torch.randn(*s, device=DEVICE, dtype=dtype) * 0.1  # noqa: E731
        x, up, down = mk(200, 96), mk(80, 8), mk(8, 96)
        a, t, b = mk(72, 8), mk(8, 8, 9), mk(8, 56)
        with mock.patch.object(tune, "_time", wraps=tune._time) as timed:
            kept = _kept(
                lambda: (
                    self.ops.lora_bypass_fwd(x, up, down, gamma=0.5),
                    self.ops.lora_tucker_fwd(a, t, b, gamma=0.5),
                )
            )
        self.assertEqual(model._measure.cache_info().currsize, 1, "probe ran")
        self.assertTrue(tuner.record, "model-ranked tuner timed a key")
        self.assertGreater(timed.call_count, 0, "planner tuner timed a key")
        self.assertTrue(kept, "generator state moved during tuning")

    def test_check_catches_a_draw(self):
        self.assertFalse(_kept(lambda: torch.randn(4, device=DEVICE)))
        self.assertFalse(_kept(lambda: torch.randn(4)))
        self.assertTrue(_kept(lambda: torch.empty(4, device=DEVICE)))

    def test_probe_without_its_generator_is_caught(self):
        dev = device_info(DEVICE)
        self.assertTrue(_kept(lambda: model._measure(dev)))
        model._measure.cache_clear()
        randn = torch.randn

        def shared(*args, generator=None, **kwargs):
            return randn(*args, **kwargs)

        with mock.patch.object(torch, "randn", shared):
            self.assertFalse(_kept(lambda: model._measure(dev)))


if __name__ == "__main__":
    unittest.main()
