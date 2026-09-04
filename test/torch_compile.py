import unittest

import torch
import torch.nn as nn

from lycoris.functional.lokr import _apply_factor_cap, kron_bypass
from lycoris.kernels.select import choose, compiled
from lycoris.modules.dylora import DyLoraModule


class TorchCompileCompatibility(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        compiled.cache_clear()
        _apply_factor_cap.cache_clear()

    def test_outer_compile_owns_backend_selection(self):
        def fn(x):
            if choose((x,), backend="compile") != "torch":
                raise AssertionError("nested backend selected")
            return x + 1

        x = torch.randn(2)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        torch.testing.assert_close(actual, x + 1)

    def test_outer_compile_does_not_start_per_op_compile(self):
        x = torch.randn(2, 16, requires_grad=True)
        w1 = torch.randn(2, 2, requires_grad=True)
        w2 = torch.randn(4, 8, requires_grad=True)

        def fn(x, w1, w2):
            return kron_bypass(
                x,
                w1,
                None,
                None,
                w2,
                None,
                None,
                scale=0.25,
                backend="compile",
            )

        expected = fn(x, w1, w2)
        compiled.cache_clear()
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x, w1, w2)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(compiled.cache_info().misses, 0)

        grad = torch.randn_like(actual)
        expected_grads = torch.autograd.grad(expected, (x, w1, w2), grad)
        actual_grads = torch.autograd.grad(actual, (x, w1, w2), grad)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_dylora_random_rank_is_fullgraph_traceable(self):
        for bypass_mode in (False, True):
            with self.subTest(bypass_mode=bypass_mode):
                base = nn.Linear(16, 16)
                adapter = DyLoraModule(
                    "test",
                    base,
                    lora_dim=8,
                    block_size=4,
                    bypass_mode=bypass_mode,
                )
                adapter.apply_to()
                with torch.no_grad():
                    for parameter in adapter.parameters():
                        parameter.normal_(std=0.1)
                x = torch.randn(2, 16, requires_grad=True)

                references = []
                for rank in (0, 4):
                    if bypass_mode:
                        down, up, _ = adapter.get_weight(rank)
                        delta = nn.functional.linear(nn.functional.linear(x, down), up)
                    else:
                        diff = adapter.get_diff_weight(rank=rank)[0]
                        delta = nn.functional.linear(x, diff)
                    references.append(adapter.org_forward(x) + delta)

                output = torch.compile(base, backend="eager", fullgraph=True)(x)
                grads = torch.autograd.grad(output.sum(), (x, *adapter.parameters()))

                self.assertEqual(output.shape, (2, 16))
                self.assertTrue(
                    any(
                        torch.allclose(output, reference, rtol=1e-5, atol=1e-6)
                        for reference in references
                    )
                )
                for grad in grads:
                    self.assertIsNotNone(grad)


if __name__ == "__main__":
    unittest.main()
