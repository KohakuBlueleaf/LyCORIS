"""The fused kernels under an enclosing torch.compile(fullgraph=True) graph.

``CompileFullgraph``: each family's ``lycoris.kernels.autograd`` entry point,
pinned to a fused backend, must (1) trace as one graph, (2) actually run the
fused op set (counted at the library boundary), and (3) match the eager fused
call in value and in every input gradient.

``CompileUnpinnedDynamic``: the unpinned functional calls, which hand an
enclosing compile the torch body, trace as one graph over a token sweep.

Usage:
    python -m unittest test.kernels.test_compile -v
"""

import unittest
from unittest import mock

import torch
from parameterized import parameterized
from torch._dynamo.eval_frame import _debug_get_cache_entry_list

from lycoris.functional import diag_oft as f_oft
from lycoris.functional import locon as f_locon
from lycoris.functional import loha as f_loha
from lycoris.functional import lokr as f_lokr
from lycoris.kernels import library
from lycoris.kernels.autograd.diag_oft import diag_oft_bypass_diff
from lycoris.kernels.autograd.dora import apply_dora
from lycoris.kernels.autograd.locon import locon_bypass_diff, locon_diff_weight
from lycoris.kernels.autograd.loha import loha_bypass_diff
from lycoris.kernels.autograd.lokr import lokr_kron_bypass
from lycoris.kernels.dispatch import fused_backends

from . import judge, refs

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda", 0) if CUDA else None
BACKENDS = list(fused_backends(DEVICE)) if CUDA else []
DTYPE = (
    torch.bfloat16
    if CUDA and torch.cuda.get_device_capability(0)[0] >= 8
    else torch.float16
)


def _mk(*shape):
    return (torch.randn(*shape, device="cuda") * 0.1).to(DTYPE)


def _cases():
    t, o, i, r = 96, 80, 64, 8

    def lora(be):
        x, d, u = _mk(t, i), _mk(r, i), _mk(o, r)
        fn = lambda x, d, u: locon_bypass_diff(  # noqa: E731
            x, d, u, gamma=0.5, backend=be
        )
        return fn, (x, d, u)

    def loha(be):
        x = _mk(t, i)
        w = (_mk(r, i), _mk(o, r), _mk(r, i), _mk(o, r))
        fn = lambda x, a, b, c, d: loha_bypass_diff(  # noqa: E731
            x, a, b, c, d, gamma=0.5, backend=be
        )
        return fn, (x, *w)

    def lokr(be):
        x, w1, w2 = _mk(t, i), _mk(8, 8), _mk(10, 8)
        fn = lambda x, w1, w2: lokr_kron_bypass(  # noqa: E731
            x, w1, None, None, w2, None, None, scale=0.5, backend=be
        )
        return fn, (x, w1, w2)

    def oft(be):
        y, blocks = _mk(t, o), _mk(10, 8, 8)
        fn = lambda y, b: diag_oft_bypass_diff(y, b, backend=be)  # noqa: E731
        return fn, (y, blocks)

    def dora(be):
        w = _mk(o, i)
        dscale = (w.float().norm(dim=1, keepdim=True) * 1.3).to(DTYPE)
        fn = lambda w, s: apply_dora(w, s, 1.0, True, backend=be)  # noqa: E731
        return fn, (w, dscale)

    def lora_merge(be):
        d, u = _mk(r, i), _mk(o, r)
        fn = lambda d, u: locon_diff_weight(d, u, gamma=0.5, backend=be)  # noqa: E731
        return fn, (d, u)

    return {
        "lora": lora,
        "loha": loha,
        "lokr": lokr,
        "oft": oft,
        "dora": dora,
        "lora_merge": lora_merge,
    }


CASES = [(name, be) for name in _cases() for be in BACKENDS]


class CompileFullgraph(unittest.TestCase):
    def setUp(self):
        if not CUDA or not BACKENDS:
            self.skipTest("needs CUDA and a fused backend")
        torch.manual_seed(0)
        torch._dynamo.reset()

    @parameterized.expand(CASES, skip_on_empty=True)
    def test_family(self, name, backend):
        fn, inputs = _cases()[name](backend)
        leaves_e = [t.clone().requires_grad_(True) for t in inputs]
        leaves_c = [t.clone().requires_grad_(True) for t in inputs]
        out_e = fn(*leaves_e)
        g = torch.randn_like(out_e)
        grads_e = torch.autograd.grad(out_e, leaves_e, g)

        calls = []
        real = library.ops_for

        def counting(be, device):
            calls.append(be)
            return real(be, device)

        compiled = torch.compile(fn, fullgraph=True)
        with mock.patch.object(library, "ops_for", counting):
            out_c = compiled(*leaves_c)
            grads_c = torch.autograd.grad(out_c, leaves_c, g)
        self.assertGreater(len(calls), 0, f"{name}: fused op never ran under compile")
        tol = 2e-2
        self.assertLess(_rel(out_c, out_e), tol, f"{name}: forward")
        for k, (a, b) in enumerate(zip(grads_c, grads_e)):
            self.assertLess(_rel(a, b), tol, f"{name}: grad {k}")


def _bypass(name, t):
    """(unpinned fn, operands) of one bypass family at token count t."""
    o, i, r = 80, 64, 8
    if name == "lora":
        x, d, u = _mk(t, i), _mk(r, i), _mk(o, r)
        return (
            lambda x, d, u: f_locon.bypass_forward_diff(x, None, d, u, None, gamma=0.5)
        ), (x, d, u)
    if name == "loha":
        w = (_mk(r, i), _mk(o, r), _mk(r, i), _mk(o, r))
        return (
            lambda x, a, b, c, d: f_loha.bypass_forward_diff(
                x, None, a, b, c, d, None, None, gamma=0.5
            )
        ), (_mk(t, i), *w)
    if name == "lokr":
        return (
            lambda x, w1, w2: f_lokr.kron_bypass(
                x, w1, None, None, w2, None, None, scale=0.5
            )
        ), (_mk(t, i), _mk(8, 8), _mk(10, 8))
    return (lambda y, b: f_oft.bypass_forward_diff(None, y, b, None)), (
        _mk(t, o),
        _mk(10, 8, 8),
    )


UNPINNED = [(n,) for n in ("lora", "loha", "lokr", "oft")]


class CompileUnpinnedDynamic(unittest.TestCase):
    """Unpinned functional calls (every LyCORIS module's default) under
    torch.compile(dynamic=True, fullgraph=True) over a token sweep: one graph
    for the sweep, and the compiled call meets the eager body's precision bar
    at every token count."""

    def setUp(self):
        if not CUDA:
            self.skipTest("needs CUDA")
        torch.manual_seed(0)
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()

    @parameterized.expand(UNPINNED, skip_on_empty=True)
    def test_sweep(self, name):
        fn, first = _bypass(name, 96)
        fn(*[x.clone().requires_grad_(True) for x in first])
        compiled = torch.compile(fn, backend="aot_eager", dynamic=True, fullgraph=True)
        for t in (96, 160, 224):
            inputs = first if t == 96 else _bypass(name, t)[1]
            errs, bar = _errors(name, compiled, inputs)
            ok = judge.passes(errs, bar)
            self.assertTrue(ok, f"{name} t={t}: {errs} vs eager {bar}")
        self.assertEqual(len(_debug_get_cache_entry_list(fn.__code__)), 1)


def _errors(name, compiled, inputs):
    """(compiled, eager) errors of output and every gradient against an fp64
    run of the eager body, one seed gradient."""
    leaves = [x.clone().requires_grad_(True) for x in inputs]
    out = compiled(*leaves)
    g = torch.randn_like(out)
    got = [out.detach(), *torch.autograd.grad(out, leaves, g)]
    ref_leaves = [x.detach().double().requires_grad_(True) for x in inputs]
    ref_out = _ref_body(name, ref_leaves)
    ref = [ref_out.detach(), *torch.autograd.grad(ref_out, ref_leaves, g.double())]
    eager_leaves = [x.clone().requires_grad_(True) for x in inputs]
    e_out = _torch_body(name, eager_leaves)
    eager = [e_out.detach(), *torch.autograd.grad(e_out, eager_leaves, g)]
    return judge.errors(got, ref), judge.errors(eager, ref)


def _torch_body(name, leaves):
    """The family's eager body (the bar's reference in the working dtype)."""
    if name == "lora":
        x, d, u = leaves
        return f_locon.bypass_forward_diff(
            x, None, d, u, None, gamma=0.5, backend="torch"
        )
    if name == "loha":
        x, a, b, c, d = leaves
        # the eager hadamard Function saves its scale, so it takes a tensor
        gamma = torch.tensor(0.5, device=x.device, dtype=x.dtype)
        return f_loha.bypass_forward_diff(
            x, None, a, b, c, d, None, None, gamma=gamma, backend="torch"
        )
    if name == "lokr":
        x, w1, w2 = leaves
        return f_lokr.kron_bypass(
            x, w1, None, None, w2, None, None, scale=0.5, backend="torch"
        )
    y, b = leaves
    return f_oft.bypass_forward_diff(None, y, b, None, backend="torch")


def _ref_body(name, leaves):
    """The fp64 reference: the eager body, or refs' Diag-OFT (eager inverts the
    Cayley factor in fp32)."""
    if name == "oft":
        y, b = leaves
        return refs.bd_fused(b, y, None, 1.0, True, False)
    return _torch_body(name, leaves)


def _rel(a, b) -> float:
    a, b = a.double(), b.double()
    return ((a - b).abs().max() / (b.abs().max() + 1e-12)).item()


if __name__ == "__main__":
    unittest.main()
