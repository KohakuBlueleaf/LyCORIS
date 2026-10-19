"""The LoKr GEMM-layout kernels (triton.lokr.gemm_kernels) build without a GPU, each
compiled to a cubin for an explicit target: for sm_120 every kernel through every w1
shape and legal mix, fp16 and bf16 and both values of each flag across the cases; for
sm_89 and sm_86 the factor-16 w1 through each mix. A kernel that fails to build fails
here, before any GPU run; the witness is shown to fail on a constructed kernel that does
not build.

Usage:
    python -m unittest test.kernels.test_lokr_gemm_build -v
"""

import unittest

from lycoris.kernels.tuner.measure import pow2

try:
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler.errors import CompilationError

    from lycoris.kernels.triton.lokr import gemm_kernels as gk
except ImportError:
    triton = None

ARCHS = (120, 89, 86)
# (a, b): a factor-16 w1 (both mixes), a rectangular w1, a 48-wide w1, factor 4.
W1 = {"f16": (16, 16), "rect": (16, 8), "wide": (48, 48), "f8": (8, 8), "f4": (4, 4)}
CT = {"fp16": "*fp16", "bf16": "*bf16"}
# The cases sm_89 and sm_86 build (sm_120 builds every case): the factor-16 w1, each
# mix once.
OTHER_ARCHS = (["f16", "batch", "fp16"], ["f16", "diag", "bf16"])


def build(fn, ptrs: dict[str, str], consts: dict, sm: int, warps: int, stages: int):
    """Kernel `fn` compiled for CUDA sm `sm`: pointer args typed by `ptrs`, the
    constexprs from `consts`, `scale` fp32, every other argument i32."""
    names = fn.arg_names
    sig = {}
    for n in names:
        if n in consts:
            sig[n] = "constexpr"
        elif n in ptrs:
            sig[n] = ptrs[n]
        else:
            sig[n] = "fp32" if n == "scale" else "i32"
    src = ASTSource(
        fn, sig, constexprs={(names.index(n),): v for n, v in consts.items()}
    )
    return triton.compile(
        src,
        target=GPUTarget("cuda", sm, 32),
        options={"num_warps": warps, "num_stages": stages},
    )


def mixes(a: int, b: int) -> list[tuple[str, int]]:
    """(mix, BT) of every legal mix of w1 (a, b), BT as the GPU tests take it."""
    pa, pb = pow2(a), pow2(b)
    bt = max(1, 128 // max(pa, pb))
    out = [("diag", bt)]
    if min(pa, pb) >= 16:
        out.append(("batch", bt))
    return out


PAIRS = (("fp16", True), ("bf16", False))


def cases():
    """(label, fn, ptrs, consts) of every kernel, w1 shape and mix: the factor-16 w1 as
    fp16 with every flag on and as bf16 with every flag off, the other shapes in one
    of the two, in turn."""
    out, turn = [], 0
    for shape, (a, b) in W1.items():
        for mix, bt in mixes(a, b):
            pairs = PAIRS if shape == "f16" else (PAIRS[turn % 2],)
            turn += shape != "f16"
            for dt, flag in pairs:
                out += _kernels(f"{shape} {mix} {dt}", CT[dt], flag, mix, bt, a, b)
    return out


def _kernels(tag: str, ct: str, flag: bool, mix: str, bt: int, a: int, b: int):
    base = {"MIX": mix, "BT": bt, "PA": pow2(a), "PB": pow2(b)}
    base |= {"prec": "tf32", "group_scale": flag}
    acc = {"sg_ptr": "*fp32", "part_ptr": "*fp32"}
    fwd = {n: ct for n in ("x_ptr", "w1_ptr", "w2_ptr", "b_ptr", "y_ptr")}
    fwd["sg_ptr"] = "*fp32" if flag else ct
    dx = {n: ct for n in ("g_ptr", "x_ptr", "w1_ptr", "w2_ptr", "gx_ptr")}
    gw2 = {n: ct for n in ("g_ptr", "x_ptr", "w1_ptr")}
    return [
        (
            f"fwd {tag}",
            gk._gemm_fwd_kernel,
            fwd,
            base | {"has_base": not flag, "BN": 64, "BK": 32},
        ),
        (
            f"dx {tag}",
            gk._gemm_dx_kernel,
            dx | acc,
            base | {"want_gw1": True, "BD": 64, "BK": 32},
        ),
        (f"gw2 {tag}", gk._gemm_gw2_kernel, gw2 | acc, base | {"BC": 64, "BD": 64}),
    ]


if triton is not None:

    @triton.jit
    def _plain_local_rows(
        out_ptr, PA: tl.constexpr, PB: tl.constexpr, transpose: tl.constexpr
    ):
        """The broken variant: a row count held in a plain local, not a constexpr."""
        if transpose:
            rows = PA
        else:
            rows = PB
        r = tl.arange(0, rows)
        tl.store(out_ptr + r, r)

    @triton.jit
    def _constexpr_rows(
        out_ptr, PA: tl.constexpr, PB: tl.constexpr, transpose: tl.constexpr
    ):
        """The same kernel with the row count a constexpr."""
        rows: tl.constexpr = PA if transpose else PB
        r = tl.arange(0, rows)
        tl.store(out_ptr + r, r)


class GemmKernelsBuild(unittest.TestCase):
    def setUp(self):
        if triton is None:
            self.skipTest("needs Triton")

    def test_every_kernel_builds_for_each_arch(self):
        failed = []
        for sm in ARCHS:
            todo = [c for c in cases() if sm == 120 or c[0].split()[1:] in OTHER_ARCHS]
            for label, fn, ptrs, consts in todo:
                try:
                    build(fn, ptrs, consts, sm, 4, 2)
                except Exception as e:  # noqa: BLE001 - every failure is listed
                    failed.append(
                        f"sm_{sm} {label}: {type(e).__name__}: {str(e)[:300]}"
                    )
        self.assertFalse(failed, "\n".join(failed))

    def test_witness_fails_on_a_kernel_that_does_not_build(self):
        consts = {"PA": 16, "PB": 32, "transpose": True}
        build(_constexpr_rows, {"out_ptr": "*i32"}, consts, 120, 4, 2)
        with self.assertRaises(CompilationError):
            build(_plain_local_rows, {"out_ptr": "*i32"}, consts, 120, 4, 2)


if __name__ == "__main__":
    unittest.main()
