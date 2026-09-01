"""Shared per-case measurement: four arms x (fwd, fwd+bwd) x vram x error.

An arm spec is {"fwd": fn, "fwdbwd": fn|None, "out": fn, "leaves": [...]} —
``out`` returns the forward result for the error column; ``fwdbwd`` clears
its leaves' grads itself and must leave them populated so the backward is
provably run once before timing.
"""

import torch

from .harness import assert_backward_ran, interleave, peak_vram, rel_err, ulp_err


def make_arm(fn, tensors, grad):
    """One arm over ``fn(*tensors)``; the fwd+bwd closure clears its leaves
    every iteration, since .backward() accumulates and that read-modify-write
    tax is proportional to the gradient bytes an arm owns — it does not cancel
    in a ratio between two arms."""
    leaves = [t.clone().requires_grad_(True) for t in tensors]

    def fwdbwd():
        for leaf in leaves:
            leaf.grad = None
        fn(*leaves).backward(grad)

    return {
        "fwd": lambda: fn(*tensors),
        "fwdbwd": fwdbwd,
        "out": lambda: fn(*tensors),
        "leaves": leaves,
    }


def measure_case(
    family: str,
    case: str,
    params: dict,
    arms: dict,
    ref64: torch.Tensor,
    logical_bytes: float,
    logical_flops: float,
    dram_gbps: float,
    dtype=torch.float16,
    ulp_mode: str = "rms",
) -> list[dict]:
    rows = []
    fwd_arms = {n: a["fwd"] for n, a in arms.items()}
    fwd = interleave(fwd_arms)
    bwd_arms = {n: a["fwdbwd"] for n, a in arms.items() if a.get("fwdbwd")}
    for name, spec in arms.items():
        if spec.get("fwdbwd"):
            spec["fwdbwd"]()
            assert_backward_ran(spec["leaves"])
    bwd = interleave(bwd_arms) if bwd_arms else {}
    for name, spec in arms.items():
        ms, hms, gms = fwd[name]
        fb = bwd.get(name, (None, None, None))
        got = spec["out"]()
        row = {
            "family": family,
            "case": case,
            "arm": name,
            **params,
            "fwd_ms": ms,
            "fwd_host_ms": hms,
            "fwd_dev_ms": gms,
            "fwdbwd_ms": fb[0],
            "fwdbwd_host_ms": fb[1],
            "fwdbwd_dev_ms": fb[2],
            "eff_gbps": logical_bytes / ms * 1e3 / 1e9,
            "eff_tflops": logical_flops / ms * 1e3 / 1e12,
            "pct_bw": logical_bytes / ms * 1e3 / 1e9 / dram_gbps * 100,
            "vram_fwd": peak_vram(spec["fwd"]),
            "vram_fwdbwd": peak_vram(spec["fwdbwd"]) if spec.get("fwdbwd") else None,
            "rel_err": rel_err(got, ref64),
            "ulp": ulp_err(got, ref64, dtype, ulp_mode),
        }
        # A rate whose wall is mostly host dispatch is a floor, not a
        # measurement; the flag rides on the row so the figure can ring it.
        row["host_share"] = hms / ms if ms else float("nan")
        row["host_bound"] = bool(hms >= ms / 2)
        rows.append(row)
        print(
            f"{family:9s} {case:22s} {name:9s} fwd={row['fwd_ms']:.3f}ms "
            f"(host {100 * row['host_share']:.0f}%, dev {gms:.3f}ms) "
            f"fb={row['fwdbwd_ms'] or float('nan'):.3f}ms "
            f"bw={row['pct_bw']:.0f}% ulp={row['ulp']:.2f}"
        )
    return rows


def compiled(fn):
    """A compile arm that is actually compiled at every case.

    ``dynamic=False`` re-specializes per shape on one shared code object, and
    the cases here exceed the 8-recompile budget — past which Dynamo stops
    compiling the frame and the arm silently measures eager. ``dynamic=None``
    (automatic) plus a per-case cache reset keeps every case compiled.
    """
    torch._dynamo.reset()
    return torch.compile(fn, dynamic=None)


def assert_compiled(fn, *args) -> None:
    """Refuse a compile arm that fell back: a bench must not report eager's
    speed under compile's name."""
    import torch._dynamo as dynamo

    explain = dynamo.explain(fn)(*args)
    if explain.graph_count == 0:
        raise RuntimeError("compile arm produced no graph — it fell back to eager")
