"""Fused Triton/TileLang kernels for LyCORIS algorithms.

The functional and module APIs reach these through ``lycoris.kernels.select``,
which picks a backend per call; the per-algorithm autograd functions in
``lycoris.kernels.autograd`` and the dispatch in ``lycoris.kernels.dispatch``
are the entry points for calling one directly. ``set_tuning`` turns config
timing on or off for the process. Docs live in ``docs/kernels/``.
"""

from .dispatch import available_backends, fused_backends, resolve_backend
from .plans.tune import set_tuning, tuning_enabled

__all__ = [
    "available_backends",
    "fused_backends",
    "resolve_backend",
    "set_tuning",
    "tuning_enabled",
]
