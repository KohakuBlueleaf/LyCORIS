"""Fused Triton/TileLang kernels for LyCORIS algorithms.

The functional and module APIs reach these through ``lycoris.kernels.select``,
which picks a backend per call; the per-algorithm autograd functions in
``lycoris.kernels.autograd`` and the dispatch in ``lycoris.kernels.dispatch``
are the entry points for calling one directly. Docs live in ``docs/kernels/``
and the internal working notes in ``.internal/fused-kernels/``.
"""

from .dispatch import available_backends, fused_backends, resolve_backend

__all__ = ["available_backends", "fused_backends", "resolve_backend"]
