"""Backend op-set resolution (importlib so optional backends stay optional)."""

import functools
import importlib

from .dispatch import FUSED, resolve_backend


@functools.cache
def get_ops(backend: str | None = None):
    name = resolve_backend(backend)
    if name not in FUSED:
        raise RuntimeError(
            f"the {name} backend has no fused ops; use the functional API, "
            f"which dispatches it through lycoris.kernels.select"
        )
    return importlib.import_module(f"lycoris.kernels.{name}.ops")
