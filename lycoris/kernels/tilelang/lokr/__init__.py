import math

from .apply import lokr_bypass_fwd
from .grad import lokr_bypass_bwd
from .kron import lokr_full_merge_bwd
from .merge import lokr_full_merge_fwd, lokr_merge_bwd, lokr_merge_fwd

# The apply op covers every factor size (the torch composition past its tiles).
MAX_KRON_FACTOR = math.inf

__all__ = [
    "MAX_KRON_FACTOR",
    "lokr_bypass_bwd",
    "lokr_bypass_fwd",
    "lokr_full_merge_bwd",
    "lokr_full_merge_fwd",
    "lokr_merge_bwd",
    "lokr_merge_fwd",
]
