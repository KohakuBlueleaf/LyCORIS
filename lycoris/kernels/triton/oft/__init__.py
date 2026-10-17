from .apply import oft_bwd, oft_fwd

# Largest block of the fused apply.
MAX_BLOCK = 32

__all__ = ["MAX_BLOCK", "oft_bwd", "oft_fwd"]
