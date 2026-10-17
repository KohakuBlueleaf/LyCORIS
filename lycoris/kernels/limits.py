"""Scope bounds of the fused kernels that callers test before taking a fused backend."""

# Largest LoHa rank the fused hadamard backward kernels (merge and bypass) take.
MAX_HADA_RANK = 128
