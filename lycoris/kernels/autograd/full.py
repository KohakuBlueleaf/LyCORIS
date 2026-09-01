"""Full (native diff) module: fused merge add (bypass unsupported by design)."""

from .norms import AddScaledFn


def full_diff_weight(org_weight, diff, multiplier=1.0, backend=None):
    return AddScaledFn.apply(org_weight, diff, float(multiplier), backend)
