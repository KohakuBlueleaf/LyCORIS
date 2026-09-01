"""DyLoRA: LoCon rebuild on rank-sliced factors (runtime rank, no re-tune keys)."""

from .locon import locon_diff_weight


def dylora_diff_weight(down, up, rank=None, gamma=1.0, backend=None):
    if rank is not None:
        down = down[:rank]
        up = up[:, :rank]
    return locon_diff_weight(down, up, None, gamma, backend)
