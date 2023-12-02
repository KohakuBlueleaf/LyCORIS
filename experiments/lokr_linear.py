import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from einops import rearrange


def lokr_linear(h, a, b, c):
    vq = a.size(1)
    uq = c.size(1)

    h_in_group = rearrange(h, "b ... (uq vq) -> b ... uq vq", uq=uq, vq=vq)
    ha = F.linear(h_in_group, a)
    hb = F.linear(ha, b)

    h_cross_group = hb.transpose(-1, -2)
    hc = F.linear(h_cross_group, c)

    h = rearrange(hc, "b ... vp up -> b ... (up vp)")

    return h


def lokr_rebuild(h, a, b, c):
    rebuild_weight = torch.kron(c, b @ a)
    return F.linear(h, rebuild_weight)


h = torch.randn(3, 77, 768)
a = torch.randn(2, 64)
b = torch.randn(32, 2)
c = torch.randn(24, 12)

result1 = lokr_linear(h, a, b, c)
result2 = lokr_rebuild(h, a, b, c)

print(F.mse_loss(result1, result2))
