import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def input_norm(x):
    return x.transpose(0, 1).norm(dim=1, keepdim=True).transpose(0, 1)


RANK = 1


w1 = torch.randn(128, 144) * math.sqrt(2/144)
w2 = torch.randn(128, 144) * math.sqrt(2/144)
diff_w = w2 - w1
u, s, vh = torch.linalg.svd(diff_w)
low_rank_diff = u[:, :RANK] @ torch.diag(s[:RANK]) @ vh[:RANK, :]
reconstruction1 = w1 + low_rank_diff

w1_norm = input_norm(w1)
w2_norm = input_norm(w2)

normed_w2 = w2 / w2_norm
normed_w1 = w1 / w1_norm
normed_diff_w = normed_w2 - normed_w1

u, s, vh = torch.linalg.svd(normed_diff_w)
low_rank_normed_diff = u[:, :RANK] @ torch.diag(s[:RANK]) @ vh[:RANK, :]
normed_reconstruction = (normed_w1 + low_rank_normed_diff)
reconstruction2 = normed_reconstruction / input_norm(normed_reconstruction) * w2_norm


print(F.mse_loss(reconstruction1, w2))
print(F.mse_loss(reconstruction2, w2))