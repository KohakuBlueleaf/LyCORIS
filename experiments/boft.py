from math import log2, floor

import torch
from einops import rearrange


def pow_of_2(value):
    while value > 1:
        if value & 1:
            return False
        value = value >> 1
    return True


# BOFT(m, b)
m = -1
b = 10
dim = 320

assert b % 2 == 0
assert pow_of_2(dim // b)

real_b = b // 2
if m == -1:
    m = floor(log2(dim / real_b))
print(m, b, dim)
assert dim >= 2**m * real_b


# inp is the W in Y = WX + B
# we use identity matrix to check how much element in final weight (BOFT(m, b)) is non zero
# dim, dim | simulate the weight of linear layer ↓
inp = torch.eye(dim, dim).float().transpose(-1, -2)
w = torch.randn(m, dim // b, b, b)  # Should be orthgonal matrixies


for i in range(m):
    inp = rearrange(inp, "(c g k) ... -> (c k g) ...", g=2, k=2**i * real_b)
    inp = rearrange(inp, "(d b) ... -> d b ...", b=b)
    inp = torch.einsum("b i j, b j ... -> b i ...", w[i], inp)
    inp = rearrange(inp, "d b ... -> (d b) ...")
    inp = rearrange(inp, "(c k g) ... -> (c g k) ...", g=2, k=2**i * real_b)

print(torch.count_nonzero(inp).item(), inp.shape.numel())


# inp here is the (WX) in Y = WX + B
# we use identity matrix to check how much element in final weight (BOFT(m, b)) is non zero
inp = torch.eye(77, dim).float().transpose(-1, -2)
inp = inp.repeat(4, 1, 1)  # 4, 77, dim | simulate the input
w = torch.randn(m, dim // b, b, b)  # Should be orthgonal matrixies


for i in range(m):
    inp = rearrange(inp, "... (c g k) d -> ...(c k g) d", g=2, k=2**i * real_b)
    inp = rearrange(inp, "... (k b) d -> ... k b d", b=b)
    inp = w[i] @ inp
    inp = rearrange(inp, "... k b d -> ... (k b) d")
    inp = rearrange(inp, "... (c k g) d -> ...(c g k) d", g=2, k=2**i * real_b)

print(torch.count_nonzero(inp).item(), inp.shape.numel())
