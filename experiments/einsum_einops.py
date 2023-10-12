import torch
from einops import einsum, rearrange


diag_oft = torch.zeros(3, 4, 4)
weight = torch.randn(12, 16)
eye = torch.eye(4).unsqueeze(0)

weight = rearrange(weight, '(k n) ... -> k n ...', k=3, n=4)
result = torch.einsum('k n m, k n ... -> k m ...', diag_oft+eye, weight)
print(result)