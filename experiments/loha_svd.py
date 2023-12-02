import torch
import torch.nn.functional as F

# constants
loha_dim = 16
lora_dim = loha_dim * 2  # same parameter count
weight_rank = 256

# make norm=1 low rank weight for test
test_weight = torch.randn(1280, weight_rank) @ torch.randn(weight_rank, 1280)
test_weight = test_weight / torch.norm(test_weight)
sqrt_weight = torch.sqrt(torch.abs(test_weight))

# low rank aproximation
e, s, v = torch.svd(test_weight)
a = e[:, :lora_dim]
b = torch.diag(s[:lora_dim]) @ v[:, :lora_dim].T
diff = test_weight - a @ b

print(
    torch.mean(diff**2),
    torch.mean(diff),
    torch.max(torch.abs(diff)),
    torch.min(torch.abs(diff)),
)


# hadamard product
a = sqrt_weight * torch.sign(test_weight)
b = sqrt_weight

e, s, v = torch.svd(a)
a = e[:, :loha_dim] @ torch.diag(s[:loha_dim]) @ v[:, :loha_dim].T

e, s, v = torch.svd(b)
b = e[:, :loha_dim] @ torch.diag(s[:loha_dim]) @ v[:, :loha_dim].T
diff = test_weight - a * b

print(
    torch.mean(diff**2),
    torch.mean(diff),
    torch.max(torch.abs(diff)),
    torch.min(torch.abs(diff)),
)
