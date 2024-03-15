import torch
import torch.nn as nn

from .base import ModuleCustomSD


class IA3Module(ModuleCustomSD):
    """
    Hadamard product Implementaion for Low Rank Adaptation
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        train_on_input=False,
        **kwargs
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.tucker = False

        self.shape = org_module.weight.shape
        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            if train_on_input:
                train_dim = in_dim
            else:
                train_dim = out_dim
            self.weight = nn.Parameter(torch.empty(1, train_dim, 1, 1))
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            if train_on_input:
                train_dim = in_dim
            else:
                train_dim = out_dim

            self.weight = nn.Parameter(torch.empty(train_dim))

        # Need more experiences on init method
        torch.nn.init.constant_(self.weight, 0)

        self.multiplier = multiplier
        self.org_forward = None
        self.org_module = [org_module]  # remove in applying
        self.grad_ckpt = False
        self.train_input = train_on_input
        self.register_buffer("on_input", torch.tensor(int(train_on_input)))

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    @torch.enable_grad()
    def forward(self, x):
        if self.train_input:
            x = x * (1 + self.weight * self.multiplier)
        out = self.org_forward(x)
        dtype = out.dtype
        if not self.train_input:
            out = out * (1 + self.weight * self.multiplier)
            out = out.to(dtype)
        return out
