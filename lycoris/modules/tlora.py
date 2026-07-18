"""
T-LoRA: Timestep-dependent LoRA with SVD-based orthogonal initialization.

Based on the paper "T-LoRA: Timestep-aware LoRA for Diffusion Models"
Key features:
1. SVD-based orthogonal initialization from original weights or random matrix
2. Learnable singular values (lambda_layer)
3. Timestep-dependent rank masking (set externally via set_timestep_mask)
4. Residual subtraction from base state for zero-shot preservation

The mask is applied as: output = P(Q(x) * λ * mask) - P_base(Q_base(x) * λ_base * mask)
This ensures that when weights haven't changed from init, the output contribution is zero.
"""

import gc
import math
from functools import cache
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..logging import logger


SigType = Literal["principal", "last", "middle"]

# Thread-local storage for timestep mask (set by training loop)
_timestep_mask_storage: dict[int, torch.Tensor] = {}


def set_timestep_mask(mask: torch.Tensor, group_id: int = 0) -> None:
    """
    Set the timestep mask for T-LoRA modules.

    Call this before the forward pass with the appropriate mask for the current timestep.
    The mask should be shape (1, rank) with 1s for active ranks and 0s for masked ranks.

    Args:
        mask: Binary mask tensor of shape (1, max_rank) or (batch, max_rank)
        group_id: Optional group ID for multi-network scenarios
    """
    _timestep_mask_storage[group_id] = mask


def get_timestep_mask(group_id: int = 0) -> Optional[torch.Tensor]:
    """Get the current timestep mask, or None if not set."""
    return _timestep_mask_storage.get(group_id, None)


def clear_timestep_mask(group_id: int = 0) -> None:
    """Clear the timestep mask after forward pass."""
    _timestep_mask_storage.pop(group_id, None)


def compute_timestep_mask(
    timestep: int,
    max_timestep: int,
    max_rank: int,
    min_rank: int = 1,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Compute a binary rank mask based on timestep.

    At high noise (high timestep): fewer ranks active (structure-level adaptation)
    At low noise (low timestep): more ranks active (detail-level adaptation)

    Args:
        timestep: Current denoising timestep
        max_timestep: Maximum timestep (e.g., 1000)
        max_rank: Maximum rank (all ranks active at t=0)
        min_rank: Minimum rank (active even at t=max_timestep)
        alpha: Scaling exponent (1.0 = linear, >1 = more aggressive at high noise)

    Returns:
        Binary mask of shape (1, max_rank)
    """
    r = int(((max_timestep - timestep) / max_timestep) ** alpha * (max_rank - min_rank)) + min_rank
    r = min(r, max_rank)  # Clamp to max_rank
    mask = torch.zeros((1, max_rank))
    mask[:, :r] = 1.0
    return mask


@cache
def log_tlora_init():
    return logger.info(
        "T-LoRA: Using SVD-based orthogonal initialization with timestep-dependent rank masking"
    )


class TLoraModule(LycorisBaseModule):
    """
    T-LoRA module with SVD-orthogonal initialization and timestep-dependent rank masking.

    The weight delta is computed as:
        ΔW = P @ diag(λ * mask) @ Q - P_base @ diag(λ_base * mask) @ Q_base

    Where P, Q are orthogonal matrices from SVD, λ are learnable singular values,
    and mask is the timestep-dependent binary mask.
    """

    name = "tlora"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = [
        "q_layer.weight",
        "p_layer.weight",
        "lambda_layer",
        "alpha",
    ]
    weight_list_det = ["lambda_layer"]  # Unique identifier for T-LoRA

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        rank_dropout_scale: bool = False,
        bypass_mode: bool = None,
        sig_type: SigType = "principal",
        use_data_init: bool = True,
        mask_group_id: int = 0,
        **kwargs,
    ):
        """
        Initialize T-LoRA module.

        Args:
            lora_name: Unique name for this LoRA module
            org_module: Original module to wrap
            multiplier: Output multiplier
            lora_dim: Rank of the LoRA decomposition
            alpha: Alpha scaling factor
            dropout: Dropout probability
            rank_dropout: Rank-wise dropout probability
            module_dropout: Module-level dropout probability
            rank_dropout_scale: Whether to scale by dropout rate
            bypass_mode: Use bypass forward mode
            sig_type: Which singular vectors to use ("principal", "last", "middle")
            use_data_init: If True, use SVD of original weights; if False, use random matrix
            mask_group_id: Group ID for timestep mask lookup
        """
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            dropout,
            rank_dropout,
            module_dropout,
            rank_dropout_scale,
            bypass_mode,
        )

        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in T-LoRA algo.")

        self.lora_dim = lora_dim
        self.sig_type = sig_type
        self.use_data_init = use_data_init
        self.mask_group_id = mask_group_id

        log_tlora_init()

        # Determine dimensions based on module type
        if self.module_type.startswith("conv"):
            self.isconv = True
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding

            # For convolutions, we work with reshaped weights
            # Original weight shape: (out_channels, in_channels, *kernel_size)
            # We treat it as (out_dim, in_dim * prod(kernel_size)) for SVD
            self.conv_shape = (out_dim, in_dim, *k_size)
            flat_in_dim = in_dim * math.prod(k_size)

            # Q projects from input space to rank space
            # For conv, we use 1x1 convs to avoid kernel complexity in LoRA
            self.q_layer = self.module(in_dim, lora_dim, 1, bias=False)
            self.p_layer = self.module(lora_dim, out_dim, 1, bias=False)

            # Store original conv params for the actual forward
            self.down_op = self.op
            self.up_op = self.op
            self.kw_dict_down = {
                "stride": stride,
                "padding": padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups,
            }
            self.kw_dict_up = {
                "stride": (1,) * len(k_size),
                "padding": (0,) * len(k_size),
                "dilation": (1,) * len(k_size),
                "groups": 1,
            }
        else:
            self.isconv = False
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            flat_in_dim = in_dim

            self.q_layer = nn.Linear(in_dim, lora_dim, bias=False)
            self.p_layer = nn.Linear(lora_dim, out_dim, bias=False)
            self.down_op = F.linear
            self.up_op = F.linear

        # Learnable singular values
        self.lambda_layer = nn.Parameter(torch.ones(1, lora_dim))

        # SVD-based orthogonal initialization
        self._initialize_from_svd(org_module, in_dim, out_dim, flat_in_dim)

        # Store frozen base state for residual subtraction
        self.register_buffer("base_q", self.q_layer.weight.data.clone())
        self.register_buffer("base_p", self.p_layer.weight.data.clone())
        self.register_buffer("base_lambda", self.lambda_layer.data.clone())

        # Alpha scaling (same as standard LoRA)
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # Dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def _initialize_from_svd(
        self,
        org_module: nn.Module,
        in_dim: int,
        out_dim: int,
        flat_in_dim: int,
    ) -> None:
        """Initialize Q and P layers using SVD."""
        if self.use_data_init:
            # SVD of original weights (data-dependent init)
            weight = org_module.weight.data.float()
            if self.isconv:
                # Reshape conv weight to 2D: (out_dim, in_dim * kernel_size)
                weight = weight.reshape(out_dim, -1)

            u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        else:
            # SVD of random matrix (data-independent init)
            rand_weight = torch.normal(
                mean=0,
                std=1 / self.lora_dim,
                size=(out_dim, flat_in_dim),
            )
            u, s, vh = torch.linalg.svd(rand_weight, full_matrices=False)

        # Select singular vectors based on sig_type
        if self.sig_type == "principal":
            # Use top singular vectors (largest singular values)
            q_init = vh[:self.lora_dim]  # (lora_dim, in_dim)
            p_init = u[:, :self.lora_dim]  # (out_dim, lora_dim)
            lambda_init = s[:self.lora_dim]
        elif self.sig_type == "last":
            # Use bottom singular vectors (smallest singular values)
            q_init = vh[-self.lora_dim:]
            p_init = u[:, -self.lora_dim:]
            lambda_init = s[-self.lora_dim:]
        elif self.sig_type == "middle":
            # Use middle singular vectors
            start_q = (vh.shape[0] - self.lora_dim) // 2
            start_p = (u.shape[1] - self.lora_dim) // 2
            start_s = (s.shape[0] - self.lora_dim) // 2
            q_init = vh[start_q:start_q + self.lora_dim]
            p_init = u[:, start_p:start_p + self.lora_dim]
            lambda_init = s[start_s:start_s + self.lora_dim]
        else:
            raise ValueError(f"Unknown sig_type: {self.sig_type}")

        # Handle case where rank is larger than available singular values
        if q_init.shape[0] < self.lora_dim:
            pad_size = self.lora_dim - q_init.shape[0]
            q_init = F.pad(q_init, (0, 0, 0, pad_size))
            p_init = F.pad(p_init, (0, pad_size))
            lambda_init = F.pad(lambda_init, (0, pad_size), value=1e-6)

        # Assign to layers
        if self.isconv:
            # For conv, q_layer is 1x1 conv: weight shape (lora_dim, in_channels, 1, ...)
            # We need to reshape q_init from (lora_dim, in_dim) to conv format
            kernel_ones = [1] * (len(self.conv_shape) - 2)
            self.q_layer.weight.data = q_init[:, :self.shape[1]].reshape(
                self.lora_dim, self.shape[1], *kernel_ones
            ).contiguous()
            self.p_layer.weight.data = p_init[:self.shape[0], :].reshape(
                self.shape[0], self.lora_dim, *kernel_ones
            ).contiguous()
        else:
            # For linear: q_layer.weight is (lora_dim, in_features)
            # p_layer.weight is (out_features, lora_dim)
            self.q_layer.weight.data = q_init.contiguous()
            self.p_layer.weight.data = p_init.contiguous()

        self.lambda_layer.data = lambda_init.unsqueeze(0).contiguous()

        # Cleanup
        del u, s, vh
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_mask(self, device: torch.device) -> torch.Tensor:
        """Get the current timestep mask, defaulting to all ones."""
        mask = get_timestep_mask(self.mask_group_id)
        if mask is None:
            mask = torch.ones(1, self.lora_dim)
        # Ensure mask covers our rank (in case max_rank > lora_dim)
        if mask.shape[1] > self.lora_dim:
            mask = mask[:, :self.lora_dim]
        elif mask.shape[1] < self.lora_dim:
            # Pad with ones if mask is smaller
            mask = F.pad(mask, (0, self.lora_dim - mask.shape[1]), value=1.0)
        return mask.to(device)

    @classmethod
    def make_module_from_state_dict(
        cls,
        lora_name: str,
        orig_module: nn.Module,
        q_weight: torch.Tensor,
        p_weight: torch.Tensor,
        lambda_weight: torch.Tensor,
        alpha: torch.Tensor,
    ):
        """Reconstruct module from saved state dict."""
        lora_dim = q_weight.shape[0]
        module = cls(
            lora_name,
            orig_module,
            multiplier=1.0,
            lora_dim=lora_dim,
            alpha=float(alpha),
            use_data_init=True,
        )
        module.q_layer.weight.data.copy_(q_weight)
        module.p_layer.weight.data.copy_(p_weight)
        module.lambda_layer.data.copy_(lambda_weight)
        return module

    def custom_state_dict(self):
        """Return state dict for saving."""
        return {
            "q_layer.weight": self.q_layer.weight,
            "p_layer.weight": self.p_layer.weight,
            "lambda_layer": self.lambda_layer,
            "alpha": self.alpha,
        }

    def get_diff_weight(self, multiplier=1.0, shape=None, device=None):
        """
        Compute the weight difference: current - base.

        For T-LoRA: ΔW = P @ diag(λ*mask) @ Q - P_base @ diag(λ_base*mask) @ Q_base
        """
        if device is None:
            device = self.q_layer.weight.device

        mask = self._get_mask(device)

        # Current weights
        q = self.q_layer.weight.to(device)  # (lora_dim, in_dim) or conv shape
        p = self.p_layer.weight.to(device)  # (out_dim, lora_dim) or conv shape
        lam = self.lambda_layer.to(device) * mask  # (1, lora_dim)

        # Base weights
        q_base = self.base_q.to(device)
        p_base = self.base_p.to(device)
        lam_base = self.base_lambda.to(device) * mask

        if self.isconv:
            # For conv, reshape to 2D for matmul
            # q: (lora_dim, in_ch, 1, ...) -> (lora_dim, in_ch)
            # p: (out_ch, lora_dim, 1, ...) -> (out_ch, lora_dim)
            q_2d = q.reshape(self.lora_dim, -1)
            p_2d = p.reshape(self.shape[0], self.lora_dim)
            q_base_2d = q_base.reshape(self.lora_dim, -1)
            p_base_2d = p_base.reshape(self.shape[0], self.lora_dim)

            # Current: P @ diag(λ) @ Q
            # diag(λ) @ Q = λ.T * Q (broadcasting)
            curr = p_2d @ (lam.T * q_2d)  # (out_ch, in_ch)
            base = p_base_2d @ (lam_base.T * q_base_2d)

            # Reshape back to conv shape with 1x1 kernel
            kernel_ones = [1] * (len(self.shape) - 2)
            diff = (curr - base).reshape(self.shape[0], self.shape[1], *kernel_ones)
        else:
            # For linear: P @ diag(λ) @ Q
            # p: (out_features, lora_dim), λ: (1, lora_dim), q: (lora_dim, in_features)
            curr = p @ (lam.T * q)  # (out_features, in_features)
            base = p_base @ (lam_base.T * q_base)
            diff = curr - base

        diff = diff * self.scale * multiplier

        if shape is not None:
            diff = diff.view(shape)

        return diff, None

    def get_merged_weight(self, multiplier=1.0, shape=None, device=None):
        """Get original weight + LoRA delta."""
        diff, _ = self.get_diff_weight(multiplier=multiplier, shape=shape, device=device)
        weight = self.org_weight
        if device is not None:
            weight = weight.to(device)

        # For conv with non-1x1 kernel, we need to handle shape mismatch
        if self.isconv and diff.shape != weight.shape:
            # diff is 1x1, weight has kernel - can't directly add
            # This is a limitation; for full merge, would need to expand diff
            # For now, return weight + diff padded to kernel size
            kernel_size = weight.shape[2:]
            if all(k == 1 for k in kernel_size):
                merged = weight + diff
            else:
                # Expand 1x1 diff to kernel size by padding
                pad_dims = []
                for k in reversed(kernel_size):
                    pad_dims.extend([0, k - 1])
                diff_expanded = F.pad(diff, pad_dims)
                merged = weight + diff_expanded
        else:
            merged = weight + diff

        return merged, None

    def orthogonality_regularization(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute orthogonality regularization loss.

        Encourages P and Q to remain orthogonal:
        L_p = ||P^T @ P - I||_F^2
        L_q = ||Q @ Q^T - I||_F^2

        Returns:
            (p_reg, q_reg): Regularization losses for P and Q
        """
        device = self.q_layer.weight.device

        if self.isconv:
            q = self.q_layer.weight.reshape(self.lora_dim, -1)
            p = self.p_layer.weight.reshape(self.shape[0], self.lora_dim)
        else:
            q = self.q_layer.weight
            p = self.p_layer.weight

        # P^T @ P should be identity (lora_dim x lora_dim)
        p_reg = torch.sum(
            (p.T @ p - torch.eye(self.lora_dim, device=device)) ** 2
        )

        # Q @ Q^T should be identity (lora_dim x lora_dim)
        q_reg = torch.sum(
            (q @ q.T - torch.eye(self.lora_dim, device=device)) ** 2
        )

        return p_reg, q_reg

    def bypass_forward_diff(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Compute LoRA contribution in bypass mode."""
        device = x.device
        dtype = x.dtype
        mask = self._get_mask(device)

        lam = self.lambda_layer.to(device) * mask
        lam_base = self.base_lambda.to(device) * mask

        if self.isconv:
            # Current path: x -> Q -> scale by λ -> P
            q_out = self.down_op(x, self.q_layer.weight.to(dtype), None, **self.kw_dict_down)
            q_out_scaled = q_out * lam.view(1, -1, *([1] * (q_out.dim() - 2)))
            curr_out = self.up_op(q_out_scaled, self.p_layer.weight.to(dtype), None, **self.kw_dict_up)

            # Base path
            q_base_out = self.down_op(x, self.base_q.to(dtype), None, **self.kw_dict_down)
            q_base_scaled = q_base_out * lam_base.view(1, -1, *([1] * (q_base_out.dim() - 2)))
            base_out = self.up_op(q_base_scaled, self.base_p.to(dtype), None, **self.kw_dict_up)
        else:
            # Current path
            q_out = self.down_op(x, self.q_layer.weight.to(dtype), None)
            q_out_scaled = q_out * lam
            curr_out = self.up_op(q_out_scaled, self.p_layer.weight.to(dtype), None)

            # Base path
            q_base_out = self.down_op(x, self.base_q.to(dtype), None)
            q_base_scaled = q_base_out * lam_base
            base_out = self.up_op(q_base_scaled, self.base_p.to(dtype), None)

        diff = curr_out - base_out
        return self.dropout(diff * self.scale * scale)

    def bypass_forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """Forward with bypass mode (compute LoRA separately)."""
        return self.org_forward(x) + self.bypass_forward_diff(x, scale=scale)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass with timestep-dependent rank masking."""
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)

        if self.bypass_mode:
            return self.bypass_forward(x, scale=self.multiplier)

        # Standard forward: compute full weight and apply
        base = self.org_forward(x, *args, **kwargs)
        device = x.device

        base_weight = self._current_weight().to(device)
        diff_weight, _ = self.get_diff_weight(multiplier=self.multiplier, device=device)
        diff_weight = diff_weight.to(base_weight.dtype)

        # For conv with different kernel sizes, use bypass mode logic
        if self.isconv and diff_weight.shape != base_weight.shape:
            return self.bypass_forward(x, scale=self.multiplier)

        new_weight = base_weight + diff_weight
        delta_weight = new_weight - base_weight

        delta = self.op(x, delta_weight, None, **self.kw_dict)
        return base + delta

    @torch.no_grad()
    def apply_max_norm(self, max_norm: float, device=None):
        """Apply max norm regularization to weights."""
        diff, _ = self.get_diff_weight(multiplier=1.0, device=device)
        orig_norm = diff.norm() * self.scale
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            # Scale the lambda layer to reduce norm
            self.lambda_layer.data *= ratio

        return scaled, orig_norm * ratio
