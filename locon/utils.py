import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.linalg as linalg

from tqdm import tqdm


def extract_conv(
    weight: nn.Parameter|torch.Tensor,
    lora_rank = 8
) -> tuple[nn.Parameter, nn.Parameter]:
    out_ch, in_ch, kernel_size, _ = weight.shape
    lora_rank = min(out_ch, in_ch, lora_rank)
    
    U, S, Vh = linalg.svd(weight.reshape(out_ch, -1))
    
    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    
    Vh = Vh[:lora_rank, :]
    
    extract_weight_A = Vh.reshape(lora_rank, in_ch, kernel_size, kernel_size).cpu()
    extract_weight_B = U.reshape(out_ch, lora_rank, 1, 1).cpu()
    return nn.Parameter(extract_weight_A), nn.Parameter(extract_weight_B)


def merge_conv(
    weight_a: nn.Parameter|torch.Tensor,
    weight_b: nn.Parameter|torch.Tensor,
):
    rank, in_ch, kernel_size, k_ = weight_a.shape
    out_ch, rank_, _, _ = weight_b.shape
    
    assert rank == rank_ and kernel_size == k_
    
    merged = weight_b.reshape(out_ch, -1) @ weight_a.reshape(rank, -1)
    weight = merged.reshape(out_ch, in_ch, kernel_size, kernel_size)
    return nn.Parameter(weight)


def extract_linear(
    weight: nn.Parameter|torch.Tensor,
    lora_rank = 8
) -> tuple[nn.Parameter, nn.Parameter]:
    out_ch, in_ch = weight.shape
    lora_rank = min(out_ch, in_ch, lora_rank)
    
    try:
        U, S, Vh = linalg.svd(weight)
    except:
        print()
        print(weight.shape)
        U, S, Vh = linalg.svd(weight.cpu())
    
    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]
    
    extract_weight_A = Vh.reshape(lora_rank, in_ch).cpu()
    extract_weight_B = U.reshape(out_ch, lora_rank).cpu()
    return nn.Parameter(extract_weight_A), nn.Parameter(extract_weight_B)


def merge_linear(
    weight_a: nn.Parameter|torch.Tensor,
    weight_b: nn.Parameter|torch.Tensor,
):
    rank, in_ch = weight_a.shape
    out_ch, rank_ = weight_b.shape
    
    assert rank == rank_
    
    weight = weight_b @ weight_a
    return nn.Parameter(weight)