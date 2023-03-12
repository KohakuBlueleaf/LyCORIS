from typing import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.linalg as linalg

from tqdm import tqdm


def make_sparse(t: torch.Tensor, sparsity=0.95):
    abs_t = torch.abs(t)
    np_array = abs_t.detach().cpu().numpy()
    quan = float(np.quantile(np_array, sparsity))
    sparse_t = t.masked_fill(abs_t < quan, 0)
    return sparse_t


def extract_conv(
    weight: Union[torch.Tensor, nn.Parameter],
    mode = 'fixed',
    mode_param = 0,
    device = 'cpu',
) -> Tuple[nn.Parameter, nn.Parameter]:
    weight = weight.to(device)
    out_ch, in_ch, kernel_size, _ = weight.shape
    
    U, S, Vh = linalg.svd(weight.reshape(out_ch, -1))
    
    if mode=='fixed':
        lora_rank = mode_param
    elif mode=='threshold':
        assert mode_param>=0
        lora_rank = torch.sum(S>mode_param)
    elif mode=='ratio':
        assert 1>=mode_param>=0
        min_s = torch.max(S)*mode_param
        lora_rank = torch.sum(S>min_s)
    elif mode=='quantile' or mode=='percentile':
        assert 1>=mode_param>=0
        s_cum = torch.cumsum(S, dim=0)
        min_cum_sum = mode_param * torch.sum(S)
        lora_rank = torch.sum(s_cum<min_cum_sum)
    else:
        raise NotImplementedError('Extract mode should be "fixed", "threshold", "ratio" or "quantile"')
    lora_rank = max(1, lora_rank)
    lora_rank = min(out_ch, in_ch, lora_rank)
    
    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]
    
    diff = (weight - (U @ Vh).reshape(out_ch, in_ch, kernel_size, kernel_size)).detach()
    extract_weight_A = Vh.reshape(lora_rank, in_ch, kernel_size, kernel_size).detach()
    extract_weight_B = U.reshape(out_ch, lora_rank, 1, 1).detach()
    del U, S, Vh, weight
    return extract_weight_A, extract_weight_B, diff


def merge_conv(
    weight_a: Union[torch.Tensor, nn.Parameter],
    weight_b: Union[torch.Tensor, nn.Parameter],
    device = 'cpu'
):
    rank, in_ch, kernel_size, k_ = weight_a.shape
    out_ch, rank_, _, _ = weight_b.shape
    assert rank == rank_ and kernel_size == k_
    
    wa = weight_a.to(device)
    wb = weight_b.to(device)
    
    if device == 'cpu':
        wa = wa.float()
        wb = wb.float()
    
    merged = wb.reshape(out_ch, -1) @ wa.reshape(rank, -1)
    weight = merged.reshape(out_ch, in_ch, kernel_size, kernel_size)
    del wb, wa
    return weight


def extract_linear(
    weight: Union[torch.Tensor, nn.Parameter],
    mode = 'fixed',
    mode_param = 0,
    device = 'cpu',
) -> Tuple[nn.Parameter, nn.Parameter]:
    weight = weight.to(device)
    out_ch, in_ch = weight.shape
    
    U, S, Vh = linalg.svd(weight)
    
    if mode=='fixed':
        lora_rank = mode_param
    elif mode=='threshold':
        assert mode_param>=0
        lora_rank = torch.sum(S>mode_param)
    elif mode=='ratio':
        assert 1>=mode_param>=0
        min_s = torch.max(S)*mode_param
        lora_rank = torch.sum(S>min_s)
    elif mode=='quantile' or mode=='percentile':
        assert 1>=mode_param>=0
        s_cum = torch.cumsum(S, dim=0)
        min_cum_sum = mode_param * torch.sum(S)
        lora_rank = torch.sum(s_cum<min_cum_sum)
    else:
        raise NotImplementedError('Extract mode should be "fixed", "threshold", "ratio" or "quantile"')
    lora_rank = max(1, lora_rank)
    lora_rank = min(out_ch, in_ch, lora_rank)
    
    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]
    
    diff = (weight - U @ Vh).detach()
    extract_weight_A = Vh.reshape(lora_rank, in_ch).detach()
    extract_weight_B = U.reshape(out_ch, lora_rank).detach()
    del U, S, Vh, weight
    return extract_weight_A, extract_weight_B, diff


def merge_linear(
    weight_a: Union[torch.Tensor, nn.Parameter],
    weight_b: Union[torch.Tensor, nn.Parameter],
    device = 'cpu'
):
    rank, in_ch = weight_a.shape
    out_ch, rank_ = weight_b.shape
    assert rank == rank_
    
    wa = weight_a.to(device)
    wb = weight_b.to(device)
    
    if device == 'cpu':
        wa = wa.float()
        wb = wb.float()
    
    weight = wb @ wa
    del wb, wa
    return weight


def extract_diff(
    base_model,
    db_model,
    mode = 'fixed',
    linear_mode_param = 0,
    conv_mode_param = 0,
    extract_device = 'cpu',
    use_bias = False,
    sparsity = 0.98,
    small_conv = True
):
    UNET_TARGET_REPLACE_MODULE = [
        "Transformer2DModel", 
        "Attention", 
        "ResnetBlock2D", 
        "Downsample2D", 
        "Upsample2D"
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'
    def make_state_dict(
        prefix, 
        root_module: torch.nn.Module,
        target_module: torch.nn.Module,
        target_replace_modules
    ):
        loras = {}
        temp = {}
        
        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                temp[name] = {}
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ not in {'Linear', 'Conv2d'}:
                        continue
                    temp[name][child_name] = child_module.weight
        
        for name, module in tqdm(list(target_module.named_modules())):
            if name in temp:
                weights = temp[name]
                for child_name, child_module in module.named_modules():
                    lora_name = prefix + '.' + name + '.' + child_name
                    lora_name = lora_name.replace('.', '_')
                    
                    layer = child_module.__class__.__name__
                    if layer == 'Linear':
                        extract_a, extract_b, diff = extract_linear(
                            (child_module.weight - weights[child_name]),
                            mode,
                            linear_mode_param,
                            device = extract_device,
                        )
                    elif layer == 'Conv2d':
                        is_linear = (child_module.weight.shape[2] == 1
                                     and child_module.weight.shape[3] == 1)
                        extract_a, extract_b, diff = extract_conv(
                            (child_module.weight - weights[child_name]), 
                            mode,
                            linear_mode_param if is_linear else conv_mode_param,
                            device = extract_device,
                        )
                        if small_conv and not is_linear:
                            dim = extract_a.size(0)
                            extract_c, extract_a, _ = extract_conv(
                                extract_a.transpose(0, 1), 
                                'fixed', dim, 
                                extract_device
                            )
                            extract_a = extract_a.transpose(0, 1)
                            extract_c = extract_c.transpose(0, 1)
                            loras[f'{lora_name}.lora_mid.weight'] = extract_c.detach().cpu().contiguous().half()
                            diff = child_module.weight - torch.einsum(
                                'i j k l, j r, p i -> p r k l', 
                                extract_c, extract_a.flatten(1, -1), extract_b.flatten(1, -1)
                            )
                            del extract_c
                    else:
                        continue
                    loras[f'{lora_name}.lora_down.weight'] = extract_a.detach().cpu().contiguous().half()
                    loras[f'{lora_name}.lora_up.weight'] = extract_b.detach().cpu().contiguous().half()
                    loras[f'{lora_name}.alpha'] = torch.Tensor([extract_a.shape[0]]).half()
                    
                    if use_bias:
                        diff = diff.detach().cpu().reshape(extract_b.size(0), -1)
                        sparse_diff = make_sparse(diff, sparsity).to_sparse()
                        
                        indices = sparse_diff.indices().to(torch.int16)
                        values = sparse_diff.values().half()
                        loras[f'{lora_name}.bias_indices'] = indices
                        loras[f'{lora_name}.bias_values'] = values
                        loras[f'{lora_name}.bias_size'] = torch.tensor(diff.shape).to(torch.int16)
                    del extract_a, extract_b, diff
        return loras
    
    text_encoder_loras = make_state_dict(
        LORA_PREFIX_TEXT_ENCODER, 
        base_model[0], db_model[0], 
        TEXT_ENCODER_TARGET_REPLACE_MODULE
    )
    
    unet_loras = make_state_dict(
        LORA_PREFIX_UNET,
        base_model[2], db_model[2], 
        UNET_TARGET_REPLACE_MODULE
    )
    print(len(text_encoder_loras), len(unet_loras))
    return text_encoder_loras|unet_loras


def merge_locon(
    base_model,
    locon_state_dict: Dict[str, torch.TensorType],
    scale: float = 1.0,
    device = 'cpu'
):
    UNET_TARGET_REPLACE_MODULE = [
        "Transformer2DModel", 
        "Attention", 
        "ResnetBlock2D", 
        "Downsample2D", 
        "Upsample2D"
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'
    def merge(
        prefix, 
        root_module: torch.nn.Module,
        target_replace_modules
    ):
        temp = {}
        
        for name, module in tqdm(list(root_module.named_modules())):
            if module.__class__.__name__ in target_replace_modules:
                temp[name] = {}
                for child_name, child_module in module.named_modules():
                    layer = child_module.__class__.__name__
                    if layer not in {'Linear', 'Conv2d'}:
                        continue
                    lora_name = prefix + '.' + name + '.' + child_name
                    lora_name = lora_name.replace('.', '_')
                    
                    down = locon_state_dict[f'{lora_name}.lora_down.weight'].float()
                    up = locon_state_dict[f'{lora_name}.lora_up.weight'].float()
                    alpha = locon_state_dict[f'{lora_name}.alpha'].float()
                    rank = down.shape[0]
                    
                    if layer == 'Conv2d':
                        delta = merge_conv(down, up, device)
                        child_module.weight.requires_grad_(False)
                        child_module.weight += (alpha.to(device)/rank * scale * delta).cpu()
                        del delta
                    elif layer == 'Linear':
                        delta = merge_linear(down, up, device)
                        child_module.weight.requires_grad_(False)
                        child_module.weight += (alpha.to(device)/rank * scale * delta).cpu()
                        del delta
    
    merge(
        LORA_PREFIX_TEXT_ENCODER, 
        base_model[0], 
        TEXT_ENCODER_TARGET_REPLACE_MODULE
    )
    merge(
        LORA_PREFIX_UNET,
        base_model[2], 
        UNET_TARGET_REPLACE_MODULE
    )


def merge_loha(
    base_model,
    loha_state_dict: Dict[str, torch.TensorType],
    scale: float = 1.0,
    device = 'cpu'
):
    UNET_TARGET_REPLACE_MODULE = [
        "Transformer2DModel", 
        "Attention", 
        "ResnetBlock2D", 
        "Downsample2D", 
        "Upsample2D"
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'
    def merge(
        prefix, 
        root_module: torch.nn.Module,
        target_replace_modules
    ):
        temp = {}
        
        for name, module in tqdm(list(root_module.named_modules())):
            if module.__class__.__name__ in target_replace_modules:
                temp[name] = {}
                for child_name, child_module in module.named_modules():
                    layer = child_module.__class__.__name__
                    if layer not in {'Linear', 'Conv2d'}:
                        continue
                    lora_name = prefix + '.' + name + '.' + child_name
                    lora_name = lora_name.replace('.', '_')
                    
                    w1a = loha_state_dict[f'{lora_name}.hada_w1_a'].float().to(device)
                    w1b = loha_state_dict[f'{lora_name}.hada_w1_b'].float().to(device)
                    w2a = loha_state_dict[f'{lora_name}.hada_w2_a'].float().to(device)
                    w2b = loha_state_dict[f'{lora_name}.hada_w2_b'].float().to(device)
                    alpha = loha_state_dict[f'{lora_name}.alpha'].float().to(device)
                    dim = w1b.shape[0]
                    
                    delta = (w1a @ w1b) * (w2a @ w2b)
                    delta = delta.reshape(child_module.weight.shape)
                    
                    if layer == 'Conv2d':
                        child_module.weight.requires_grad_(False)
                        child_module.weight += (alpha.to(device)/dim * scale * delta).cpu()
                    elif layer == 'Linear':
                        child_module.weight.requires_grad_(False)
                        child_module.weight += (alpha.to(device)/dim * scale * delta).cpu()
                    del delta
    
    merge(
        LORA_PREFIX_TEXT_ENCODER, 
        base_model[0], 
        TEXT_ENCODER_TARGET_REPLACE_MODULE
    )
    merge(
        LORA_PREFIX_UNET,
        base_model[2], 
        UNET_TARGET_REPLACE_MODULE
    )