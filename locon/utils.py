import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.linalg as linalg

from tqdm import tqdm


def extract_conv(
    weight: nn.Parameter|torch.Tensor,
    lora_rank = 8,
    singular_threshold = 0.1,
    use_threshold = False,
) -> tuple[nn.Parameter, nn.Parameter]:
    out_ch, in_ch, kernel_size, _ = weight.shape
    lora_rank = min(out_ch, in_ch, lora_rank)
    
    U, S, Vh = linalg.svd(weight.reshape(out_ch, -1))
    
    if use_threshold:
        lora_rank = torch.sum(S>singular_threshold)
        lora_rank = max(1, lora_rank)
        print(lora_rank)
    
    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]
    
    extract_weight_A = Vh.reshape(lora_rank, in_ch, kernel_size, kernel_size).cpu()
    extract_weight_B = U.reshape(out_ch, lora_rank, 1, 1).cpu()
    del U, S, Vh, weight
    return extract_weight_A, extract_weight_B


def merge_conv(
    weight_a: nn.Parameter|torch.Tensor,
    weight_b: nn.Parameter|torch.Tensor,
):
    rank, in_ch, kernel_size, k_ = weight_a.shape
    out_ch, rank_, _, _ = weight_b.shape
    
    assert rank == rank_ and kernel_size == k_
    
    merged = weight_b.reshape(out_ch, -1) @ weight_a.reshape(rank, -1)
    weight = merged.reshape(out_ch, in_ch, kernel_size, kernel_size)
    return weight


def extract_linear(
    weight: nn.Parameter|torch.Tensor,
    lora_rank = 8,
    singular_threshold = 0.1,
    use_threshold = False,
) -> tuple[nn.Parameter, nn.Parameter]:
    out_ch, in_ch = weight.shape
    lora_rank = min(out_ch, in_ch, lora_rank)
    
    U, S, Vh = linalg.svd(weight)
    
    if use_threshold:
        lora_rank = torch.sum(S>singular_threshold).item()
        lora_rank = max(1, lora_rank)
        print(lora_rank, singular_threshold)
    
    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]
    
    extract_weight_A = Vh.reshape(lora_rank, in_ch).cpu()
    extract_weight_B = U.reshape(out_ch, lora_rank).cpu()
    del U, S, Vh, weight
    return extract_weight_A, extract_weight_B


def merge_linear(
    weight_a: nn.Parameter|torch.Tensor,
    weight_b: nn.Parameter|torch.Tensor,
):
    rank, in_ch = weight_a.shape
    out_ch, rank_ = weight_b.shape
    
    assert rank == rank_
    
    weight = weight_b @ weight_a
    return weight


def extract_diff(
    base_model,
    db_model,
    lora_dim=4, 
    conv_lora_dim=4,
    use_threshold = False,
    threshold_linear = 0.1,
    threshold_conv = 0.1
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
        
        for name, module in list(target_module.named_modules()):
            if name in temp:
                weights = temp[name]
                for child_name, child_module in module.named_modules():
                    lora_name = prefix + '.' + name + '.' + child_name
                    lora_name = lora_name.replace('.', '_')
                    if child_module.__class__.__name__ == 'Linear':
                        extract_a, extract_b = extract_linear(
                            (child_module.weight - weights[child_name]),
                            lora_dim,
                            threshold_linear,
                            use_threshold,
                        )
                    elif child_module.__class__.__name__ == 'Conv2d':
                        extract_a, extract_b = extract_conv(
                            (child_module.weight - weights[child_name]), 
                            conv_lora_dim,
                            threshold_conv,
                            use_threshold,
                        )
                    else:
                        continue
                    loras[f'{lora_name}.lora_down.weight'] = extract_a.detach().cpu().half()
                    loras[f'{lora_name}.lora_up.weight'] = extract_b.detach().cpu().half()
                    loras[f'{lora_name}.alpha'] = torch.Tensor([int(extract_a.shape[0])]).detach().cpu().half()
                    del extract_a, extract_b
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