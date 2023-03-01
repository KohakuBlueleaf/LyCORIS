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
    device = 'cpu',
) -> tuple[nn.Parameter, nn.Parameter]:
    out_ch, in_ch, kernel_size, _ = weight.shape
    lora_rank = min(out_ch, in_ch, lora_rank)
    
    U, S, Vh = linalg.svd(weight.reshape(out_ch, -1).to(device))
    
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
    weight: nn.Parameter|torch.Tensor,
    lora_rank = 8,
    singular_threshold = 0.1,
    use_threshold = False,
    device = 'cpu',
) -> tuple[nn.Parameter, nn.Parameter]:
    out_ch, in_ch = weight.shape
    lora_rank = min(out_ch, in_ch, lora_rank)
    
    U, S, Vh = linalg.svd(weight.to(device))
    
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
    lora_dim=4, 
    conv_lora_dim=4,
    use_threshold = False,
    use_threshold_conv = False,
    threshold_linear = 0.1,
    threshold_conv = 0.1,
    extract_device = 'cpu'
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
                        extract_a, extract_b = extract_linear(
                            (child_module.weight - weights[child_name]),
                            lora_dim,
                            threshold_linear,
                            use_threshold,
                            device = extract_device,
                        )
                    elif layer == 'Conv2d':
                        is_linear = (child_module.weight.shape[2] == 1
                                     and child_module.weight.shape[3] == 1)
                        extract_a, extract_b = extract_conv(
                            (child_module.weight - weights[child_name]), 
                            conv_lora_dim,
                            threshold_linear if is_linear else threshold_conv,
                            use_threshold if is_linear else use_threshold_conv,
                            device = extract_device,
                        )
                    else:
                        continue
                    loras[f'{lora_name}.lora_down.weight'] = extract_a.detach().cpu().contiguous().half()
                    loras[f'{lora_name}.lora_up.weight'] = extract_b.detach().cpu().contiguous().half()
                    loras[f'{lora_name}.alpha'] = torch.Tensor([extract_a.shape[0]]).half()
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


def merge_locon(
    base_model,
    locon_state_dict: dict[str, torch.TensorType],
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