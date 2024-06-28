import re
import hashlib
from io import BytesIO
from typing import Dict, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as linalg

import safetensors.torch

from tqdm import tqdm
from .general import *


def load_bytes_in_safetensors(tensors):
    bytes = safetensors.torch.save(tensors)
    b = BytesIO(bytes)

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)

    return b.read()


def precalculate_safetensors_hashes(state_dict):
    # calculate each tensor one by one to reduce memory usage
    hash_sha256 = hashlib.sha256()
    for tensor in state_dict.values():
        single_tensor_sd = {"tensor": tensor}
        bytes_for_tensor = load_bytes_in_safetensors(single_tensor_sd)
        hash_sha256.update(bytes_for_tensor)

    return f"0x{hash_sha256.hexdigest()}"


def str_bool(val):
    return str(val).lower() != "false"


def default(val, d):
    return val if val is not None else d


def make_sparse(t: torch.Tensor, sparsity=0.95):
    abs_t = torch.abs(t)
    np_array = abs_t.detach().cpu().numpy()
    quan = float(np.quantile(np_array, sparsity))
    sparse_t = t.masked_fill(abs_t < quan, 0)
    return sparse_t


def extract_conv(
    weight: Union[torch.Tensor, nn.Parameter],
    mode="fixed",
    mode_param=0,
    device="cpu",
    is_cp=False,
) -> Tuple[nn.Parameter, nn.Parameter]:
    weight = weight.to(device)
    out_ch, in_ch, kernel_size, _ = weight.shape

    U, S, Vh = linalg.svd(weight.reshape(out_ch, -1))

    if mode == "full":
        return weight, "full"
    elif mode == "fixed":
        lora_rank = mode_param
    elif mode == "threshold":
        assert mode_param >= 0
        lora_rank = torch.sum(S > mode_param)
    elif mode == "ratio":
        assert 1 >= mode_param >= 0
        min_s = torch.max(S) * mode_param
        lora_rank = torch.sum(S > min_s)
    elif mode == "quantile" or mode == "percentile":
        assert 1 >= mode_param >= 0
        s_cum = torch.cumsum(S, dim=0)
        min_cum_sum = mode_param * torch.sum(S)
        lora_rank = torch.sum(s_cum < min_cum_sum)
    else:
        raise NotImplementedError(
            'Extract mode should be "fixed", "threshold", "ratio" or "quantile"'
        )
    lora_rank = max(1, lora_rank)
    lora_rank = min(out_ch, in_ch, lora_rank)
    if lora_rank >= out_ch / 2 and not is_cp:
        return weight, "full"

    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S).to(device)
    Vh = Vh[:lora_rank, :]

    diff = (weight - (U @ Vh).reshape(out_ch, in_ch, kernel_size, kernel_size)).detach()
    extract_weight_A = Vh.reshape(lora_rank, in_ch, kernel_size, kernel_size).detach()
    extract_weight_B = U.reshape(out_ch, lora_rank, 1, 1).detach()
    del U, S, Vh, weight
    return (extract_weight_A, extract_weight_B, diff), "low rank"


def extract_linear(
    weight: Union[torch.Tensor, nn.Parameter],
    mode="fixed",
    mode_param=0,
    device="cpu",
) -> Tuple[nn.Parameter, nn.Parameter]:
    weight = weight.to(device)
    out_ch, in_ch = weight.shape

    U, S, Vh = linalg.svd(weight)

    if mode == "full":
        return weight, "full"
    elif mode == "fixed":
        lora_rank = mode_param
    elif mode == "threshold":
        assert mode_param >= 0
        lora_rank = torch.sum(S > mode_param)
    elif mode == "ratio":
        assert 1 >= mode_param >= 0
        min_s = torch.max(S) * mode_param
        lora_rank = torch.sum(S > min_s)
    elif mode == "quantile" or mode == "percentile":
        assert 1 >= mode_param >= 0
        s_cum = torch.cumsum(S, dim=0)
        min_cum_sum = mode_param * torch.sum(S)
        lora_rank = torch.sum(s_cum < min_cum_sum)
    else:
        raise NotImplementedError(
            'Extract mode should be "fixed", "threshold", "ratio" or "quantile"'
        )
    lora_rank = max(1, lora_rank)
    lora_rank = min(out_ch, in_ch, lora_rank)
    if lora_rank >= out_ch / 2:
        return weight, "full"

    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S).to(device)
    Vh = Vh[:lora_rank, :]

    diff = (weight - U @ Vh).detach()
    extract_weight_A = Vh.reshape(lora_rank, in_ch).detach()
    extract_weight_B = U.reshape(out_ch, lora_rank).detach()
    del U, S, Vh, weight
    return (extract_weight_A, extract_weight_B, diff), "low rank"


@torch.no_grad()
def extract_diff(
    base_tes,
    db_tes,
    base_unet,
    db_unet,
    mode="fixed",
    linear_mode_param=0,
    conv_mode_param=0,
    extract_device="cpu",
    use_bias=False,
    sparsity=0.98,
    small_conv=True,
):
    UNET_TARGET_REPLACE_MODULE = [
        "Linear",
        "Conv2d",
        "LayerNorm",
        "GroupNorm",
        "GroupNorm32",
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = [
        "Embedding",
        "Linear",
        "Conv2d",
        "LayerNorm",
        "GroupNorm",
        "GroupNorm32",
    ]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    def make_state_dict(
        prefix,
        root_module: torch.nn.Module,
        target_module: torch.nn.Module,
        target_replace_modules,
    ):
        loras = {}
        temp = {}

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                temp[name] = module

        for name, module in tqdm(
            list((n, m) for n, m in target_module.named_modules() if n in temp)
        ):
            weights = temp[name]
            lora_name = prefix + "." + name
            lora_name = lora_name.replace(".", "_")
            layer = module.__class__.__name__

            if layer in {
                "Linear",
                "Conv2d",
                "LayerNorm",
                "GroupNorm",
                "GroupNorm32",
                "Embedding",
            }:
                root_weight = module.weight
                if torch.allclose(root_weight, weights.weight):
                    continue
            else:
                continue
            module = module.to(extract_device)
            weights = weights.to(extract_device)

            if mode == "full":
                decompose_mode = "full"
            elif layer == "Linear":
                weight, decompose_mode = extract_linear(
                    (root_weight - weights.weight),
                    mode,
                    linear_mode_param,
                    device=extract_device,
                )
                if decompose_mode == "low rank":
                    extract_a, extract_b, diff = weight
            elif layer == "Conv2d":
                is_linear = root_weight.shape[2] == 1 and root_weight.shape[3] == 1
                weight, decompose_mode = extract_conv(
                    (root_weight - weights.weight),
                    mode,
                    linear_mode_param if is_linear else conv_mode_param,
                    device=extract_device,
                )
                if decompose_mode == "low rank":
                    extract_a, extract_b, diff = weight
                if small_conv and not is_linear and decompose_mode == "low rank":
                    dim = extract_a.size(0)
                    (extract_c, extract_a, _), _ = extract_conv(
                        extract_a.transpose(0, 1),
                        "fixed",
                        dim,
                        extract_device,
                        True,
                    )
                    extract_a = extract_a.transpose(0, 1)
                    extract_c = extract_c.transpose(0, 1)
                    loras[f"{lora_name}.lora_mid.weight"] = (
                        extract_c.detach().cpu().contiguous().half()
                    )
                    diff = (
                        (
                            root_weight
                            - torch.einsum(
                                "i j k l, j r, p i -> p r k l",
                                extract_c,
                                extract_a.flatten(1, -1),
                                extract_b.flatten(1, -1),
                            )
                        )
                        .detach()
                        .cpu()
                        .contiguous()
                    )
                    del extract_c
            else:
                module = module.to("cpu")
                weights = weights.to("cpu")
                continue

            if decompose_mode == "low rank":
                loras[f"{lora_name}.lora_down.weight"] = (
                    extract_a.detach().cpu().contiguous().half()
                )
                loras[f"{lora_name}.lora_up.weight"] = (
                    extract_b.detach().cpu().contiguous().half()
                )
                loras[f"{lora_name}.alpha"] = torch.Tensor([extract_a.shape[0]]).half()
                if use_bias:
                    diff = diff.detach().cpu().reshape(extract_b.size(0), -1)
                    sparse_diff = make_sparse(diff, sparsity).to_sparse().coalesce()

                    indices = sparse_diff.indices().to(torch.int16)
                    values = sparse_diff.values().half()
                    loras[f"{lora_name}.bias_indices"] = indices
                    loras[f"{lora_name}.bias_values"] = values
                    loras[f"{lora_name}.bias_size"] = torch.tensor(diff.shape).to(
                        torch.int16
                    )
                del extract_a, extract_b, diff
            elif decompose_mode == "full":
                if "Norm" in layer:
                    w_key = "w_norm"
                    b_key = "b_norm"
                else:
                    w_key = "diff"
                    b_key = "diff_b"
                weight_diff = module.weight - weights.weight
                loras[f"{lora_name}.{w_key}"] = (
                    weight_diff.detach().cpu().contiguous().half()
                )
                if getattr(weights, "bias", None) is not None:
                    bias_diff = module.bias - weights.bias
                    loras[f"{lora_name}.{b_key}"] = (
                        bias_diff.detach().cpu().contiguous().half()
                    )
            else:
                raise NotImplementedError
            module = module.to("cpu")
            weights = weights.to("cpu")
        return loras

    all_loras = {}

    all_loras |= make_state_dict(
        LORA_PREFIX_UNET,
        base_unet,
        db_unet,
        UNET_TARGET_REPLACE_MODULE,
    )
    del base_unet, db_unet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for idx, (te1, te2) in enumerate(zip(base_tes, db_tes)):
        if len(base_tes) > 1:
            prefix = f"{LORA_PREFIX_TEXT_ENCODER}{idx+1}"
        else:
            prefix = LORA_PREFIX_TEXT_ENCODER
        all_loras |= make_state_dict(
            prefix,
            te1,
            te2,
            TEXT_ENCODER_TARGET_REPLACE_MODULE,
        )
        del te1, te2

    all_lora_name = set()
    for k in all_loras:
        lora_name, weight = k.rsplit(".", 1)
        all_lora_name.add(lora_name)
    print(len(all_lora_name))
    return all_loras


re_digits = re.compile(r"\d+")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "norm1": "in_layers_0",
        "norm2": "out_layers_0",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    },
}


def convert_diffusers_name_to_compvis(key):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_conv_in(.*)"):
        return f"lora_unet_input_blocks_0_0{m[0]}"

    if match(m, r"lora_unet_conv_out(.*)"):
        return f"lora_unet_out_2{m[0]}"

    if match(m, r"lora_unet_time_embedding_linear_(\d+)(.*)"):
        return f"lora_unet_time_embed_{m[0] * 2 - 2}{m[1]}"

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"lora_unet_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return (
            f"lora_unet_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"
        )

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"lora_unet_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"lora_unet_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"lora_unet_output_blocks_{2 + m[0] * 3}_2_conv"
    return key


@torch.no_grad()
def merge(tes, unet, lyco_state_dict, scale: float = 1.0, device="cpu"):
    from ..modules import make_module, get_module

    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    merged = 0

    def merge_state_dict(
        prefix,
        root_module: torch.nn.Module,
        lyco_state_dict: Dict[str, torch.Tensor],
    ):
        nonlocal merged
        for child_name, child_module in tqdm(
            list(root_module.named_modules()), desc=f"Merging {prefix}"
        ):
            lora_name = prefix + "." + child_name
            lora_name = lora_name.replace(".", "_")

            lyco_type, params = get_module(lyco_state_dict, lora_name)
            if lyco_type is None:
                continue
            module = make_module(lyco_type, params, lora_name, child_module)
            if module is None:
                continue
            module.to(device)
            module.merge_to(scale)
            key_dict.pop(convert_diffusers_name_to_compvis(lora_name), None)
            key_dict.pop(lora_name, None)
            merged += 1

    key_dict = {}
    for k, v in tqdm(list(lyco_state_dict.items()), desc="Converting Dtype and Device"):
        module, weight_key = k.split(".", 1)
        convert_key = convert_diffusers_name_to_compvis(module)
        if convert_key != module and len(tes) > 1:
            # kohya's format for sdxl is as same as SGM, not diffusers
            del lyco_state_dict[k]
            key_dict[convert_key] = key_dict.get(convert_key, []) + [k]
            k = f"{convert_key}.{weight_key}"
        else:
            key_dict[module] = key_dict.get(module, []) + [k]
        lyco_state_dict[k] = v.float().cpu()

    for idx, te in enumerate(tes):
        if len(tes) > 1:
            prefix = LORA_PREFIX_TEXT_ENCODER + str(idx + 1)
        else:
            prefix = LORA_PREFIX_TEXT_ENCODER
        merge_state_dict(
            prefix,
            te,
            lyco_state_dict,
        )
        torch.cuda.empty_cache()
    merge_state_dict(
        LORA_PREFIX_UNET,
        unet,
        lyco_state_dict,
    )
    torch.cuda.empty_cache()
    print(f"Unused state dict key: {key_dict}")
    print(f"{merged} Modules been merged")
