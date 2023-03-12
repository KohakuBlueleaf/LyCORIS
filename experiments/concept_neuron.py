import os, sys
sys.path.insert(0, os.getcwd())
import math
from warnings import warn
import os
from typing import List
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

from lycoris.kohya_utils import *
from lycoris.kohya_model_utils import *
from lycoris.locon import LoConModule
from lycoris.loha import LohaModule


class ConceptModule(nn.Module):
    def __init__(
        self, 
        lora_name, org_module: nn.Module, 
        multiplier=1.0,
    ):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == 'Conv2d':
            # For general LoCon
            stride = org_module.stride
            padding = org_module.padding
            self.op = F.conv2d
            self.extra_args = {
                'stride': stride,
                'padding': padding
            }
        else:
            self.op = F.linear
            self.extra_args = {}
        
        self.concept_mask = nn.Parameter(torch.ones_like(
            org_module.weight
        ))
        
        self.multiplier = multiplier
        self.org_module = [org_module]

    def apply_to(self):
        self.org_module[0].forward = self.forward

    def forward(self, x):
        bias = None if self.org_module[0].bias is None else self.org_module[0].bias.data
        return self.op(
            x, self.org_module[0].weight.data * (1 + (self.concept_mask-1) * self.multiplier),
            bias, **self.extra_args
        )


class ConceptNetwork(torch.nn.Module):
    '''
    LoRA + LoCon
    '''
    # Ignore proj_in or proj_out, their channels is only a few.
    UNET_TARGET_REPLACE_MODULE = [
        "Transformer2DModel", 
        "Attention",
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention"]
    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'

    def __init__(
        self, 
        text_encoder, unet, 
        multiplier=1.0,
        **kwargs
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        
        total_params = 0
        # create module instances
        def create_modules(prefix, root_module: torch.nn.Module, target_replace_modules) -> List[ConceptModule]:
            nonlocal total_params
            print('Create Concept Module')
            loras = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        if child_name.rsplit('.', 1)[-1] not in ['to_k', 'to_v', 'k_proj', 'v_proj']:
                            continue
                        lora_name = prefix + '.' + name + '.' + child_name
                        lora_name = lora_name.replace('.', '_')
                        total_params += child_module.weight.size().numel()
                        lora = ConceptModule(
                            lora_name, child_module, self.multiplier, 
                        )
                        loras.append(lora)
            return loras

        self.text_encoder_loras = create_modules(
            ConceptNetwork.LORA_PREFIX_TEXT_ENCODER,
            text_encoder, 
            ConceptNetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
        )
        print(f"create Concept for Text Encoder: {len(self.text_encoder_loras)} modules.")

        self.unet_loras = create_modules(ConceptNetwork.LORA_PREFIX_UNET, unet, ConceptNetwork.UNET_TARGET_REPLACE_MODULE)
        print(f"create Concept for U-Net: {len(self.unet_loras)} modules.")

        self.weights_sd = None
        
        print(f'Total concept param: {total_params}')
        
        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier
            
    def load_weights(self, file):
        if os.path.splitext(file)[1] == '.safetensors':
            from safetensors.torch import load_file, safe_open
            self.weights_sd = load_file(file)
        else:
            self.weights_sd = torch.load(file, map_location='cpu')

    def apply_to(self, text_encoder, unet, apply_text_encoder=None, apply_unet=None):
        if self.weights_sd:
            weights_has_text_encoder = weights_has_unet = False
            for key in self.weights_sd.keys():
                if key.startswith(ConceptNetwork.LORA_PREFIX_TEXT_ENCODER):
                    weights_has_text_encoder = True
                elif key.startswith(ConceptNetwork.LORA_PREFIX_UNET):
                    weights_has_unet = True

            if apply_text_encoder is None:
                apply_text_encoder = weights_has_text_encoder
            else:
                assert apply_text_encoder == weights_has_text_encoder, f"text encoder weights: {weights_has_text_encoder} but text encoder flag: {apply_text_encoder} / 重みとText Encoderのフラグが矛盾しています"

            if apply_unet is None:
                apply_unet = weights_has_unet
            else:
                assert apply_unet == weights_has_unet, f"u-net weights: {weights_has_unet} but u-net flag: {apply_unet} / 重みとU-Netのフラグが矛盾しています"
        else:
            assert apply_text_encoder is not None and apply_unet is not None, f"internal error: flag not set"

        if apply_text_encoder:
            print("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        if self.weights_sd:
            # if some weights are not in state dict, it is ok because initial LoRA does nothing (lora_up is initialized by zeros)
            info = self.load_state_dict(self.weights_sd, False)
            print(f"weights are loaded: {info}")

    def enable_gradient_checkpointing(self):
        # not supported
        def make_ckpt(module):
            if isinstance(module, torch.nn.Module):
                module.grad_ckpt = True
        self.apply(make_ckpt)
        pass

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr):
        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        self.requires_grad_(True)
        all_params = []

        if self.text_encoder_loras:
            param_data = {'params': enumerate_params(self.text_encoder_loras)}
            if text_encoder_lr is not None:
                param_data['lr'] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_loras:
            param_data = {'params': enumerate_params(self.unet_loras)}
            if unet_lr is not None:
                param_data['lr'] = unet_lr
            all_params.append(param_data)

        return all_params

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == '.safetensors':
            from safetensors.torch import save_file

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)