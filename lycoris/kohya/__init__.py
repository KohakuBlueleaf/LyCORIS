# network module for kohya
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

import math
from warnings import warn
import os
from typing import List
import torch
import torch.utils.checkpoint as checkpoint

from .utils import *
from ..modules.locon import LoConModule
from ..modules.loha import LohaModule
from ..modules.ia3 import IA3Module
from ..modules.lokr import LokrModule
from ..modules.dylora import DyLoraModule
from ..modules.glora import GLoRAModule
from ..modules.hypernet import ImgWeightGenerator, TextWeightGenerator

from ..config import PRESET
from ..utils.preset import read_preset


def create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs):
    if network_dim is None:
        network_dim = 4  # default
    conv_dim = int(kwargs.get('conv_dim', network_dim) or network_dim)
    conv_alpha = float(kwargs.get('conv_alpha', network_alpha) or network_alpha)
    dropout = float(kwargs.get('dropout', 0.) or 0.)
    rank_dropout = float(kwargs.get("rank_dropout", 0.) or 0.)
    module_dropout = float(kwargs.get("module_dropout", 0.) or 0.)
    algo = (kwargs.get('algo', 'lora') or 'lora').lower()
    use_cp = (not kwargs.get('disable_conv_cp', True)
              or kwargs.get('use_conv_cp', False))
    block_size = int(kwargs.get('block_size', 4) or 4)
    network_module = {
        'lora': LoConModule,
        'locon': LoConModule,
        'loha': LohaModule,
        'ia3': IA3Module,
        'lokr': LokrModule,
        'dylora': DyLoraModule,
        'glora': GLoRAModule,
    }[algo]

    preset = kwargs.get('preset', 'full')
    if preset not in PRESET:
        preset = read_preset(preset)
    else:
        preset = PRESET[preset]
    assert preset is not None
    LycorisNetwork.apply_preset(preset)

    print(f'Using rank adaptation algo: {algo}')

    if ((algo == 'loha')
            and not kwargs.get('no_dim_warn', False)
            and (network_dim > 64 or conv_dim > 64)):
        print('=' * 20 + 'WARNING' + '=' * 20)
        warning_type = {
            'loha': "Hadamard Product representation"
        }
        warning_msg = f"""You are not supposed to use dim>64 (64*64 = 4096, it already has enough rank)\n
            in {warning_type[algo]}!\n
            Please consider use lower dim or disable this warning with --network_args no_dim_warn=True\n
            If you just want to use high dim {algo}, please consider use lower lr.
        """
        warn(warning_msg, stacklevel=2)
        print('=' * 20 + 'WARNING' + '=' * 20)

    if algo == 'ia3':
        network = IA3Network(
            text_encoder, unet,
            multiplier=multiplier,
        )
    else:
        network = LycorisNetwork(
            text_encoder, unet,
            multiplier=multiplier,
            lora_dim=network_dim, conv_lora_dim=conv_dim,
            alpha=network_alpha, conv_alpha=conv_alpha,
            dropout=dropout, rank_dropout=rank_dropout, module_dropout=module_dropout,
            use_cp=use_cp,
            network_module=network_module,
            decompose_both=kwargs.get('decompose_both', False),
            factor=kwargs.get('factor', -1),
            block_size=block_size
        )

    if algo == 'dylora':
        # dylora didn't support scale weight norm yet
        delattr(network, 'apply_max_norm_regularization')

    return network


def create_network_from_weights(multiplier, file, vae, text_encoder, unet, weights_sd=None, for_inference=False,
                                network_dim=4, network_alpha=1, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # get dim/alpha mapping
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        else:
            # elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim
        # else:
        #     print(f"Key: '{key}'")
        # print(lora_name, value.size(), dim)

    # support old LoRA without alpha
    for key in modules_dim.keys():
        if key not in modules_alpha:
            modules_alpha[key] = modules_dim[key]

    if network_dim is None:
        network_dim = 4  # default
    conv_dim = int(kwargs.get('conv_dim', network_dim) or network_dim)
    conv_alpha = float(kwargs.get('conv_alpha', network_alpha) or network_alpha)
    dropout = float(kwargs.get('dropout', 0.) or 0.)
    rank_dropout = float(kwargs.get("rank_dropout", 0.) or 0.)
    module_dropout = float(kwargs.get("module_dropout", 0.) or 0.)
    algo = (kwargs.get('algo', 'lora') or 'lora').lower()
    use_cp = (not kwargs.get('disable_conv_cp', True)
              or kwargs.get('use_conv_cp', False))
    block_size = int(kwargs.get('block_size', 4) or 4)
    network_module = {
        'lora': LoConModule,
        'locon': LoConModule,
        'loha': LohaModule,
        'ia3': IA3Module,
        'lokr': LokrModule,
        'dylora': DyLoraModule,
        'glora': GLoRAModule,
    }[algo]

    preset = kwargs.get('preset', 'full')
    if preset not in PRESET:
        preset = read_preset(preset)
    else:
        preset = PRESET[preset]
    assert preset is not None
    LycorisNetwork.apply_preset(preset)

    print(f'Using rank adaptation algo: {algo}')

    if ((algo == 'loha')
            and not kwargs.get('no_dim_warn', False)
            and (network_dim > 64 or conv_dim > 64)):
        print('=' * 20 + 'WARNING' + '=' * 20)
        warning_type = {
            'loha': "Hadamard Product representation"
        }
        warning_msg = f"""You are not supposed to use dim>64 (64*64 = 4096, it already has enough rank)\n
            in {warning_type[algo]}!\n
            Please consider use lower dim or disable this warning with --network_args no_dim_warn=True\n
            If you just want to use high dim {algo}, please consider use lower lr.
        """
        warn(warning_msg, stacklevel=2)
        print('=' * 20 + 'WARNING' + '=' * 20)

    if algo == 'ia3':
        network = IA3Network(
            text_encoder, unet,
            multiplier=multiplier,
        )
    else:
        network = LycorisNetwork(
            text_encoder, unet,
            multiplier=multiplier,
            lora_dim=network_dim, conv_lora_dim=conv_dim,
            alpha=network_alpha, conv_alpha=conv_alpha,
            dropout=dropout, rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            use_cp=use_cp,
            network_module=network_module,
            decompose_both=kwargs.get('decompose_both', False),
            factor=kwargs.get('factor', -1),
            block_size=block_size, modules_dim=modules_dim, modules_alpha=modules_alpha
        )

    if algo == 'dylora':
        # dylora didn't support scale weight norm yet
        delattr(network, 'apply_max_norm_regularization')
    return network, weights_sd


def create_hypernetwork(multiplier, network_dim, network_alpha, vae, text_encoder, unet, vocab_size, **kwargs):
    if network_dim is None:
        network_dim = 4
    dropout = float(kwargs.get('dropout', 0.) or 0.)
    rank_dropout = float(kwargs.get("rank_dropout", 0.) or 0.)
    module_dropout = float(kwargs.get("module_dropout", 0.) or 0.)
    algo = (kwargs.get('algo', 'lora') or 'lora').lower()
    use_cp = (not kwargs.get('disable_conv_cp', True)
              or kwargs.get('use_conv_cp', False))
    block_size = int(kwargs.get('block_size', 4) or 4)
    down_dim = int(kwargs.get('down_dim', 128) or 128)
    up_dim = int(kwargs.get('up_dim', 64) or 64)
    delta_iters = int(kwargs.get('delta_iters', 5) or 5)
    decoder_blocks = int(kwargs.get('decoder_blocks', 4) or 4)
    network_module = {
        'lora': LoConModule,
        'locon': LoConModule,
    }[algo]

    print(f'Using rank adaptation algo: {algo}')

    return HyperDreamNetwork(
        text_encoder, unet,
        multiplier=multiplier,
        lora_dim=network_dim, alpha=network_alpha,
        use_cp=use_cp,
        dropout=dropout, rank_dropout=rank_dropout, module_dropout=module_dropout,
        network_module=network_module,
        down_dim=down_dim, up_dim=up_dim, delta_iters=delta_iters,
        decoder_blocks=decoder_blocks, vocab_size=vocab_size,
        decompose_both=kwargs.get('decompose_both', False),
        factor=kwargs.get('factor', -1),
        block_size=block_size
    )


class LycorisNetwork(torch.nn.Module):
    '''
    LoRA + LoCon
    '''
    # Ignore proj_in or proj_out, their channels is only a few.
    ENABLE_CONV = True
    UNET_TARGET_REPLACE_MODULE = [
        "Transformer2DModel",
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D"
    ]
    UNET_TARGET_REPLACE_NAME = [
        "conv_in",
        "conv_out",
        "time_embedding.linear_1",
        "time_embedding.linear_2",
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'
    MODULE_ALGO_MAP = {}
    NAME_ALGO_MAP = {}

    @classmethod
    def apply_preset(cls, preset):
        if 'enable_conv' in preset:
            cls.ENABLE_CONV = preset['enable_conv']
        if 'unet_target_module' in preset:
            cls.UNET_TARGET_REPLACE_MODULE = preset['unet_target_module']
        if 'unet_target_name' in preset:
            cls.UNET_TARGET_REPLACE_NAME = preset['unet_target_name']
        if 'text_encoder_target_module' in preset:
            cls.TEXT_ENCODER_TARGET_REPLACE_MODULE = preset['text_encoder_target_module']
        if 'text_encoder_target_name' in preset:
            cls.TEXT_ENCODER_TARGET_REPLACE_NAME = preset['text_encoder_target_name']
        if 'module_algo_map' in preset:
            cls.MODULE_ALGO_MAP = preset['module_algo_map']
        if 'name_algo_map' in preset:
            cls.NAME_ALGO_MAP = preset['name_algo_map']
        return cls

    def __init__(
            self,
            text_encoder, unet,
            multiplier=1.0,
            lora_dim=4, conv_lora_dim=4,
            alpha=1, conv_alpha=1,
            use_cp=False,
            dropout=0, rank_dropout=0, module_dropout=0,
            network_module=LoConModule,
            modules_dim=None,
            modules_alpha=None,
            **kwargs,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim

        if not self.ENABLE_CONV:
            conv_lora_dim = 0

        self.conv_lora_dim = int(conv_lora_dim)
        if modules_dim is not None:
            print(f"Creating LoRA network from weights")

        if self.conv_lora_dim and self.conv_lora_dim != self.lora_dim:
            print('Apply different lora dim for conv layer')
            print(f'Conv Dim: {conv_lora_dim}, Linear Dim: {lora_dim}')
        elif self.conv_lora_dim == 0:
            print('Disable conv layer')

        self.alpha = alpha
        self.conv_alpha = float(conv_alpha)
        if self.conv_lora_dim and self.alpha != self.conv_alpha:
            print('Apply different alpha value for conv layer')
            print(f'Conv alpha: {conv_alpha}, Linear alpha: {alpha}')

        if 1 >= dropout >= 0:
            print(f'Use Dropout value: {dropout}')
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        # create module instances
        def create_modules(
                prefix,
                root_module: torch.nn.Module,
                target_replace_modules,
                target_replace_names=[]
        ) -> List[network_module]:
            print('Create LyCORIS Module')
            loras = []
            for name, module in root_module.named_modules():
                module_name = module.__class__.__name__
                if module_name in target_replace_modules:
                    if module_name in self.MODULE_ALGO_MAP:
                        algo = self.MODULE_ALGO_MAP[module_name]
                    else:
                        algo = network_module
                    for child_name, child_module in module.named_modules():
                        lora_name = prefix + '.' + name + '.' + child_name
                        lora_name = lora_name.replace('.', '_')

                        # print(modules_dim, lora_name)
                        if modules_dim is not None and lora_name in modules_dim:
                            dim = modules_dim[lora_name]
                            alpha = modules_alpha[lora_name]
                            # print(f"Found dim: {dim} alpha: {alpha}")
                        else:
                            dim = self.lora_dim
                            alpha = self.alpha

                        if child_module.__class__.__name__ == 'Linear' and lora_dim > 0:
                            lora = algo(
                                lora_name, child_module, self.multiplier,
                                dim, alpha,
                                self.dropout, self.rank_dropout, self.module_dropout,
                                use_cp,
                                **kwargs
                            )
                        elif child_module.__class__.__name__ == 'Conv2d':
                            k_size, *_ = child_module.kernel_size
                            if k_size == 1 and lora_dim > 0:
                                lora = algo(
                                    lora_name, child_module, self.multiplier,
                                    dim, alpha,
                                    self.dropout, self.rank_dropout, self.module_dropout,
                                    use_cp,
                                    **kwargs
                                )
                            elif conv_lora_dim > 0:
                                lora = algo(
                                    lora_name, child_module, self.multiplier,
                                    dim, alpha,
                                    self.dropout, self.rank_dropout, self.module_dropout,
                                    use_cp,
                                    **kwargs
                                )
                            else:
                                continue
                        else:
                            continue
                        loras.append(lora)
                elif name in target_replace_names:
                    if name in self.NAME_ALGO_MAP:
                        algo = self.NAME_ALGO_MAP[name]
                    else:
                        algo = network_module
                    lora_name = prefix + '.' + name
                    lora_name = lora_name.replace('.', '_')

                    if modules_dim is not None and lora_name in modules_dim:
                        dim = modules_dim[lora_name]
                        alpha = modules_alpha[lora_name]
                        print(f"Found dim: {dim} alpha: {alpha}")
                    else:
                        dim = self.lora_dim
                        alpha = self.alpha

                    if module.__class__.__name__ == 'Linear' and lora_dim > 0:
                        lora = algo(
                            lora_name, module, self.multiplier,
                            dim, alpha,
                            self.dropout, self.rank_dropout, self.module_dropout,
                            use_cp,
                            **kwargs
                        )
                    elif module.__class__.__name__ == 'Conv2d':
                        k_size, *_ = module.kernel_size
                        if k_size == 1 and lora_dim > 0:
                            lora = algo(
                                lora_name, module, self.multiplier,
                                dim, alpha,
                                self.dropout, self.rank_dropout, self.module_dropout,
                                use_cp,
                                **kwargs
                            )
                        elif conv_lora_dim > 0:
                            lora = algo(
                                lora_name, module, self.multiplier,
                                dim, alpha,
                                self.dropout, self.rank_dropout, self.module_dropout,
                                use_cp,
                                **kwargs
                            )
                        else:
                            continue
                    else:
                        continue
                    loras.append(lora)
            return loras

        if network_module == GLoRAModule:
            print('GLoRA enabled, only train transformer')
            # only train transformer (for GLoRA)
            LycorisNetwork.UNET_TARGET_REPLACE_MODULE = [
                "Transformer2DModel",
                "Attention",
            ]
            LycorisNetwork.UNET_TARGET_REPLACE_NAME = []

        if isinstance(text_encoder, list):
            text_encoders = text_encoder
            use_index = True
        else:
            text_encoders = [text_encoder]
            use_index = False

        self.text_encoder_loras = []
        for i, te in enumerate(text_encoders):
            self.text_encoder_loras.extend(create_modules(
                LycorisNetwork.LORA_PREFIX_TEXT_ENCODER + (f'{i + 1}' if use_index else ''),
                te,
                LycorisNetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
            ))
        print(f"create LyCORIS for Text Encoder: {len(self.text_encoder_loras)} modules.")

        self.unet_loras = create_modules(LycorisNetwork.LORA_PREFIX_UNET, unet,
                                         LycorisNetwork.UNET_TARGET_REPLACE_MODULE)
        print(f"create LyCORIS for U-Net: {len(self.unet_loras)} modules.")

        self.weights_sd = None

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
                if key.startswith(LycorisNetwork.LORA_PREFIX_TEXT_ENCODER):
                    weights_has_text_encoder = True
                elif key.startswith(LycorisNetwork.LORA_PREFIX_UNET):
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
            print("enable LyCORIS for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LyCORIS for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        if self.weights_sd:
            # if some weights are not in state dict, it is ok because initial LoRA does nothing (lora_up is initialized by zeros)
            info = self.load_state_dict(self.weights_sd, False)
            print(f"weights are loaded: {info}")

    def apply_max_norm_regularization(self, max_norm_value, device):
        key_scaled = 0
        norms = []
        for model in self.unet_loras:
            if hasattr(model, 'apply_max_norm'):
                scaled, norm = model.apply_max_norm(max_norm_value, device)
                norms.append(norm)
                key_scaled += scaled

        for model in self.text_encoder_loras:
            if hasattr(model, 'apply_max_norm'):
                scaled, norm = model.apply_max_norm(max_norm_value, device)
                norms.append(norm)
                key_scaled += scaled

        return key_scaled, sum(norms) / len(norms), max(norms)

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

    def apply_max_norm_regularization(self, max_norm_value, device):
        norms = []
        scaled = 0

        for lora in self.unet_loras:
            if hasattr(lora, 'apply_max_norm'):
                scaled, norm = lora.apply_max_norm(max_norm_value, device)
                norms.append(norm)
                scaled += int(scaled)

        for lora in self.text_encoder_loras:
            if hasattr(lora, 'apply_max_norm'):
                scaled, norm = lora.apply_max_norm(max_norm_value, device)
                norms.append(norm)
                scaled += int(scaled)

        return scaled, sum(norms) / len(norms), max(norms)


class HyperDreamNetwork(torch.nn.Module):
    '''
    HyperDreamBooth hypernetwork part
    only train Attention right now
    '''
    UNET_TARGET_REPLACE_MODULE = [
        "Attention",
    ]
    UNET_TARGET_REPLACE_NAME = []
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention"]
    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'

    def __init__(
            self,
            text_encoder, unet,
            multiplier=1.0,
            lora_dim=4, alpha=1,
            use_cp=False,
            dropout=0, rank_dropout=0, module_dropout=0,
            network_module=LoConModule,
            down_dim=100, up_dim=50, delta_iters=5, decoder_blocks=4, vocab_size=49408,
            **kwargs,
    ) -> None:
        super().__init__()
        self.gradient_ckpt = False
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha

        if 1 >= dropout >= 0:
            print(f'Use Dropout value: {dropout}')
        if network_module != LoConModule:
            print('HyperDreamBooth only support LoRA at this time')
            raise NotImplementedError
        if lora_dim * (down_dim + up_dim) > 4096:
            print('weight elements > 4096 (dim * (down_dim + up_dim)) is not recommended!')

        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        # create module instances
        def create_modules(
                prefix,
                root_module: torch.nn.Module,
                target_replace_modules,
                target_replace_names=[]
        ) -> List[network_module]:
            print('Create LyCORIS Module')
            loras = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        lora_name = prefix + '.' + name + '.' + child_name
                        lora_name = lora_name.replace('.', '_')
                        if child_module.__class__.__name__ == 'Linear' and lora_dim > 0:
                            lora = network_module(
                                lora_name, child_module, self.multiplier,
                                self.lora_dim, self.alpha,
                                self.dropout, self.rank_dropout, self.module_dropout,
                                use_cp,
                                **kwargs
                            )
                        elif child_module.__class__.__name__ == 'Conv2d':
                            k_size, *_ = child_module.kernel_size
                            if k_size == 1 and lora_dim > 0:
                                lora = network_module(
                                    lora_name, child_module, self.multiplier,
                                    self.lora_dim, self.alpha,
                                    self.dropout, self.rank_dropout, self.module_dropout,
                                    use_cp,
                                    **kwargs
                                )
                            else:
                                continue
                        else:
                            continue
                        loras.append(lora)
                elif name in target_replace_names:
                    lora_name = prefix + '.' + name
                    lora_name = lora_name.replace('.', '_')
                    if module.__class__.__name__ == 'Linear' and lora_dim > 0:
                        lora = network_module(
                            lora_name, module, self.multiplier,
                            self.lora_dim, self.alpha,
                            self.dropout, self.rank_dropout, self.module_dropout,
                            use_cp,
                            **kwargs
                        )
                    elif module.__class__.__name__ == 'Conv2d':
                        k_size, *_ = module.kernel_size
                        if k_size == 1 and lora_dim > 0:
                            lora = network_module(
                                lora_name, module, self.multiplier,
                                self.lora_dim, self.alpha,
                                self.dropout, self.rank_dropout, self.module_dropout,
                                use_cp,
                                **kwargs
                            )
                        else:
                            continue
                    else:
                        continue
                    loras.append(lora)
            return loras

        if isinstance(text_encoder, list):
            text_encoders = text_encoder
            use_index = True
        else:
            text_encoders = [text_encoder]
            use_index = False

        self.text_encoder_loras = []
        for i, te in enumerate(text_encoders):
            self.text_encoder_loras.extend(create_modules(
                LycorisNetwork.LORA_PREFIX_TEXT_ENCODER + (f'{i + 1}' if use_index else ''),
                te,
                LycorisNetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
            ))
        print(f"create LyCORIS for Text Encoder: {len(self.text_encoder_loras)} modules.")

        self.unet_loras = create_modules(LycorisNetwork.LORA_PREFIX_UNET, unet,
                                         LycorisNetwork.UNET_TARGET_REPLACE_MODULE)
        print(f"create LyCORIS for U-Net: {len(self.unet_loras)} modules.")

        self.loras: list[LoConModule] = self.text_encoder_loras + self.unet_loras
        self.img_weight_generater = ImgWeightGenerator(
            weight_dim=(down_dim + up_dim) * lora_dim,
            weight_num=len(self.unet_loras),
            sample_iters=delta_iters,
            decoder_blocks=decoder_blocks,
        )
        self.text_weight_generater = TextWeightGenerator(
            weight_dim=(down_dim + up_dim) * lora_dim,
            weight_num=len(self.text_encoder_loras),
            sample_iters=delta_iters,
            decoder_blocks=decoder_blocks,
        )
        self.split = (down_dim * lora_dim, up_dim * lora_dim)
        self.lora_dim = lora_dim

        self.weights_sd = None

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

        self.checkpoint = torch.nn.Parameter(torch.tensor(0.0))

        with torch.no_grad():
            self.update_reference(
                torch.randn(1, 3, *self.img_weight_generater.ref_size),
                ["test"]
            )

        # for lora in self.loras:
        #     assert torch.all(lora.data[0]==0)

    def gen_weight(self, ref_img, caption, iter=None, ensure_grad=0):
        unet_weights = self.img_weight_generater(ref_img, iter, ensure_grad=ensure_grad)
        unet_weights = unet_weights + self.checkpoint
        unet_weights = [i.split(self.split, dim=-1) for i in unet_weights.split(1, dim=1)]
        text_weights = self.text_weight_generater(caption, iter, ensure_grad=ensure_grad)
        text_weights = text_weights + self.checkpoint
        text_weights = [i.split(self.split, dim=-1) for i in text_weights.split(1, dim=1)]
        return unet_weights, text_weights

    def update_reference(self, ref_img, caption, iter=None):
        # use idx for aux weight seed
        if self.gradient_ckpt and self.training:
            ensure_grad = torch.zeros(1, device=ref_img.device, requires_grad=True)
            unet_weights_list, text_weights_list = checkpoint.checkpoint(
                self.gen_weight, ref_img, caption, iter, ensure_grad
            )
        else:
            unet_weights_list, text_weights_list = self.gen_weight(ref_img, caption, iter)

        for idx, (lora, weight) in enumerate(zip(self.unet_loras, unet_weights_list)):
            assert lora.multiplier > 0, f"multiplier must be positive: {lora.multiplier}"
            # weight: [batch, 1, weight_dim]
            # if weight.dim()==3:
            #     weight = weight.squeeze(1)
            lora.update_weights(*weight, idx)

        for idx, (lora, weight) in enumerate(zip(self.text_encoder_loras, text_weights_list)):
            assert lora.multiplier > 0, f"multiplier must be positive: {lora.multiplier}"
            # weight: [batch, 1, weight_dim]
            # if weight.dim()==3:
            #     weight = weight.squeeze(1)
            lora.update_weights(*weight, idx)

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
                if key.startswith(LycorisNetwork.LORA_PREFIX_TEXT_ENCODER):
                    weights_has_text_encoder = True
                elif key.startswith(LycorisNetwork.LORA_PREFIX_UNET):
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
            print("enable LyCORIS for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LyCORIS for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to(is_hypernet=True)

    def enable_gradient_checkpointing(self):
        self.gradient_ckpt = True

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, learning_rate):
        self.requires_grad_(True)
        all_params = []

        if self.text_encoder_loras:
            all_params.append({
                'params': (
                        [p for p in self.text_weight_generater.decoder_model.parameters()]
                        + [p for p in self.text_weight_generater.pos_emb_proj.parameters()]
                        + [p for p in self.text_weight_generater.feature_proj.parameters()]
                        + ([p for p in self.text_weight_generater.encoder_model.parameters()]
                           if self.text_weight_generater.train_encoder else [])
                ),
                'lr': text_encoder_lr
            })
        if self.unet_loras:
            all_params.append({
                'params': (
                        [p for p in self.img_weight_generater.decoder_model.parameters()]
                        + [p for p in self.img_weight_generater.pos_emb_proj.parameters()]
                        + [p for p in self.img_weight_generater.feature_proj.parameters()]
                        + ([p for p in self.img_weight_generater.encoder_model.parameters()]
                           if self.img_weight_generater.train_encoder else [])
                ),
                'lr': unet_lr
            })
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

        state_dict = self.img_weight_generater.state_dict()
        if not self.img_weight_generater.train_encoder:
            for k in self.img_weight_generater.encoder_model.state_dict().keys():
                state_dict.pop(f'encoder_model.{k}')
        state_dict = {f'img_weight_generater.{i}': v for i, v in state_dict.items()}

        state_dict = self.text_weight_generater.state_dict()
        if not self.text_weight_generater.train_encoder:
            for k in self.text_weight_generater.encoder_model.state_dict().keys():
                state_dict.pop(f'encoder_model.{k}')
        state_dict = {f'text_weight_generater.{i}': v for i, v in state_dict.items()}

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


class IA3Network(torch.nn.Module):
    '''
    IA3 network
    '''
    # Ignore proj_in or proj_out, their channels is only a few.
    UNET_TARGET_REPLACE_MODULE = []
    UNET_TARGET_REPLACE_NAME = ["to_k", "to_v", "ff.net.2"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = []
    TEXT_ENCODER_TARGET_REPLACE_NAME = ["k_proj", "v_proj", "mlp.fc2"]
    TRAIN_INPUT = ["mlp.fc2", "ff.net.2"]
    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'

    def __init__(
            self,
            text_encoder, unet,
            multiplier=1.0,
            **kwargs,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier

        # create module instances
        def create_modules(
                prefix,
                root_module: torch.nn.Module,
                target_replace_modules,
                target_replace_names=[],
                target_train_input=[]
        ) -> List[IA3Module]:
            print('Create LyCORIS Module')
            loras = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        lora_name = prefix + '.' + name + '.' + child_name
                        lora_name = lora_name.replace('.', '_')
                        if child_module.__class__.__name__ in {'Linear', 'Conv2d'}:
                            lora = IA3Module(
                                lora_name, child_module, self.multiplier,
                                name in target_train_input,
                                **kwargs,
                            )
                            loras.append(lora)
                elif any(i in name for i in target_replace_names):
                    lora_name = prefix + '.' + name
                    lora_name = lora_name.replace('.', '_')
                    if module.__class__.__name__ in {'Linear', 'Conv2d'}:
                        lora = IA3Module(
                            lora_name, module, self.multiplier,
                            name in target_train_input,
                            **kwargs,
                        )
                        loras.append(lora)
            return loras

        self.text_encoder_loras = create_modules(
            IA3Network.LORA_PREFIX_TEXT_ENCODER,
            text_encoder,
            IA3Network.TEXT_ENCODER_TARGET_REPLACE_MODULE,
            IA3Network.TEXT_ENCODER_TARGET_REPLACE_NAME,
            IA3Network.TRAIN_INPUT
        )
        print(f"create LyCORIS for Text Encoder: {len(self.text_encoder_loras)} modules.")

        self.unet_loras = create_modules(
            IA3Network.LORA_PREFIX_UNET,
            unet,
            IA3Network.UNET_TARGET_REPLACE_MODULE,
            IA3Network.UNET_TARGET_REPLACE_NAME,
            IA3Network.TRAIN_INPUT
        )
        print(f"create LyCORIS for U-Net: {len(self.unet_loras)} modules.")

        self.weights_sd = None

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
                if key.startswith(LycorisNetwork.LORA_PREFIX_TEXT_ENCODER):
                    weights_has_text_encoder = True
                elif key.startswith(LycorisNetwork.LORA_PREFIX_UNET):
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
            print("enable LyCORIS for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            print("enable LyCORIS for U-Net")
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
