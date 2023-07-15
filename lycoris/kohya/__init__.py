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

from .kohya_utils import *
from ..modules.locon import LoConModule
from ..modules.loha import LohaModule
from ..modules.ia3 import IA3Module
from ..modules.lokr import LokrModule
from ..modules.dylora import DyLoraModule
from ..modules.glora import GLoRAModule
from ..modules.hypernet import ImgWeightGenerator


def create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs):
    if network_dim is None:
        network_dim = 4                     # default
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
        'ia3':  IA3Module,
        'lokr': LokrModule,
        'dylora': DyLoraModule,
        'glora': GLoRAModule,
    }[algo]
    
    print(f'Using rank adaptation algo: {algo}')
    
    if ((algo == 'loha' or algo == 'lokr')
        and not kwargs.get('no_dim_warn', False) 
        and (network_dim>64 or conv_dim>64)):
        print('='*20 + 'WARNING' + '='*20)
        warning_type ={
            'loha': "Hadamard Product representation",
            'lokr': "Kronecker Product representation"
        }
        warning_msg = f"""You are not supposed to use dim>64 (64*64 = 4096, it already has enough rank)\n
            in {warning_type[algo]}!\n
            Please consider use lower dim or disable this warning with --network_args no_dim_warn=True\n
            If you just want to use high dim {algo}, please consider use lower lr.
        """
        warn(warning_msg, stacklevel=2)
        print('='*20 + 'WARNING' + '='*20)
    
    if algo == 'ia3':
        network = IA3Network(
            text_encoder, unet,
            multiplier = multiplier, 
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
            block_size = block_size
        )
    
    return network


def create_hypernetwork(multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs):
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
        down_dim=down_dim, up_dim=up_dim, delta_iters=delta_iters, decoder_blocks=decoder_blocks,
        decompose_both=kwargs.get('decompose_both', False),
        factor=kwargs.get('factor', -1),
        block_size = block_size
    )


class LycorisNetwork(torch.nn.Module):
    '''
    LoRA + LoCon
    '''
    # Ignore proj_in or proj_out, their channels is only a few.
    UNET_TARGET_REPLACE_MODULE = [
        "Transformer2DModel", 
        "Attention", 
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

    def __init__(
        self, 
        text_encoder, unet, 
        multiplier=1.0, 
        lora_dim=4, conv_lora_dim=4, 
        alpha=1, conv_alpha=1,
        use_cp = False,
        dropout = 0, rank_dropout = 0, module_dropout = 0,
        network_module = LoConModule,
        **kwargs,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.conv_lora_dim = int(conv_lora_dim)
        if self.conv_lora_dim != self.lora_dim: 
            print('Apply different lora dim for conv layer')
            print(f'Conv Dim: {conv_lora_dim}, Linear Dim: {lora_dim}')
            
        self.alpha = alpha
        self.conv_alpha = float(conv_alpha)
        if self.alpha != self.conv_alpha: 
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
            target_replace_names = []
        ) -> List[network_module]:
            print('Create LyCORIS Module')
            loras = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        lora_name = prefix + '.' + name + '.' + child_name
                        lora_name = lora_name.replace('.', '_')
                        if child_module.__class__.__name__ == 'Linear' and lora_dim>0:
                            lora = network_module(
                                lora_name, child_module, self.multiplier, 
                                self.lora_dim, self.alpha, 
                                self.dropout, self.rank_dropout, self.module_dropout, 
                                use_cp,
                                **kwargs
                            )
                        elif child_module.__class__.__name__ == 'Conv2d':
                            k_size, *_ = child_module.kernel_size
                            if k_size==1 and lora_dim>0:
                                lora = network_module(
                                    lora_name, child_module, self.multiplier, 
                                    self.lora_dim, self.alpha, 
                                    self.dropout, self.rank_dropout, self.module_dropout, 
                                    use_cp,
                                    **kwargs
                                )
                            elif conv_lora_dim>0:
                                lora = network_module(
                                    lora_name, child_module, self.multiplier, 
                                    self.conv_lora_dim, self.conv_alpha, 
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
                    if module.__class__.__name__ == 'Linear' and lora_dim>0:
                        lora = network_module(
                            lora_name, module, self.multiplier, 
                            self.lora_dim, self.alpha, 
                            self.dropout, self.rank_dropout, self.module_dropout, 
                            use_cp,
                            **kwargs
                        )
                    elif module.__class__.__name__ == 'Conv2d':
                        k_size, *_ = module.kernel_size
                        if k_size==1 and lora_dim>0:
                            lora = network_module(
                                lora_name, module, self.multiplier, 
                                self.lora_dim, self.alpha, 
                                self.dropout, self.rank_dropout, self.module_dropout, 
                                use_cp,
                                **kwargs
                            )
                        elif conv_lora_dim>0:
                            lora = network_module(
                                lora_name, module, self.multiplier, 
                                self.conv_lora_dim, self.conv_alpha, 
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
                LycorisNetwork.LORA_PREFIX_TEXT_ENCODER + (f'{i+1}' if use_index else ''),
                te, 
                LycorisNetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
            ))
        print(f"create LyCORIS for Text Encoder: {len(self.text_encoder_loras)} modules.")

        self.unet_loras = create_modules(LycorisNetwork.LORA_PREFIX_UNET, unet, LycorisNetwork.UNET_TARGET_REPLACE_MODULE)
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
        
        return key_scaled, sum(norms)/len(norms), max(norms)

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
        
        return scaled, sum(norms)/len(norms), max(norms)


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
        use_cp = False,
        dropout = 0, rank_dropout = 0, module_dropout = 0,
        network_module = LoConModule,
        down_dim = 100, up_dim = 50, delta_iters = 5, decoder_blocks = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        
        if 1 >= dropout >= 0:
            print(f'Use Dropout value: {dropout}')
        if network_module != LoConModule:
            print('HyperDreamBooth only support LoRA at this time')
            raise NotImplementedError
        if lora_dim*(down_dim+up_dim) > 4096:
            print('weight elements > 4096 (dim * (down_dim + up_dim)) is not recommended!')
        
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        
        # create module instances
        def create_modules(
            prefix, 
            root_module: torch.nn.Module, 
            target_replace_modules,
            target_replace_names = []
        ) -> List[network_module]:
            print('Create LyCORIS Module')
            loras = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        lora_name = prefix + '.' + name + '.' + child_name
                        lora_name = lora_name.replace('.', '_')
                        if child_module.__class__.__name__ == 'Linear' and lora_dim>0:
                            lora = network_module(
                                lora_name, child_module, self.multiplier, 
                                self.lora_dim, self.alpha, 
                                self.dropout, self.rank_dropout, self.module_dropout, 
                                use_cp,
                                **kwargs
                            )
                        elif child_module.__class__.__name__ == 'Conv2d':
                            k_size, *_ = child_module.kernel_size
                            if k_size==1 and lora_dim>0:
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
                    if module.__class__.__name__ == 'Linear' and lora_dim>0:
                        lora = network_module(
                            lora_name, module, self.multiplier, 
                            self.lora_dim, self.alpha, 
                            self.dropout, self.rank_dropout, self.module_dropout, 
                            use_cp,
                            **kwargs
                        )
                    elif module.__class__.__name__ == 'Conv2d':
                        k_size, *_ = module.kernel_size
                        if k_size==1 and lora_dim>0:
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
                LycorisNetwork.LORA_PREFIX_TEXT_ENCODER + (f'{i+1}' if use_index else ''),
                te, 
                LycorisNetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
            ))
        print(f"create LyCORIS for Text Encoder: {len(self.text_encoder_loras)} modules.")

        self.unet_loras = create_modules(LycorisNetwork.LORA_PREFIX_UNET, unet, LycorisNetwork.UNET_TARGET_REPLACE_MODULE)
        print(f"create LyCORIS for U-Net: {len(self.unet_loras)} modules.")
        
        self.loras: list[LoConModule] = self.text_encoder_loras + self.unet_loras
        self.weight_generater = ImgWeightGenerator(
            weight_dim=(down_dim+up_dim)*lora_dim,
            weight_num=len(self.loras),
            sample_iters=delta_iters,
            decoder_blocks=decoder_blocks,
        )
        self.split = (down_dim* lora_dim, up_dim  * lora_dim)
        self.lora_dim = lora_dim

        self.weights_sd = None

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)
        
        self.checkpoint = torch.nn.Parameter(torch.tensor(0.0))
        
        with torch.no_grad():
            self.update_reference(torch.randn(1, 3, *self.weight_generater.ref_size))
        
        for lora in self.loras:
            assert torch.all(lora.data[0]==0)

    def update_reference(self, ref_img, iter=None):
        # use idx for aux weight seed
        weights = self.weight_generater(ref_img, iter)
        weights = weights + self.checkpoint
        for idx, (lora, weight) in enumerate(zip(self.loras, weights.split(1, dim=1))):
            assert lora.multiplier > 0, f"multiplier must be positive: {lora.multiplier}"
            # weight: [batch, 1, weight_dim]
            # if weight.dim()==3:
            #     weight = weight.squeeze(1)
            lora.update_weights(*weight.split(self.split, dim=-1), idx)

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
        # not supported
        pass

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr):
        self.requires_grad_(True)
        all_params = []
        all_params.append({
            'params': (
                [p for p in self.weight_generater.decoder_model.parameters()]
                + ([p for p in self.weight_generater.encoder_model.parameters()] 
                   if self.weight_generater.train_encoder else [])
            ), 
            'lr': text_encoder_lr
        })
        all_params.append({
            'params': (
                [p for p in self.weight_generater.pos_emb_proj.parameters()]
                + [p for p in self.weight_generater.feature_proj.parameters()]
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

        state_dict = self.weight_generater.state_dict()
        if not self.weight_generater.train_encoder:
            for k in self.weight_generater.encoder_model.state_dict().keys():
                state_dict.pop(f'encoder_model.{k}')

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
    TEXT_ENCODER_TARGET_REPLACE_NAME= ["k_proj", "v_proj", "mlp.fc2"]
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
            target_replace_names = [],
            target_train_input = []
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
