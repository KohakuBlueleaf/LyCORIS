import os
import re
import logging
from typing import Any, List

import torch

from .utils import precalculate_safetensors_hashes
from .wrapper import LycorisNetwork, network_module_dict, deprecated_arg_dict
from .modules.locon import LoConModule
from .modules.loha import LohaModule
from .modules.ia3 import IA3Module
from .modules.lokr import LokrModule
from .modules.dylora import DyLoraModule
from .modules.glora import GLoRAModule
from .modules.norms import NormModule
from .modules.full import FullModule
from .modules.diag_oft import DiagOFTModule
from .modules.boft import ButterflyOFTModule
from .modules import make_module, get_module

from .config import PRESET
from .utils.preset import read_preset
from .utils import str_bool
from .logging import logger


def create_network(
    multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs
):
    for key, value in kwargs.items():
        if key in deprecated_arg_dict:
            logger.warning(
                f"{key} is deprecated. Please use {deprecated_arg_dict[key]} instead.",
                stacklevel=2,
            )
            kwargs[deprecated_arg_dict[key]] = value
    if network_dim is None:
        network_dim = 4  # default
    conv_dim = int(kwargs.get("conv_dim", network_dim) or network_dim)
    conv_alpha = float(kwargs.get("conv_alpha", network_alpha) or network_alpha)
    dropout = float(kwargs.get("dropout", 0.0) or 0.0)
    rank_dropout = float(kwargs.get("rank_dropout", 0.0) or 0.0)
    module_dropout = float(kwargs.get("module_dropout", 0.0) or 0.0)
    algo = (kwargs.get("algo", "lora") or "lora").lower()
    use_tucker = str_bool(
        not kwargs.get("disable_conv_cp", True)
        or kwargs.get("use_conv_cp", False)
        or kwargs.get("use_cp", False)
        or kwargs.get("use_tucker", False)
    )
    use_scalar = str_bool(kwargs.get("use_scalar", False))
    block_size = int(kwargs.get("block_size", None) or 4)
    train_norm = str_bool(kwargs.get("train_norm", False))
    constraint = float(kwargs.get("constraint", None) or 0)
    rescaled = str_bool(kwargs.get("rescaled", False))
    weight_decompose = str_bool(kwargs.get("dora_wd", False))
    full_matrix = str_bool(kwargs.get("full_matrix", False))
    bypass_mode = str_bool(kwargs.get("bypass_mode", False))
    rs_lora = str_bool(kwargs.get("rs_lora", False))
    unbalanced_factorization = str_bool(kwargs.get("unbalanced_factorization", False))

    if unbalanced_factorization:
        logger.info("Unbalanced factorization for LoKr is enabled")

    if bypass_mode:
        logger.info("Bypass mode is enabled")

    if weight_decompose:
        logger.info("Weight decomposition is enabled")

    if full_matrix:
        logger.info("Full matrix mode for LoKr is enabled")

    preset_str = kwargs.get("preset", "full")
    if preset_str not in PRESET:
        preset = read_preset(preset_str)
    else:
        preset = PRESET[preset_str]
    assert preset is not None
    LycorisNetworkKohya.apply_preset(preset)

    logger.info(f"Using rank adaptation algo: {algo}")

    if algo == "ia3" and preset_str != "ia3":
        logger.warning("It is recommended to use preset ia3 for IA^3 algorithm")

    network = LycorisNetworkKohya(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        conv_lora_dim=conv_dim,
        alpha=network_alpha,
        conv_alpha=conv_alpha,
        dropout=dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        use_tucker=use_tucker,
        use_scalar=use_scalar,
        network_module=algo,
        train_norm=train_norm,
        decompose_both=kwargs.get("decompose_both", False),
        factor=kwargs.get("factor", -1),
        block_size=block_size,
        constraint=constraint,
        rescaled=rescaled,
        weight_decompose=weight_decompose,
        full_matrix=full_matrix,
        bypass_mode=bypass_mode,
        rs_lora=rs_lora,
        unbalanced_factorization=unbalanced_factorization,
    )

    return network


def create_network_from_weights(
    multiplier,
    file,
    vae,
    text_encoder,
    unet,
    weights_sd=None,
    for_inference=False,
    **kwargs,
):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # get dim/alpha mapping
    unet_loras = {}
    te_loras = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if lora_name.startswith(LycorisNetworkKohya.LORA_PREFIX_UNET):
            unet_loras[lora_name] = None
        elif lora_name.startswith(LycorisNetworkKohya.LORA_PREFIX_TEXT_ENCODER):
            te_loras[lora_name] = None

    for name, modules in unet.named_modules():
        lora_name = f"{LycorisNetworkKohya.LORA_PREFIX_UNET}_{name}".replace(".", "_")
        if lora_name in unet_loras:
            unet_loras[lora_name] = modules

    if isinstance(text_encoder, list):
        text_encoders = text_encoder
        use_index = True
    else:
        text_encoders = [text_encoder]
        use_index = False

    for idx, te in enumerate(text_encoders):
        if use_index:
            prefix = f"{LycorisNetworkKohya.LORA_PREFIX_TEXT_ENCODER}{idx+1}"
        else:
            prefix = LycorisNetworkKohya.LORA_PREFIX_TEXT_ENCODER
        for name, modules in te.named_modules():
            lora_name = f"{prefix}_{name}".replace(".", "_")
            if lora_name in te_loras:
                te_loras[lora_name] = modules

    original_level = logger.level
    logger.setLevel(logging.ERROR)
    network = LycorisNetworkKohya(text_encoder, unet)
    network.unet_loras = []
    network.text_encoder_loras = []
    logger.setLevel(original_level)

    logger.info("Loading UNet Modules from state dict...")
    for lora_name, orig_modules in unet_loras.items():
        if orig_modules is None:
            continue
        lyco_type, params = get_module(weights_sd, lora_name)
        module = make_module(lyco_type, params, lora_name, orig_modules)
        if module is not None:
            network.unet_loras.append(module)
    logger.info(f"{len(network.unet_loras)} Modules Loaded")

    logger.info("Loading TE Modules from state dict...")
    for lora_name, orig_modules in te_loras.items():
        if orig_modules is None:
            continue
        lyco_type, params = get_module(weights_sd, lora_name)
        module = make_module(lyco_type, params, lora_name, orig_modules)
        if module is not None:
            network.text_encoder_loras.append(module)
    logger.info(f"{len(network.text_encoder_loras)} Modules Loaded")

    for lora in network.unet_loras + network.text_encoder_loras:
        lora.multiplier = multiplier

    return network, weights_sd


class LycorisNetworkKohya(LycorisNetwork):
    """
    LoRA + LoCon
    """

    # Ignore proj_in or proj_out, their channels is only a few.
    ENABLE_CONV = True
    UNET_TARGET_REPLACE_MODULE = [
        "Transformer2DModel",
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
        "HunYuanDiTBlock",
    ]
    UNET_TARGET_REPLACE_NAME = [
        "conv_in",
        "conv_out",
        "time_embedding.linear_1",
        "time_embedding.linear_2",
    ]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = [
        "CLIPAttention",
        "CLIPMLP",
        "MT5Block",
        "BertLayer",
    ]
    TEXT_ENCODER_TARGET_REPLACE_NAME = []
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    MODULE_ALGO_MAP = {}
    NAME_ALGO_MAP = {}

    @classmethod
    def apply_preset(cls, preset):
        if "enable_conv" in preset:
            cls.ENABLE_CONV = preset["enable_conv"]
        if "unet_target_module" in preset:
            cls.UNET_TARGET_REPLACE_MODULE = preset["unet_target_module"]
        if "unet_target_name" in preset:
            cls.UNET_TARGET_REPLACE_NAME = preset["unet_target_name"]
        if "text_encoder_target_module" in preset:
            cls.TEXT_ENCODER_TARGET_REPLACE_MODULE = preset[
                "text_encoder_target_module"
            ]
        if "text_encoder_target_name" in preset:
            cls.TEXT_ENCODER_TARGET_REPLACE_NAME = preset["text_encoder_target_name"]
        if "module_algo_map" in preset:
            cls.MODULE_ALGO_MAP = preset["module_algo_map"]
        if "name_algo_map" in preset:
            cls.NAME_ALGO_MAP = preset["name_algo_map"]
        return cls

    def __init__(
        self,
        text_encoder,
        unet,
        multiplier=1.0,
        lora_dim=4,
        conv_lora_dim=4,
        alpha=1,
        conv_alpha=1,
        use_tucker=False,
        dropout=0,
        rank_dropout=0,
        module_dropout=0,
        network_module: str = "locon",
        norm_modules=NormModule,
        train_norm=False,
        **kwargs,
    ) -> None:
        torch.nn.Module.__init__(self)
        root_kwargs = kwargs
        self.multiplier = multiplier
        self.lora_dim = lora_dim

        if not self.ENABLE_CONV:
            conv_lora_dim = 0

        self.conv_lora_dim = int(conv_lora_dim)
        if self.conv_lora_dim and self.conv_lora_dim != self.lora_dim:
            logger.info("Apply different lora dim for conv layer")
            logger.info(f"Conv Dim: {conv_lora_dim}, Linear Dim: {lora_dim}")
        elif self.conv_lora_dim == 0:
            logger.info("Disable conv layer")

        self.alpha = alpha
        self.conv_alpha = float(conv_alpha)
        if self.conv_lora_dim and self.alpha != self.conv_alpha:
            logger.info("Apply different alpha value for conv layer")
            logger.info(f"Conv alpha: {conv_alpha}, Linear alpha: {alpha}")

        if 1 >= dropout >= 0:
            logger.info(f"Use Dropout value: {dropout}")
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        self.use_tucker = use_tucker

        def create_single_module(
            lora_name: str,
            module: torch.nn.Module,
            algo_name,
            dim=None,
            alpha=None,
            use_tucker=self.use_tucker,
            **kwargs,
        ):
            for k, v in root_kwargs.items():
                if k in kwargs:
                    continue
                kwargs[k] = v

            if train_norm and "Norm" in module.__class__.__name__:
                return norm_modules(
                    lora_name,
                    module,
                    self.multiplier,
                    self.rank_dropout,
                    self.module_dropout,
                    **kwargs,
                )
            lora = None
            if isinstance(module, torch.nn.Linear) and lora_dim > 0:
                dim = dim or lora_dim
                alpha = alpha or self.alpha
            elif isinstance(
                module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
            ):
                k_size, *_ = module.kernel_size
                if k_size == 1 and lora_dim > 0:
                    dim = dim or lora_dim
                    alpha = alpha or self.alpha
                elif conv_lora_dim > 0 or dim:
                    dim = dim or conv_lora_dim
                    alpha = alpha or self.conv_alpha
                else:
                    return None
            else:
                return None
            lora = network_module_dict[algo_name](
                lora_name,
                module,
                self.multiplier,
                dim,
                alpha,
                self.dropout,
                self.rank_dropout,
                self.module_dropout,
                use_tucker,
                **kwargs,
            )
            return lora

        def create_modules_(
            prefix: str,
            root_module: torch.nn.Module,
            algo,
            configs={},
        ):
            loras = {}
            lora_names = []
            for name, module in root_module.named_modules():
                module_name = module.__class__.__name__
                if module_name in self.MODULE_ALGO_MAP and module is not root_module:
                    next_config = self.MODULE_ALGO_MAP[module_name]
                    next_algo = next_config.get("algo", algo)
                    new_loras, new_lora_names = create_modules_(
                        f"{prefix}_{name}", module, next_algo, next_config
                    )
                    for lora_name, lora in zip(new_lora_names, new_loras):
                        if lora_name not in loras:
                            loras[lora_name] = lora
                            lora_names.append(lora_name)
                    continue
                if name:
                    lora_name = prefix + "." + name
                else:
                    lora_name = prefix
                lora_name = lora_name.replace(".", "_")
                if lora_name in loras:
                    continue

                lora = create_single_module(lora_name, module, algo, **configs)
                if lora is not None:
                    loras[lora_name] = lora
                    lora_names.append(lora_name)
            return [loras[lora_name] for lora_name in lora_names], lora_names

        # create module instances
        def create_modules(
            prefix,
            root_module: torch.nn.Module,
            target_replace_modules,
            target_replace_names=[],
        ) -> List:
            logger.info("Create LyCORIS Module")
            loras = []
            next_config = {}
            for name, module in root_module.named_modules():
                module_name = module.__class__.__name__
                if module_name in target_replace_modules and not any(
                    re.match(t, name) for t in target_replace_names
                ):
                    if module_name in self.MODULE_ALGO_MAP:
                        next_config = self.MODULE_ALGO_MAP[module_name]
                        algo = next_config.get("algo", network_module)
                    else:
                        algo = network_module
                    loras.extend(
                        create_modules_(f"{prefix}_{name}", module, algo, next_config)[
                            0
                        ]
                    )
                    next_config = {}
                elif name in target_replace_names or any(
                    re.match(t, name) for t in target_replace_names
                ):
                    conf_from_name = self.find_conf_for_name(name)
                    if conf_from_name is not None:
                        next_config = conf_from_name
                        algo = next_config.get("algo", network_module)
                    elif module_name in self.MODULE_ALGO_MAP:
                        next_config = self.MODULE_ALGO_MAP[module_name]
                        algo = next_config.get("algo", network_module)
                    else:
                        algo = network_module
                    lora_name = prefix + "." + name
                    lora_name = lora_name.replace(".", "_")
                    lora = create_single_module(lora_name, module, algo, **next_config)
                    next_config = {}
                    if lora is not None:
                        loras.append(lora)
            return loras

        if network_module == GLoRAModule:
            logger.info("GLoRA enabled, only train transformer")
            # only train transformer (for GLoRA)
            LycorisNetworkKohya.UNET_TARGET_REPLACE_MODULE = [
                "Transformer2DModel",
                "Attention",
            ]
            LycorisNetworkKohya.UNET_TARGET_REPLACE_NAME = []

        if isinstance(text_encoder, list):
            text_encoders = text_encoder
            use_index = True
        else:
            text_encoders = [text_encoder]
            use_index = False

        self.text_encoder_loras = []
        for i, te in enumerate(text_encoders):
            self.text_encoder_loras.extend(
                create_modules(
                    LycorisNetworkKohya.LORA_PREFIX_TEXT_ENCODER
                    + (f"{i+1}" if use_index else ""),
                    te,
                    LycorisNetworkKohya.TEXT_ENCODER_TARGET_REPLACE_MODULE,
                    LycorisNetworkKohya.TEXT_ENCODER_TARGET_REPLACE_NAME,
                )
            )
        logger.info(
            f"create LyCORIS for Text Encoder: {len(self.text_encoder_loras)} modules."
        )

        self.unet_loras = create_modules(
            LycorisNetworkKohya.LORA_PREFIX_UNET,
            unet,
            LycorisNetworkKohya.UNET_TARGET_REPLACE_MODULE,
            LycorisNetworkKohya.UNET_TARGET_REPLACE_NAME,
        )
        logger.info(f"create LyCORIS for U-Net: {len(self.unet_loras)} modules.")

        algo_table = {}
        for lora in self.text_encoder_loras + self.unet_loras:
            algo_table[lora.__class__.__name__] = (
                algo_table.get(lora.__class__.__name__, 0) + 1
            )
        logger.info(f"module type table: {algo_table}")

        self.weights_sd = None

        self.loras = self.text_encoder_loras + self.unet_loras
        # assertion
        names = set()
        for lora in self.loras:
            assert (
                lora.lora_name not in names
            ), f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def find_conf_for_name(
        self,
        name: str,
    ) -> dict[str, Any]:
        if name in self.NAME_ALGO_MAP.keys():
            return self.NAME_ALGO_MAP[name]

        for key, value in self.NAME_ALGO_MAP.items():
            if re.match(key, name):
                return value

        return None

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            self.weights_sd = load_file(file)
        else:
            self.weights_sd = torch.load(file, map_location="cpu")
        missing, unexpected = self.load_state_dict(self.weights_sd, strict=False)
        state = {}
        if missing:
            state["missing keys"] = missing
        if unexpected:
            state["unexpected keys"] = unexpected
        return state

    def apply_to(self, text_encoder, unet, apply_text_encoder=None, apply_unet=None):
        assert (
            apply_text_encoder is not None and apply_unet is not None
        ), f"internal error: flag not set"

        if apply_text_encoder:
            logger.info("enable LyCORIS for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable LyCORIS for U-Net")
        else:
            self.unet_loras = []

        self.loras = self.text_encoder_loras + self.unet_loras

        for lora in self.loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        if self.weights_sd:
            # if some weights are not in state dict, it is ok because initial LoRA does nothing (lora_up is initialized by zeros)
            info = self.load_state_dict(self.weights_sd, False)
            logger.info(f"weights are loaded: {info}")

    # TODO refactor to common function with apply_to
    def merge_to(self, text_encoder, unet, weights_sd, dtype, device):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(LycorisNetworkKohya.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(LycorisNetworkKohya.LORA_PREFIX_UNET):
                apply_unet = True

        if apply_text_encoder:
            logger.info("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        self.loras = self.text_encoder_loras + self.unet_loras
        super().merge_to(1)

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, learning_rate):
        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        self.requires_grad_(True)
        all_params = []

        if self.text_encoder_loras:
            param_data = {"params": enumerate_params(self.text_encoder_loras)}
            if text_encoder_lr is not None:
                param_data["lr"] = text_encoder_lr
            all_params.append(param_data)

        if self.unet_loras:
            param_data = {"params": enumerate_params(self.unet_loras)}
            if unet_lr is not None:
                param_data["lr"] = unet_lr
            all_params.append(param_data)

        return all_params

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash = precalculate_safetensors_hashes(state_dict)
            metadata["sshs_model_hash"] = model_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)
