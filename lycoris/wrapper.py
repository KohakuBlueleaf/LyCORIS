# General LyCORIS wrapper based on kohya-ss/sd-scripts' style

import math
import os
import regex as re
import sys

sys.setrecursionlimit(10000)
from typing import List

import torch

from .utils import *
from .modules.locon import LoConModule
from .modules.loha import LohaModule
from .modules.lokr import LokrModule
from .modules.dylora import DyLoraModule
from .modules.glora import GLoRAModule
from .modules.norms import NormModule
from .modules.full import FullModule
from .modules.diag_oft import DiagOFTModule
from .modules.boft import ButterflyOFTModule
from .modules import make_module

from .config import PRESET
from .utils.preset import read_preset
from .utils import get_module, str_bool
from .logging import logger


network_module_dict = {
    "lora": LoConModule,
    "locon": LoConModule,
    "loha": LohaModule,
    "lokr": LokrModule,
    "dylora": DyLoraModule,
    "glora": GLoRAModule,
    "full": FullModule,
    "diag-oft": DiagOFTModule,
    "boft": ButterflyOFTModule,
}


def create_lycoris(module, multiplier, linear_dim, linear_alpha, **kwargs):
    if linear_dim is None:
        linear_dim = 4  # default
    conv_dim = int(kwargs.get("conv_dim", linear_dim) or linear_dim)
    conv_alpha = float(kwargs.get("conv_alpha", linear_alpha) or linear_alpha)
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
    if "disable_conv_cp" in kwargs or "use_cp" in kwargs or "use_conv_cp" in kwargs:
        logger.warning(
            "disable_conv_cp and use_cp are deprecated. Please use use_tucker instead.",
            stacklevel=2,
        )
    use_scalar = str_bool(kwargs.get("use_scalar", False))
    block_size = int(kwargs.get("block_size", 4) or 4)
    train_norm = str_bool(kwargs.get("train_norm", False))
    constrain = float(kwargs.get("constrain", 0) or 0)
    rescaled = str_bool(kwargs.get("rescaled", False))

    if algo == "glora" and conv_dim > 0:
        conv_dim = 0
        logger.info("Disable conv layer for GLoRA")

    preset = kwargs.get("preset", "full")
    if preset not in PRESET:
        preset = read_preset(preset)
    else:
        preset = PRESET[preset]
    assert preset is not None
    LycorisNetwork.apply_preset(preset)

    logger.info(f"Using rank adaptation algo: {algo}")

    if (
        (algo == "loha")
        and not kwargs.get("no_dim_warn", False)
        and (linear_dim > 64 or conv_dim > 64)
    ):
        warning_type = {"loha": "Hadamard Product representation"}
        warning_msg = (
            "You are not supposed to use dim>64 (64*64 = 4096, it already has enough rank)\n"
            f"in {warning_type[algo]}!\n"
            "Please consider use lower dim or disable this warning with --network_args no_dim_warn=True\n"
            f"If you just want to use high dim {algo}, please consider use lower lr."
        )
        logger.warning(warning_msg, stacklevel=2)

    network = LycorisNetwork(
        module,
        multiplier=multiplier,
        lora_dim=linear_dim,
        conv_lora_dim=conv_dim,
        alpha=linear_alpha,
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
        constrain=constrain,
        rescaled=rescaled,
    )

    if algo == "dylora":
        # dylora didn't support scale weight norm yet
        delattr(type(network), "apply_max_norm_regularization")

    return network


def create_lycoris_from_weights(multiplier, file, module, weights_sd=None, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # get dim/alpha mapping
    loras = {}
    for key in weights_sd:
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        loras[lora_name] = None

    for name, modules in module.named_modules():
        lora_name = f"{LycorisNetwork.LORA_PREFIX}_{name}".replace(".", "_")
        if lora_name in loras:
            loras[lora_name] = modules

    network = LycorisNetwork(module, init_only=True)
    network.multiplier = multiplier
    network.loras = []

    for lora_name, orig_modules in loras.items():
        if orig_modules is None:
            continue
        lyco_type, params = get_module(weights_sd, lora_name)
        module = make_module(lyco_type, params, lora_name, orig_modules)
        if module is not None:
            network.loras.append(module)
            network.algo_table[module.__class__.__name__] = (
                network.algo_table.get(module.__class__.__name__, 0) + 1
            )

    for lora in network.loras:
        lora.multiplier = multiplier

    return network, weights_sd


class LycorisNetwork(torch.nn.Module):
    ENABLE_CONV = True
    TARGET_REPLACE_MODULE = []
    TARGET_REPLACE_NAME = []
    LORA_PREFIX = "lycoris"
    MODULE_ALGO_MAP = {}
    NAME_ALGO_MAP = {}

    @classmethod
    def apply_preset(cls, preset):
        if "enable_conv" in preset:
            cls.ENABLE_CONV = preset["enable_conv"]
        if "target_module" in preset:
            cls.TARGET_REPLACE_MODULE = preset["target_module"]
        if "target_name" in preset:
            cls.TARGET_REPLACE_NAME = preset["target_name"]
        if "module_algo_map" in preset:
            cls.MODULE_ALGO_MAP = preset["module_algo_map"]
        if "name_algo_map" in preset:
            cls.NAME_ALGO_MAP = preset["name_algo_map"]
        return cls

    def __init__(
        self,
        module: nn.Module,
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
        init_only=False,
        **kwargs,
    ) -> None:
        super().__init__()
        root_kwargs = kwargs
        if init_only:
            self.multiplier = 1
            self.lora_dim = 0
            self.alpha = 1
            self.conv_lora_dim = 0
            self.conv_alpha = 1
            self.dropout = 0
            self.rank_dropout = 0
            self.module_dropout = 0
            self.use_tucker = False
            self.loras = []
            self.algo_table = {}
            return
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
            if module.__class__.__name__ == "Linear" and lora_dim > 0:
                dim = dim or lora_dim
                alpha = alpha or self.alpha
            elif module.__class__.__name__ == "Conv2d":
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
                if module_name in self.MODULE_ALGO_MAP:
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
                lora_name = prefix + "." + name
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
                if module_name in target_replace_modules:
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
                    if name in self.NAME_ALGO_MAP:
                        next_config = self.NAME_ALGO_MAP[name]
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

        self.loras = create_modules(
            LycorisNetwork.LORA_PREFIX,
            module,
            LycorisNetwork.TARGET_REPLACE_MODULE,
            LycorisNetwork.TARGET_REPLACE_NAME,
        )
        logger.info(f"create LyCORIS: {len(self.loras)} modules.")

        algo_table = {}
        for lora in self.loras:
            algo_table[lora.__class__.__name__] = (
                algo_table.get(lora.__class__.__name__, 0) + 1
            )
        logger.info(f"module type table: {algo_table}")

        self.weights_sd = None

        # assertion
        names = set()
        for lora in self.loras:
            assert (
                lora.lora_name not in names
            ), f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.loras:
            lora.multiplier = self.multiplier

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

    def apply_to(self):
        for lora in self.loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        if self.weights_sd:
            # if some weights are not in state dict, it is ok because initial LoRA does nothing (lora_up is initialized by zeros)
            info = self.load_state_dict(self.weights_sd, False)
            logger.info(f"weights are loaded: {info}")

    def is_mergeable(self):
        return True

    def restore(self):
        for lora in self.loras:
            lora.restore()

    def merge_to(self, weight=1.0):
        for lora in self.loras:
            lora.merge_to(weight)

    def apply_max_norm_regularization(self, max_norm_value, device):
        key_scaled = 0
        norms = []
        for model in self.loras:
            if hasattr(model, "apply_max_norm"):
                scaled, norm = model.apply_max_norm(max_norm_value, device)
                norms.append(norm)
                key_scaled += scaled

        if key_scaled == 0:
            return key_scaled, 0, 0

        return key_scaled, sum(norms) / len(norms), max(norms)

    def enable_gradient_checkpointing(self):
        # not supported
        def make_ckpt(module):
            if isinstance(module, torch.nn.Module):
                module.grad_ckpt = True

        self.apply(make_ckpt)
        pass

    def prepare_optimizer_params(self, lr):
        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        self.requires_grad_(True)
        all_params = []

        param_data = {"params": enumerate_params(self.loras)}
        if lr is not None:
            param_data["lr"] = lr
        all_params.append(param_data)
        return all_params

    def prepare_grad_etc(self, *args):
        self.requires_grad_(True)

    def on_epoch_start(self, *args):
        self.train()

    def get_trainable_params(self, *args):
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

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)
