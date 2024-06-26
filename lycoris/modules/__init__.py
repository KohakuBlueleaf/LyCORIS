import torch
import torch.nn as nn

from .base import LycorisBaseModule
from .locon import LoConModule
from .loha import LohaModule
from .lokr import LokrModule
from .full import FullModule
from .norms import NormModule
from .diag_oft import DiagOFTModule
from .boft import ButterflyOFTModule
from .glora import GLoRAModule
from .dylora import DyLoraModule
from .ia3 import IA3Module

from ..functional.general import factorization


MODULE_LIST = [
    LoConModule,
    LohaModule,
    IA3Module,
    LokrModule,
    FullModule,
    NormModule,
    DiagOFTModule,
    ButterflyOFTModule,
    GLoRAModule,
    DyLoraModule,
]


def get_module(lyco_state_dict, lora_name):
    for module in MODULE_LIST:
        if module.algo_check(lyco_state_dict, lora_name):
            return module, tuple(module.extract_state_dict(lyco_state_dict, lora_name))
    return None, None


@torch.no_grad()
def make_module(lyco_type: LycorisBaseModule, params, lora_name, orig_module):
    try:
        module = lyco_type.make_module_from_state_dict(lora_name, orig_module, *params)
    except NotImplementedError:
        module = None
    return module
