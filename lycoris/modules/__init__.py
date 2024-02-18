import torch

from .full import FullModule
from .norms import NormModule
from .locon import LoConModule
from .loha import LohaModule
from .lokr import LokrModule, factorization
from .ia3 import IA3Module


@torch.no_grad()
def make_module(lyco_type, params, lora_name, orig_module):
    module = None
    if lyco_type == "locon":
        up, down, mid, alpha = params
        module = LoConModule(
            lora_name,
            orig_module,
            1,
            down.size(0),
            float(alpha),
            use_cp=mid is not None,
        )
        module.lora_up.weight.data.copy_(up)
        module.lora_down.weight.data.copy_(down)
        if mid is not None:
            module.lora_mid.weight.data.copy_(mid)
    elif lyco_type == "hada":
        w1a, w1b, w2a, w2b, t1, t2, alpha = params
        module = LohaModule(
            lora_name, orig_module, 1, w1b.size(0), float(alpha), use_cp=t1 is not None
        )
        module.hada_w1_a.copy_(w1a)
        module.hada_w1_b.copy_(w1b)
        module.hada_w2_a.copy_(w2a)
        module.hada_w2_b.copy_(w2b)
        if t1 is not None:
            module.hada_t1.copy_(t1)
            module.hada_t2.copy_(t2)
    elif lyco_type == "kron":
        w1, w1a, w1b, w2, w2a, w2b, _, t2, alpha = params
        if w1a is not None:
            lora_dim = w1a.size(1)
        elif w2a is not None:
            lora_dim = w2a.size(1)
        else:
            lora_dim = 10000000000000

        if w1 is None:
            out_dim = w1a.size(0)
            in_dim = w1b.size(1)
        else:
            out_dim, in_dim = w1.shape

        shape_s = [out_dim, in_dim]

        if w2 is None:
            out_dim *= w2a.size(0)
            in_dim *= w2b.size(1)
        else:
            out_dim *= w2.size(0)
            in_dim *= w2.size(1)

        if (
            shape_s[0] == factorization(out_dim, -1)[0]
            and shape_s[1] == factorization(in_dim, -1)[0]
        ):
            factor = -1
        else:
            factor = max(w1.shape) if w1 is not None else max(w1a.size(0), w1b.size(1))

        module = LokrModule(
            lora_name,
            orig_module,
            1,
            lora_dim,
            float(alpha),
            use_cp=t2 is not None,
            decompose_both=w1 is None and w2 is None,
            factor=factor,
        )
        if w1 is not None:
            module.lokr_w1.copy_(w1)
        else:
            module.lokr_w1_a.copy_(w1a)
            module.lokr_w1_b.copy_(w1b)
        if w2 is not None:
            module.lokr_w2.copy_(w2)
        else:
            module.lokr_w2_a.copy_(w2a)
            module.lokr_w2_b.copy_(w2b)
        if t2 is not None:
            module.lokr_t2.copy_(t2)
    elif lyco_type == "norm":
        w_norm, b_norm = params
        module = NormModule(
            lora_name,
            orig_module,
            1,
        )
        module.w_norm.copy_(w_norm)
        if b_norm is not None:
            module.b_norm.copy_(b_norm)
    elif lyco_type == "full":
        diff, diff_b = params
        module = FullModule(
            lora_name,
            orig_module,
            1,
        )
        module.diff.copy_(diff)
        if diff_b is not None:
            module.diff_b.copy_(diff_b)
    elif lyco_type == "ia3":
        pass
    else:
        return None

    return module
