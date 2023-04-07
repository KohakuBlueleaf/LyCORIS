'''
Modified version for full net lora
(Lora for ResBlock and up/down sample block)
'''
import os, sys
import re
import torch

from modules import shared, devices, sd_models
now_dir = os.path.dirname(os.path.abspath(__file__))
lora_path = os.path.join(now_dir, '..', '..', '..', 'extensions-builtin/Lora')
sys.path.insert(0, lora_path)
import lora
new_lora = 'lora_calc_updown' in dir(lora)

from locon_compvis import LoConModule, LoConNetworkCompvis, create_network_and_apply_compvis


try:
    '''
    Hijack Additional Network extension
    '''
    # skip addnet since don't support new version
    raise
    now_dir = os.path.dirname(os.path.abspath(__file__))
    addnet_path = os.path.join(now_dir, '..', '..', 'sd-webui-additional-networks/scripts')
    sys.path.append(addnet_path)
    import lora_compvis
    import scripts
    scripts.lora_compvis = lora_compvis
    scripts.lora_compvis.LoRAModule = LoConModule
    scripts.lora_compvis.LoRANetworkCompvis = LoConNetworkCompvis
    scripts.lora_compvis.create_network_and_apply_compvis = create_network_and_apply_compvis
    print('LoCon Extension hijack addnet extension successfully')
except:
    print('Additional Network extension not installed, Only hijack built-in lora')


'''
Hijack sd-webui LoRA
'''
re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")

re_unet_conv_in = re.compile(r"lora_unet_conv_in(.+)")
re_unet_conv_out = re.compile(r"lora_unet_conv_out(.+)")
re_unet_time_embed = re.compile(r"lora_unet_time_embedding_linear_(\d+)(.+)")

re_unet_down_blocks = re.compile(r"lora_unet_down_blocks_(\d+)_attentions_(\d+)_(.+)")
re_unet_mid_blocks = re.compile(r"lora_unet_mid_block_attentions_(\d+)_(.+)")
re_unet_up_blocks = re.compile(r"lora_unet_up_blocks_(\d+)_attentions_(\d+)_(.+)")

re_unet_down_blocks_res = re.compile(r"lora_unet_down_blocks_(\d+)_resnets_(\d+)_(.+)")
re_unet_mid_blocks_res = re.compile(r"lora_unet_mid_block_resnets_(\d+)_(.+)")
re_unet_up_blocks_res = re.compile(r"lora_unet_up_blocks_(\d+)_resnets_(\d+)_(.+)")

re_unet_downsample = re.compile(r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv(.+)")
re_unet_upsample = re.compile(r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv(.+)")

re_text_block = re.compile(r"lora_te_text_model_encoder_layers_(\d+)_(.+)")


def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex):
        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []
    
    if match(m, re_unet_conv_in):
        return f'diffusion_model_input_blocks_0_0{m[0]}'
    
    if match(m, re_unet_conv_out):
        return f'diffusion_model_out_2{m[0]}'
    
    if match(m, re_unet_time_embed):
        return f"diffusion_model_time_embed_{m[0]*2-2}{m[1]}"
    
    if match(m, re_unet_down_blocks):
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_unet_mid_blocks):
        return f"diffusion_model_middle_block_1_{m[1]}"

    if match(m, re_unet_up_blocks):
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[1]}_1_{m[2]}"

    if match(m, re_unet_down_blocks_res):
        block = f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[1]}_0_"
        if m[2].startswith('conv1'):
            return f"{block}in_layers_2{m[2][len('conv1'):]}"
        elif m[2].startswith('conv2'):
            return f"{block}out_layers_3{m[2][len('conv2'):]}"
        elif m[2].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[2][len('time_emb_proj'):]}"
        elif m[2].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[2][len('conv_shortcut'):]}"

    if match(m, re_unet_mid_blocks_res):
        block = f"diffusion_model_middle_block_{m[0]*2}_"
        if m[1].startswith('conv1'):
            return f"{block}in_layers_2{m[1][len('conv1'):]}"
        elif m[1].startswith('conv2'):
            return f"{block}out_layers_3{m[1][len('conv2'):]}"
        elif m[1].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[1][len('time_emb_proj'):]}"
        elif m[1].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[1][len('conv_shortcut'):]}"

    if match(m, re_unet_up_blocks_res):
        block = f"diffusion_model_output_blocks_{m[0] * 3 + m[1]}_0_"
        if m[2].startswith('conv1'):
            return f"{block}in_layers_2{m[2][len('conv1'):]}"
        elif m[2].startswith('conv2'):
            return f"{block}out_layers_3{m[2][len('conv2'):]}"
        elif m[2].startswith('time_emb_proj'):
            return f"{block}emb_layers_1{m[2][len('time_emb_proj'):]}"
        elif m[2].startswith('conv_shortcut'):
            return f"{block}skip_connection{m[2][len('conv_shortcut'):]}"

    if match(m, re_unet_downsample):
        return f"diffusion_model_input_blocks_{m[0]*3+3}_0_op{m[1]}"

    if match(m, re_unet_upsample):
        return f"diffusion_model_output_blocks_{m[0]*3 + 2}_{1+(m[0]!=0)}_conv{m[1]}"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return key


class LoraOnDisk:
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename


class LoraModule:
    def __init__(self, name):
        self.name = name
        self.multiplier = 1.0
        self.modules = {}
        self.mtime = None


class FakeModule(torch.nn.Module):
    def __init__(self, weight, func):
        super().__init__()
        self.weight = weight
        self.func = func
    
    def forward(self, x):
        return self.func(x)


class FullModule:
    def __init__(self):
        self.weight = None
        self.alpha = None
        self.op = None
        self.extra_args = {}
        self.shape = None
        self.up = None
    
    def down(self, x):
        return x
    
    def inference(self, x):
        return self.op(x, self.weight, **self.extra_args)


class LoraUpDownModule:
    def __init__(self):
        self.up_model = None
        self.mid_model = None
        self.down_model = None
        self.alpha = None
        self.dim = None
        self.op = None
        self.extra_args = {}
        self.shape = None
        self.bias = None
        self.up = None
    
    def down(self, x):
        return x
    
    def inference(self, x):
        if hasattr(self, 'bias') and isinstance(self.bias, torch.Tensor):
            out_dim = self.up_model.weight.size(0)
            rank = self.down_model.weight.size(0)
            rebuild_weight = (
                self.up_model.weight.reshape(out_dim, -1) @ self.down_model.weight.reshape(rank, -1)
                + self.bias
            ).reshape(self.shape)
            return self.op(
                x, rebuild_weight,
                **self.extra_args
            )
        else:
            if self.mid_model is None:
                return self.up_model(self.down_model(x))
            else:
                return self.up_model(self.mid_model(self.down_model(x)))


def pro3(t, wa, wb):
    temp = torch.einsum('i j k l, j r -> i r k l', t, wb)
    return torch.einsum('i j k l, i r -> r j k l', temp, wa)


def pro3_outer(t, wa, b): # for outer product
    raise NotImplemented()
    # temp = torch.einsum('i j k l, j r -> i r k l', t, wb)
    # return torch.einsum('i j k l, i r -> r j k l', temp, wa)


class LoraHadaModule:
    def __init__(self):
        self.t1 = None
        self.w1a = None
        self.w1b = None
        self.t2 = None
        self.w2a = None
        self.w2b = None
        self.alpha = None
        self.dim = None
        self.op = None
        self.extra_args = {}
        self.shape = None
        self.bias = None
        self.up = None
    
    def down(self, x):
        return x
    
    def inference(self, x):
        if hasattr(self, 'bias') and isinstance(self.bias, torch.Tensor):
            bias = self.bias
        else:
            bias = 0
        
        if self.t1 is None:
            return self.op(
                x,
                ((self.w1a @ self.w1b) * (self.w2a @ self.w2b) + bias).view(self.shape),
                **self.extra_args
            )
        else:
            return self.op(
                x,
                (pro3(self.t1, self.w1a, self.w1b) 
                 * pro3(self.t2, self.w2a, self.w2b) + bias).view(self.shape),
                **self.extra_args
            )

class LoraKronModule:
    def __init__(self):
        self.t1 = None
        self.w1a = None
        self.w1b = None
        self.t2 = None
        self.w2a = None
        self.w2b = None
        self.alpha = None
        self.dim = None
        self.op = None
        self.extra_args = {}
        self.shape = None
        self.bias = None
        self.up = None
    
    def down(self, x):
        return x
    
    def inference(self, x):
        if hasattr(self, 'bias') and isinstance(self.bias, torch.Tensor):
            bias = self.bias
        else:
            bias = 0
        
        if self.t1 is None:
            return self.op(
                x,
                (torch.kron(self.w1a@self.w1b, self.w2a@self.w2b) + bias).view(self.shape),
                **self.extra_args
            )
        else:
            # will raise NotImplemented Error
            return self.op(
                x,
                (pro3_outer(self.t1, self.w1a, self.w1b) 
                 * pro3_outer(self.t2, self.w2a, self.w2b) + bias).view(self.shape),
                **self.extra_args
            )



CON_KEY = {
    "lora_up.weight",
    "lora_down.weight",
    "lora_mid.weight"
}
HADA_KEY = {
    "hada_t1",
    "hada_w1_a",
    "hada_w1_b",
    "hada_t2",
    "hada_w2_a",
    "hada_w2_b",
}
KRON_KEY = {
    "lokr_t1",
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_t2",
    "lokr_w2_a",
    "lokr_w2_b",
}

def load_lora(name, filename):
    lora = LoraModule(name)
    lora.mtime = os.path.getmtime(filename)

    sd = sd_models.read_state_dict(filename)
    is_sd2 = 'model_transformer_resblocks' in shared.sd_model.lora_layer_mapping

    keys_failed_to_match = []

    for key_diffusers, weight in sd.items():
        fullkey = convert_diffusers_name_to_compvis(key_diffusers, is_sd2)
        key, lora_key = fullkey.split(".", 1)
        
        sd_module = shared.sd_model.lora_layer_mapping.get(key, None)
        
        if sd_module is None:
            m = re_x_proj.match(key)
            if m:
                sd_module = shared.sd_model.lora_layer_mapping.get(m.group(1), None)
        
        if sd_module is None:
            print(key)
            keys_failed_to_match.append(key_diffusers)
            continue

        lora_module = lora.modules.get(key, None)
        if lora_module is None:
            lora_module = LoraUpDownModule()
            lora.modules[key] = lora_module

        if lora_key == "alpha":
            lora_module.alpha = weight.item()
            continue
        
        if lora_key == "diff":
            weight = weight.to(device=devices.device, dtype=devices.dtype)
            weight.requires_grad_(False)
            lora_module = FullModule()
            lora.modules[key] = lora_module
            lora_module.weight = weight
            lora_module.alpha = weight.size(1)
            lora_module.up = FakeModule(
                weight,
                lora_module.inference
            )
            lora_module.up.to(device=devices.cpu if new_lora else devices.device, dtype=devices.dtype)
            if len(weight.shape)==2:
                lora_module.op = torch.nn.functional.linear
                lora_module.extra_args = {
                    'bias': None
                }
            else:
                lora_module.op = torch.nn.functional.conv2d
                lora_module.extra_args = {
                    'stride': sd_module.stride,
                    'padding': sd_module.padding,
                    'bias': None
                }
            continue
        
        if 'bias_' in lora_key:
            if lora_module.bias is None:
                lora_module.bias = [None, None, None]
            if 'bias_indices' == lora_key:
                lora_module.bias[0] = weight
            elif 'bias_values' == lora_key:
                lora_module.bias[1] = weight
            elif 'bias_size' == lora_key:
                lora_module.bias[2] = weight
            
            if all((i is not None) for i in lora_module.bias):
                print('build bias')
                lora_module.bias = torch.sparse_coo_tensor(
                    lora_module.bias[0],
                    lora_module.bias[1],
                    tuple(lora_module.bias[2]),
                ).to(device=devices.cpu if new_lora else devices.device, dtype=devices.dtype)
                lora_module.bias.requires_grad_(False)
            continue
        
        if lora_key in CON_KEY:
            if (type(sd_module) == torch.nn.Linear
                or type(sd_module) == torch.nn.modules.linear.NonDynamicallyQuantizableLinear
                or type(sd_module) == torch.nn.MultiheadAttention):
                weight = weight.reshape(weight.shape[0], -1)
                module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
                lora_module.op = torch.nn.functional.linear
            elif type(sd_module) == torch.nn.Conv2d:
                if lora_key == "lora_down.weight":
                    if len(weight.shape) == 2:
                        weight = weight.reshape(weight.shape[0], -1, 1, 1)
                    if weight.shape[2] != 1 or weight.shape[3] != 1:
                        module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], sd_module.kernel_size, sd_module.stride, sd_module.padding, bias=False)
                    else:
                        module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
                elif lora_key == "lora_mid.weight":
                    module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], sd_module.kernel_size, sd_module.stride, sd_module.padding, bias=False)
                elif lora_key == "lora_up.weight":
                    module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
                lora_module.op = torch.nn.functional.conv2d
                lora_module.extra_args = {
                    'stride': sd_module.stride,
                    'padding': sd_module.padding
                }
            else:
                assert False, f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}'
            
            if hasattr(sd_module, 'weight'):
                lora_module.shape = sd_module.weight.shape
            with torch.no_grad():
                module.weight.copy_(weight)

            module.to(device=devices.cpu if new_lora else devices.device, dtype=devices.dtype)
            module.requires_grad_(False)

            if lora_key == "lora_up.weight":
                lora_module.up_model = module
                lora_module.up = FakeModule(
                    lora_module.up_model.weight,
                    lora_module.inference
                )
            elif lora_key == "lora_mid.weight":
                lora_module.mid_model = module
            elif lora_key == "lora_down.weight":
                lora_module.down_model = module
                lora_module.dim = weight.shape[0]
            else:
                print(lora_key)
        elif lora_key in HADA_KEY:
            if type(lora_module) != LoraHadaModule:
                alpha = lora_module.alpha
                bias = lora_module.bias
                lora_module = LoraHadaModule()
                lora_module.alpha = alpha
                lora_module.bias = bias
                lora.modules[key] = lora_module
            if hasattr(sd_module, 'weight'):
                lora_module.shape = sd_module.weight.shape
            
            weight = weight.to(device=devices.cpu if new_lora else devices.device, dtype=devices.dtype)
            weight.requires_grad_(False)
            
            if lora_key == 'hada_w1_a':
                lora_module.w1a = weight
                if lora_module.up is None:
                    lora_module.up = FakeModule(
                        lora_module.w1a,
                        lora_module.inference
                    )
            elif lora_key == 'hada_w1_b':
                lora_module.w1b = weight
                lora_module.dim = weight.shape[0]
            elif lora_key == 'hada_w2_a':
                lora_module.w2a = weight
            elif lora_key == 'hada_w2_b':
                lora_module.w2b = weight
            elif lora_key == 'hada_t1':
                lora_module.t1 = weight
                lora_module.up = FakeModule(
                    lora_module.t1,
                    lora_module.inference
                )
            elif lora_key == 'hada_t2':
                lora_module.t2 = weight
            
            if (type(sd_module) == torch.nn.Linear
                or type(sd_module) == torch.nn.modules.linear.NonDynamicallyQuantizableLinear
                or type(sd_module) == torch.nn.MultiheadAttention):
                lora_module.op = torch.nn.functional.linear
            elif type(sd_module) == torch.nn.Conv2d:
                lora_module.op = torch.nn.functional.conv2d
                lora_module.extra_args = {
                    'stride': sd_module.stride,
                    'padding': sd_module.padding
                }
            else:
                assert False, f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}'
                
        elif lora_key in KRON_KEY:
            if not isinstance(lora_module, LoraKronModule):
                alpha = lora_module.alpha
                bias = lora_module.bias
                lora_module = LoraKronModule()
                lora_module.alpha = alpha
                lora_module.bias = bias
                lora.modules[key] = lora_module
            if hasattr(sd_module, 'weight'):
                lora_module.shape = sd_module.weight.shape
            
            weight = weight.to(device=devices.cpu if new_lora else devices.device, dtype=devices.dtype)
            weight.requires_grad_(False)
            
            if lora_key == 'lokr_w1_a':
                lora_module.w1a = weight
                if lora_module.up is None:
                    lora_module.up = FakeModule(
                        lora_module.w1a,
                        lora_module.inference
                    )
            elif lora_key == 'lokr_w1_b':
                lora_module.w1b = weight
                lora_module.dim = weight.shape[0]
            elif lora_key == 'lokr_w2_a':
                lora_module.w2a = weight
            elif lora_key == 'lokr_w2_b':
                lora_module.w2b = weight
            elif lora_key == 'lokr_t1':
                lora_module.t1 = weight
                lora_module.up = FakeModule(
                    lora_module.t1,
                    lora_module.inference
                )
            elif lora_key == 'lokr_t2':
                lora_module.t2 = weight
            
            if (any(isinstance(sd_module, torch_layer) for torch_layer in 
                    [torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear, torch.nn.MultiheadAttention])):
                lora_module.op = torch.nn.functional.linear
            elif isinstance(sd_module, torch.nn.Conv2d):
                lora_module.op = torch.nn.functional.conv2d
                lora_module.extra_args = {
                    'stride': sd_module.stride,
                    'padding': sd_module.padding
                }
            else:
                assert False, f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}'
        
        else:
            assert False, f'Bad Lora layer name: {key_diffusers} - must end in lora_up.weight, lora_down.weight or alpha'

    if len(keys_failed_to_match) > 0:
        print(shared.sd_model.lora_layer_mapping)
        print(f"Failed to match keys when loading Lora {filename}: {keys_failed_to_match}")

    return lora


def lora_forward(module, input, res):
    if len(lora.loaded_loras) == 0:
        return res
    
    lora_layer_name = getattr(module, 'lora_layer_name', None)
    for lora_m in lora.loaded_loras:
        module = lora_m.modules.get(lora_layer_name, None)
        if module is not None and lora_m.multiplier:
            if hasattr(module, 'up'):
                scale = lora_m.multiplier * (module.alpha / module.up.weight.size(1) if module.alpha else 1.0)
            else:
                scale = lora_m.multiplier * (module.alpha / module.dim if module.alpha else 1.0)
            
            if shared.opts.lora_apply_to_outputs and res.shape == input.shape:
                x = res
            else:
                x = input
            
            if hasattr(module, 'inference'):
                res = res + module.inference(x) * scale
            elif hasattr(module, 'up'):
                res = res + module.up(module.down(x)) * scale
            else:
                raise NotImplementedError(
                    "Your settings, extensions or models are not compatible with each other."
                )
    return res


def _rebuild_conventional(up, down, shape):
    return (up.reshape(up.size(0), -1) @ down.reshape(down.size(0), -1)).reshape(shape)


def _rebuild_cp_decomposition(up, down, mid):
    up = up.reshape(up.size(0), -1)
    down = down.reshape(down.size(0), -1)
    return torch.einsum('n m k l, i n, m j -> i j k l', mid, up, down)


def rebuild_weight(module, orig_weight: torch.Tensor) -> torch.Tensor:
    if isinstance(module, LoraUpDownModule):
        up = module.up_model.weight.to(orig_weight.device, dtype=orig_weight.dtype)
        down = module.down_model.weight.to(orig_weight.device, dtype=orig_weight.dtype)
        
        output_shape = [up.size(0), down.size(1)]
        if (mid:=module.mid_model) is not None:
            # cp-decomposition
            mid = mid.weight.to(orig_weight.device, dtype=orig_weight.dtype)
            updown = _rebuild_cp_decomposition(up, down, mid)
            output_shape += mid.shape[2:]
        else:
            if len(down.shape) == 4:
                output_shape += down.shape[2:]
            updown = _rebuild_conventional(up, down, output_shape)
        
    elif isinstance(module, LoraHadaModule):
        w1a = module.w1a.to(orig_weight.device, dtype=orig_weight.dtype)
        w1b = module.w1b.to(orig_weight.device, dtype=orig_weight.dtype)
        w2a = module.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
        w2b = module.w2b.to(orig_weight.device, dtype=orig_weight.dtype)
        
        output_shape = [w1a.size(0), w1b.size(1)]
        
        if module.t1 is not None:
            output_shape = [w1a.size(1), w1b.size(1)]
            t1 = module.t1.to(orig_weight.device, dtype=orig_weight.dtype)
            updown1 = pro3(t1, w1a, w1b)
            output_shape += t1.shape[2:]
        else:
            if len(w1b.shape) == 4:
                output_shape += w1b.shape[2:]
            updown1 = _rebuild_conventional(w1a, w1b, output_shape)
        
        if module.t2 is not None:
            t2 = module.t2.to(orig_weight.device, dtype=orig_weight.dtype)
            updown2 = pro3(t2, w2a, w2b)
        else:
            updown2 = _rebuild_conventional(w2a, w2b, output_shape)
        updown = updown1 * updown2
    
    elif isinstance(module, LoraKronModule):
        w1a = module.w1a.to(orig_weight.device, dtype=orig_weight.dtype)
        w1b = module.w1b.to(orig_weight.device, dtype=orig_weight.dtype)
        w2a = module.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
        w2b = module.w2b.to(orig_weight.device, dtype=orig_weight.dtype)
        
        output_shape = [w1a.size(0), w1b.size(1)]
        output_shape_1 = output_shape
        output_shape_2 = [w2a.size(0), w2b.size(1)]
        
        if module.t1 is not None:
            output_shape = [w1a.size(1), w1b.size(1)] # [rank, rank]
            t1 = module.t1.to(orig_weight.device, dtype=orig_weight.dtype)
            updown1 = pro3(t1, w1a, w1b)
            output_shape += t1.shape[2:] # [rank, rank, *kernel]
        else:
            if len(w1b.shape) == 4:
                output_shape_1 += w1b.shape[2:]
            updown1 = _rebuild_conventional(w1a, w1b, output_shape_1)
        
        if module.t2 is not None:
            t2 = module.t2.to(orig_weight.device, dtype=orig_weight.dtype)
            updown2 = pro3(t2, w2a, w2b)
        else:
            updown2 = _rebuild_conventional(w2a, w2b, output_shape_2)
        updown = torch.kron(updown1, updown2)
    
    elif isinstance(module, FullModule):
        output_shape = module.weight.shape
        updown = module.weight.to(orig_weight.device, dtype=orig_weight.dtype)
    
    if hasattr(module, 'bias') and module.bias != None:
        updown = updown.reshape(module.bias.shape)
        updown += module.bias.to(orig_weight.device, dtype=orig_weight.dtype)
        updown = updown.reshape(output_shape)
    
    if len(output_shape) == 4:
        updown = updown.reshape(output_shape)
    
    if orig_weight.size().numel() == updown.size().numel():
        updown = updown.reshape(orig_weight.shape)
    
    return updown


def lora_calc_updown(lora, module, target):
    with torch.no_grad():
        updown = rebuild_weight(module, target)
        updown = updown * lora.multiplier * (module.alpha / module.up.weight.shape[1] if module.alpha else 1.0)
        return updown


lora.convert_diffusers_name_to_compvis = convert_diffusers_name_to_compvis
lora.load_lora = load_lora
lora.lora_forward = lora_forward
lora.lora_calc_updown = lora_calc_updown
print('LoCon Extension hijack built-in lora successfully')