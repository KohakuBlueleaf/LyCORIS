import torch
import torch.nn as nn
import torch.nn.functional as F

from lycoris.kohya.model_utils import load_models_from_stable_diffusion_checkpoint
import lycoris.kohya as network_module


clip, vae, unet = load_models_from_stable_diffusion_checkpoint(
    False, "./test_model/ACertainty.ckpt"
)
net = network_module.create_network(1, 1, 1, vae, clip, unet, preset="full")
net = network_module.create_network(1, 1, 1, vae, clip, unet, preset="attn-mlp")
net = network_module.create_network(1, 1, 1, vae, clip, unet, preset="attn-only")

total = 0
for name, module in unet.named_modules():
    if isinstance(module, nn.Linear):
        total += 1
    if isinstance(module, nn.Conv2d):
        total += 1

print(total)
total = 0
for name, module in clip.named_modules():
    if isinstance(module, nn.Linear):
        total += 1
    if isinstance(module, nn.Conv2d):
        total += 1

print(total)
