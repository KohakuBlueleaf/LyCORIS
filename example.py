import torch.nn as nn

from lycoris.wrapper import create_lycoris


net = nn.Sequential(
    nn.Linear(512, 2048),
    nn.Mish(),
    nn.Linear(2048, 512)
)
lycoris_net = create_lycoris(
    net, 1.0,
    linear_dim = 16, linear_alpha= 2.0,
    algo = 'lokr'
)
lycoris_net.apply_to()


print(sum(p.numel() for p in net.parameters()))
print(sum(p.numel() for p in lycoris_net.parameters()))