import torch.nn as nn

from lycoris.wrapper import create_lycoris, LycorisNetwork


class DemoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.test_1 = nn.Linear(512, 512)
        self.te_2st = nn.Linear(256, 256)
        self._3test = nn.Linear(128, 128)

net = DemoNet()
LycorisNetwork.apply_preset({'target_name': ['.*te.*']})
lycoris_net1 = create_lycoris(net, 1.0, linear_dim = 16, linear_alpha = 2.0, algo = 'lokr')
lycoris_net1.apply_to()

LycorisNetwork.apply_preset({'target_name': ['.*es.*']})
lycoris_net2 = create_lycoris(net, 1.0, linear_dim = 16, linear_alpha = 2.0, algo = 'lokr')
lycoris_net2.apply_to()

print(f'#Modules of net1: {len(lycoris_net1.loras)}')
print(f'#Modules of net2: {len(lycoris_net2.loras)}')

print('Total params:', sum(p.numel() for p in net.parameters()))
print('Net1 Params:', sum(p.numel() for p in lycoris_net1.parameters()))
print('Net2 Params:', sum(p.numel() for p in lycoris_net2.parameters()))