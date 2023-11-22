import torch.nn as nn

from lycoris.wrapper import create_lycoris, LycorisNetwork


class DemoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.test_1 = nn.Linear(512, 512)
        self.te_2st = nn.Linear(256, 256)
        self._3test = nn.Linear(128, 128)

net = DemoNet()
LycorisNetwork.apply_preset({
    'target_name':[
        '.*te.*'
    ]
})
lycoris_net = create_lycoris(
    net, 1.0,
    linear_dim = 16, linear_alpha= 2.0,
    algo = 'lokr'
)
lycoris_net.apply_to()


print(sum(p.numel() for p in net.parameters()))
print(sum(p.numel() for p in lycoris_net.parameters()))