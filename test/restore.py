import sys

sys.setrecursionlimit(10000000)

import torch
import torch.nn as nn
import torch.nn.functional as F

from lycoris import create_lycoris


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.fc1 = nn.Linear(8, 1024)
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def test_lycoris_restore():
    net = TestNet().cuda()
    compiled_net = torch.compile(net)

    test_x = torch.randn(1, 8).cuda()
    out = compiled_net(test_x)
    print("\nOriginal\n", out)

    lycoris = create_lycoris(
        net, algo="lokr", factor=2, bypass_mode=True, full_matrix=True
    )
    lycoris.apply_to()
    lycoris.cuda()
    for parameter in lycoris.parameters():
        parameter.data = torch.randn_like(parameter) * 0.1
    net = torch.compile(net)
    out = net(test_x)
    print("\nBypass Lokr Applied\n", out)

    lycoris.restore()
    # lycoris.set_multiplier(0.0)
    out = net(test_x)
    print("\nRestored\n", out)

    lycoris.apply_to()
    # lycoris.set_multiplier(1.0)
    out = net(test_x)
    print("\nBypass Lokr Applied\n", out)


if __name__ == "__main__":
    test_lycoris_restore()
