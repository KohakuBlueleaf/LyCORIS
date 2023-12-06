from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Lambda, Compose

from lycoris import create_lycoris, LycorisNetwork


class DemoNet(nn.Module):
    """
    A Simple Pytorch Module for demo
    """

    def __init__(self):
        super().__init__()
        self.test_1 = nn.Linear(784, 2048)
        self.te_2st = nn.Linear(2048, 784)
        self._3test = nn.Linear(784, 10)

    def forward(self, x):
        h = self.test_1(x)
        h = F.mish(h)
        h = self.te_2st(h)
        h = x + h
        h = self._3test(h)
        return h


# LyCORIS wrapper demo
net = DemoNet()
LycorisNetwork.apply_preset({"target_name": [".*te.*"]})
lycoris_net1 = create_lycoris(net, 1.0, linear_dim=16, linear_alpha=2.0, algo="lokr")
lycoris_net1.apply_to()

LycorisNetwork.apply_preset({"target_name": [".*es.*"]})
lycoris_net2 = create_lycoris(net, 1.0, linear_dim=16, linear_alpha=2.0, algo="lokr")
lycoris_net2.apply_to()

print(f"#Modules of net1: {len(lycoris_net1.loras)}")
print(f"#Modules of net2: {len(lycoris_net2.loras)}")

print("Total params:", sum(p.numel() for p in net.parameters()))
print("Net1 Params:", sum(p.numel() for p in lycoris_net1.parameters()))
print("Net2 Params:", sum(p.numel() for p in lycoris_net2.parameters()))


# Training loop demo
trns = Compose([ToTensor(), Lambda(lambda x: x.view(-1))])
train_ds = MNIST(root="data", download=True, train=True, transform=trns)
test_ds = MNIST(root="data", download=True, train=False, transform=trns)
train_loader = Data.DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = Data.DataLoader(test_ds, batch_size=32)

optimizer = torch.optim.AdamW(
    chain(lycoris_net1.parameters(), lycoris_net2.parameters()), lr=0.005
)
ema_loss = 0


for i, (x, t) in enumerate(train_loader):
    optimizer.zero_grad()
    y = net(x)
    loss = F.cross_entropy(y, t)
    loss.backward()
    optimizer.step()
    ema_decay = min(0.999, i / 1000)
    ema_loss = ema_decay * ema_loss + (1 - ema_decay) * loss.item()
    if i % 100 == 0:
        print(i, ema_loss)


total_correct = 0
total_num = 0
for i, (x, t) in enumerate(test_loader):
    y = net(x)
    pred = y.argmax(dim=1)
    total_correct += (pred == t).sum().item()
    total_num += len(t)
print(total_correct / total_num)
