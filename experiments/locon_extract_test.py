import os, sys
sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F

from lycoris.utils import extract_conv


KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
IN_CH = 1280
OUT_CH = 1280
LORA_RANK = 32
SIZE = 32

convA = nn.Conv2d(IN_CH, LORA_RANK, 3, 1, 1, bias=False)
convB = nn.Conv2d(LORA_RANK, OUT_CH, 1, bias=False)

conv_orig = nn.Conv2d(IN_CH, OUT_CH, 3, 1, 1, bias=False)

extract_a, extract_b = extract_conv(conv_orig.weight, LORA_RANK)
convA.weight = nn.Parameter(extract_a)
convB.weight = nn.Parameter(extract_b)


test_x = torch.randn(1, IN_CH, SIZE, SIZE)
test_out_lora = convB(convA(test_x))
test_out_orig = conv_orig(test_x)

print('MSE Loss: ', F.mse_loss(test_out_orig, test_out_lora))
print('L1 Loss : ', F.l1_loss(test_out_orig, test_out_lora))
print('Distance: ', torch.dist(test_out_orig, test_out_lora))