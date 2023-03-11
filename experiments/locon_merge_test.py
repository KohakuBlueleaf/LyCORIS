import torch
import torch.nn as nn
import torch.nn.functional as F

from lycoris.utils import merge_conv


KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
IN_CH = 1280
OUT_CH = 1280
LORA_RANK = 256
SIZE = 32

convA = nn.Conv2d(IN_CH, LORA_RANK, 3, 1, 1, bias=False)
convB = nn.Conv2d(LORA_RANK, OUT_CH, 1, bias=False)

conv_orig = nn.Conv2d(IN_CH, OUT_CH, 3, 1, 1, bias=False)
conv_orig.weight = merge_conv(convA.weight, convB.weight)


test_x = torch.randn(1, IN_CH, SIZE, SIZE)
test_out_lora = convB(convA(test_x))
test_out_orig = conv_orig(test_x)

print('MSE Loss: ', F.mse_loss(test_out_orig, test_out_lora))
print('L1 Loss : ', F.l1_loss(test_out_orig, test_out_lora))
print('Distance: ', torch.dist(test_out_orig, test_out_lora))