import torch
import torch.nn as nn
import torch.nn.functional as F


def make_pro3(t, x1, x2):
    x1 = x1.reshape(x1.size(0), -1)
    x2 = x2.reshape(x2.size(0), -1).transpose(0, 1)
    
    temp = (t.transpose(0, 3) @ x2).transpose(0, 3)
    return (temp.transpose(1, 3) @ x1).transpose(1, 3)


KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
IN_CH = 1280
OUT_CH = 1280
LORA_RANK = 256
SIZE = 32


conv_down = nn.Conv2d(IN_CH, LORA_RANK, 1, bias=False)
conv_main = nn.Conv2d(LORA_RANK, LORA_RANK, 3, 1, 1, bias=False)
conv_up = nn.Conv2d(LORA_RANK, OUT_CH, 1, bias=False)

conv_orig = nn.Conv2d(IN_CH, OUT_CH, 3, 1, 1, bias=False)
conv_orig.weight = nn.Parameter(make_pro3(
    conv_main.weight, conv_down.weight, conv_up.weight
))


test_x = torch.randn(1, IN_CH, SIZE, SIZE)
test_out_lora = conv_up(conv_main(conv_down(test_x)))
test_out_orig = conv_orig(test_x)

print('MSE Loss: ', F.mse_loss(test_out_orig, test_out_lora))
print('L1 Loss : ', F.l1_loss(test_out_orig, test_out_lora))
print('Distance: ', torch.dist(test_out_orig, test_out_lora))


test_1 = torch.randn(256, 256)/16
test_2 = torch.randn(256, 256)/16
test_3 = torch.randn(256, 256)/16

test_a = test_1@test_3 + test_2@test_3
test_b = (test_1+test_2) @ test_3

print('MSE Loss: ', F.mse_loss(test_a, test_b))
print('L1 Loss : ', F.l1_loss(test_a, test_b))
print('Distance: ', torch.dist(test_a, test_b))