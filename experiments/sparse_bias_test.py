import os, sys
sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F

from lycoris.utils import extract_conv
from pytorch_lightning import seed_everything


def make_sparse(t, sparsity=0.95):
    sparse_t = t.masked_fill(
        torch.abs(t) < torch.quantile(torch.abs(t), sparsity), 
        0
    )
    return sparse_t

seed_everything(0)

KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
IN_CH = 1280
OUT_CH = 1280
LORA_RANK = 32
SIZE = 32

conv_orig = nn.Conv2d(IN_CH, OUT_CH, 3, 1, 1, bias=False)
conv_orig.weight = nn.Parameter(
    (torch.randn(OUT_CH, LORA_RANK*2) @ torch.randn(LORA_RANK*2, IN_CH*9) * 0.05).reshape(conv_orig.weight.shape)
)
conv_rebd = nn.Conv2d(IN_CH, OUT_CH, 3, 1, 1, bias=False)

extract_a, extract_b, diff = extract_conv(conv_orig.weight, 'fixed', LORA_RANK)
out_weight = extract_b.reshape(OUT_CH, -1) @ extract_a.reshape(LORA_RANK, -1)
out_weight = out_weight.reshape(conv_orig.weight.shape)


conv_rebd.weight = nn.Parameter(out_weight)
print()
print('without sparse bias')
print('MSE Loss: ', F.mse_loss(conv_orig.weight, conv_rebd.weight))
print('L1 Loss : ', F.l1_loss(conv_orig.weight, conv_rebd.weight))
print('Distance: ', torch.dist(conv_orig.weight, conv_rebd.weight))


conv_rebd.weight = nn.Parameter(out_weight + make_sparse(diff, 0.99))
print()
print('with sparse bias, sparsity 99% ')
print('MSE Loss: ', F.mse_loss(conv_orig.weight, conv_rebd.weight))
print('L1 Loss : ', F.l1_loss(conv_orig.weight, conv_rebd.weight))
print('Distance: ', torch.dist(conv_orig.weight, conv_rebd.weight))


conv_rebd.weight = nn.Parameter(out_weight + make_sparse(diff, 0.98))
print()
print('with sparse bias, sparsity 98% ')
print('MSE Loss: ', F.mse_loss(conv_orig.weight, conv_rebd.weight))
print('L1 Loss : ', F.l1_loss(conv_orig.weight, conv_rebd.weight))
print('Distance: ', torch.dist(conv_orig.weight, conv_rebd.weight))


conv_rebd.weight = nn.Parameter(out_weight + make_sparse(diff, 0.95))
print()
print('with sparse bias, sparsity 95% ')
print('MSE Loss: ', F.mse_loss(conv_orig.weight, conv_rebd.weight))
print('L1 Loss : ', F.l1_loss(conv_orig.weight, conv_rebd.weight))
print('Distance: ', torch.dist(conv_orig.weight, conv_rebd.weight))