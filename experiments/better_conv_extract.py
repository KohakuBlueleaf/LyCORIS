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

def make_cp(t, x1, x2):
    x1 = x1.reshape(x1.size(0), -1)
    x2 = x2.reshape(x2.size(0), -1)
    
    # [rank, rank, k, k] * [rank, out]
    temp = torch.einsum('n m k l, i n -> i m k l', t, x2)
    
    # [out, rank, k, k] * [rank, in]
    result = torch.einsum('i m k l, m j -> i j k l', temp, x1)
    print(result.shape)
    return result


seed_everything(0)

KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
IN_CH = 1280
OUT_CH = 640
LORA_RANK = 32
SIZE = 32

conv_orig = nn.Conv2d(IN_CH, OUT_CH, 3, 1, 1, bias=False).cuda()
conv_orig.weight = nn.Parameter(
    (torch.randn(OUT_CH, LORA_RANK) 
     @ torch.randn(LORA_RANK, IN_CH*9) 
     * 0.05
     ).reshape(conv_orig.weight.shape).cuda()
)
conv_rebd = nn.Conv2d(IN_CH, OUT_CH, 3, 1, 1, bias=False).cuda()

extract_a, extract_b, diff = extract_conv(conv_orig.weight, 'fixed', LORA_RANK, 'cuda')
out_weight = extract_b.reshape(OUT_CH, -1) @ extract_a.reshape(LORA_RANK, -1)
out_weight = out_weight.reshape(conv_orig.weight.shape)

extract_a, extract_c, diff = extract_conv(extract_a.transpose(0, 1), 'fixed', LORA_RANK, 'cuda')
extract_a = extract_a.transpose(0, 1)
extract_c = extract_c.transpose(0, 1)
print(extract_a.shape, extract_b.shape, extract_c.shape)


conv_rebd.weight = nn.Parameter(make_cp(
    extract_a, extract_c, extract_b
))
print()
print('without sparse bias')
print('MSE Loss: ', F.mse_loss(conv_orig.weight, conv_rebd.weight))
print('L1 Loss : ', F.l1_loss(conv_orig.weight, conv_rebd.weight))
print('Distance: ', torch.dist(conv_orig.weight, conv_rebd.weight))