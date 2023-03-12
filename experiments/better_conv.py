import torch
import torch.nn as nn
import torch.nn.functional as F


class HadaWeightCP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, orig_weight, t1, w1a, w1b, t2, w2a, w2b, scale=torch.tensor(1)):
        ctx.save_for_backward(t1, w1a, w1b, t2, w2a, w2b, scale)
        
        temp = torch.einsum('i j k l, j r -> i r k l', t1, w1b)
        rebuild1 = torch.einsum('i j k l, i r -> r j k l', temp, w1a)
        
        temp = torch.einsum('i j k l, j r -> i r k l', t2, w2b)
        rebuild2 = torch.einsum('i j k l, i r -> r j k l', temp, w2a)
        
        return orig_weight + rebuild1*rebuild2*scale

    @staticmethod
    def backward(ctx, grad_out):
        (t1, w1a, w1b, t2, w2a, w2b, scale) = ctx.saved_tensors
        
        grad_out = grad_out*scale
        
        temp = torch.einsum('i j k l, j r -> i r k l', t2, w2b)
        rebuild = torch.einsum('i j k l, i r -> r j k l', temp, w2a)
        
        grad_w = rebuild*grad_out
        del rebuild
        
        grad_w1a = torch.einsum('r j k l, i j k l -> r i', temp, grad_w)
        grad_temp = torch.einsum('i j k l, i r -> r j k l', grad_w, w1a.T)
        del grad_w, temp
        
        grad_w1b = torch.einsum('i r k l, i j k l -> r j', t1, grad_temp)
        grad_t1 = torch.einsum('i j k l, j r -> i r k l', grad_temp, w1b.T)
        del grad_temp
        
        temp = torch.einsum('i j k l, j r -> i r k l', t1, w1b)
        rebuild = torch.einsum('i j k l, i r -> r j k l', temp, w1a)
        
        grad_w = rebuild*grad_out
        del rebuild
        
        grad_w2a = torch.einsum('r j k l, i j k l -> r i', temp, grad_w)
        grad_temp = torch.einsum('i j k l, i r -> r j k l', grad_w, w2a.T)
        del grad_w, temp
        
        grad_w2b = torch.einsum('i r k l, i j k l -> r j', t2, grad_temp)
        grad_t2 = torch.einsum('i j k l, j r -> i r k l', grad_temp, w2b.T)
        del grad_temp
        return grad_out, grad_t1, grad_w1a, grad_w1b, grad_t2, grad_w2a, grad_w2b, None


def make_weight_cp(orig_weight, t1, w1a, w1b, t2, w2a, w2b, scale=torch.tensor(0.25)):
    return HadaWeightCP.apply(orig_weight, t1, w1a, w1b, t2, w2a, w2b, scale)


def make_cp(orig_weight, t1, w1a, w1b, t2, w2a, w2b, scale=torch.tensor(0.25)):
    temp = torch.einsum('i j k l, j r -> i r k l', t1, w1b)
    rebuild1 = torch.einsum('i j k l, i r -> r j k l', temp, w1a)
    
    temp = torch.einsum('i j k l, j r -> i r k l', t2, w2b)
    rebuild2 = torch.einsum('i j k l, i r -> r j k l', temp, w2a)
    
    return orig_weight + rebuild1*rebuild2 * scale


KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
IN_CH = 1280
OUT_CH = 1208
LORA_RANK = 4
SIZE = 32


t1 = nn.Parameter(torch.randn(LORA_RANK, LORA_RANK, 3, 3))
w1a = nn.Parameter(torch.randn(LORA_RANK, OUT_CH))
w1b = nn.Parameter(torch.randn(LORA_RANK, IN_CH))

t2 = nn.Parameter(torch.randn(LORA_RANK, LORA_RANK, 3, 3))
w2a = nn.Parameter(torch.randn(LORA_RANK, OUT_CH))
w2b = nn.Parameter(torch.randn(LORA_RANK, IN_CH))

orig = nn.Parameter(torch.randn(OUT_CH, IN_CH, 3, 3))

test_x = torch.randn(1, 4, 64, 64)
test_t = torch.randn(1, 4, 64, 64)


w1 = make_cp(orig, t1, w1a, w1b, t2, w2a, w2b)
w2 = make_weight_cp(orig, t1, w1a, w1b, t2, w2a, w2b)

torch.mean(w1).backward()
grad1 = t1.grad.clone()
t1.grad = None

torch.mean(w2).backward()
grad2 = t1.grad.clone()


print('MSE Loss: ', F.mse_loss(grad1, grad2))
print('L1 Loss : ', F.l1_loss(grad1, grad2))
print('Distance: ', torch.dist(grad1, grad2))