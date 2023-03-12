from timeit import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F


def ein(t, x1, x2, grad):
    #inference
    rebuild = torch.einsum('i j k l, j r, i p -> p r k l', t, x1, x2)
    # temp = torch.einsum('i j k l, j r -> i r k l', t, x1)
    # rebuild = torch.einsum('i j k l, i r -> r j k l', temp, x2)
    
    #backward
    temp = torch.einsum('i j k l, j r -> i r k l', t, x1)
    rebuild = torch.einsum('i j k l, i r -> r j k l', temp, x2)
    
    grad_w = rebuild*grad
    grad_x2 = torch.einsum('r j k l, i j k l -> r i', temp, grad_w)
    grad_temp = torch.einsum('i j k l, i r -> r j k l', grad_w, x2.T)
    
    grad_x1 = torch.einsum('i r k l, i j k l -> r j', t, grad_temp)
    grad_t = torch.einsum('i j k l, j r -> i r k l', grad_temp, x1.T)
    
    # assert t.shape == grad_t.shape
    # assert x1.shape == grad_x1.shape
    # assert x2.shape == grad_x2.shape


def mat(t, x1, x2, grad):
    #inference
    temp = (t.transpose(1, 3) @ x1).transpose(1, 3)
    rebuild = (temp.transpose(0, 3) @ x2).transpose(0, 3)
    
    #backward
    temp = (t.transpose(1, 3) @ x1).transpose(1, 3)
    rebuild = (temp.transpose(0, 3) @ x2).transpose(0, 3)
    
    grad_w = rebuild*grad
    grad_x2 = torch.einsum('r j k l, i j k l -> r i', temp, grad_w)
    grad_temp = (grad_w.transpose(0, 3) @ x2.T).transpose(0, 3)
    
    grad_x1 = torch.einsum('i r k l, i j k l -> r j', t, grad_temp)
    grad_t = (grad_temp.transpose(1, 3) @ x1.T).transpose(1, 3)
    
    # assert t.shape == grad_t.shape
    # assert x1.shape == grad_x1.shape
    # assert x2.shape == grad_x2.shape


def fold(w1a, w1b, grad):
    #inference
    rebuild = (w1a@w1b).reshape(OUT_DIM, IN_DIM, K_SIZE, K_SIZE)
    
    #backward
    rebuild = (w1a@w1b)
    grad_w1 = rebuild * grad.reshape(rebuild.shape)
    grad_w1a = grad_w1 @ w1b.T
    grad_w1b = w1a.T @ grad_w1
    
    # assert w1a.shape == grad_w1a.shape
    # assert w1b.shape == grad_w1b.shape


IN_DIM = 1280
OUT_DIM = 1280
K_SIZE = 3
RANK = 32

t = torch.randn(RANK, RANK, K_SIZE, K_SIZE).cuda()
x1 = torch.randn(RANK, IN_DIM).cuda()
x2 = torch.randn(RANK, OUT_DIM).cuda()

w1a = torch.randn(OUT_DIM, RANK).cuda()
w1b = torch.randn(RANK, IN_DIM*K_SIZE**2).cuda()

grad = torch.randn(OUT_DIM, IN_DIM, K_SIZE, K_SIZE).cuda()

ein(t, x1, x2, grad)
mat(t, x1, x2, grad)
fold(w1a, w1b, grad)

print(f'Proposition 1 Params: {w1a.size().numel() + w1b.size().numel()}')
print(f'Proposition 3 Params: {t.size().numel() + x1.size().numel() + x2.size().numel()}')

NUM = 5000
print('CP einsum', timeit('ein(t, x1, x2, grad)', globals=globals(), number=NUM)/NUM)
print('CP matmul', timeit('mat(t, x1, x2, grad)', globals=globals(), number=NUM)/NUM)
print('Pro1 matmul', timeit('fold(w1a, w1b, grad)', globals=globals(), number=NUM)/NUM)