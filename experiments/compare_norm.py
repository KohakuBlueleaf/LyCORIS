import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from safetensors import safe_open


with safe_open("./sd1.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensor = f.get_tensor(k)
        if len(tensor.shape)>1 and max(tensor.shape)>1000:
            print(torch.norm(tensor), tensor.shape)

print('='*100)
print('-'*100)
with safe_open("./xl.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensor = f.get_tensor(k)
        if len(tensor.shape)>1 and max(tensor.shape)>1000:
            print(torch.norm(tensor), tensor.shape)