import torch
import torch.nn
import torch.nn.functional as F


from matplotlib import pyplot as plt
import numpy as np
from collections import Counter


def check_diff_singular_value(
    a: torch.TensorType,
    b: torch.TensorType
):
    assert a.shape == b.shape
    
    diff = a-b
    U, S, Vh = torch.linalg.svd(diff)
    
    return S


def load_models(a, b):
    model_a = torch.load(a)['state_dict']
    model_b = torch.load(b)['state_dict']
    
    linear_singular = []
    linear_layer = {}
    conv_singular = []
    conv_layer = {}
    for k, va, vb in ((k, model_a[k], model_b[k]) for k in model_a if k in model_b):
        if va.shape != vb.shape:
            continue
        if torch.allclose(va.float(), vb.float()): continue
        if 'first_stage' in k: continue
        
        if len(va.shape) == 2:
            out, in_ = va.shape
            if out>10000 or in_>10000:
                continue
            linear_singular.append(
                check_diff_singular_value(
                    va.float().cuda(), 
                    vb.float().cuda()
                ).cpu()
            )
            linear_layer[k] = torch.sum(linear_singular[-1]>0.05).item()
        elif len(va.shape) == 4:
            out_ch, in_ch, kw, kh = va.shape
            conv_singular.append(
                check_diff_singular_value(
                    va.reshape(out_ch, -1).float().cuda(),
                    vb.reshape(out_ch, -1).float().cuda()
                ).cpu()
            )
            conv_layer[k] = torch.sum(conv_singular[-1]>0.1).item()
        else:
            continue
        
        print(k)
    
    linear_singular = torch.stack([i[:128] for i in linear_singular if i.shape[0]>100])
    rank_counter = Counter([torch.sum(i>0.05).item() for i in linear_singular if i.shape[0]>100])
    linear_singular_rank = [rank_counter[i] for i in range(128)]
    
    conv_singular = torch.stack([i[:128] for i in conv_singular if i.shape[0]>100])
    rank_counter = Counter([torch.sum(i>0.1).item() for i in conv_singular if i.shape[0]>100])
    conv_singular_rank = [rank_counter[i] for i in range(128)]
    
    lin_std, lin_mean = torch.std_mean(linear_singular, dim=0)
    lin_max = torch.max(linear_singular, dim=0).values
    lin_min = torch.min(linear_singular, dim=0).values
    conv_std, conv_mean = torch.std_mean(conv_singular, dim=0)
    conv_max = torch.max(conv_singular, dim=0).values
    conv_min = torch.min(conv_singular, dim=0).values
    
    fig, ax = plt.subplots()
    fig, ax_conv = plt.subplots()
    
    x = np.arange(1, 129)
    # ax.plot(x, np.array(lin_mean))
    # ax.fill_between(x, np.array(lin_mean-lin_std), np.array(lin_mean+lin_std), alpha=0.5)
    # ax.fill_between(x, lin_min.numpy(), lin_max.numpy(), alpha=0.2)
    # ax_conv.plot(x, np.array(conv_mean))
    # ax_conv.fill_between(x, np.array(conv_mean-conv_std), np.array(conv_mean+conv_std), alpha=0.5)
    # ax_conv.fill_between(x, conv_min.numpy(), conv_max.numpy(), alpha=0.2)
    
    # ax.plot(list(linear_layer.keys()), list(linear_layer.values()))
    ax_conv.plot(list(conv_layer.keys()), list(conv_layer.values()))
    plt.show()


if __name__ == '__main__':
    load_models(
        './test_model/umamusume0224-continue-ep00-gs123386.ckpt',
        './test_model/ACertainty.ckpt',
    )