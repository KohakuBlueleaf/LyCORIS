# LoCon - LoRA for Convolution Network

## Motivation
convolution is matmul so there should be a lora version for it.

## Why Convolution is matmul?

im2col:

## What I done
* A demo for LoRA on Convolution network(This repo)
* A network module for kohya_ss/sd-script
* An [Extension](https://github.com/KohakuBlueleaf/a1111-sd-webui-locon) for using this↑ in sd-webui


## usage
### For kohya script
move locon folder into kohya-ss/sd-scripts
and use 
```bash
python3 sd-scripts/train_network.py \
  --network_module=locon.locon_kohya
```
to train locon+lora for SD model

### For a1111's sd-webui
download [Extension](https://github.com/KohakuBlueleaf/a1111-sd-webui-locon) into sd-webui, and then use locon model as how you use lora model.

### Example Model
Onimai LoRA:

---
## Some calculation
LoRA for Linear:
$$
Y_{out*batch} = W_{out*in}‧X_{in*batch}\\
Y'_{out*batch} = W_{out*in}‧X_{in*batch} + Wa_{out*rank}‧Wb_{rank*in}‧X_{in*batch}\\
$$

Convolution img2col:
$$
X:[\bold{c}hannel, \bold{w}idth, \bold{h}eight]
\xrightarrow{reorder}[c*kw*kh, outw*outh]\\
Kernels: [out, c, kw, kh] \xrightarrow{reshape} [out, c*kw*kh]\\

Conv(X, Kernels) = Kernels * X \xrightarrow{reshape} [out, outw, outh]
$$

LoRA for Convolution:
$$
Conv(in, out, ksize, padding, stride)\\
\xrightarrow{}Conv(rank, out, 1)\circ Conv(in, rank, ksize, padding, stride)
$$


Another form:
```
[out_ch, in_ch*size**2] x [in_ch*size**2, out_h * out_w]
↓
[out_ch, LoRA_rank] x [LoRA_rank, in_ch*size**2] x [in_ch*size**2, out_h * out_w]
↓
[out_ch, LoRA_rank] x ([LoRA_rank, in_ch*size**2] x [in_ch*size**2, out_h * out_w])
↓
[out_ch, LoRA_rank] x [LoRA_rank, out_h * out_w]
```

FLOPS:
* before = out_ch \* in_ch \* size\*\*2 \* out_h \* out_w

* after  = out_ch \* LoRA_rank \* out_h \* out_w + LoRA_rank \* in_ch \* size\*\*2 \* out_h \* out_w

* after = (out_ch \* LoRA_rank + LoRA_rank \* in_ch \* size\*\*2) \* out_h \* out_w

Params to train:
* before = out_ch \* in_ch \* size\*\*2

* after  = LoRA_rank \* in_ch \* size\*\*2 + LoRA_rank \* out_ch