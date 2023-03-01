# LoCon - LoRA for Convolution Network

## Motivation
convolution is matmul so there should be a lora version for it.

## Why Convolution is matmul?

im2col:
![image](https://user-images.githubusercontent.com/59680068/221547963-c821b9fa-2825-4b8d-8192-c3109268417f.png)
![image](https://user-images.githubusercontent.com/59680068/221547996-4be14700-1392-4859-9e29-e3e669142a09.png)


## What I done
* A demo for LoRA on Convolution network(This repo)
* A network module for kohya_ss/sd-script
* An [Extension](https://github.com/KohakuBlueleaf/a1111-sd-webui-locon) for using this↑ in sd-webui


## Difference from training LoRA on Stable Diffusion
normally most of people train LoRA with kohya-ss/sd-scripts, (me too)

but lora only train green part, locon can train yellow part. Combine them can cover almost all of the layers in the model.

(I skip the porj in and porj out since they have very small channels, if you want to f/t them, maybe just f/t it without any trick.)
![image](https://user-images.githubusercontent.com/59680068/221555165-7b0a1b96-0cc4-4ec4-bdd7-559a43002c65.png)



## usage
### For kohya script
move locon folder into kohya-ss/sd-scripts
and use 
```bash
python3 sd-scripts/train_network.py \
  --network_module locon.locon_kohya \
  --network_args "conv_dim=RANK_FOR_CONV" "conv_alpha=ALPHA_FOR_CONV" \
  --network_dim "RANK_FOR_TRANSFORMER" --network_alpha "ALPHA_FOR_TRANSFORMER"
```
to train locon+lora for SD model

### For a1111's sd-webui
download [Extension](https://github.com/KohakuBlueleaf/a1111-sd-webui-locon) into sd-webui, and then use locon model as how you use lora model.

### Example Model
Onimai LoRA:
https://huggingface.co/KBlueLeaf/onimai-locon-test
![05510-2023-02-27_dc50ca8f4b_download_TTRH_3334316821_c1054458-576x832](https://user-images.githubusercontent.com/59680068/221551622-e26477a7-f929-42a3-9cd5-937ca1595daf.png)

---
## Some calculation
LoRA for Linear:
```math
Y_{out*batch} = W_{out*in}‧X_{in*batch}
```
```math
Y'_{out*batch} = W_{out*in}‧X_{in*batch} + Wa_{out*rank}‧Wb_{rank*in}‧X_{in*batch}
```

Convolution img2col:
```math
X:[channel, width, height]
```
```math
\xrightarrow{reorder}[c*kw*kh, outw*outh]
```
```math
Kernels: [out, c, kw, kh] \xrightarrow{reshape} [out, c*kw*kh]
```
```math
Conv(X, Kernels) = Kernels * X \xrightarrow{reshape} [out, outw, outh]
```

LoRA for Convolution:
```math
Conv(in, out, ksize, padding, stride)
```
```math
\xrightarrow{}Conv(rank, out, 1)\circ Conv(in, rank, ksize, padding, stride)
```


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


## Citation

```bibtex
@misc{LoCon,
  author       = "Shih-Ying Yeh (Kohaku-BlueLeaf)",
  title        = "LoCon - LoRA for Convolution Network",
  howpublished = "\url{https://github.com/KohakuBlueleaf/LoCon}",
  month        = "Feb",
  year         = "2023"
}
```
