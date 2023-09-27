# Demo, Example, Comparing


## expand LoRA to Convolution Layer
lora for transformer only
vs
lora for whole model

Yog-Sothoth LoRA/LoCon:
https://civitai.com/models/14878/loconlora-yog-sothoth-depersonalization

LoRA rank=1:
![00159](https://user-images.githubusercontent.com/59680068/222422792-b37648c3-af2e-4bee-9f82-b14a5d8e5f5d.png)
LoCon rank=1
![00164](https://user-images.githubusercontent.com/59680068/222422830-4ec9f550-cdff-4314-b694-1658bf9f1c83.png)

xy grid:
![image](https://user-images.githubusercontent.com/59680068/222424002-5ce2572c-9102-4e2d-83f2-100bc41ec272.png)


With some experiments from community, finetuning whole model can learn "More". But in some case this means it will overfit heavily.

---

## Hadamard product vs Conventional
We introduced LoRA finetuning with Hadamard Product representation from FedPara.
And based on this [experiments](https://civitai.com/models/17336/roukin8-character-lohaloconfullckpt-8), LoHa with same size (and dim>2, or rank>4) can get better result in some situation.

![Image](https://i.imgur.com/l3P0TgM.jpg)

This is some comments from the experiments:
```
Why LoHa?

The following comparison should be quite convincing.

(Left 4 LoHa with different number of training steps; Right 4 same thing but for LoCon; same seed same training parameters for the two training)


In addition to the five characters I also train a bunch of style into the model (what I have always been doing actually).

However, LoRa and LoCon do not combine styles with characters that well (characters are only trained with anime images) and this capacity gets largely improved in LoHa.

Note that the two files have almost the same size (around 30mb). For this I set (linear_dim, conv_dim) to (16,8) for LoCon and (8,4) for LoHa. However with Hadamard product the resulting matrix could now be of rank 8x8=64 for linear layers and 4x4=16 for convolutional layers.
```

And there is more example about LoHA vs LoCon in same file size. (diff < 200KB)
![xyz_grid-0330-2023-03-08_d4d1ef62c3_KBlueLeaf_KBlueLeaf-v1 1_4114224331_c2af89d9-2160x2789](https://user-images.githubusercontent.com/59680068/223930255-a1e4b91e-25da-41ce-9dd2-12bd0e976afc.jpg)
![xyz_grid-0324-2023-03-08_d4d1ef62c3_KBlueLeaf_KBlueLeaf-v1 1_2496183095_a91c0526-2160x2789](https://user-images.githubusercontent.com/59680068/223930265-f1f1d658-3722-46ce-93e6-fdec18a58c19.jpg)

![xyz_grid-0300-2023-03-08_b38775f1cf_download_谈秋-v2_3652865965_ed37959c-2640x2047](https://user-images.githubusercontent.com/59680068/223930346-c32a062c-4b3a-40a6-83e3-853127129b7d.jpg)
![xyz_grid-0296-2023-03-08_d4d1ef62c3_KBlueLeaf_KBlueLeaf-v1 1_4001890147_ed37959c-2640x2047](https://user-images.githubusercontent.com/59680068/223930351-45653f67-7b0c-42e6-9a81-cc691d2b9dbf.jpg)

