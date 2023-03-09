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
And based on this [experiments](https://civitai.com/models/17336/roukin8-character-lohaloconfullckpt-8), LoHa with same size (and dim>2, or rank>4) can get definitely better result.

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
