# Algorithms Implemented in LyCORIS

See [Algo-Details.md](Algo-Details.md) and [Demo.md](Demo.md) for more examples and explanation

All the methods below except for GLoKr are supported in a1111/sd-webui.
However, newer methods may only be available in the latest release / the dev branch.

### Conventional LoRA

* Trigged by `algo=lora` or `algo=locon` (Just alias)
* Includes Conv layer implementation from LoCon.
* Recommended settings
  * dim <= 64
  * alpha from 1 (or lower, like 0.3) to half dimension
* Ref: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### LoHa

* Trigged by `algo=loha`
* Originally designed for federated learning, but has some cool property like rank<=dim^2 so should be good for parameter-efficient fine-tuning.
  * Conventional LoRA is rank<=dim
* Recommended settings
  * dim <= 32
  * alpha from 1 (or lower) to half dimension
* Seems to have the strongest dampening effect. May be suitable for easier concepts, multi-concept, and the pursuit of better generalization ability.
* Ref: [FedPara Low-Rank Hadamard Product For Communication-Efficient Federated Learning](https://arxiv.org/abs/2108.06098)
* **WARNING: High dim with LoHa may cause unstable loss or loss going to NaN. If you want to use high dim LoHa, please use lower lr.**

### LoKr

* Trigged by `algo=lokr`
* Basically same idea of LoHa, but uses Kronecker product.
* Recommended settings
  * Small LoKr: factor=-1 (from 900~2500 KB)
  * Large LoKr: factor~8, full dimension (LoRA like)
* Full dimension is triggered by setting dimension to sufficiently large number (like 10000). This prevents the second block from being decomposed. alpha is ignored in this case.
* Results learned with smaller LoKr (the default setup) may be harder to transfer (i.e. you can only get reasonable result on the model you trained on).
* Use smaller factor will produce bigger file, you can tune it if you think 2.5 MB full rank is not good enough.
* Ref: [KronA: Parameter Efficient Tuning with Kronecker Adapter](https://arxiv.org/abs/2212.10650)

### Native Fine-Tuning

* Trigged by `algo=full`
* Also known as dreambooth. This just uses full matrix without further decomposition.
* With our implementation you can use it as LoRA, change base model, and merge on the fly.
* Can potentially give the best result if everything is tuned correctly, but is probably also the most sensitive to hyperparameter adjustment.
* Produces the biggest file, but does _not_ train slower than other methods.

### (IA)^3

* Trigged by `algo=ia3`
* **Experimental:** you can try this method with dev version package or install from source code.
* This method needs much higher lr (about 5e-3~1e-2).
* This method is good at learning style, but hard to transfer.
* This method produces very tiny file (less than 1 MB).
* No network arguments for this method (even preset is not applicable here).
* **Can be regarded as a special case of Diag-OFT listed below.**
* Ref: [Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning](https://arxiv.org/abs/2205.05638)

### DyLoRA

* Trigged by `algo=dylora`
* Basically a training trick of LoRA.
* Every step, only update one row/col of LoRA weight.
* When we want to update the kth row/col, we only use 0~k row/col to rebuild the weight (0<=k<=dim-1).
* You can easily resize DyLoRA to target and get similar or even better result than LoRA trained at target dim. (And you don't need to train lot LoRAs with different dim to check which is better).
* You should use large dim with alpha=dim/4~dim (1 or dim is not very recommended).
  * Example: dim=128, alpha=64
* Since we only update 1 row/col each step, you will need more steps to get reasonable result. If you want to train it with fewer steps, you may need to set block_size (number of updated rows/cols per step) to a higher value (default=0).
* Using gradient accumulation with batch size 1 is recommended in the original paper.
* Ref: [DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low Rank Adaptation](https://arxiv.org/abs/2210.07558)

### GLoRA

* Triggered by `algo=glora`
* Ref: [One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning](https://arxiv.org/abs/2306.07967)
* TODO

### GLoKr

- TODO

### Diag-OFT

* Triggered by `algo=diag-oft`
* It preserves the hyperspherical energy by training orthogonal transformations that apply to outputs of each layer.
* It converges faster than LoRA according to the original paper, but experiments are still needed.
* `dim` corresponds to block size: we fix block size instead of block number here to make it more comparable to LoRA
* Set `constraint` to get COFT and set `rescaled` to get rescaled OFT
* Ref: [Controlling Text-to-Image Diffusion by Orthogonal Finetuning](https://arxiv.org/abs/2306.07280)

### BOFT

* Triggered by `algo=boft`
* An advanced version of Diag-OFT which use butterfly operation to get full orthogonol matrix.
* With some extreme hyperparameter settings, it can become diag-oft or oft. Its capbility is between this 2 algorithm.
* constraint and rescaled also supported.
* TODO
* Ref: [Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization](https://arxiv.org/abs/2311.06243)

### T-LoRA

* Triggered by `algo=tlora`
* Timestep-aware LoRA designed specifically for diffusion models.
* Uses SVD-based orthogonal initialization for independent, non-interfering rank components.
* Supports timestep-dependent rank masking: fewer ranks at high noise (structure), more at low noise (detail).
* Recommended settings:
  * dim: 4-16 (orthogonal init is more parameter-efficient)
  * alpha: 1 to dim
  * sig_type: "principal" (default), "last", or "middle"
* Training frameworks must call `set_timestep_mask()` before each forward pass.
* Residual subtraction ensures zero-shot preservation (LoRA contributes nothing at initialization).
* Ref: [T-LoRA: Timestep-aware LoRA for Diffusion Models](https://github.com/rosinality/T-LoRA)
