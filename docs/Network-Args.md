# Network Arguments

Arguments to put in `network_args` for kohya sd scripts

### Algo

- Set with `algo=ALGO_NAME`
- Check [List of Implemented Algorithms](Algo-List.md) for algorithms to use

### Preset

- Set with `preset=PRESET/CONFIG_FILE`
- Pre-implemented: `full` (default), `attn-mlp`, `attn-only` etc.
- Valid for all but (IA)^3
- Use `preset=xxx.toml` to choose config file (for LyCORIS module settings)
- More info in [Preset](Preset.md)

### Dimension

- Dimension of the linear layers is set with the _script argument_ `network_dim`
- Dimension of the convolutional layers is set with `conv_dim=INT`
- Valid for all but (IA)^3 and native fine-tuning
- For LoKr, setting dimension to sufficiently large value (>10240/2) prevents the second block from being further decomposed

### Alpha

- Alpha of the linear layers is set with the _script argument_ `network_alpha`
- Alpha of the convolutional layers is set with `conv_alpha=FLOAT`
- Valid for all but (IA)^3 and native fine-tuning, ignored by full dimension LoKr as well
- Merge ratio is alpha/dimension, check Appendix B.1 of our [paper](https://arxiv.org/abs/2309.14859) for relation between alpha and learning rate / initialization

### Dropouts

- Set with `dropout=FLOAT`, `rank_dropout=FLOAT`, `module_dropout=FLOAT`
- Set the dropout rate, the types of dropout that are valid could vary from method to method

### Factor

- Set with `factor=INT`
- Valid for LoKr
- Use `-1` to get the smallest decomposition

### Decompose both

- Enabled with `decompose_both=True`
- Valid for LoKr
- Perform LoRA decomposition of both matrices resulting from LoKr decomposition (by default only the larger matrix is decomposed)

### Block Size

- Set with `block_size=INT`
- Valid for DyLoRA
- Set the "unit" of DyLoRA (i.e. how many rows / columns to update each time)

### Tucker Decomposition

- Enabled with `use_tucker=True`
- Valid for all but (IA)^3 and native fine-tuning
- It was given the wrong name `use_cp=` in older version

### Scalar

- Enabled with `use_scalar=True`
- Valid for LoRA, LoHa, and LoKr.
- Train an additional scalar in front of the weight difference
- Use a different weight initialization strategy

### Weight Decompose

* Enabled with `dora_wd=True`
* Valid for LoRA, LoHa, and LoKr
* Enable the DoRA method for these algorithms.
* Will force `bypass_mode=False`

### Bypass Mode

* Enabled with `bypass_mode=True`
* Valid for LoRA, LoHa, LoKr
* Use $Y = WX + \Delta WX$  instead of $Y=(W+\Delta W)X$
* Designed for bnb 8bit/4bit linear layer. (QLyCORIS)

### Normalization Layers

- Enabled with `train_norm=True`
- Valid for all but (IA)^3

### Rescaled OFT

- Enabled with `rescaled=True`
- Valid for Diag-OFT

### Constrained OFT

- Enabled with `constraint=FLOAT`
- Valid for Diag-OFT

### Singular Vector Type (T-LoRA)

- Set with `sig_type=STRING`
- Valid for T-LoRA
- Options: `principal` (default), `last`, `middle`
- Controls which singular vectors from SVD are used for initialization:
  - `principal`: Top-k singular vectors (largest singular values)
  - `last`: Bottom-k singular vectors (smallest singular values)
  - `middle`: Middle-k singular vectors

### Data-Dependent Initialization (T-LoRA)

- Enabled with `use_data_init=True` (default)
- Valid for T-LoRA
- When True, performs SVD on the original layer weights
- When False, performs SVD on a random matrix (data-independent)

### Timestep Mask Group (T-LoRA)

- Set with `mask_group_id=INT`
- Valid for T-LoRA
- Default: 0
- For multi-network scenarios, allows different networks to use different timestep masks
