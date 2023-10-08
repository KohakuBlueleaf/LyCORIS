# Network Arguments

Arguments to put in `network_args` for kohya sd  scripts

### Algo
- Set with `algo=ALGO_NAME`
- check [List of Implemented Algorithms](Algo-List.md) for algorithms to use

### Preset
- Set with `preset=PRESET/CONFIG_FILE`
- Pre-impelmented: `full` (default), `attn-mlp`, `attn-only` etc.
- Valid for all but (IA)^3
- Use `preset=xxx.toml` to choose config file (for LyCORIS module settings)
- More info in [Preset](Preset.md)

### Dimension
- Set with `dim=INT`
- Valid for all but (IA)^3 and native fine-tuning
- For LoKr, setting dimension to sufficiently large value (10240) prevents the second block from being further decomposed

### Alpha
- Set with `alpha=NUMBER`
- Valid for all but (IA)^3 and native fine-tuning, ignored by full dimension LoKr as well
- Merge ratio is alpha/dimension, check Appendix B.1 of our [paper](https://arxiv.org/abs/2309.14859) for relation between alpha and learning rate / initialization

### Factor
- Set with `factor=INT`
- Valid for LoKr
- Use `-1` to get the smallest decomposition

### Block Size
- Set with `block_size=INT`
- Valid for DyLoRA
- Set the "unit" of DyLoRA (i.e. how many rows / columns to update each time)

### Tucker Decomposition
- Enabled with `use_tucker=True`
- Valid for all but (IA)^3 and native fine-tuning
- It was given the wrong name`use_cp=` in older version

### Scalar
- Enabled with `use_scalar=True`
- Train an additional scalar in front of the weight difference
- Use a different weight initialization strategy

### Normalization Layers
- Enabled with `train_norm=True`
