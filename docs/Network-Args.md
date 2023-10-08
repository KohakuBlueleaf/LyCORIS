# Network Arguments

Arguments to put in `network_args` for kohya sd  scripts

### Algo
- set with `algo=ALGO_NAME`
- check [List of Implemented Algorithms](Algo-List.md) for algorithms to use

### Preset
- set with `preset=PRESET/CONFIG_FILE`
- pre-impelmented: `full` (default), `attn-mlp`, `attn-only` etc.
- Valid for all but (IA)^3
- Use `preset=xxx.toml` to choose config file(for LyCORIS module settings)
- More info in Preset.md

### Dimension
- set with `dim=INT`
- Valid for all but (IA)^3 and native fine-tuning
- For LoKr, setting dimension to sufficiently large value (10240) prevents the second block from being further decomposed


### Alpha
- set with `alpha=NUMBER`
- Valid for all but (IA)^3 and native fine-tuning, ignored by full dimension LoKr as well
- Merge ratio is alpha/dimension, check Appendix B.1 of our [paper](https://arxiv.org/abs/2309.14859) for relation between alpha and learning rate / initialization

### Factor
- set with `factor=INT`
- Valid for LoKr
- use `-1` to get smallest decomposition.
- super high or super low value will give you full finetune.

### Block Size
- set with `block_size=INT`
- Valid for DyLoRA
- set the "unit" of DyLoRA

### Tucker Decomposition
- Enable with `use_tucker=True`
- Valid for all but (IA)^3 and native fine-tuning
- it was given the wrong name`use_cp=` in older version

### Scalar
- Enable with `use_scalar=True`
- Use a different weight initial strategy

### Normalization Layers
- Enable with `train_norm=True`
