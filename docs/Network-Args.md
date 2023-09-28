# Network Arguments

Arguments to put in `network_args` for kohya sd  scripts

### Algo
- set with `algo=`
- check [List of Implemented Algorithms](Algo-List.md) for algorithms to use

### Preset
- set with `preset=`
- pre-impelmented: `full` (default), `attn-mlp`, `attn-only`
- Valid for all but (IA)^3
- Used for ...

### Dimension
- set with `dim=`
- Valid for all but (IA)^3 and native fine-tuning
- For LoKr, setting dimension to sufficiently large value (XXX for SD1, XXX for SDXL) prevents the second block from being further decomposed


### Alpha
- set with `alpha=`
- Valid for all but (IA)^3 and native fine-tuning, ignored by full dimension LoKr as well
- Merge ratio is alpha/dimension, check Appendix B.1 of our [paper](https://arxiv.org/abs/2309.14859) for relation between alpha and learning rate / initialization

### Factor
- set with `factor=`
- Valid for LoKr

### Block Size
- set with `block_size=`
- Valid for DyLoRA

### Tucker Decomposition
- set with `use_tucker=`
- Valid for all but (IA)^3 and native fine-tuning
- it was given the wrong name`use_cp=` in older version

### Scalar
- set with `use_scalar=`

### Normalization Layers
- set with `train_norm=`
