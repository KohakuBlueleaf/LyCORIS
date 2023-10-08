# Preset/Config system

preset/config system is added after LyCORIS 1.9.0 for more detailed control.

## Preset
LyCORIS provide few presets for common users

* `preset=full`
  * Default preset, train all the layers in the UNet and CLIP.
* `preset=full-lin`
  * `full` but skip convolutional layers.
* `preset=attn-mlp`
  * "kohya preset", train all the transformer block.
* `preset=attn-only`
  * only attention layer will be trained, lot of papers only do training on attn layer.
* `preset=unet-transformer-only`
  * as same as kohya_ss/sd_scripts with disabled TE, or, attn-mlp preset with train_unet_only enabled.
* `preset=unet-convblock-only`
  * only ResBlock, UpSample, DownSample will be trained.


## Configs
You can write a `config.toml` for more detail control

config system allows you to:
* Choose different algorithm for specific module type/module
* Use different setting for specific module type/module
* Enable training for specific module type/module

You can check [example config](example_configs/preset_configs/example.toml) for example usage.