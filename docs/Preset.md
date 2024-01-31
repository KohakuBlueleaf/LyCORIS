# Preset/Config system

Preset/config system is added after LyCORIS 1.9.0 for more fine-grained control.

## Preset

LyCORIS provides a few presets for common users

* `preset=full`
  * default preset, train all the layers in the UNet and CLIP.
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

You can check [example config](../example_configs/preset_configs/example.toml) for example usage.


### arguments

* enable_conv `bool`
  * Enable training for convolution layers or not.
* unet_target_module `list[str]`
  * A list of name of the module classes you want to train.
    * In the example, we train almost all the blocks in the UNet.
* unet_target_name `list[str]`
  * A list of name of the modules you want to train.
    * In the example, we train the few layers which is not under the blocks noted above.
  * Regex is ok.
* text_encoder_target_module `list[str]`
  * As same as unet_target_module but for TE.
* text_encoder_target_name `list[str]`
  * As same as unet_target_name but for TE.
* module_algo_map/name_algo_map `dict[str, str]`
  * to apply different settings (for example: algo, dim, alpha, ...etc) to different class/name.
    * only enabled if the module/name is trained. (Ensure the target_module/name include it)
  * Check [example.toml](../example_configs/preset_configs/example.toml) for the full format.
