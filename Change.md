# Change Log

## 2023/12/15 quick fixes of 2.0.2

* Fix bugs in full module.
* Related: Fix bugs in `stable-diffusion-webui/extensions-builtin/Lora`
  * The [PR](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14300)

## 2023/12/14 quick fixes of 2.0.1

* Support merge sdxl loras which trained on plain diffusers with Kohya's LoRA implementation.
  * Can be found in LECO or other similar projects.
* Refactor the batch convert scripts for pivotal bundle and hcp.
* Change the class name `lycoris.kohya.LycorisNetwork` to `lycoris.kohya.LycorisNetworkKohya` to avoid confusion.
* Fix bugs in merge scripts for Norm module and LoKr module.
* Fix bugs in scaled weight norms of OFT.
* Fix bugs in extract scripts for SDXL.
* Fix bugs in full module which consume 2x vram.
* Fix bugs in `create_network_from_weights` which caused bugs in "resume" feature for SDXL.

## 2023/12/02 update to 2.0.0

* Start supporting [HCP-Diffusion](https://github.com/IrisRainbowNeko/HCP-Diffusion) (The reason to name this version "2.0.0")
  * Now LyCORIS support LoHa/LoKr/Diag-OFT algorithm in HCP-Diffusion
  * Add Pivotal tuning utilities
  * Add hcp convert utilities
  * Have no plan at this time to support full/lora and train_norms since HCP can do them natively
* Add Diag-OFT modules
* Add standalone usage support
  * Can wrap any pytorch module which contains Linear/Conv2d/LayerNorm/GroupNorm modules
  * Will support more module in the future
* Add SDXL support in Merge script
* Add SDXL support in Extract-locon
* More efficient (speed/vram) implementation for full module
* Better implementation of custom state_dict
* Fix errors of dropouts
* Fix errors of apply_max_norms
* Fix errors of resume

---

## 2023/09/27 update to 1.9.0

* Add norm modules (for training LayerNorm and GroupNorm, which should be good for style)
* Add full modules (So you can "native finetune" with lycoris now, should be convinient to try different weight)
* Add preset config system
* Add custom config system
* Support resuming from models
* Merge script support norm and full modules
* Fix errors with optional requirements
* Fix errors with not necessary import
* Fix wrong factorization behaviours

## 2023/07/27 update to 1.8.2

* Update utils in kohya-ss/sd-scripts

## 2023/07/27 update to 1.8.1

* Add config/preset system
* Improve the project structure

## 2023/07/19 update to 1.8.0

* reimplement weight init method
* implement HyperDreamBooth into LyCORIS
* better file structure

## 2023/06/28 update to 1.7.1

* **rearrange the version format, previous 0.1.7 should be 1.7.0**
* fix the bug in scale weight norm

## 2023/06/26 Update to 0.1.7

* Add support for rank_dropout and module_dropout on LoCon/LoHa/LoKr
* Add support for scale_weight_norms on LoCon/LoHa/LoKr
* Will support SDXL on 0.1.8 (you can follow the dev branch)

## 2023/06/04 update to 0.1.6

* add dylora and IA^3 algorithm

## 2023/03/29 Update to 0.1.4

* cp decomposition is default to disable now
* add 4 more layer to train (conv_in/out, time_embedding)

## 2023/03/12 Update to 0.1.0

* Add cp-decomposition implementation for convolution layer
  * Both LoRA(LoCon) and LoHa can use this more parameter-efficient decomposition
* Add sparse bias for extracted LoRA
  * Will add to training in the future (Maybe)
* Change weight initialization method in LoHa
  * Use lower std to avoid loss to go high or NaN when using normal lr (like 0.5 in Dadap)
