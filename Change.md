# Change Log

## 2024/06/xx update to 3.0.0 - Brand New Functional API, Parametrize API and Module API

### The reasons of 3.0.0

We reconstruct the whole library with new Class definition and brand new Functional API system.

We also removed lot of redundant/unused modules.

Since the whole library are changed significantly. We decide to call it 3.0.0 as a new major version.

### Major Changes

* New Module API
* Add Parametrize API
* Add Functional API
  * LoCon/LoHa/LoKr/Diag-OFT/BOFT only.
* Remove optional deps from install_requires
* Remove lot of redundant/deprecated modules
* Better testing
* HunYuan DiT Support

### Full change log

#### New Features

* LyCORIS now have consistent API for different algorithm like `bypass_forward_diff` or `get_diff_weight` method. Developers of other project can utilize these API to do more tricks or integrate LyCORIS into their framework more easily.
* LyCORIS now have parametrize API which utilize `torch.nn.utils.parametrize.register_parametrization` to directly patch individual parameters. Which can be useful for MHA layer or other tricky modules.
  * Currently only support 2~5D tensors. And LyCORIS will pretend these weights are weight of Linear/Conv1,2,3D then send it into LyCORIS modules
  * More native implementation or more detailed control will be added in the future.
* LyCORIS now have functional API. Developers who prefer functional more than Module things can utilize this feature.
  * Functional API also allow developers who don't want to introduce new dependencies. Just copy-paste the source code and utilizing it. (with Apache-2 License, directly copy-paste is totally allowed)
* Add support for Conv1d and Conv3d module on LoCon/LoHa/LoKr/Full/OFT/BOFT/GLoRA (not All algo in LyCORIS support them, you may receive error when apply unsopported algo), support inherited module (for example: `LoRACompatibleConv` or `LoRACompatibleLinear` from [`huggingface/diffusers`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/lora.py))
* HunYuan DiT support.

#### Improvements, Fixes, Slight Changes

* Drop dependencies related to kohya-ss/sd-scripts:
  * We now take kohya-ss/sd-scripts as optional dependency
  * Which means `transformers`, `diffusers` and anything related to kohya are all optional deps now.
* The definition of dropout and rank_dropout in each algorithm are changed. Since some concept of original rank_dropout in the lora of kohya-ss/sd-script is hard to applied to other algorithm. We can only design the dropout for each module seperatedly.
* `apply_max_norm` issue are all fixed.
* DyLoRA, (IA)^3, GLoRA are all rewritten and support Linear/Conv1,2,3d.
* (IA)^3, GLoRA, Diag-OFT, BOFT are supported in `create_lycoris_from_weights`
  * `lycoris.kohya.create_network_from_weights` also support them as well.
* `create_lycoris_from_weights` and `create_network_from_weights` now have correct logging infos.
* `get_module` and `make_module` are moved into modules' API.

#### Deprecation

* HCP modules are dropped. We will wait until HCP have better wrapper API.
* HyperNetwork-related modules like `hypernet/`, `attention.py`, `lilora.py` are removed.
* Uncompleted GLoKr are removed.
* code copied from kohya-ss/sd-scripts are removed. The original sd-scripts repo is now an optional dependency.

---

## 2024/03/15 update to 2.2.0 - QLyCORIS and DoRA

#### New Algo

* DoRA
  * Ref: [DoRA: Weight-Decomposed Low-Rank Adaptation](https://github.com/KohakuBlueleaf/LyCORIS)
* Weight decompose for LoHa and LoKr. (A.K.A DoHa/DoKr)
  * DoRA/DoHa/DoKr will require smaller Learning rate!

#### New Features

* Support "bypass" (a.k.a. adapter) mode for LoHa/LoKr/OFT/BOFT
  * LoHa will require 2xFLOPs since we rebuild full diff weight and then do one more forward.
  * LoKr, OFT, BOFT should be more efficient than LoHa in bypass mode.
* Support [bnb 8bit/4bit Linear layer](https://github.com/TimDettmers/bitsandbytes) (a.k.a. QLyCORIS) with LoHa/LoKr/OFT/BOFT.
  * This will force module to enable bypass mode.

#### Fixes, slight changes

* Refine some details about code quality. Based on the report from GitRoll. (Thx you gitroll!)
* Remove redundant calculation in BOFT
* rank_dropout has been removed from OFT/BOFT temporarily untill we ensure how to apply it.
* Fix bugs in lokr when `lokr_w1_a` not exist.
* Fix bugs in conversion scritps.

## 2024/02/18 update to 2.1.0

#### New Algo

* [BOFT (Butterfly OFT)](https://arxiv.org/abs/2311.06243)

#### Improvements

* Faster, better extract script
* support kohya-ss/sd-scripts image gen
* support regex name in kohya-ss/sd-scripts
* support resume on:
  * full
  * loha
  * oft
  * boft
* Add logger into LyCORIS

#### Fixes, slight changes

* Update HCP convert for the case where only UNet or TE is trained.
* Change arg names for conversion scripts.
* Fix wrong TE prefix in merge scripts.
* Fix warnings and confusing logging.

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
