# Change Log

## 2023/09/xx update to 1.9.0
* Add norm modules (for training LayerNorm and GroupNorm, which should be good for style)
* Add full modules (So you can "native finetune" with lycoris now, should be convinient to try different weight)
* Add custom preset system
* Merge script support norm and full modules
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