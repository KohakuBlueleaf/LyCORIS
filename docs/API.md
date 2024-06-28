# API Reference [WIP]

## Module

### Class: `LycorisBaseModule`:

* classmethod

  * `parametrize`
  * `algo_check`
  * `extract_state_dict`
  * `make_module_from_state_dict`
* property

  * `dtype`
  * `device`
  * `org_weight`
* methods

  * `apply_to`
  * `restore`
  * `merge_to`
  * `get_diff_weight`
  * `get_merged_weight`
  * `apply_max_norm`
  * `bypass_forward_diff`
  * `bypass_forward`
  * `parametrize_forward`
  * `forward`

#### Subclasses

* `LoConModule`
* `LohaModule`
* `LokrModule`
* `DyLoraModule`
* `GLoRAModule`
* `NormModule`
* `FullModule`
* `DiagOFTModule`

### Functions

* `create_lycoris`: see example
* `create_lycoris_from_weights`: see example

## Functional

TODO

## Others

TODO
