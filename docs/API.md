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

* `get_module`: determine the algorithm and extract corresponding weights from state dict.
* `make_module`: based on given algorithm and weights to construct modules.

## Functional

For each modules, we have 3 basic methods:

* `weight_gen`: Generate weights for corresponding algorithm
* `weight_diff`: calculate $\Delta W$
* `bypass_forward_diff`: calculate $\Delta W X$

There are some other utilities:

* `factorization`: $fact(p, factor) = (m, n)$
  * where $m \times n = p$, $m < n$, $m<=factor$ and $m, n \in \mathbb{N}$
  * This method have been used in LoKr and Diag-OFT.
* `power2factorization`: $p2fact(p, factor) = (m, n)$
  * where $m \times n = p$, $m < n$, $m<=factor$, $m=2k$, $n=2^p$ and $m, n, p, k \in \mathbb{N}$
  * This method have been used in BOFT.
* `tucker_weight` and `tucker_weight_from_conv`: Reconstruct tucker decomposed weight from tensors or conv modules.

### Usage

For all the functional API, you can directly use any kind of them with following example:

```python
from lycoris.functional import xxx
weights = xxx.weight_gen(org_weight)

def forward_with_diff_weight(x, org_weight, weights):
    return org_forward(x, org_weight + xxx.weight_diff(*weights))

def forward_with_diff_activation(x, org_weight, weights):
    org_out = org_forward(x, org_weight)
    return org_out + xxx.bypass_forward_diff(x, org_out, *weights)
```

Although different algorithm will have different extra arguments for weight_diff and bypass_forward_diff, the overall logic is same.

## Others

### wrapper

* `LycorisNetwork`: the wrapper class to patch any pytorch modules to apply LyCORIS algorithms.
* `create_lycoris`: see example
* `create_lycoris_from_weights`: see example

### kohya

* the specialized wrapper for kohya-ss/sd-scripts.
