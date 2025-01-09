![pypi](https://img.shields.io/pypi/v/lycoris-lora.svg)
![versions](https://img.shields.io/pypi/pyversions/lycoris-lora.svg)
![PyPI - License](https://img.shields.io/pypi/l/lycoris-lora)
![downloads](https://img.shields.io/pypi/dm/lycoris-lora)
![commits](https://img.shields.io/github/commit-activity/m/KohakuBlueleaf/LyCORIS/dev)
![Discord](https://img.shields.io/discord/1082218577395986452)

# LyCORIS - Lora beYond Conventional methods, Other Rank adaptation Implementations for Stable diffusion. (ICLR'24)

![banner image](docs/images/banner2.png)

A project that implements different parameter-efficient fine-tuning algorithms for Stable Diffusion.

This project originated from LoCon (see archive branch).

**If you are interested in discussing more details, you can join [our Discord server](https://discord.gg/VtTFKrj9gJ)**

[![Discord!](https://i.imgur.com/A8tOvFS.jpg)](https://discord.gg/VtTFKrj9gJ)

**If you want to check more in-depth experiment results and discussions for LyCORIS, you can check our [paper](https://openreview.net/forum?id=wfzXa8e783)**

## Algorithm Overview

LyCORIS currently contains LoRA (LoCon), LoHa, LoKr, (IA)^3, DyLoRA, Native fine-tuning (aka dreambooth).
GLoRA and GLoKr are coming soon.
Please check [List of Implemented Algorithms](docs/Algo-List.md) and [Guidelines](docs/Guidelines.md) for more details.

A simple comparison of some of these methods are provided below (to be taken with a grain of salt)

|                       | Full | LoRA | LoHa | LoKr low factor | LoKr high factor $^+$ |
| --------------------- | ---- | ---- | ---- | --------------- | ---------------------- |
| Fidelity              | ★   | ●   | ▲   | ◉              | ▲                     |
| Flexibility $^*$     | ★   | ●   | ◉   | ▲              | ● $^†$              |
| Diversity             | ▲   | ◉   | ★   | ●              | ★                     |
| Size                  | ▲   | ●   | ●   | ●              | ★                     |
| Training Speed Linear | ★   | ●   | ●   | ★              | ★                     |
| Training Speed Conv   | ●   | ★   | ▲   | ●              | ●                     |

★ > ◉ > ● > ▲
[> means better and smaller size is better]

$^+$ Usually we take `factor <= 0.5 * sqrt(dim)` as low factor and `factor >= sqrt(dim` as high factor. For example, factor<=8 for SD1.x/SD2.x/SDXL can be seen as low factor, and, factor>=16 can be seen as high factor. <br>
$^*$ Flexibility means anything related to generating images not similar to those in the training set, and combination of multiple concepts, whether they are trained together or not <br>
$^†$ It may become more difficult to switch base model or combine multiple concepts in this situation <br>

**The actual performance may vary depending on the datasets, tasks, and hyperparameters used. It is recommended to experiment with different settings to achieve optimal results.**

## Usage

### Image Generation

#### [a1111/sd-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

After sd-webui 1.5.0, LyCORIS models are officially supported by the built-in LoRA system. You can put them in either `models/Lora` or `models/LyCORIS` and use the default syntax `<lora:filename:multiplier>` to trigger it.

When we add new model types, we will always make sure they can be used with the newest version of sd-webui.

As for sd-webui with version < 1.5.0 or sd-webui-forge, please check this [extension](https://github.com/KohakuBlueleaf/a1111-sd-webui-lycoris).

#### Others

As far as we are aware, LyCORIS models are also supported in the following interfaces / online generation services (please help us complete the list!)

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [InvokeAI](https://github.com/invoke-ai/InvokeAI)
- [CivitAI](https://civitai.com/)
- [Tensor.Art](https://tensor.art/)

However, newer model types may not always be supported. If you encounter this issue, consider requesting the developers of the corresponding interface or website to include support for the new type.

### Training

There are three different ways to train LyCORIS models.

- With [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) (see a list of compatible graphical interfaces and colabs at the end of the section)
- With [Naifu-Diffusion](https://github.com/Mikubill/naifu)
- With your own script by using LyCORIS as standalone wrappers for **ANY** pytorch modules.

In any case, please install this package in the corresponding virtual environment. You can either install it

- through pip

  ```bash
  pip install lycoris-lora
  ```
- or from source

  ```bash
  git clone https://github.com/KohakuBlueleaf/LyCORIS
  cd LyCORIS
  pip install .
  ```

A detailed description of the network arguments is provided in [docs/Network-Args.md](docs/Network-Args.md).

#### kohya script

You can use this package's kohya module to run kohya's training script to train lycoris module for SD models

- with command line arguments

  ```bash
  accelerate launch train_network.py \
    --network_module lycoris.kohya \
    --network_dim "DIM_FOR_LINEAR" --network_alpha "ALPHA_FOR_LINEAR"\
    --network_args "conv_dim=DIM_FOR_CONV" "conv_alpha=ALPHA_FOR_CONV" \
    "dropout=DROPOUT_RATE" "algo=locon" \
  ```
- with `toml` files

  ```bash
  accelerate launch train_network.py \
    --config_file example_configs/training_configs/kohya/loha_config.toml \
    --dataset_config example_configs/training_configs/kohya/dataset_config.toml
  ```

  For your convenience, some example `toml` files for kohya LyCORIS training are provided in [example/training_configs/kohya](example_configs/training_configs/kohya).

#### HCP-Diffusion

**The support for HCP-Diffusion has been dropped on LyCORIS3.0.0, we will wait until HCP side finish the implementation of new wrapper**

You can use this package's hcp module to run HCP-Diffusion's training script to train lycoris module for SD models

```bash
accelerate launch -m hcpdiff.train_ac_single \
  --cfg example_configs/training_configs/hcp/hcp_diag_oft.yaml
```

For your convenience, some example `yaml` files for HCP LyCORIS training are provided in [example/training_configs/hcp](example_configs/training_configs/hcp).

For the moment being the outputs of HCP-Diffusion are not directly compatible with a1111/sdwebui.
You can perform conversion with [tools/batch_hcp_convert.py](tools/batch_hcp_convert.py).

In the case of pivotal tuning, [tools/batch_bundle_convert.py](tools/batch_bundle_convert.py) can be further used to convert to and from bundle formats.
Check [docs/Conversion-scripts.md](docs/Conversion-scripts.md) for more information.

#### As standalone wrappers

See [standalone_example.py](standalone_example.py) for full example.

Import `create_lycoris` and `LycorisNetwork` from `lycoris` library, put your preset to `LycorisNetwork` and then use `create_lycoris` to create LyCORIS module for your pytorch module.

For example:

```py
from lycoris import create_lycoris, LycorisNetwork

LycorisNetwork.apply_preset(
    {"target_name": [".*attn.*"]}
)
lycoris_net = create_lycoris(
    your_model, 
    1.0, 
    linear_dim=16, 
    linear_alpha=2.0, 
    algo="lokr"
)
lycoris_net.apply_to()

# after apply_to(), your_model() will run with LyCORIS net
lycoris_param = lycoris_net.parameters()
forward_with_lyco = your_model(x)
```

You can check my [HakuPhi](https://github.com/KohakuBlueleaf/HakuPhi) project to see how I utilize LyCORIS to finetune the Phi-1.5 models.

#### Other method

After LyCORIS3.0.0, Parametrize API and Functional API have been added, which provide more different ways on utilizing LyCORIS library.

Check API reference for more informations.
You can also take the [test suites](test/) as a kind of examples.

#### Bitsandbytes support

See [bnb_example.py](bnb_example.py) for example. Basically as same as standalone wrapper.

#### Graphical interfaces and Colabs (via kohya trainer)

You can also train LyCORIS with the following graphical interfaces

* [bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss)
* [derrian-distro/LoRA_Easy_Training_Scripts](https://github.com/derrian-distro/LoRA_Easy_Training_Scripts)
* [Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts)

and colabs (please help us complete the list!)

* [hollowstrawberry/kohya-colab](https://github.com/hollowstrawberry/kohya-colab)
* [Linaqruf/kohya-trainer](https://github.com/Linaqruf/kohya-trainer)

However, they are not guaranteed to be up-to-date. In particular, newer types may not be supported. Consider requesting the developers for support or simply use the original kohya script in this case.

## Utilities

### Extract LoCon

You can extract LoCon from a dreambooth model with its base model.

```bash
python3 extract_locon.py <settings> <base_model> <db_model> <output>
```

Use --help to get more info

```
$ python3 extract_locon.py --help
usage: extract_locon.py [-h] [--is_v2] [--is_sdxl] [--device DEVICE] [--mode MODE] [--safetensors] [--linear_dim LINEAR_DIM]
                        [--conv_dim CONV_DIM] [--linear_threshold LINEAR_THRESHOLD] [--conv_threshold CONV_THRESHOLD]
                        [--linear_ratio LINEAR_RATIO] [--conv_ratio CONV_RATIO] [--linear_quantile LINEAR_QUANTILE]
                        [--conv_quantile CONV_QUANTILE] [--use_sparse_bias] [--sparsity SPARSITY] [--disable_cp]
                        base_model db_model output_name
```

### Merge LyCORIS back to model

You can merge your LyCORIS model back to your checkpoint (base model).

```bash
python3 merge.py <settings> <base_model> <lycoris_model> <output>
```

Use --help to get more info

```
$ python3 merge.py --help
usage: merge.py [-h] [--is_v2] [--is_sdxl] [--device DEVICE] [--dtype DTYPE] [--weight WEIGHT] base_model lycoris_model output_name
```

### Conversion of LoRA, LyCORIS and full models between HCP and sd-webui format

This script allows you to use the LyCORIS models trained with HCP-Diffusion in sd-webui.

```bash
python3 batch_hcp_convert.py \
  --network_path /path/to/ckpts \
  --dst_dir /path/to/stable-diffusion-webui/models/Lora \
  --output_prefix something \
  --auto_scale_alpha --to_webui
```

See [docs/Conversion-scripts.md](docs/Conversion-scripts.md) for more information.

### Conversion from and to bundle format

This script is particularly useful in the case of pivotal tuning.

```bash
python3 batch_bundle_convert.py \
  --network_path /path/to/sd-webui-ssd/models/Lora  \
  --emb_path /path/to/ckpts \
  --dst_dir /path/to/sd-webui-ssd/models/Lora/bundle \
  --to_bundle --verbose 2 
```

See [docs/Conversion-scripts.md](docs/Conversion-scripts.md) for more information.

## Change Log

For full log, please see [Change.md](Change.md)


### 2024/12/09 update to 3.1.1

#### New Features

* use `wd_on_output=True` can enable "correct" weight-decomposition implementation which use the output dimension of weight to calc the norm. The original implementation in LyCORIS calculate things on input dimension due to ambiguos annotation in paper.

#### Improvements

* BOFT now have more efficient implementation which avoid einops.rearrange.
* `.merge_to()` will automatically match the device and dtype now.

#### Bug fixes

* `scale_weight_norm` working correctly now.

## Todo list

- [ ] Automatically selecting an algorithm based on the specific rank requirement.
- [ ] More experiments for different task, not only diffusion models.
  - [X] LoKr and LoHa have been proven to be useful for Large Language Model.
- [ ] Explore other low-rank representations or parameter-efficient methods to fine-tune either the entire model or specific parts of it.
- [ ] Documentation for whole library.

## Citation

```bibtex
@inproceedings{
  yeh2024navigating,
  title={Navigating Text-To-Image Customization: From Ly{CORIS} Fine-Tuning to Model Evaluation},
  author={SHIH-YING YEH and Yu-Guan Hsieh and Zhidong Gao and Bernard B W Yang and Giyeong Oh and Yanmin Gong},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=wfzXa8e783}
}
```
