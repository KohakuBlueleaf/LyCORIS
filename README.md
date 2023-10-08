# LyCORIS - Lora beYond Conventional methods, Other Rank adaptation Implementations for Stable diffusion.

A project that implements different parameter-efficient fine-tuning algorithms for Stable Diffusion.

This project originated from LoCon (see archive branch).

**If you are interested in discussing more details, you can join [our Discord server](https://discord.gg/VtTFKrj9gJ)**

[![Discord!](https://i.imgur.com/A8tOvFS.jpg)](https://discord.gg/VtTFKrj9gJ)

**If you want to check more in-depth experiment results and discussions for LyCORIS, you can check our [paper](https://arxiv.org/abs/2309.14859)**

## Algorithm Overview

LyCORIS currently contains LoRA (LoCon), LoHa, LoKr, (IA)^3, DyLoRA, Native fine-tuning (aka dreambooth).
GLoRA and GLoKr are coming soon.
Please check [List of Implemented Algorithms](docs/Algo-List.md) and [Guidelines](docs/Guidelines.md) for more details.

A simple comparison of some of these methods are provided below (to be taken with a grain of salt)

|                       | Full | LoRA | LoHa | LoKr low factor | LoKr high factor |
| --------------------- | ---- | ---- | ---- | --------------- | ---------------- |
| Fidelity              | ★    | ●    | ▲    | ◉               | ▲                |
| Flexibility $^*$      | ★    | ●    | ◉    | ▲               | ● $^†$           |
| Diversity             | ▲    | ◉    | ★    | ●               | ★                |
| Size                  | ▲    | ●    | ●    | ●               | ★                |
| Training Speed Linear | ★    | ●    | ●    | ★               | ★                |
| Training Speed Conv   | ●    | ★    | ▲    | ●               | ●                |

★ > ◉ > ● > ▲
[> means better and smaller size is better]

$^*$ Flexibility means anything related to generating images not similar to those in the training set, and combination of multiple concepts, whether they are trained together or not  
$^†$ It may become more difficult to switch base model or combine multiple concepts in this situation

## Usage

### Image Generation

#### [a1111/sd-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
After sd-webui 1.5.0, LyCORIS models are officially supported by the built-in LoRA system. You can put them in either `models/Lora` or `models/LyCORIS` and use the default syntax `<lora:filename:multiplier>` to trigger it.

When we add new model types, we will always make sure they can be used with the newest version of sd-webui.

As for sd-webui with version < 1.5.0, please check this [extension](https://github.com/KohakuBlueleaf/a1111-sd-webui-lycoris).

#### Others

As far as we are aware, LyCORIS models are also supported in the following interfaces / online generation services (please help us complete the list!)

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [InvokeAI](https://github.com/invoke-ai/InvokeAI)
- [CivitAI](https://civitai.com/) 
- [Tensor.Art](https://tensor.art/)

However, newer model types may not always be supported. If you encounter this issue, consider requesting the developers of the corresponding interface or website to include support for the new type.

### Training

For the time being LyCORIS is mainly trained with [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) (see a list of compatible graphical interfaces and colabs at the end of the section). Supports for other trainers are coming soon.

A detilaed description of the network arguments is provided [here](docs/Network-Args.md).

#### kohya script

1. Activate sd-scripts' venv with
    ```bash
    source PATH_TO_SDSCRIPTS_VENV/Scripts/activate
    ```
    or
    ```powershell
    PATH_TO_SDSCRIPTS_VENV\Scripts\Activate.ps1 # or .bat     for cmd
    ```

2. Install this package
    * through pip
        ```bash
        pip install lycoris_lora
        ```

    * from source
        ```bash
        git clone https://github.com/KohakuBlueleaf/LyCORIS
        cd LyCORIS
        pip install .
        ````

3. Use this package's kohya module to run kohya's training script to train lycoris module for SD models

    - with command line arguments
        ```bash
        python3 sd-scripts/train_network.py \
          --network_module lycoris.kohya \
          --network_dim "DIM_FOR_LINEAR" --network_alpha "ALPHA_FOR_LINEAR"\
          --network_args "conv_dim=DIM_FOR_CONV" "conv_alpha=ALPHA_FOR_CONV" \
          "dropout=DROPOUT_RATE" "algo=locon" \
        ```

    - with `toml` files
        ```bash
        python train_network.py --config_file XXX.toml
        ```
        For your convenience, some example `toml` files for LyCORIS training are provided in [example/training_configs](examples/training_configs).


* Tips:
  * Use network_dim=0 or conv_dim=0 to disable linear/conv layer
  * LoHa/LoKr/(IA)^3 doesn't support dropout yet.


#### Graphical Interfaces and Colabs

You can also train LyCORIS with the following graphical interfaces
* [bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss)
* [derrian-distro/LoRA_Easy_Training_Scripts](https://github.com/derrian-distro/LoRA_Easy_Training_Scripts)
* [Akegarasu/lora-scripts](https://github.com/Akegarasu/lora-scripts)

and colabs (please help us complete the list!)
* [hollowstrawberry/kohya-colab](https://github.com/hollowstrawberry/kohya-colab)
* [Linaqruf/kohya-trainer](https://github.com/Linaqruf/kohya-trainer)

However, they are not guaranteed to be up-to-date. In particular, newer types may not be supported. Consider requesting the developpers for support or simply use the original kohya script in this case.


## Utilities

### Extract LoCon
You can extract LoCon from a dreambooth model with its base model.
```bash
python3 extract_locon.py <settings> <base_model> <db_model> <output>
```
Use --help to get more info
```
$ python3 extract_locon.py --help
usage: extract_locon.py [-h] [--is_v2] [--device DEVICE] [--mode MODE] [--safetensors] [--linear_dim LINEAR_DIM] [--conv_dim CONV_DIM]
                        [--linear_threshold LINEAR_THRESHOLD] [--conv_threshold CONV_THRESHOLD] [--linear_ratio LINEAR_RATIO] [--conv_ratio CONV_RATIO]
                        [--linear_percentile LINEAR_PERCENTILE] [--conv_percentile CONV_PERCENTILE]
                        base_model db_model output_name
```

### Merge LyCORIS back to model
You can merge your LyCORIS model back to your checkpoint(base model)
```bash
python3 merge.py <settings> <base_model> <lycoris_model> <output>
```
Use --help to get more info
```
$ python3 merge.py --help
usage: merge.py [-h] [--is_v2] [--device DEVICE] [--dtype DTYPE] [--weight WEIGHT] base_model lycoris_model output_name
```



## Change Log
For full log, please see [Change.md](Change.md)

### 2023/09/27 update to 1.9.0
* Add norm modules (for training LayerNorm and GroupNorm, which should be good for style)
* Add full modules (So you can "native fine-tune" with LyCORIS now, should be convenient to try different weight)
* Add preset config system
* Add custom config system
* Merge script support norm and full modules
* Fix errors with optional requirements
* Fix errors with not necessary import
* Fix wrong factorization behaviors


## Todo list
- [ ] Module and Document for using LyCORIS in any other model, Not only SD.
- [x] Proposition3 in [FedPara](https://arxiv.org/abs/2108.06098)
  * also need custom backward to save the vram
- [ ] Low rank + sparse representation
  - [x] For extraction
  - [ ] For training
- [ ] Support more operation, not only linear and conv2d.
- [x] Configure varying ranks or dimensions for specific modules as needed.
- [ ] Automatically selecting an algorithm based on the specific rank requirement.
- [ ] Explore other low-rank representations or parameter-efficient methods to fine-tune either the entire model or specific parts of it.
- [ ] More experiments for different task, not only diffusion models.


## Citation

```bibtex
@misc{LyCORIS,
      title={Navigating Text-To-Image Customization: From LyCORIS Fine-Tuning to Model Evaluation}, 
      author={Shin-Ying Yeh and Yu-Guan Hsieh and Zhidong Gao and Bernard B W Yang and Giyeong Oh and Yanmin Gong},
      year={2023},
      eprint={2309.14859},
      archivePrefix={arXiv}
}
```
