# Tools: Conversion Scripts

This page documents the usage of the two conversion scripts [batch_hcp_convert.py](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/tools/batch_hcp_convert.py) and [batch_bundle_convert.py](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/tools/batch_bundle_convert.py)

1. [Batch HCP Convert](#Batch-HCP-Convert)
1. [Batch Bundle Convert](#Batch-Bundle-Convert)

## Batch HCP Convert

The script [batch_hcp_convert.py](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/tools/batch_hcp_convert.py) performs conversion between hcp and sd-webui format. It applies to LoRA, LyCORIS, and full model (for full model only conversion from hcp to webui is implemented).

#### Example Usage:
```bash!
python batch_hcp_convert.py \
--network_path /path/to/ckpts \
--dst_dir /path/to/stable-diffusion-webui/models/Lora \
--output_prefix my_favorite_anime \
--auto_scale_alpha --to_webui
```

With this, if `unet-loha-5000.safetensors` and `text_encoder-loha-5000.safetensors` in HCP format are found in `/path/to/ckpts`, the file `my_favorite_anime-loha-5000.safetensors` in webui format will be saved to `/path/to/stable-diffusion-webui/models/Lora`.

#### Main Arguments
- `from_webui` and `to_webui`: Whether to perform conversion from webui or to webui format. Note that in HCP format unet and text encoder are saved separately.
- `network_path`: A list of network checkpoints and folders to retrieve checkpoints from. The checkpoints should have extensions as specified in the argument `--network_ext` (defaults to `[".safetensors"]`). 
    - `to-webui`: When performing conversion from HCP to webui format, text encoder and unet checkpoints are matched automatically, assuming that their file names start with `text_encoder-` or `unet-` and have the same remaining component (i.e. the default format output by HCP-diffusion). The conversion is also possible when only unet or text encoder is trained. It throws error when multiple files with the same file names are found (even if they are located in different directories).  
    **Example Usage:** --network_path /path/to/ckpt_folder /path/to/ckpt_folder2/unet-oft-5000.safetensors /path/to/ckpt_folder2/text_encoder-oft-5000.safetensors /path/to/ckpt_folder2/unet-loha-5000.safetensors
    - `from-webui`: For the other way around, for each file `XXX.safetensors` (assuming we are using safetensors format here), two files `text_encoder-XXX.safetensors` and `unet-XXX.safetensors` are saved to `--dst_dir`. The argument `--save_network_type` determines whether the checkpoints should be saved as `lora` or `plugin` in HCP format. For all lycoris models, i.e. LoHa, LoKr, and Diag-OFT, `plugin` should be chosen.  
    **Example Usage:** --network_path /path/to/ckpt_folder /path/to/ckpt_folder2/my_favorite_anime-loha-5000.safetensors 
- `dst_dir`: The destination folder where the output checkpoints are saved.
- `output_prefix`: Prefix for output file names in `--to-webui` conversion.
- `base_path`: Path to the diffusers model that should be considered as base model when performing conversion from HCP full format to lycoris full format. This is necessary for full format conversion because in HCP weights are saved while in LyCORIS only weight differences are saved.  
**Example Usage:** --base_path /path/to/huggingface-cache/hub/models--deepghs--animefull-latest/snapshots/f3f16ff08cf978747fd8b0f1b7c640821308fe21/
- `auto_scale_alpha`: This only affects LoRA conversion and you should in theory always keep it on when converting LoRAs, unless you know what you are doing.
- `sdxl`: Argument for enabling SDXL conversion.

#### Secondary Arguments

- `save_network_type`: The network type to consider. It only affects `--from_webui` conversion.
- `network_ext`: Extension(s) for input network files.  
**Example Usage:** --network_ext .safetensors .pt
- `recursive`: Argument for reading the files recursively from the folders given in `--network_path`.
- `device`: Device used for conversion.
- `save_fp16`: Argument for saving in fp16 format.

## Batch Bundle Convert

The script [batch_bundle_convert.py](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/tools/batch_bundle_convert.py) performs conversion between bundle and non-bundle format (i.e., embeddings saved separately).

#### Example Usage:

```bash!
python batch_bundle_convert.py \
--network_path /path/to/sd-webui-ssd/models/Lora  \
--emb_path /path/to/ckpts \
--dst_dir /path/to/sd-webui-ssd/models/Lora/bundle \
--to_bundle --verbose 2 
```

With this, if `my_favorite_anime-loha-5000.safetensors` is found in `/path/to/sd-webui-ssd/models/Lora`, and `character1-5000.pt` and `character2-5000.pt` are found in `/path/to/ckpts`, the file `my_favorite_anime-loha-bundle-5000.safetensors` in bundle format, containing the corresponding embedding information with trigger words `character1` and `character2`, will be saved to `/path/to/sd-webui-ssd/models/Lora/bundle`.

#### Main Arguments:

- `from_bundle` and `to_bundle`: Whether to perform conversion from bundle or to bundle format. This can also be think of "unzip" and "zip".
- `network_path` and `emb_path`: Lists of files and folders to retrieve network checkpoints and embeddings from. A file is identified as a network checkpoint (resp. embedding) if its extension is specified in `--network_ext` (resp. `--emb_ext`).
    - `to_bundle`: In this case, network checkpoints and embeddings are grouped together using step count, assuming that their names are of the format `{name}-{step_count}.{ext}`. It throws error if multiple network checkpoints have the same step count. The resulting file is called `{network_name}-bundle-{step_count}.{network_ext}`, and for each embedding `{emb_name}-{step_count}.{emb_ext}`, it is saved with the name `{emb_name}` in the bundle file.
    - `from_bundle`: It basically does the opposite of what is explained above. Every single bundle file found in `--network_path` gets separated into one network checkpoint and multiple embedding files, while the step counts are added to the stored embedding names to form the file names. `--emb_path` has no effect here.
- `dst_dir`: The destination folder where the output checkpoints are saved.

#### Secondary Arguments

- `network_ext` and `emb_ext`: Extension(s) for input network and embedding files.
- `recursive`: Argument for reading the files recursively from the folders given in `--network_path` and `--emb_path`.
- `verbose`: Verbosity level. To be chosen from 0, 1, and 2.
- `pack_all_embeddings`: Argument for packing all found embeddings to all found network files instead of using step correspondence.
