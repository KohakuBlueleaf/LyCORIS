# See: https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/13568

import os, sys

sys.path.insert(0, os.getcwd())
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "lora_model",
        help="The model you want to pack embeddings into it.",
        default="",
        type=str,
    )
    parser.add_argument(
        "embeddings",
        help="the embeddings you want to pack into the lora model.",
        default="",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "output", help="output file name (with path)", default="", type=str, nargs="*"
    )
    parser.add_argument(
        "--safetensors",
        help="use safetensors to save bundled file",
        default=True,
        action="store_true",
    )
    return parser.parse_args()


ARGS = get_args()

import torch
from torch import load, save
from safetensors import safe_open
from safetensors.torch import save_file


def load_state_dict(file_path):
    is_safetensors = file_path.rsplit(".", 1)[-1] == "safetensors"
    if is_safetensors:
        state_dict = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        state_dict = load(file_path)
    return state_dict


def main():
    args = ARGS
    is_safetensors = args.lora_model.rsplit(".", 1)[-1] == "safetensors"

    lora_sd = load_state_dict(args.lora_model)
    embs_sd = {
        os.path.splitext(os.path.basename(x))[0]: load_state_dict(x)
        for x in args.embeddings
    }

    for emb, emb_sd in embs_sd.items():
        for key, value in emb_sd.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    lora_sd[f"bundle_emb.{emb}.{key}.{subkey}"] = subvalue
            elif isinstance(value, torch.Tensor):
                lora_sd[f"bundle_emb.{emb}.{key}"] = value

    for k in lora_sd:
        if k.startswith("bundle_emb"):
            print(k)

    if args.output:
        output_name = args.output
    else:
        output_name = f"{os.path.splitext(args.lora_model)[0]}_bundle"
        if args.safetensors:
            output_name += ".safetensors"
        else:
            output_name += ".pt"

    if is_safetensors:
        save_file(lora_sd, output_name)
    else:
        save(lora_sd, output_name)


if __name__ == "__main__":
    main()
