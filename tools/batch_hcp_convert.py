# TODO: Deal with base type in from_webui, better choice for base model

import os
import sys
import math
import argparse
from typing import List, Dict
from collections import defaultdict

import torch
from safetensors.torch import load_file
from hcpdiff.ckpt_manager import auto_manager


class LoraConverter(object):
    com_name_unet = [
        "down_blocks",
        "up_blocks",
        "mid_block",
        "transformer_blocks",
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "proj_in",
        "proj_out",
        "input_blocks",
        "middle_block",
        "output_blocks",
    ]
    com_name_te = ["self_attn", "q_proj", "v_proj", "k_proj", "out_proj", "text_model"]
    prefix_unet = "lora_unet_"
    prefix_te = "lora_te_"
    prefix_te_xl_clip_B = "lora_te1_"
    prefix_te_xl_clip_bigG = "lora_te2_"

    lora_w_map = {"lora_down.weight": "W_down", "lora_up.weight": "W_up"}

    def __init__(self, save_fp16=False):
        self.com_name_unet_tmp = [x.replace("_", "%") for x in self.com_name_unet]
        self.com_name_te_tmp = [x.replace("_", "%") for x in self.com_name_te]
        self.save_fp16 = save_fp16

    def convert_from_webui(
        self, state, network_type="lora", auto_scale_alpha=False, sdxl=False
    ):
        assert network_type in ["lora", "plugin"]
        if not sdxl:
            sd_unet = self.convert_from_webui_(
                state,
                network_type=network_type,
                prefix=self.prefix_unet,
                com_name=self.com_name_unet,
                com_name_tmp=self.com_name_unet_tmp,
            )
            sd_TE = self.convert_from_webui_(
                state,
                network_type=network_type,
                prefix=self.prefix_te,
                com_name=self.com_name_te,
                com_name_tmp=self.com_name_te_tmp,
            )
        else:
            sd_unet = self.convert_from_webui_xl_unet_(
                state,
                network_type=network_type,
                prefix=self.prefix_unet,
                com_name=self.com_name_unet,
                com_name_tmp=self.com_name_unet_tmp,
            )
            sd_TE = self.convert_from_webui_xl_te_(
                state,
                network_type=network_type,
                prefix=self.prefix_te_xl_clip_B,
                com_name=self.com_name_te,
                com_name_tmp=self.com_name_te_tmp,
            )
            sd_TE2 = self.convert_from_webui_xl_te_(
                state,
                network_type=network_type,
                prefix=self.prefix_te_xl_clip_bigG,
                com_name=self.com_name_te,
                com_name_tmp=self.com_name_te_tmp,
            )
            sd_TE.update(sd_TE2)
        if auto_scale_alpha and network_type == "lora":
            sd_unet = self.alpha_scale_from_webui(sd_unet)
            sd_te = self.alpha_scale_from_webui(sd_te)
        return {network_type: sd_unet}, {network_type: sd_te}

    def convert_to_webui(
        self, sd_unet, sd_TE, network_type="lora", auto_scale_alpha=False, sdxl=False
    ):
        assert network_type in ["lora", "plugin"]
        sd_unet = self.convert_to_webui_(
            sd_unet, network_type=network_type, prefix=self.prefix_unet
        )
        if sdxl:
            sd_te = self.convert_to_webui_xl_(
                sd_te, network_type=network_type, prefix=self.prefix_te
            )
        else:
            sd_te = self.convert_to_webui_(
                sd_te, network_type=network_type, prefix=self.prefix_te
            )
        sd_unet.update(sd_TE)
        if auto_scale_alpha and network_type == "lora":
            sd_unet = self.alpha_scale_to_webui(sd_unet)
        return sd_unet

    def convert_from_webui_(self, state, network_type, prefix, com_name, com_name_tmp):
        state = {k: v for k, v in state.items() if k.startswith(prefix)}
        prefix_len = len(prefix)
        sd_covert = {}
        for k, v in state.items():
            model_k, lora_k = k[prefix_len:].split(".", 1)
            model_k = (
                self.replace_all(model_k, com_name, com_name_tmp)
                .replace("_", ".")
                .replace("%", "_")
            )
            if self.save_fp16:
                v = v.half()
            if lora_k == "alpha" or network_type == "plugin":
                sd_covert[f"{model_k}.___.{lora_k}"] = v
            else:
                # This converts to the version after commit 9fdce2d
                sd_covert[f"{model_k}.___.layer.{self.lora_w_map[lora_k]}"] = v
        return sd_covert

    def convert_to_webui_(self, state, network_type, prefix):
        sd_covert = {}
        for k, v in state.items():
            separator = ".___."
            if network_type == "plugin" or "alpha" in k or "scale" in k:
                model_k, lora_k = k.split(separator, 1)
            # LoRA version after commit 9fdce2d
            elif k.endswith("W_down"):
                model_k, _ = k.split(separator, 1)
                lora_k = "lora_down.weight"
            elif k.endswith("W_up"):
                model_k, _ = k.split(separator, 1)
                lora_k = "lora_up.weight"
            # LoRA version before commit 9fdce2d
            else:
                separator = ".___.layer."
                model_k, lora_k = k.split(separator, 1)
            if self.save_fp16:
                v = v.half()
            sd_covert[f"{prefix}{model_k.replace('.', '_')}.{lora_k}"] = v
        return sd_covert

    def convert_to_webui_xl_(self, state, network_type, prefix):
        sd_convert = {}
        for k, v in state.items():
            separator = ".___."
            if network_type == "plugin" or "alpha" in k or "scale" in k:
                model_k, lora_k = k.split(separator, 1)
            # LoRA version after commit 9fdce2d
            elif k.endswith("W_down"):
                model_k, _ = k.split(separator, 1)
                lora_k = "lora_down.weight"
            elif k.endswith("W_up"):
                model_k, _ = k.split(separator, 1)
                lora_k = "lora_up.weight"
            # LoRA version before commit 9fdce2d
            else:
                separator = ".___.layer."
                model_k, lora_k = k.split(separator, 1)
            model_k, lora_k = k.split(separator, 1)
            new_k = f"{prefix}{model_k.replace('.', '_')}.{lora_k}"
            if "clip" in new_k:
                new_k = (
                    new_k.replace("_clip_B", "1")
                    if "clip_B" in new_k
                    else new_k.replace("_clip_bigG", "2")
                )
            if self.save_fp16:
                v = v.half()
            sd_convert[new_k] = v
        return sd_convert

    def convert_from_webui_xl_te_(
        self, state, network_type, prefix, com_name, com_name_tmp
    ):
        state = {k: v for k, v in state.items() if k.startswith(prefix)}
        sd_covert = {}
        prefix_len = len(prefix)

        for k, v in state.items():
            model_k, lora_k = k[prefix_len:].split(".", 1)
            model_k = (
                self.replace_all(model_k, com_name, com_name_tmp)
                .replace("_", ".")
                .replace("%", "_")
            )
            if prefix == "lora_te1_":
                model_k = f"clip_B.{model_k}"
            else:
                model_k = f"clip_bigG.{model_k}"

            if self.save_fp16:
                v = v.half()

            if lora_k == "alpha" or network_type == "plugin":
                sd_covert[f"{model_k}.___.{lora_k}"] = v
            else:
                # This converts to the version after commit 9fdce2d
                sd_covert[f"{model_k}.___.layer.{self.lora_w_map[lora_k]}"] = v
        return sd_covert

    def convert_from_webui_xl_unet_(
        self, state, network_type, prefix, com_name, com_name_tmp
    ):
        # Down:
        # 4 -> 1, 0  4 = 1 + 3 * 1 + 0
        # 5 -> 1, 1  5 = 1 + 3 * 1 + 1
        # 7 -> 2, 0  7 = 1 + 3 * 2 + 0
        # 8 -> 2, 1  8 = 1 + 3 * 2 + 1

        # Up
        # 0 -> 0, 0  0 = 0 * 3 + 0
        # 1 -> 0, 1  1 = 0 * 3 + 1
        # 2 -> 0, 2  2 = 0 * 3 + 2
        # 3 -> 1, 0  3 = 1 * 3 + 0
        # 4 -> 1, 1  4 = 1 * 3 + 1
        # 5 -> 1, 2  5 = 1 * 3 + 2

        down = {
            "4": [1, 0],
            "5": [1, 1],
            "7": [2, 0],
            "8": [2, 1],
        }
        up = {
            "0": [0, 0],
            "1": [0, 1],
            "2": [0, 2],
            "3": [1, 0],
            "4": [1, 1],
            "5": [1, 2],
        }

        import re

        m = []

        def match(key, regex_text):
            regex = re.compile(regex_text)
            r = re.match(regex, key)
            if not r:
                return False

            m.clear()
            m.extend(r.groups())
            return True

        state = {k: v for k, v in state.items() if k.startswith(prefix)}
        sd_covert = {}
        prefix_len = len(prefix)
        for k, v in state.items():
            model_k, lora_k = k[prefix_len:].split(".", 1)

            model_k = (
                self.replace_all(model_k, com_name, com_name_tmp)
                .replace("_", ".")
                .replace("%", "_")
            )

            if match(model_k, r"input_blocks.(\d+).1.(.+)"):
                new_k = (
                    f"down_blocks.{down[m[0]][0]}.attentions" f".{down[m[0]][1]}.{m[1]}"
                )
            elif match(model_k, r"middle_block.1.(.+)"):
                new_k = f"mid_block.attentions.0.{m[0]}"
                pass
            elif match(model_k, r"output_blocks.(\d+).(\d+).(.+)"):
                new_k = f"up_blocks.{up[m[0]][0]}.attentions" f".{up[m[0]][1]}.{m[2]}"
            else:
                raise NotImplementedError

            if self.save_fp16:
                v = v.half()

            if lora_k == "alpha" or network_type == "plugin":
                sd_covert[f"{new_k}.___.{lora_k}"] = v
            else:
                sd_covert[f"{new_k}.___.layer.{lora_k}"] = v

        return sd_covert

    @staticmethod
    def replace_all(data: str, srcs: List[str], dsts: List[str]):
        for src, dst in zip(srcs, dsts):
            data = data.replace(src, dst)
        return data

    @staticmethod
    def alpha_scale_from_webui(state):
        # Apply to "lora_down" and "lora_up" respectively to prevent overflow
        for k, v in state.items():
            if "lora_up" in k or "W_up" in k:
                state[k] = v * math.sqrt(v.shape[1])
            elif "lora_down" in k or "W_down" in k:
                state[k] = v * math.sqrt(v.shape[0])
        return state

    @staticmethod
    def alpha_scale_to_webui(state):
        for k, v in state.items():
            if "lora_up" in k:
                state[k] = v * math.sqrt(v.shape[1])
            elif "lora_down" in k:
                state[k] = v * math.sqrt(v.shape[0])
        return state


class BaseConverter(object):
    prefix_unet = "lora_unet_"
    prefix_te = "lora_te_"

    def __init__(self, base_model_path, device, save_fp16=False, sdxl=False):
        self.save_fp16 = save_fp16
        self.sdxl = sdxl
        unet_path = os.path.join(
            base_model_path, "unet", "diffusion_pytorch_model.safetensors"
        )
        text_enc_path = os.path.join(
            base_model_path, "text_encoder", "model.safetensors"
        )

        # Load models from safetensors if it exists, if it doesn't pytorch
        if os.path.exists(unet_path):
            self.unet_state_dict = load_file(unet_path, device=device)
        else:
            unet_path = os.path.join(
                base_model_path, "unet", "diffusion_pytorch_model.bin"
            )
            self.unet_state_dict = torch.load(unet_path, map_location=device)

        if os.path.exists(text_enc_path):
            self.text_enc_dict = load_file(text_enc_path, device=device)
        else:
            text_enc_path = os.path.join(
                base_model_path, "text_encoder", "pytorch_model.bin"
            )
            self.text_enc_dict = torch.load(text_enc_path, map_location=device)

    def convert_to_webui(
        self,
        sd_unet,
        sd_te,
    ):
        sd_unet = self.convert_to_webui_(
            sd_unet, base_state=self.unet_state_dict, prefix=self.prefix_unet
        )
        sd_te = self.convert_to_webui_(
            sd_te, base_state=self.text_enc_dict, prefix=self.prefix_te
        )
        sd_unet.update(sd_te)
        return sd_unet

    def convert_to_webui_(self, ft_state, base_state, prefix):
        sd_covert = {}
        for k, v in ft_state.items():
            v_base = base_state[k]
            model_k, lora_k = k.rsplit(".", 1)
            if lora_k == "weight":
                lora_k = "diff"
            else:
                lora_k = "diff_b"
            v_diff = v - v_base
            if self.save_fp16:
                v_diff = v_diff.half()
            new_k = f"{prefix}{model_k.replace('.', '_')}.{lora_k}"
            if self.sdxl and "clip" in new_k:
                new_k = (
                    new_k.replace("_clip_B", "1")
                    if "clip_B" in new_k
                    else new_k.replace("_clip_bigG", "2")
                )
            sd_covert[new_k] = v_diff
        return sd_covert


def gather_files_from_list(
    paths: List[str], extensions: List[str], recursive: bool
) -> List[str]:
    """Gather files from given paths based on specific extensions.

    Args:
        paths (List[str]): A list of paths which can be files or directories.
        extensions (List[str]): A list of file extensions to filter by.
        recursive (bool): If True, search for files recursively in directories.

    Returns:
        List[str]: A list of file paths that match the given extensions.
    """
    files = []

    def is_extension_valid(file: str) -> bool:
        return any(file.endswith(ext) for ext in extensions)

    def add_files_from_directory(directory: str):
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                if is_extension_valid(filepath):
                    files.append(filepath)
            if not recursive:
                break

    for path in paths:
        if os.path.isfile(path) and is_extension_valid(path):
            files.append(path)
        elif os.path.isdir(path):
            add_files_from_directory(path)

    return files


def get_unet_te_pairs(files: List[str]) -> Dict[str, Dict[str, str]]:
    """Get unet and text encoder pairs from a list of files.

    Args:
        files (List[str]): A list of candidate file paths.

    Returns:
        Dict[str, Dict[str, str]]:
            A dictionary where keys are file names and values are dictionaries
            containing paths to unet and text encoder files.

    Raises:
        ValueError:
            If muliple unet or text encoder files are found with the same name.
    """
    file_pairs = defaultdict(lambda: {"TE": None, "unet": None})
    for file_path in files:
        filename = os.path.basename(file_path)
        parts = os.path.splitext(filename)[0].split("-")
        if len(parts) > 1:
            prefix, name = parts[0], "-".join(parts[1:])
            if "text_encoder" in prefix:
                if file_pairs[name]["TE"] is not None:
                    raise ValueError(f"File name {name} for text encoder is repeated. ")
                file_pairs[name]["TE"] = file_path
            elif "unet" in prefix:
                if file_pairs[name]["unet"] is not None:
                    raise ValueError(f"File name {name} for unet is repeated. ")
                file_pairs[name]["unet"] = file_path
    return file_pairs


def save_and_print_path(sd, path):
    try:
        # Old HCP
        ckpt_manager = auto_manager(path)()
    except TypeError:
        ckpt_manager = auto_manager(path)
    os.makedirs(args.dump_path, exist_ok=True)
    ckpt_manager._save_ckpt(sd, save_path=path)
    print("Saved to:", path)


def get_network_types(sd_unet, sd_te):
    network_types = []
    for network_type in ["lora", "plugin", "base"]:
        if network_type in sd_unet.keys() and network_type in sd_te.keys():
            network_types.append(network_type)
    return network_types


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LoRA models.")
    parser.add_argument(
        "--lora_path",
        nargs="+",
        type=str,
        default=[],
        required=True,
        help=(
            "Paths to LoRAs or folders containing LoRA models. "
            "Both unet and text encoder paths should be provided here in case of "
            "conversion to webui format."
        ),
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="Path to base model path. Used for full model conversion.",
    )
    parser.add_argument(
        "--dump_path",
        required=True,
        type=str,
        help="Path to save the converted state dict.",
    )
    parser.add_argument(
        "--from_webui", action="store_true", help="Convert from webui format."
    )
    # TODO: implement convert from webui to base
    parser.add_argument(
        "--save_network_type",
        type=str,
        required="--from_webui" in sys.argv,
        choices=["lora", "plugin", "base"],
        help="Specify the network type for conversion.",
    )
    parser.add_argument(
        "--to_webui", action="store_true", help="Convert to webui format."
    )
    parser.add_argument(
        "--output_prefix", default="", type=str, help="Prefix for output filenames."
    )
    parser.add_argument(
        "--lora_ext",
        default=[".safetensors"],
        type=str,
        nargs="+",
        help="Extensions for LoRA files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for files in directories.",
    )
    parser.add_argument(
        "--auto_scale_alpha", action="store_true", help="Automatically scale alpha."
    )
    parser.add_argument(
        "--device", help="Which device to use for conversion", default="cpu", type=str
    )
    parser.add_argument("--save_fp16", action="store_true", help="Save in FP16 format.")
    parser.add_argument("--sdxl", action="store_true", help="Enable SDXL conversion.")
    args = parser.parse_args()

    lora_converter = LoraConverter(save_fp16=args.save_fp16)
    base_converter = None

    lora_files = gather_files_from_list(args.lora_path, args.lora_ext, args.recursive)

    if args.from_webui:
        for file_path in lora_files:
            try:
                # Old HCP
                ckpt_manager = auto_manager(file_path)()
            except TypeError:
                ckpt_manager = auto_manager(file_path)
            print(f"Converting {file_path}")
            state = ckpt_manager.load_ckpt(file_path, map_location=args.device)
            if args.save_network_type == "base":
                raise NotImplementedError(
                    "Conversion from webui to base is not yet supported."
                )
            sd_unet, sd_te = lora_converter.convert_from_webui(
                state,
                network_type=args.save_network_type,
                auto_scale_alpha=args.auto_scale_alpha,
                sdxl=args.sdxl,
            )
            filename = os.path.basename(file_path)
            te_path = os.path.join(args.dump_path, "text_encoder-" + filename)
            unet_path = os.path.join(args.dump_path, "unet-" + filename)
            save_and_print_path(sd_te, te_path)
            save_and_print_path(sd_unet, unet_path)

    elif args.to_webui:
        file_pairs = get_unet_te_pairs(lora_files)

        for name, file_paths in file_pairs.items():
            if file_paths["TE"] and file_paths["unet"]:
                # Assume here that unet and TE have the same extension
                try:
                    # Old HCP
                    ckpt_manager = auto_manager(file_paths["TE"])()
                except TypeError:
                    ckpt_manager = auto_manager(file_paths["TE"])
                sd_unet = ckpt_manager.load_ckpt(
                    file_paths["unet"], map_location=args.device
                )
                sd_te = ckpt_manager.load_ckpt(
                    file_paths["TE"], map_location=args.device
                )
                network_types = get_network_types(sd_unet, sd_te)
                for network_type in network_types:
                    print(
                        f'Converting pair: {file_paths["TE"]} and {file_paths["unet"]}'
                        f' with key "{network_type}"'
                    )
                    if network_type == "base":
                        if base_converter is None:
                            base_converter = BaseConverter(
                                args.base_path,
                                device=args.device,
                                save_fp16=args.save_fp16,
                                sdxl=args.sdxl,
                            )
                        state = base_converter.convert_to_webui(
                            sd_unet[network_type],
                            sd_te[network_type],
                        )
                    else:
                        state = lora_converter.convert_to_webui(
                            sd_unet[network_type],
                            sd_te[network_type],
                            network_type=network_type,
                            auto_scale_alpha=args.auto_scale_alpha,
                            sdxl=args.sdxl,
                        )

                    if len(network_types) > 1:
                        suffix = f"-{network_type}"
                    else:
                        suffix = ""

                    output_path = os.path.join(
                        args.dump_path,
                        f"{args.output_prefix}-{name}{suffix}.safetensors",
                    )
                    save_and_print_path(state, output_path)
