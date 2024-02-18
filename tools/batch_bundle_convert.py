import os
import argparse
import warnings
from typing import List
from collections import defaultdict

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


def save_state_dict(state, output_path):
    if output_path.endswith(".safetensors"):
        save_file(state, output_path)
    else:
        save(state, output_path)


def pack_bundle(lora, emb_dict, verbose=False):
    for emb, emb_sd in emb_dict.items():
        for key, value in emb_sd.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    lora[f"bundle_emb.{emb}.{key}.{subkey}"] = subvalue
            elif isinstance(value, torch.Tensor):
                lora[f"bundle_emb.{emb}.{key}"] = value
    if verbose:
        print("The following content has been added to lora")
        for key, value in lora.items():
            if key.startswith("bundle_emb"):
                if isinstance(value, torch.Tensor):
                    print(f" {key}: tensor of shape {value.shape}")
                else:
                    print(f" {key}: {value}")
    return lora


def unpack_bundle(lora, verbose, step="", emb_format=".pt"):
    assert emb_format in [".pt", ".safetensors"]
    if step != "":
        step = "-" + str(step)
    emb_dict = {}
    bundle_keys = []
    for lora_key, value in lora.items():
        if lora_key.startswith("bundle_emb"):
            bundle_keys.append(lora_key)
            _, emb, *rest = lora_key.split(".")
            emb = emb + step
            if emb not in emb_dict:
                emb_dict[emb] = {}
            if len(rest) == 2:
                key, subkey = rest
                if emb_format == ".pt":
                    if key not in emb_dict[emb]:
                        emb_dict[emb][key] = {}
                    emb_dict[emb][key][subkey] = value
                else:
                    emb_dict[emb][subkey] = value
            elif len(rest) == 1:
                key = rest[0]
                emb_dict[emb][key] = value
    for bundle_key in bundle_keys:
        del lora[bundle_key]
    if emb_format == ".pt":
        for emb, emb_sd in emb_dict.items():
            emb_sd["name"] = emb
    if verbose:
        print("The following embeddings have been loaded from bundle")
        print_emb_information(emb_dict)
    return lora, emb_dict


def print_emb_information(emb_dict):
    for emb, emb_sd in emb_dict.items():
        print(emb)
        for key, value in emb_sd.items():
            if isinstance(value, dict):  # Check if the value is another dictionary
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        print(f" {key}.{subkey}: tensor of shape" f" {subvalue.shape}")
                    else:
                        print(f"  {key}.{subkey}: {subvalue}")
            elif isinstance(value, torch.Tensor):
                print(f" {key}: tensor of shape {value.shape}")
            else:
                print(f" {key}: {value}")


def extract_step(file_path):
    filename = os.path.splitext(os.path.basename(file_path))[0]
    step = filename.split("-")[-1].replace("step", "")
    if step.isdigit():
        name = "-".join(filename.split("-")[:-1])
        return name, int(step)
    else:
        return name, ""


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


def get_lora_embs_step_correspondence(lora_files: List[str], emb_files: List[str]):
    """Associate LoRA model files with embedding files based on their step count.

    This function takes in lists of LoRA file paths and embedding file paths,
    extracts their step counts, and associates them based on matching steps.
    If a file's step count cannot be determined, it uses the key 'none'.

    Args:
        lora_files (List[str]): A list of file paths to LoRA model files.
        emb_files (List[str]): A list of file paths to embedding files.

    Returns:
        Dict[str, Dict[str, Union[str, List[str]]]]: A dictionary where keys are
        step counts (or 'none') and values are dictionaries containing 'lora'
        (path to the LoRA model) and 'embs' (a list of paths to associated
        embedding files).
    """
    lora_embs = defaultdict(lambda: {"lora": None, "embs": []})
    for network_path in lora_files:
        _, step = extract_step(network_path)
        if step in lora_embs:
            raise ValueError(
                "Find two Lora files with the same" f" step count {step}, abort"
            )
        lora_embs[step]["lora"] = network_path
    for emb_path in emb_files:
        _, step = extract_step(emb_path)
        if step in lora_embs:
            lora_embs[step]["embs"].append(emb_path)
        else:
            print(f"Warning: no corresponding lora found for {emb_path}")
    return lora_embs


def convert_lora_name(network_path, dst_dir, to_bundle):
    name, step = extract_step(network_path)
    if step != "":
        step = "-" + str(step)
    if to_bundle:
        name = name + "-bundle"
    elif name.endswith("-bundle"):
        name = name[:-7]
    lora_save_path = os.path.join(
        dst_dir, name + step + os.path.splitext(network_path)[1]
    )
    return lora_save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool for packing and unpacking LoRA and embeddings."
    )
    parser.add_argument(
        "--network_path",
        nargs="+",
        type=str,
        default=[],
        help="Paths to LoRAs or folders containing LoRA models.",
    )
    parser.add_argument(
        "--emb_path",
        nargs="+",
        type=str,
        default=[],
        help="Paths to embedding files or folders containing embedding files.",
    )
    parser.add_argument(
        "--dst_dir",
        default=None,
        type=str,
        help="Destination directory for output files.",
    )
    parser.add_argument(
        "--from_bundle", action="store_true", help="Unpack from bundle."
    )
    parser.add_argument("--to_bundle", action="store_true", help="Pack to bundle.")
    parser.add_argument(
        "--network_ext",
        nargs="+",
        type=str,
        default=[".safetensors"],
        help="Extensions for LoRA files.",
    )
    parser.add_argument(
        "--emb_ext",
        nargs="+",
        type=str,
        default=[".pt"],
        help="Extensions for embedding files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for files in directories.",
    )
    parser.add_argument(
        "--pack_all_embeddings",
        action="store_true",
        help=(
            "Pack all embeddings to all LoRA files"
            " instead of using step correspondence."
        ),
    )
    parser.add_argument("--verbose", default=1, type=int, help="Verbosity level.")

    # Deprecated
    parser.add_argument(
        "--lora_path",
        nargs="*",
        type=str,
        default=None,
        help="Deprecated. Please use --network_path instead.",
    )
    parser.add_argument(
        "--lora_ext",
        nargs="*",
        default=None,
        type=str,
        help="Deprecated. Please use --network_ext instead.",
    )

    args = parser.parse_args()

    # Deprecation warnings
    if args.lora_path is not None:
        warnings.warn(
            "The --lora_path argument is deprecated and will be removed in the future. "
            "Please use --network_path instead.",
            DeprecationWarning,
        )
        args.network_path = args.lora_path
    if args.lora_ext is not None:
        warnings.warn(
            "The --lora_ext argument is deprecated and will be removed in the future. "
            "Please use --network_ext instead.",
            DeprecationWarning,
        )
        args.network_ext = args.lora_ext

    network_paths = gather_files_from_list(
        args.network_path, args.network_ext, args.recursive
    )

    if args.from_bundle:
        dst_dir = "bundles_unpack" if args.dst_dir is None else args.dst_dir
        os.makedirs(dst_dir, exist_ok=True)
        for network_path in network_paths:
            if args.verbose >= 1:
                print(f"Unpacking {network_path}")
            lora = load_state_dict(network_path)
            _, step = extract_step(network_path)
            lora, emb_dict = unpack_bundle(
                lora, args.verbose >= 2, step=step, emb_format=args.emb_ext[0]
            )
            lora_save_path = convert_lora_name(network_path, dst_dir, to_bundle=False)
            save_state_dict(lora, lora_save_path)
            for emb, emb_sd in emb_dict.items():
                emb_save_path = os.path.join(dst_dir, emb + args.emb_ext[0])
                save_state_dict(emb_sd, emb_save_path)
    elif args.to_bundle:
        if args.emb_path == []:
            args.emb_path = args.network_path
        emb_paths = gather_files_from_list(args.emb_path, args.emb_ext, args.recursive)
        dst_dir = "bundles" if args.dst_dir is None else args.dst_dir
        os.makedirs(dst_dir, exist_ok=True)
        lora_embs_dict = {}
        if args.pack_all_embeddings:
            for i, network_path in enumerate(network_paths):
                lora_embs_dict[i] = {"lora": network_path, "embs": emb_paths}
        else:
            lora_embs_dict = get_lora_embs_step_correspondence(network_paths, emb_paths)
        for _, lora_embs_pair in lora_embs_dict.items():
            network_path = lora_embs_pair["lora"]
            if args.verbose >= 1:
                print(f"Packing {network_path}")
            lora = load_state_dict(network_path)
            emb_dict = {}
            for emb_path in lora_embs_pair["embs"]:
                name, _ = extract_step(emb_path)
                emb_dict[name] = load_state_dict(emb_path)
            bundle = pack_bundle(lora, emb_dict, args.verbose >= 2)
            lora_save_path = convert_lora_name(network_path, dst_dir, to_bundle=True)
            save_state_dict(bundle, lora_save_path)
