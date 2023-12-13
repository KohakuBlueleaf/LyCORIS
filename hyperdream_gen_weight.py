import os, sys

sys.path.insert(0, os.getcwd())
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_model",
        help="The model you want to build hypernetwork on it",
        default="",
        type=str,
    )
    parser.add_argument(
        "hyperdream_model",
        help="the hypernetwork you want to use to generate lora weight",
        default="",
        type=str,
    )
    parser.add_argument(
        "image_path", help="the image for generate weight", default="", type=str
    )
    parser.add_argument(
        "output_name", help="the output model", default="./out.pt", type=str
    )
    parser.add_argument(
        "--is_v2",
        help="Your base model is sd v2 or not",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--device",
        help="Which device you want to use to merge the weight",
        default="cpu",
        type=str,
    )
    parser.add_argument("--dtype", help="dtype to save", default="float", type=str)
    parser.add_argument(
        "--weight", help="weight for the lyco model to merge", default="1.0", type=float
    )
    return parser.parse_args()


ARGS = get_args()


from lycoris.kohya.model_utils import load_file, load_models_from_stable_diffusion_checkpoint
from lycoris.kohya import create_hypernetwork

import torch
import torch.nn as nn

from torchvision.transforms.functional import resize, to_tensor
from PIL import Image

from safetensors.torch import save_file


@torch.no_grad()
def main():
    te, vae, unet = load_models_from_stable_diffusion_checkpoint(
        ARGS.is_v2, ARGS.base_model
    )
    if ARGS.hyperdream_model.rsplit(".", 1)[-1] == "safetensors":
        lyco = load_file(ARGS.hyperdream_model)
    else:
        lyco = torch.load(ARGS.hyperdream_model)

    dtype_str = ARGS.dtype.replace("fp", "float").replace("loat", "").strip()
    dtype = {
        "f": torch.float,
        "f16": torch.float16,
        "f32": torch.float32,
        "f64": torch.float64,
        "bf": torch.bfloat16,
        "bf16": torch.bfloat16,
    }.get(dtype_str, None)
    if dtype is None:
        raise ValueError(f'Cannot Find the dtype "{ARGS.dtype}"')

    hyperdream = (
        create_hypernetwork(
            1.0,
            4,
            1,
            vae,
            te,
            unet,
            down_dim=128,
            up_dim=64,
            delta_iters=4,
            decoder_blocks=4,
        )
        .to(dtype)
        .to(ARGS.device)
    )
    (missing_keys, unexpected_keys) = hyperdream.load_state_dict(lyco, strict=False)

    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    for k in missing_keys:
        if k == "checkpoint" or "block_pos_emb" in k or "encoder_model" in k:
            continue
        assert False, f"Cannot find {k}, only keys in encoder_model can be missing"

    ref_img = Image.open(ARGS.image_path).convert("RGB")
    ref_img = resize(ref_img, hyperdream.weight_generater.ref_size)
    ref_img = to_tensor(ref_img).unsqueeze(0).to(dtype).to(ARGS.device) * 2 - 1

    with torch.autocast(ARGS.device, dtype=dtype):
        hyperdream.update_reference(ref_img, 7)

    state_dict = {}
    for lora in hyperdream.loras:
        down, up = lora.make_lightweight(*lora.data)
        if down.dim() == 3:
            down = down.mean(dim=0)
        if up.dim() == 3:
            up = up.mean(dim=0)

        test_updown = up @ down
        print(torch.norm(test_updown))
        if isinstance(lora.lora_down, nn.Conv2d):
            down = down.unsqueeze(-1).unsqueeze(-1)
            up = up.unsqueeze(-1).unsqueeze(-1)
        state_dict[f"{lora.lora_name}.lora_down.weight"] = down
        state_dict[f"{lora.lora_name}.lora_up.weight"] = up
        state_dict[f"{lora.lora_name}.alpha"] = torch.tensor(1)

    if ARGS.output_name.endswith(".safetensors"):
        save_file(state_dict, ARGS.output_name)
    else:
        torch.save(state_dict, ARGS.output_name)


if __name__ == "__main__":
    main()
