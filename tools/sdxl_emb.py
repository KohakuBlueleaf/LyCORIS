import os, sys

sys.path.insert(0, os.getcwd())
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_model",
        help="The model you want to use to generate embedding.",
        default="",
        type=str,
    )
    parser.add_argument(
        "emb_content",
        help="The text content of the emb you want to generate.",
        default="",
        type=str,
    )
    parser.add_argument(
        "emb_file_name",
        help="The output file name for the embedding files",
        default="",
        type=str,
    )
    return parser.parse_args()


args = ARGS = get_args()


from library.sdxl_model_util import (
    load_models_from_sdxl_checkpoint,
)

import torch
from safetensors.torch import save_file
from transformers import CLIPTokenizer


@torch.no_grad()
def main():
    clip_l, clip_g, *_ = load_models_from_sdxl_checkpoint(
        None, args.base_model, map_location="cpu"
    )
    l_emb = clip_l.text_model.embeddings
    g_emb = clip_g.text_model.embeddings

    tokenizer_l = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer_g = CLIPTokenizer.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    )

    tokenizer_g.pad_token_id = 0

    token_l = torch.tensor(tokenizer_l.encode(args.emb_content))
    token_g = torch.tensor(tokenizer_g.encode(args.emb_content))

    data = {
        "clip_l": l_emb(token_l)[0],
        "clip_g": g_emb(token_g)[0],
    }

    save_file(data, args.emb_file_name or "emb.safetensors")


if __name__ == "__main__":
    main()
