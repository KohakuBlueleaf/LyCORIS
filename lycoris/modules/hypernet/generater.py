from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms.functional import resize

from timm import create_model
from einops import rearrange

from lycoris.modules.attention import TransformerBlock


def _get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""
    # TODO: make it with torch instead of numpy

    def get_position_angle_vec(position):
        # this part calculate the position In brackets
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    # [:, 0::2] are all even subscripts, is dim_2i
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class WeightDecoder(nn.Module):
    def __init__(
        self, weight_dim: int = 150, weight_num: int = 54, decoder_blocks: int = 4
    ):
        super(WeightDecoder, self).__init__()
        self.weight_num = weight_num
        self.weight_dim = weight_dim

        self.register_buffer(
            "block_pos_emb", _get_sinusoid_encoding_table(weight_num * 2, weight_dim)
        )

        # calc heads for mem-eff or flash_attn
        heads = 1
        while weight_dim % heads == 0 and weight_dim // heads > 64:
            heads *= 2

        self.pos_emb_proj = nn.Linear(weight_dim, weight_dim, bias=False)
        self.decoder_model = nn.ModuleList(
            TransformerBlock(
                weight_dim,
                heads,
                weight_dim // heads,
                context_dim=weight_dim,
                gated_ff=False,
            )
            for _ in range(decoder_blocks)
        )
        # self.delta_proj = nn.Linear(weight_dim, weight_dim, bias=False)
        self.delta_proj = nn.Sequential(
            nn.LayerNorm(weight_dim), nn.Linear(weight_dim, weight_dim, bias=False)
        )
        self.init_weights()

    def init_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        torch.nn.init.constant_(self.delta_proj[1].weight, 0)
        # advice from Nataniel Ruiz, looks like 1e-3 is small enough
        # torch.nn.init.normal_(self.delta_proj[1].weight, std=1e-3)

    def forward(self, weight, features):
        pos_emb = self.pos_emb_proj(
            self.block_pos_emb[:, : weight.size(1)].clone().detach()
        )
        h = weight + pos_emb
        for decoder in self.decoder_model:
            h = decoder(h, context=features)
        weight = weight + self.delta_proj(h)
        return weight


class ImgWeightGenerator(nn.Module):
    def __init__(
        self,
        encoder_model_name: str = "vit_base_patch16_224",
        train_encoder: bool = False,
        reference_size: Tuple[int] = (224, 224),
        weight_dim: int = 150,  # 100+50 in paper
        weight_num: int = 54,  # 21*2 in SD UNet + 12 in CLIP-L TE
        decoder_blocks: int = 4,
        sample_iters: int = 1,
    ):
        super(ImgWeightGenerator, self).__init__()
        self.ref_size = reference_size
        self.weight_num = weight_num
        self.weight_dim = weight_dim
        self.sample_iters = sample_iters
        self.train_encoder = train_encoder

        self.register_buffer(
            "block_pos_emb", _get_sinusoid_encoding_table(weight_num * 2, weight_dim)
        )

        self.encoder_model: nn.Module = create_model(
            encoder_model_name, pretrained=True
        )
        for p in self.encoder_model.parameters():
            p.requires_grad_(train_encoder)

        test_input = torch.randn(1, 3, *reference_size)
        test_output = self.encoder_model.forward_features(test_input)
        if isinstance(test_output, list):
            test_output = test_output[-1]
        if len(test_output.shape) == 4:
            # B, C, H, W -> B, L, C
            test_output = test_output.view(1, test_output.size(1), -1).transpose(1, 2)

        self.feature_proj = nn.Linear(test_output.shape[-1], weight_dim, bias=False)
        self.pos_emb_proj = nn.Linear(weight_dim, weight_dim, bias=False)
        self.decoder_model = WeightDecoder(weight_dim, weight_num, decoder_blocks)

    def forward(self, ref_img, iters=None, weight=None, ensure_grad=0):
        ref_img = resize(ref_img, self.ref_size, antialias=True)
        if not self.train_encoder:
            with torch.no_grad():
                features = self.encoder_model.forward_features(ref_img)
        else:
            features = self.encoder_model.forward_features(ref_img + ensure_grad)
        if isinstance(features, list):
            features = features[-1]
        if len(features.shape) == 4:
            # B, C, H, W -> B, L, C
            features = features.view(features.size(0), features.size(1), -1).transpose(
                1, 2
            )
        features = self.feature_proj(features + ensure_grad)

        if weight is None:
            weight = torch.zeros(
                ref_img.size(0), self.weight_num, self.weight_dim, device=ref_img.device
            )

        for iter in range(iters or self.sample_iters):
            weight = self.decoder_model(weight, features)
        return weight


class TextWeightGenerator(nn.Module):
    def __init__(
        self,
        train_encoder: bool = False,
        reference_size: Tuple[int] = (224, 224),
        weight_dim: int = 150,  # 100+50 in paper
        weight_num: int = 54,  # 21*2 in SD UNet + 12 in CLIP-L TE
        decoder_blocks: int = 4,
        sample_iters: int = 1,
    ):
        from .text_encoder import FrozenOpenCLIPEmbedder

        super(TextWeightGenerator, self).__init__()
        self.ref_size = reference_size
        self.weight_num = weight_num
        self.weight_dim = weight_dim
        self.sample_iters = sample_iters
        self.train_encoder = train_encoder

        self.register_buffer(
            "block_pos_emb", _get_sinusoid_encoding_table(weight_num * 2, weight_dim)
        )

        self.encoder_model: nn.Module = FrozenOpenCLIPEmbedder()
        for p in self.encoder_model.parameters():
            p.requires_grad_(train_encoder)

        test_input = ["test"]
        test_output = self.encoder_model(test_input)
        if isinstance(test_output, list):
            test_output = test_output[-1]
        if len(test_output.shape) == 4:
            # B, C, H, W -> B, L, C
            test_output = test_output.view(1, test_output.size(1), -1).transpose(1, 2)

        self.feature_proj = nn.Linear(test_output.shape[-1], weight_dim, bias=False)
        self.pos_emb_proj = nn.Linear(weight_dim, weight_dim, bias=False)
        self.decoder_model = WeightDecoder(weight_dim, weight_num, decoder_blocks)

    def forward(self, caption, iters=None, weight=None, ensure_grad=0):
        if not self.train_encoder:
            with torch.no_grad():
                features = self.encoder_model(caption)
        else:
            features = self.encoder_model(caption)
        if isinstance(features, list):
            features = features[-1]
        if len(features.shape) == 4:
            # B, C, H, W -> B, L, C
            features = features.view(features.size(0), features.size(1), -1).transpose(
                1, 2
            )
        features = self.feature_proj(features + ensure_grad)

        if weight is None:
            weight = torch.zeros(
                features.size(0),
                self.weight_num,
                self.weight_dim,
                device=features.device,
            )

        for iter in range(iters or self.sample_iters):
            weight = self.decoder_model(weight, features)
        return weight
