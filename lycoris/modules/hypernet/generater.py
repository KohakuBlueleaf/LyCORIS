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
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy

    def get_position_angle_vec(position):
        # this part calculate the position In brackets
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    # [:, 0::2] are all even subscripts, is dim_2i
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class WeightGenerator(nn.Module):
    def __init__(
        self, 
        encoder_model_name: str = "vit_base_patch16_224",
        train_encoder: bool = False,
        reference_size: Tuple[int] = (224, 224), 
        weight_dim: int = 150,                      # 100+50 in paper
        weight_num: int = 54,                       # 21*2 in SD UNet + 12 in CLIP-L TE
        decoder_blocks: int = 4,
        sample_iters: int = 1,
    ):
        super(WeightGenerator, self).__init__()
        self.ref_size = reference_size
        self.weight_num = weight_num
        self.weight_dim = weight_dim
        self.sample_iters = sample_iters
        
        self.register_buffer(
            'block_pos_emb', 
            _get_sinusoid_encoding_table(weight_num*2, weight_dim)
        )
        
        self.encoder_model = create_model(encoder_model_name, pretrained=True)
        self.encoder_model.requires_grad_(train_encoder)
        
        test_input = torch.randn(1, 3, *reference_size)
        test_output = self.encoder_model.forward_features(test_input)
        if isinstance(test_output, list):
            test_output = test_output[-1]
        if len(test_output.shape) == 4:
            # B, C, H, W -> B, L, C
            test_output = test_output.view(1, test_output.size(1), -1).transpose(1, 2)
        
        self.feature_proj = nn.Linear(test_output.shape[-1], weight_dim)
        self.decoder_model = nn.ModuleList(
            TransformerBlock(weight_dim, 1, weight_dim, context_dim=weight_dim, gated_ff=False)
            for _ in range(decoder_blocks)
        )
        
        self.weight_proj = nn.Linear(weight_dim, weight_dim, bias=False)
        nn.init.constant_(self.weight_proj.weight, 0)
    
    def forward(self, ref_img, weight=None):
        features = self.encoder_model.forward_features(resize(ref_img, self.ref_size))
        if isinstance(features, list):
            features = features[-1]
        if len(features.shape) == 4:
            # B, C, H, W -> B, L, C
            features = features.view(features.size(0), features.size(1), -1).transpose(1, 2)
        features = self.feature_proj(features)
        
        if weight is None:
            weight = torch.zeros(
                ref_img.size(0), self.weight_num, self.weight_dim, device=ref_img.device
            )
        weight = weight 
        h = weight + self.block_pos_emb[:, :self.weight_num].clone().detach()
        for iter in range(self.sample_iters):
            for decoder in self.decoder_model:
                h = decoder(h, context=features)
            weight = weight + h
        weight = self.weight_proj(weight)
        return weight