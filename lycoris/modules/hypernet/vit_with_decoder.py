from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model
from einops import rearrange

from lycoris.modules.attention import TransformerBlock


class WeightGenerator(nn.Module):
    def __init__(
        self, 
        encoder_model_name: str,
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
        
        self.encoder_model = create_model(encoder_model_name, pretrained=True, features_only=True)
        self.encoder_model.requires_grad_(train_encoder)
        
        test_input = torch.randn(1, 3, *reference_size)
        test_output = self.encoder_model(test_input)
        if isinstance(test_output, list):
            test_output = test_output[-1]
        if len(test_output.shape) == 4:
            # B, C, H, W -> B, L, C
            test_output = test_output.view(1, test_output.size(1), -1).transpose(1, 2)
        
        self.feature_proj = nn.Linear(test_output.shape[2], weight_dim)
        self.decoder_model = nn.ModuleList(
            TransformerBlock(weight_dim, 8, weight_dim // 8, context_dim=weight_dim, gated_ff=False)
            for _ in range(decoder_blocks)
        )
    
    def forward(self, ref_img):
        features = self.encoder_model(ref_img)
        if isinstance(features, list):
            features = features[-1]
        if len(features.shape) == 4:
            # B, C, H, W -> B, L, C
            features = features.view(features.size(0), features.size(1), -1).transpose(1, 2)
        features = self.feature_proj(features)
        
        weight = torch.zeros(
            ref_img.size(0), self.weight_num, self.weight_dim, device=ref_img.device
        )
        for iter in range(self.sample_iters):
            for decoder in self.decoder_model:
                weight = decoder(weight, context=features)
        
        return weight