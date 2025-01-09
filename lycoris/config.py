PRESET = {
    "full": {
        "enable_conv": True,
        "unet_target_module": [
            "Transformer2DModel",
            "ResnetBlock2D",
            "Downsample2D",
            "Upsample2D",
            "HunYuanDiTBlock", #HunYuanDiT
            "DoubleStreamBlock", #Flux
            "SingleStreamBlock", #Flux
            "SingleDiTBlock", #SD3.5
            "MMDoubleStreamBlock", #HunYuanVideo
            "MMSingleStreamBlock", #HunYuanVideo
        ],
        "unet_target_name": [
            "conv_in",
            "conv_out",
            "time_embedding.linear_1",
            "time_embedding.linear_2",
        ],
        "text_encoder_target_module": [
            "CLIPAttention",
            "CLIPSdpaAttention",
            "CLIPMLP",
            "MT5Block",
            "BertLayer",
        ],
        "text_encoder_target_name": [],
    },
    "full-lin": {
        "enable_conv": False,
        "unet_target_module": [
            "Transformer2DModel",
            "ResnetBlock2D",
            "HunYuanDiTBlock",
            "DoubleStreamBlock",
            "SingleStreamBlock",
            "SingleDiTBlock",
            "MMDoubleStreamBlock", #HunYuanVideo
            "MMSingleStreamBlock", #HunYuanVideo
        ],
        "unet_target_name": [
            "time_embedding.linear_1",
            "time_embedding.linear_2",
        ],
        "text_encoder_target_module": [
            "CLIPAttention",
            "CLIPSdpaAttention",
            "CLIPMLP",
            "MT5Block",
            "BertLayer",
        ],
        "text_encoder_target_name": [],
    },
    "attn-mlp": {
        "enable_conv": False,
        "unet_target_module": [
            "Transformer2DModel",
            "HunYuanDiTBlock",
            "DoubleStreamBlock",
            "SingleStreamBlock",
            "SingleDiTBlock",
            "MMDoubleStreamBlock", #HunYuanVideo
            "MMSingleStreamBlock", #HunYuanVideo
        ],
        "unet_target_name": [],
        "text_encoder_target_module": [
            "CLIPAttention",
            "CLIPSdpaAttention",
            "CLIPMLP",
            "MT5Block",
            "BertLayer",
        ],
        "text_encoder_target_name": [],
    },
    "attn-only": {
        "enable_conv": False,
        "unet_target_module": [
            "CrossAttention",
            "SelfAttention",
        ],
        "unet_target_name": [],
        "text_encoder_target_module": [
            "CLIPAttention",
            "CLIPSdpaAttention",
            "BertAttention",
            "MT5LayerSelfAttention",
        ],
        "text_encoder_target_name": [],
    },
    "unet-only": {
        "enable_conv": True,
        "unet_target_module": [
            "Transformer2DModel",
            "ResnetBlock2D",
            "Downsample2D",
            "Upsample2D",
            "HunYuanDiTBlock",
            "DoubleStreamBlock",
            "SingleStreamBlock",
            "SingleDiTBlock",
            "MMDoubleStreamBlock", #HunYuanVideo
            "MMSingleStreamBlock", #HunYuanVideo
        ],
        "unet_target_name": [
            "conv_in",
            "conv_out",
            "time_embedding.linear_1",
            "time_embedding.linear_2",
        ],
        "text_encoder_target_module": [],
        "text_encoder_target_name": [],
    },
    "unet-transformer-only": {
        "enable_conv": False,
        "unet_target_module": [
            "Transformer2DModel",
            "HunYuanDiTBlock",
            "DoubleStreamBlock",
            "SingleStreamBlock",
            "SingleDiTBlock",
            "MMDoubleStreamBlock", #HunYuanVideo
            "MMSingleStreamBlock", #HunYuanVideo
        ],
        "unet_target_name": [],
        "text_encoder_target_module": [],
        "text_encoder_target_name": [],
    },
    "unet-convblock-only": {
        "enable_conv": True,
        "unet_target_module": ["ResnetBlock2D", "Downsample2D", "Upsample2D"],
        "unet_target_name": [
            "conv_in",
            "conv_out",
        ],
        "text_encoder_target_module": [],
        "text_encoder_target_name": [],
    },
    "ia3": {
        "enable_conv": False,
        "unet_target_module": [],
        "unet_target_name": ["to_k", "to_v", "ff.net.2"],
        "text_encoder_target_module": [],
        "text_encoder_target_name": ["k_proj", "v_proj", "mlp.fc2"],
        "name_algo_map": {
            "mlp.fc2": {"train_on_input": True},
            "ff.net.2": {"train_on_input": True},
        },
    },
}
