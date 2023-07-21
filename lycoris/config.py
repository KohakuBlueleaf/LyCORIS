PRESET = {
    'full':{
        'enable_conv': True,
        'unet_target_module':[
            "Transformer2DModel", 
            "Attention", 
            "ResnetBlock2D", 
            "Downsample2D", 
            "Upsample2D"
        ],
        'unet_target_name':[
            "conv_in",
            "conv_out",
            "time_embedding.linear_1",
            "time_embedding.linear_2",
        ],
        'text_encoder_target_module': [
            "CLIPAttention", "CLIPMLP"
        ],
        'text_encoder_target_name': [],
    },
    'full-lin':{
        'enable_conv': False,
        'unet_target_module':[
            "Transformer2DModel", 
            "Attention", 
            "ResnetBlock2D",
        ],
        'unet_target_name':[
            "time_embedding.linear_1",
            "time_embedding.linear_2",
        ],
        'text_encoder_target_module': [
            "CLIPAttention", "CLIPMLP"
        ],
        'text_encoder_target_name': [],
    },
    'lora':{
        'enable_conv': False,
        'unet_target_module':[
            "Transformer2DModel", 
            "Attention", 
        ],
        'unet_target_name':[],
        'text_encoder_target_module': [
            "CLIPAttention", "CLIPMLP"
        ],
        'text_encoder_target_name': [],
    },
    'unet-only':{
        'enable_conv': True,
        'unet_target_module':[
            "Transformer2DModel", 
            "Attention", 
            "ResnetBlock2D", 
            "Downsample2D", 
            "Upsample2D"
        ],
        'unet_target_name':[
            "conv_in",
            "conv_out",
            "time_embedding.linear_1",
            "time_embedding.linear_2",
        ],
        'text_encoder_target_module': [],
        'text_encoder_target_name': [],
    },
    'unet-transformer-only':{
        'enable_conv': False,
        'unet_target_module':[
            "Transformer2DModel", 
            "Attention", 
        ],
        'unet_target_name':[],
        'text_encoder_target_module': [],
        'text_encoder_target_name': [],
    },
    'unet-convblock-only':{
        'enable_conv': True,
        'unet_target_module':[
            "ResnetBlock2D", 
            "Downsample2D", 
            "Upsample2D"
        ],
        'unet_target_name':[
            "conv_in",
            "conv_out",
        ],
        'text_encoder_target_module': [],
        'text_encoder_target_name': [],
    },
}
