"""Real layer shapes the benches sweep. Square-only benching is banned.

Linear shapes are (out, in) from SDXL UNet attention/FF blocks, Flux-class
DiT blocks, and 1-8B LLM projections; conv shapes are SDXL residual blocks.
Token counts cover text encoders, 1024px latents and LLM contexts.
"""

LINEAR = {
    "sdxl_attn": (640, 640),
    "sdxl_attn_xl": (1280, 1280),
    "sdxl_ff_in": (5120, 1280),
    "sdxl_ff_out": (1280, 5120),
    "dit_qkv": (3072, 3072),
    "dit_mlp_in": (12288, 3072),
    "llm_qkv": (4096, 4096),
    "llm_mlp": (11008, 4096),
}

CONV = {
    "sdxl_res320": (320, 320, 3, 3),
    "sdxl_res640": (640, 640, 3, 3),
    "sdxl_res1280": (1280, 1280, 3, 3),
}

TOKENS = (77, 1024, 4096, 16384)
RANKS = (4, 16, 64)
LOKR_FACTORS = (-1, 8)
OFT_DIMS = (4, 16)
