import os, sys
sys.path.append(os.getcwd())

from lycoris.utils import merge_locon
from lycoris.kohya_model_utils import (
    load_models_from_stable_diffusion_checkpoint,
    save_stable_diffusion_checkpoint,
    load_file
)

import torch


BASE_MODEL = 'PATH_TO_BASE_MODEL'
LOCON_MODEL = 'PATH_TO_LOCON_MODEL'
OUTPUT_NAME = 'NAME_FOR_OUTPUT'
SAFE_TENSOR = False
WEIGHT = 1.0
V2 = False
DEVICE = 'cuda'
DTYPE = torch.float16


base = load_models_from_stable_diffusion_checkpoint(V2, BASE_MODEL)
if SAFE_TENSOR:
    locon = load_file(LOCON_MODEL)
else:
    locon = torch.load(LOCON_MODEL)

merge_locon(
    base,
    locon,
    WEIGHT,
    DEVICE
)

save_stable_diffusion_checkpoint(
    V2, OUTPUT_NAME, 
    base[0], base[2], 
    None, 0, 0, DTYPE, 
    base[1]
)
