from locon.utils import extract_diff
from locon.kohya_model_utils import load_models_from_stable_diffusion_checkpoint

import torch


DEVICE = 'cuda'

BASE_MODEL = 'PATH_TO_BASE_MODEL'
DB_MODEL = 'PATH_TO_DREAMBOOTH_MODEL'
OUTPUT_NAME = 'PATH_TO_OUTPUT_MODEL'

# rank for extracted locon
LORA_RANK = 80
CONV_RANK = 48

# special settings
# Extract LoCon by singular value threshold
# This can auto choose rank for different layer
# higher threshold = smaller file size
# if enabled, RANK setting will be ignored
USE_THRESHOLD = False
LINEAR_THRESHOLD = 0.07
USE_THRESHOLD_CONV = False
CONV_TRESHOLD = 0.45


base = load_models_from_stable_diffusion_checkpoint(False, BASE_MODEL)
db = load_models_from_stable_diffusion_checkpoint(False, DB_MODEL)

state_dict = extract_diff(
    base, db,
    LORA_RANK, CONV_RANK,
    USE_THRESHOLD,
    USE_THRESHOLD_CONV,
    LINEAR_THRESHOLD,
    CONV_TRESHOLD,
    DEVICE
)
torch.save(state_dict, OUTPUT_NAME)