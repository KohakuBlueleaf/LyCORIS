from locon.utils import extract_diff
from locon.kohya_model_utils import load_models_from_stable_diffusion_checkpoint

import torch


BASE_MODEL = 'PATH_TO_BASE_MODEL'
DB_MODEL = 'PATH_TO_DREAMBOOTH_MODEL'
LORA_RANK = 40
CONV_RANK = 24
OUTPUT_NAME = 'PATH_TO_OUTPUT_MODEL'


base = load_models_from_stable_diffusion_checkpoint(False, BASE_MODEL)
db = load_models_from_stable_diffusion_checkpoint(False, DB_MODEL)

state_dict = extract_diff(base, db, LORA_RANK, CONV_RANK)
torch.save(state_dict, OUTPUT_NAME)