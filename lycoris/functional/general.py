import torch
import torch.nn as nn
import torch.nn.functional as F


def rebuild_tucker(t, wa, wb):
    rebuild2 = torch.einsum("i j ..., i p, j r -> p r ...", t, wa, wb)
    return rebuild2
