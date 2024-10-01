import torch
import torch.nn as nn
import torch.nn.functional as F

from lycoris.functional import loha, lokr


org_model = nn.Linear(128, 128)
# Call the Functional API to get weights
lokr_weights = lokr.weight_gen(org_model.weight)
loha_weights = loha.weight_gen(org_model.weight)

test_x = torch.randn(1, 128)
test_out = org_model(test_x)

# Use 2 different type of functional API to do the forward
test_out_lokr_diff = test_out + lokr.bypass_forward_diff(
    test_x, test_out, *lokr_weights
)
test_out_loha_diff = test_out + loha.bypass_forward_diff(
    test_x, test_out, *loha_weights
)
test_out_lokr_diff_weight = F.linear(
    test_x, org_model.weight + lokr.diff_weight(*lokr_weights), org_model.bias
)
test_out_loha_diff_weight = F.linear(
    test_x, org_model.weight + loha.diff_weight(*loha_weights), org_model.bias
)


# The init value should somehow ensure the difference is 0
print(F.mse_loss(test_out, test_out_lokr_diff))
print(F.mse_loss(test_out, test_out_loha_diff))
print(F.mse_loss(test_out, test_out_lokr_diff_weight))
print(F.mse_loss(test_out, test_out_loha_diff_weight))
