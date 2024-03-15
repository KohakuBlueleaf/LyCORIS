from transformers import AutoModelForCausalLM
from lycoris import create_lycoris, LycorisNetwork

model = AutoModelForCausalLM.from_pretrained("KBlueLeaf/DanTagGen", load_in_4bit=True)


LycorisNetwork.apply_preset(
    {"target_name": [".*proj.*"]}
)
lycoris_net = create_lycoris(
    model, 
    1.0, 
    linear_dim=16, 
    linear_alpha=2.0, 
    algo="lokr",
    factor=8
)
lycoris_net.apply_to()

print(model)
print(lycoris_net)