from __future__ import annotations

import torch
import torch.nn as nn

from lycoris import LycorisNetwork, create_lycoris


def reset_preset() -> None:
    LycorisNetwork.apply_preset(
        {
            "enable_conv": True,
            "target_module": [
                "Linear",
                "Conv1d",
                "Conv2d",
                "Conv3d",
                "GroupNorm",
                "LayerNorm",
            ],
            "target_name": [],
            "lora_prefix": "lycoris",
            "module_algo_map": {},
            "name_algo_map": {},
            "use_fnmatch": False,
            "exclude_name": [],
        }
    )


def build_wrappers(module: nn.Module) -> list[LycorisNetwork]:
    wrappers: list[LycorisNetwork] = []
    for algo in ("lora", "loha"):
        wrapper = create_lycoris(
            module,
            multiplier=1.0,
            linear_dim=8,
            linear_alpha=1.0,
            algo=algo,
        )
        wrapper.apply_to()
        with torch.no_grad():
            for param in wrapper.parameters():
                torch.nn.init.normal_(param)
        wrappers.append(wrapper)
    return wrappers


def run_cycles(module: nn.Linear, wrappers: list[LycorisNetwork], *, precise: bool, cycles: int) -> float:
    original = module.weight.detach().clone()
    for _ in range(cycles):
        for wrapper in wrappers:
            wrapper.merge_to(1.0, precise=precise)
        for wrapper in reversed(wrappers):
            wrapper.merge_to(-1.0, precise=precise)
    return (module.weight - original).abs().max().item()


def main() -> None:
    reset_preset()
    torch.manual_seed(0)

    base = nn.Linear(32, 32)
    wrappers = build_wrappers(base)
    standard_drift = run_cycles(base, wrappers, precise=False, cycles=200)

    # fresh copy for precise run
    reset_preset()
    torch.manual_seed(0)
    base_precise = nn.Linear(32, 32)
    wrappers_precise = build_wrappers(base_precise)
    precise_drift = run_cycles(base_precise, wrappers_precise, precise=True, cycles=200)

    print(f"Standard merge drift (200 cycles): {standard_drift:.3e}")
    print(f"Precise merge drift   (200 cycles): {precise_drift:.3e}")


if __name__ == "__main__":
    main()
