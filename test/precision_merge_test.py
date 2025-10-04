import unittest

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


class MergePrecisionTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_merge_round_trip_drift(self) -> None:
        drift = self._run_merge_cycle(precise=False, cycles=1000)
        self.assertLessEqual(
            drift,
            1e-9,
            msg=f"Observed {drift:.12e} max drift after 1000 cycles",
        )

    def test_precise_merge_round_trip(self) -> None:
        baseline = self._run_merge_cycle(precise=False, cycles=200)
        precise = self._run_merge_cycle(precise=True, cycles=200)

        self.assertLessEqual(
            precise,
            baseline + 1e-9,
            msg=(
                f"Precise merge drift {precise:.12e} exceeded baseline {baseline:.12e}"
            ),
        )

    @staticmethod
    def _run_merge_cycle(*, precise: bool, cycles: int) -> float:
        reset_preset()
        torch.manual_seed(0)

        base = nn.Linear(32, 32)
        wrappers = [
            create_lycoris(
                base,
                multiplier=1.0,
                linear_dim=8,
                linear_alpha=1.0,
                algo="lora",
            ),
            create_lycoris(
                base,
                multiplier=1.0,
                linear_dim=8,
                linear_alpha=1.0,
                algo="loha",
            ),
        ]

        for wrapper in wrappers:
            wrapper.apply_to()
            with torch.no_grad():
                for param in wrapper.parameters():
                    torch.nn.init.normal_(param)

        original_weight = base.weight.detach().clone()

        for _ in range(cycles):
            for wrapper in wrappers:
                wrapper.merge_to(1.0, precise=precise)
            for wrapper in reversed(wrappers):
                wrapper.merge_to(-1.0, precise=precise)

        drift = (base.weight - original_weight).abs().max().item()
        return drift


if __name__ == "__main__":
    unittest.main()
