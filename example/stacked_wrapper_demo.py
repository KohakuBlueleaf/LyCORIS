from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Lambda, ToTensor

from lycoris import LycorisNetwork, create_lycoris


class DemoNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.test_1 = nn.Linear(784, 2048)
        self.te_2st = nn.Linear(2048, 784)
        self._3test = nn.Linear(784, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.test_1(x)
        h = F.mish(h)
        h = self.te_2st(h)
        h = x + h
        return self._3test(h)


@dataclass
class StackSummary:
    model: nn.Module
    wrappers: list[LycorisNetwork]
    sample: torch.Tensor
    base_out: torch.Tensor
    outputs: list[torch.Tensor]


def apply_wrapper(
    model: nn.Module, preset: dict[str, list[str]], **kwargs
) -> LycorisNetwork:
    LycorisNetwork.apply_preset(preset)
    wrapper = create_lycoris(model, **kwargs)
    wrapper.apply_to()
    return wrapper


def verify_stack(model: nn.Module, seeds: list[int]) -> StackSummary:
    sample = torch.randn(4, 784)
    model.eval()
    with torch.no_grad():
        base_out = model(sample)

    wrappers: list[LycorisNetwork] = []
    outputs = [base_out]
    configs = [
        ({"target_name": [".*test_1.*"]}, {"algo": "lokr"}),
        ({"target_name": [".*te_2st.*"]}, {"algo": "loha"}),
        ({"target_name": [".*_3test.*"]}, {"algo": "lokr"}),
        ({"target_name": [".*te.*"]}, {"algo": "loha"}),
    ]

    seed_iter = iter(seeds)
    for preset, extra in configs:
        try:
            torch.manual_seed(next(seed_iter))
        except StopIteration:
            raise ValueError("Not enough seeds supplied for wrapper configuration")
        wrapper = apply_wrapper(
            model,
            preset=preset,
            multiplier=1.0,
            linear_dim=16,
            linear_alpha=2.0,
            **extra,
        )
        wrappers.append(wrapper)
        with torch.no_grad():
            outputs.append(model(sample))

    for wrapper in reversed(wrappers):
        wrapper.restore()
    with torch.no_grad():
        restored = model(sample)
    assert torch.allclose(restored, base_out, atol=1e-5)

    for wrapper in wrappers:
        wrapper.apply_to()

    return StackSummary(
        model=model,
        wrappers=wrappers,
        sample=sample,
        base_out=base_out,
        outputs=outputs,
    )


def describe_stack(summary: StackSummary) -> None:
    print(f"Applied wrappers: {len(summary.wrappers)}")
    for idx, (wrapper, output) in enumerate(
        zip(summary.wrappers, summary.outputs[1:]), 1
    ):
        delta = F.mse_loss(output, summary.outputs[idx - 1]).item()
        print(
            f"  Wrapper {idx}: {wrapper.loras[0].__class__.__name__} delta={delta:.6f}"
        )


def remove_wrappers(summary: StackSummary, indices: list[int]) -> torch.Tensor:
    """Remove selected wrappers, capture the output, then reapply them."""
    indices_set = set(indices)

    # Detach all wrappers to start from the base network.
    for wrapper in reversed(summary.wrappers):
        wrapper.restore()

    # Apply only the wrappers we want to keep, preserving order.
    for idx, wrapper in enumerate(summary.wrappers):
        if idx not in indices_set:
            wrapper.apply_to()

    summary.model.eval()
    with torch.no_grad():
        output = summary.model(summary.sample)

    # Restore original full stack order.
    for wrapper in reversed(summary.wrappers):
        wrapper.restore()
    for wrapper in summary.wrappers:
        wrapper.apply_to()

    summary.model.train()
    return output


def train_on_mnist(summary: StackSummary, *, data_root: str) -> None:
    transform = Compose([ToTensor(), Lambda(lambda x: x.view(-1))])
    train_ds = MNIST(root=data_root, download=True, train=True, transform=transform)
    test_ds = MNIST(root=data_root, download=True, train=False, transform=transform)
    train_loader = data.DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = data.DataLoader(test_ds, batch_size=32)

    optimiser = torch.optim.AdamW(
        chain.from_iterable(wrapper.parameters() for wrapper in summary.wrappers),
        lr=5e-3,
    )

    summary.model.train()
    ema_loss = 0.0
    for step, (inputs, targets) in enumerate(train_loader):
        optimiser.zero_grad()
        outputs = summary.model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimiser.step()
        ema_decay = min(0.999, step / 1000)
        ema_loss = ema_decay * ema_loss + (1 - ema_decay) * loss.item()
        if step % 100 == 0:
            print(step, ema_loss)

    def evaluate() -> float:
        total_correct = 0
        total_seen = 0
        summary.model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = summary.model(inputs)
                preds = outputs.argmax(dim=1)
                total_correct += (preds == targets).sum().item()
                total_seen += len(targets)
        return total_correct / max(total_seen, 1)

    print("Eval accuracy:", evaluate())

    for wrapper in summary.wrappers:
        wrapper.restore()
        wrapper.merge_to(1.0)
    print("Merged accuracy:", evaluate())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate stacking and selective removal of LyCORIS wrappers"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
        help="Seeds for wrapper initialisation (must match wrapper count)",
    )
    parser.add_argument(
        "--remove",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Wrapper indices to remove (0-based order)",
    )
    parser.add_argument(
        "--data-root", default="data", help="Directory to download MNIST into"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only demonstrate stacking/removal, skip the training loop",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    net = DemoNet()
    summary = verify_stack(net, seeds=args.seeds)
    describe_stack(summary)
    removed_output = remove_wrappers(summary, indices=args.remove)
    print(
        f"Output after removing wrappers {args.remove}:",
        removed_output.norm().item(),
    )
    if not args.skip_training:
        train_on_mnist(summary, data_root=args.data_root)
