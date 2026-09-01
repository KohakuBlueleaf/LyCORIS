# LyCORIS Documentation

Start here. Every document in this tree is listed below with what it answers,
so you can jump straight to the one you need.

## I want to …

| Goal | Read |
| --- | --- |
| Pick an algorithm | [algorithms/README.md](algorithms/README.md), then [algorithms/guidelines.md](algorithms/guidelines.md) |
| Understand the math behind an algorithm | [algorithms/details.md](algorithms/details.md) |
| Configure a training run | [usage/network-args.md](usage/network-args.md) and [usage/presets.md](usage/presets.md) |
| Call LyCORIS from my own code | [api/README.md](api/README.md) |
| Make training and merging faster | [kernels/README.md](kernels/README.md) |
| Know which backend is running | [kernels/backends.md](kernels/backends.md) |
| Check numerical accuracy | [kernels/precision.md](kernels/precision.md) |
| Reproduce the speed numbers | [kernels/benchmarks.md](kernels/benchmarks.md) |
| Convert between formats | [usage/conversion-scripts.md](usage/conversion-scripts.md) |
| Work on LyCORIS itself, or cut a release | [development/README.md](development/README.md) |
| See sample results | [resources/demo.md](resources/demo.md) |
| Learn more about fine-tuning in general | [resources/README.md](resources/README.md) |

## Tree

```
docs/
├── algorithms/     what LyCORIS implements
│   ├── README.md   list of implemented algorithms, with shapes and parameters
│   ├── details.md  the math of each algorithm
│   └── guidelines.md  which algorithm to pick, and with what hyperparameters
├── usage/          how to drive it
│   ├── network-args.md       every network argument, per algorithm
│   ├── presets.md            preset files and layer selection
│   └── conversion-scripts.md format conversion tools
├── api/
│   └── README.md   module, functional and wrapper APIs
├── kernels/        the fused kernel backends (experimental)
│   ├── README.md      what is fused, what it buys, how to switch it off
│   ├── backends.md    selection order, per-op scope, environment variables
│   ├── precision.md   dtype policy and measured accuracy
│   └── benchmarks.md  the measurement harness and current results
├── development/
│   └── README.md   checks, versioning, release channels, workflows
└── resources/
    ├── README.md   external reading on fine-tuning
    └── demo.md     example outputs
```

## Source map

| Path | Contents |
| --- | --- |
| `lycoris/modules/` | one `nn.Module` wrapper per algorithm, plus `base.py` |
| `lycoris/functional/` | the same algorithms as plain functions (`weight_gen`, `diff_weight`, `bypass_forward_diff`) |
| `lycoris/kernels/` | fused Triton and TileLang kernels, the autograd layer, and backend selection |
| `lycoris/kernels/plans/` | the analytic tile planner and the measured tuner |
| `lycoris/wrapper.py` | `LycorisNetwork`, `create_lycoris` |
| `lycoris/kohya.py` | the kohya-ss/sd-scripts entry point |
| `scripts/bench/kernels/` | the kernel benchmark suite (measure scripts write JSON, plot scripts draw it) |
| `scripts/ci/` | the checks and version helpers the workflows call |
| `test/` | functional, module, wrapper and kernel test suites |
