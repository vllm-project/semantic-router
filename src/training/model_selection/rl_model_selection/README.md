# Experimental Router-Model Training

This directory contains independent prototypes inspired by Router-R1 and
GMTRouter. They explore learned routing policies and are not wired into the
semantic-router runtime or its supported configuration schema.

## Contents

| File | Purpose |
|---|---|
| `train_router_r1.py` | fine-tune a causal language model to emit routing actions |
| `train_gmtrouter.py` | train a heterogeneous-graph routing model |
| `router_r1_server.py` | standalone `/route` demonstration server |
| `automix_verifier.py` | standalone answer-verification demonstration server |
| `configs/` | example training configuration |
| `data/` | small example JSON inputs |
| `scripts/` | GPU-oriented wrappers for the two trainers |

The example data is for exercising the pipelines. Replace it with a reviewed
dataset before drawing conclusions about routing quality.

## Setup

This directory is a uv project (`pyproject.toml` and `uv.lock`). Create the
environment from the lock:

```bash
cd src/training/model_selection/rl_model_selection
uv sync --locked
```

`pyproject.toml` bounds the packages these scripts import: `torch`,
`torch-geometric`, `transformers`, `trl`, `numpy`, `tqdm`, and `pyyaml`.
`uv.lock` pins the exact versions. `requirements.txt` stays for anyone who
installs with pip. Names in that file that these scripts do not import,
including `torch-sparse` and `torch-scatter`, are not part of the lock.

`trl` is installed, but current releases no longer export `PPOConfig` or
`PPOTrainer` from `trl`. `train_router_r1.py` catches that import error and
uses its custom training loop. Changing that import is outside this lock.

The lock was created with uv 0.13.0 on macOS 26.7.1 (Darwin arm64) using
CPython 3.12.15. It covers Python 3.10 and newer. Torch is the default PyPI
build, with no extra index. On Linux that wheel is the CUDA build: it still
imports on a machine with no GPU, and `torch.cuda.is_available()` is false
there. On macOS Apple Silicon the wheel is the CPU/MPS build. A smaller
CPU-only Linux wheel is the pip index noted in
`src/training/model_selection/ml_model_selection/requirements.txt`. Choosing
CPU, CUDA, or ROCm from uv is [issue 5026](https://github.com/vllm-project/semantic-router/issues/5026)
and is not configured in this project.

There are no tests in this directory. Confirm the scripts start from the
repository root:

```bash
uv run --project src/training/model_selection/rl_model_selection \
  python src/training/model_selection/rl_model_selection/train_router_r1.py --help

uv run --project src/training/model_selection/rl_model_selection \
  python src/training/model_selection/rl_model_selection/train_gmtrouter.py --help

uv run --project src/training/model_selection/rl_model_selection \
  python src/training/model_selection/rl_model_selection/router_r1_server.py --help

uv run --project src/training/model_selection/rl_model_selection \
  python src/training/model_selection/rl_model_selection/automix_verifier.py --help
```

Review the selected config and data paths, then run a trainer from this
directory:

```bash
uv run python train_router_r1.py \
  --config configs/router_r1_config.yaml \
  --output_dir checkpoints/router_r1

uv run python train_gmtrouter.py \
  --config configs/gmtrouter_config.yaml \
  --data_path data/interactions.json \
  --output_dir checkpoints/gmtrouter
```

`train_gmtrouter.py --use_synthetic` is a code-path smoke mode. It is not an
evaluation dataset.

## Standalone Demos

The servers do not proxy requests to the selected backend and do not configure
semantic-router. They expose the prototype decision or verification output for
manual inspection:

```bash
uv run python router_r1_server.py --model microsoft/phi-2 --port 8888
uv run python automix_verifier.py --model microsoft/phi-2 --port 8889
```

Use each script's `--test` flag for a local demonstration without starting the
HTTP server.

## Evaluation Boundary

Before proposing runtime integration, add held-out evaluation with explicit
quality, cost, latency, and safety metrics; document the candidate model pool;
and compare against deterministic router selectors. Do not describe generated
checkpoints as supported `algorithm.type` values until the public config,
runtime loader, tests, and user documentation exist.

## References

- [Router-R1](https://arxiv.org/abs/2506.09033)
- [GMTRouter](https://arxiv.org/abs/2511.08590)
- [AutoMix](https://arxiv.org/abs/2310.12963)
