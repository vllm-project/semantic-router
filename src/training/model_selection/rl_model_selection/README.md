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

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Review the selected config and data paths, then run a trainer from this
directory:

```bash
python train_router_r1.py \
  --config configs/router_r1_config.yaml \
  --output_dir checkpoints/router_r1

python train_gmtrouter.py \
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
python router_r1_server.py --model microsoft/phi-2 --port 8888
python automix_verifier.py --model microsoft/phi-2 --port 8889
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
