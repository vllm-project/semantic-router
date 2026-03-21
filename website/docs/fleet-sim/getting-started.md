---
title: Getting Started
---

# Getting Started

This page covers the maintained ways to run `vllm-sr-sim`.

## Recommended local workflow: sidecar with `vllm-sr serve`

Use the repository-native flow when you want the simulator and dashboard wired
together locally:

```bash
make vllm-sr-dev
cd src/vllm-sr
vllm-sr serve --image-pull-policy never
```

That flow:

- builds the router image
- builds the `vllm-sr-sim` image
- installs both CLIs in editable mode
- starts the simulator sidecar automatically on the shared runtime network

## Standalone CLI

Use the standalone CLI when you only need local planning commands:

```bash
cd src/fleet-sim
pip install -e .[dev]

vllm-sr-sim --version
vllm-sr-sim optimize --cdf data/azure_cdf.json --lam 200 --slo 500 --b-short 6144
```

## Standalone service

Use service mode when the dashboard or another caller should talk to the simulator
over HTTP:

```bash
cd src/fleet-sim
pip install -e .[dev]

vllm-sr-sim serve --host 0.0.0.0 --port 8000
```

When the simulator runs externally, point the dashboard or `vllm-sr serve` at it
with `TARGET_FLEET_SIM_URL`.

## External service instead of the local sidecar

If `TARGET_FLEET_SIM_URL` is already set, `vllm-sr serve` uses that external service
instead of starting the local sidecar.

If you want to turn off the default sidecar behavior without providing an external
service, set:

```bash
export VLLM_SR_SIM_ENABLED=false
```
