---
title: Getting Started
---

# Getting started

Use the local stack when you want the dashboard. Install the Python package
directly when you only need the CLI or simulator API.

## Run with the local vLLM Semantic Router stack

From the repository root:

```bash
make vllm-sr-dev
vllm-sr serve --image-pull-policy never
```

The CLI starts Fleet Sim as a sibling container by default and connects the
dashboard backend to it on the runtime network. Open the dashboard and choose
**Fleet Sim** to manage workloads, fleet definitions, and runs.

This flow uses locally built images. It does not require a separate Python
installation for Fleet Sim.

## Install the standalone CLI

From a source checkout:

```bash
cd src/fleet-sim
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
vllm-sr-sim --version
```

Windows PowerShell users can activate the environment with
`.venv\Scripts\Activate.ps1`.

### Run a first study

The following command searches for a low-cost two-pool fleet and DES-checks the
top candidates:

```bash
vllm-sr-sim optimize \
  --cdf data/azure_cdf.json \
  --lam 200 \
  --slo 500 \
  --b-short 6144 \
  --verify-top 3 \
  --n-sim-req 30000
```

Interpret the inputs before the output:

- `--cdf` describes the cumulative token-length distribution.
- `--lam` is the assumed arrival rate in requests per second.
- `--slo` is the P99 TTFT target in milliseconds.
- `--b-short` sends requests at or below the threshold to the short pool.
- `--verify-top` selects analytical candidates for DES validation.

The resulting GPU counts and costs are estimates based on the selected built-in
profiles. Replace those profiles or treat the result as relative guidance until
you have calibrated the target deployment.

## Choose a command

| Command | Use it to |
| --- | --- |
| `optimize` | Search a two-pool fleet and optionally DES-check candidates |
| `simulate` | Run DES for fixed short- and long-pool counts |
| `whatif` | Sweep arrival rates or built-in GPU profiles |
| `pareto` | Compare token thresholds from the workload CDF |
| `compare-routers` | Compare the CLI's length, compress-and-route, and random policies on one fixed fleet |
| `disagg` | Size separate prefill and decode pools |
| `grid-flex` | Estimate latency while reducing modeled concurrency and power |
| `tok-per-watt` | Compare modeled energy efficiency |
| `simulate-fleet` | Simulate an arbitrary multi-pool JSON topology |
| `serve` | Start the Fleet Sim HTTP service |

Run `vllm-sr-sim <command> --help` for the current options. Add `--out FILE`
to supported commands when you need machine-readable JSON.

## Start the standalone service

Install the API dependencies and start FastAPI:

```bash
cd src/fleet-sim
python -m pip install -e '.[api]'
vllm-sr-sim serve --host 127.0.0.1 --port 8000
```

Then check:

```bash
curl -sS http://127.0.0.1:8000/healthz
```

Interactive API documentation is available at
`http://127.0.0.1:8000/api/docs`, and the OpenAPI document is at
`/api/openapi.json`.

Use `--host 0.0.0.0` only when another host or container must connect, and put
authentication and network controls in front of the service. Fleet Sim's
FastAPI application does not add its own authentication layer.

## Use an external service with the dashboard

Set the service URL before starting the local stack:

```bash
export TARGET_FLEET_SIM_URL=http://fleet-sim.internal:8000
vllm-sr serve --image-pull-policy never
```

When that variable is present, the CLI does not start its Fleet Sim sidecar. To
disable Fleet Sim without providing an external service:

```bash
export VLLM_SR_SIM_ENABLED=false
```

Continue with [capacity-planning workflows](./use-cases) before treating a
sample result as a deployment recommendation.
