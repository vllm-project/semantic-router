# vllm-sr-sim

`vllm-sr-sim` is the maintained fleet simulator for this repository. It sizes heterogeneous GPU fleets, evaluates routing strategies, and exposes a service mode that the dashboard can call across containers.

Repository-maintained docs now live in the website:

- https://vllm-sr.ai/docs/fleet-sim/overview
- https://vllm-sr.ai/docs/fleet-sim/getting-started
- https://vllm-sr.ai/docs/fleet-sim/use-cases

## Install

```bash
cd src/fleet-sim
pip install -e .
```

Install the service extras when you want to run the simulator API:

```bash
pip install -e .[api]
```

For local development and tests:

```bash
pip install -e .[dev]
```

## CLI

```bash
vllm-sr-sim --version

vllm-sr-sim optimize \
  --cdf data/azure_cdf.json \
  --lam 200 --slo 500 --b-short 6144 \
  --verify-top 3 --n-sim-req 30000

vllm-sr-sim mixture-optimize \
  --scenario data/workload_mixture_burst.json \
  --lam 200 --slo 500 --b-short 6144 \
  --max-total-gpus 300 --out mixture-report.json

vllm-sr-sim whatif \
  --cdf data/azure_cdf.json \
  --lam-range 50 100 200 500 1000 \
  --slo 500 --b-short 6144

vllm-sr-sim serve --host 0.0.0.0 --port 8000
```

`mixture-optimize` keeps the analytical method on the repository's
`FleetOptimizer` class while evaluating versioned workload-archetype scenarios.
It reports aggregate-CDF, individual-archetype, nominal-mixture, drift/burst
window, robust worst-case, sensitivity, and infeasibility diagnostics without
changing production request-path routing.

`vllm-sr serve` also starts `vllm-sr-sim` by default as a sibling container on the shared runtime network so the dashboard can proxy it without rebuilding the router image.

## Layout

- `fleet_sim/`: simulation engine, optimizers, routing, hardware, workload, and service package
- `run_sim.py`: unified CLI entrypoint used by `vllm-sr-sim`
- `tests/`: simulator and service test coverage
- `data/`: reference workload traces used by the examples and dashboard integration
- `examples/`: sample scripts and multi-pool input files

## Docs

Long-form simulator docs are maintained in the repository website. Keep the package README focused on installation, CLI usage, and source layout.
