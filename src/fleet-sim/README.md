# vllm-sr-sim

`vllm-sr-sim` is the maintained standalone fleet simulator for this repository. It sizes heterogeneous GPU fleets, evaluates routing strategies, and exposes an optional HTTP API for automation and custom clients.

Long-form guides live on the project website:

- [Overview](https://vllm-sr.ai/docs/fleet-sim/overview)
- [Getting started](https://vllm-sr.ai/docs/fleet-sim/getting-started)
- [Use cases](https://vllm-sr.ai/docs/fleet-sim/use-cases)

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

vllm-sr-sim whatif \
  --cdf data/azure_cdf.json \
  --lam-range 50 100 200 500 1000 \
  --slo 500 --b-short 6144

vllm-sr-sim serve --host 0.0.0.0 --port 8000
```

Fleet Sim is not started by `vllm-sr serve` and is not embedded in the Semantic Router dashboard. Run the CLI or HTTP service explicitly when you need a planning study.

## Layout

- `fleet_sim/`: simulation engine, optimizers, routing, hardware, workload, and service package
- `run_sim.py`: unified CLI entrypoint used by `vllm-sr-sim`
- `tests/`: simulator and service test coverage
- `data/`: reference workload traces used by the examples
- `examples/`: sample scripts and multi-pool input files
