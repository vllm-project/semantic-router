# Local soak baseline harness

`run-soak-local.sh` runs long-running load with a memory timeseries against a
full local stack: `tools/mock-vllm` on `:8010` behind the
`bench/openai_fault_proxy.py` latency injector on `:8000`, the router with
metrics on `:9190` and pprof on `:6060`, and Envoy on `:8801`.

## Usage

- `make soak-local` — full baseline run
- `make soak-help` — harness flags and the Make knobs
- `SOAK_ARGS="..."` — extra flags forwarded to `bin/soak`
  (`SOAK_ARGS="-quick"` for a ~9-minute smoke run)
- `make soak-test` — vet and unit-test the harness (no running stack required)

## Derived router config

By default the script derives the router config at runtime from
`e2e/config/config.e2e.yaml` into `${SOAK_LOG_DIR}/config.soak.yaml`, with
three deltas:

- every model backend endpoint is rewritten to the fault proxy
  (`127.0.0.1:${SOAK_BACKEND_PORT}`)
- `global.services.observability.profiling` is enabled so the sampler can pull
  heap profiles
- `global.stores.response_cache` is disabled, because the fixed prompt
  rotation would otherwise short-circuit at a warm semantic cache

Like `config.e2e.yaml` it leaves `global.model_catalog` unset, so the real
Candle/CGO classifiers load and stay hot — that is the memory behaviour the
soak measures. Set `SOAK_CONFIG=<path>` to skip derivation and use your own
config.

## Derived envoy config

The harness fronts the router with `deploy/local/envoy.yaml`. That file is
never modified; the script derives a copy under `${SOAK_LOG_DIR}` with the
fail-closed Authorino `ext_authz` filter dropped (nothing starts Authorino
locally, so every request would be rejected with HTTP 403 before `ext_proc`)
and the listeners rebound to loopback.

## Outputs

Each run writes to `SOAK_OUT_DIR`:

- `timeseries.json` — the memory timeseries
- `summary.json` — includes `measured_requests`; a low success rate is
  recorded in `notes`
- `summary.bench` — benchstat-ready
- `profiles/` — heap profiles
- `router-config.yaml`, `envoy-config.yaml`, `run-env.txt` — for reproduction

## Known blind spot

The stack terminates at a buffered mock backend, so SSE/streaming response
handling is not covered.
