# Deployment options

Use this reference with the [main skill](https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md) for GPU platforms,
Envoy, engine mode, Kubernetes, Dashboard access, a second stack, upgrades and
removal. Installed `vllm-sr serve --help` lists every option, grouped by where
it applies.

## GPU platforms

`--platform amd` selects the `vllm-sr-rocm` image (about 20 GB on disk), passes
`/dev/kfd` and `/dev/dri` into the Router container, and runs the Router's own
models on the GPU by default. AMD Instinct MI300X and MI325X are validated.
`--platform nvidia` selects `vllm-sr-cuda` and passes `--gpus all`; it needs
the NVIDIA Container Toolkit and works, but is not yet validated. On macOS the
Docker target runs on the CPU only.

- The Router container sees every GPU. To keep some for the user's own model
  server, set `VLLM_SR_AMD_ROUTER_VISIBLE_DEVICES` (it becomes
  `ROCR_VISIBLE_DEVICES`, for example `1`) before `serve`.
- A model the Router runs is a `global.model_catalog.deployments` entry with
  `provider: model_runtime` and `device: rocm:N`, `cuda:N`, `cpu` or `auto` (the
  first validated GPU with enough free memory, else the CPU). A named GPU must
  exist, or the model fails to load with the reason. See
  [Choose a model](https://vllm-sr.ai/docs/model-runtime/choose-a-model).
- `config init`'s configuration loads no Router model, so `--platform amd`
  alone exercises nothing; the main skill's step 6 checks the GPU in engine
  mode instead.
- Router models load after `serve` returns, and `/ready` is green before they
  answer. Until a model signal is available, requests take the fallback route.
  `vllm-sr route preview --model vllm-sr/auto --prompt '…' --trace --json`
  shows each condition in `eval_trace`; a leaf with `"state": "unknown"` and a
  `signal_error` such as `decision_unavailable` is still loading. Repeat until
  it isn't. With the model cached that took about 11 s; the first start also
  downloads it into `models/` next to `config.yaml`.

## Engine mode

`vllm-sr serve MODEL` serves Router models with the model runtime alone, without
the Router: decision models (`POST /v1/decisions`), classifiers
(`/v1/classify`), embedders (`/v1/embeddings`) and rerankers (`/v1/rerank`),
with `/health` and `/v1/models`. It runs in the foreground in a container named
`vllm-sr-engine-PORT`, published on `127.0.0.1:8100` unless `--host` or
`--port` say otherwise; Ctrl-C or SIGINT stops it and removes the container.
`--device` takes `cpu`, `rocm[:N]` with `--platform amd` or `cuda[:N]` with
`--platform nvidia`. Downloads persist in `~/.cache/vllm-sr/models`. For the
Router container to reach an engine-mode server, start it with
`--host 0.0.0.0` and use `http://host.docker.internal:PORT` as the deployment's
`endpoint`. See the
[model runtime Quickstart](https://vllm-sr.ai/docs/model-runtime/quickstart).

## Envoy in front: `--gateway extproc`

Standalone serves the same routing core without Envoy. Use `extproc` for
Envoy's rate limiting, mTLS, JWT or OIDC and advanced route matching, for an
Envoy-based gateway the user runs, or for per-user authorization when no
listener trusts identity headers
([What needs extproc](https://vllm-sr.ai/docs/installation/gateway-modes#what-needs-extproc)).

On Docker, switching is a restart on the active configuration: `vllm-sr serve
--config config.yaml --gateway extproc`, and plain `vllm-sr serve --config
config.yaml` to come back; each took about 5 s. With Envoy, responses carry
`server: envoy` and `vllm-sr logs envoy` works; in standalone it explains that
there is no Envoy container.

## Kubernetes

Use a complete configuration (main skill, step 4); setup mode isn't available
on Kubernetes. Model endpoints must be reachable from the pods: a Service DNS
name or an address on the cluster's network, not `host.docker.internal`.

You need `kubectl` with a context for the target cluster, and `helm` 3. A pip
or curl install carries no chart: `serve --target kubernetes` installs
`deploy/helm/semantic-router` when the current directory has one, and otherwise
the published chart that matches the CLI (`0.0.0-latest` for a dev build, the
release number for a release); `--chart-dir` names another. `status`, `logs`,
`stop` and `dashboard` need no chart.

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --target kubernetes --config config.yaml --namespace NAMESPACE
```

On Kubernetes, set the listener's `address` to `0.0.0.0`: the Service, not the
host, publishes it.
Add `--context` to name a kubectl context, `--platform amd` or `nvidia` for the
GPU image and a request for one `amd.com/gpu` or `nvidia.com/gpu`,
`--gateway extproc` for an Envoy-based gateway in the cluster, and `--minimal`
to leave out the observability stack (Prometheus, Grafana, Jaeger and their
exporters). The Router keeps its models on a persistent volume of storage class
`standard`; `--profile dev` asks for the cluster's default class instead (and
enables the Dashboard). Check `kubectl get storageclass` first. `serve` runs
`helm upgrade --install semantic-router` with `--wait` and `--atomic`, so one
pod that doesn't become ready rolls the release back, and prints a
port-forward command:

```bash
vllm-sr status --target kubernetes --namespace NAMESPACE
kubectl port-forward -n NAMESPACE svc/semantic-router 8899:8899
```

With the port forward running, the main skill's checks 2 and 4 apply
unchanged. `vllm-sr stop --target kubernetes --namespace NAMESPACE` uninstalls
the release. Later Helm upgrades keep a document the Dashboard or Router API
saved; see
[Configuration Workflows](https://vllm-sr.ai/docs/installation/configuration-workflows#helm).

## Dashboard and access

The Dashboard and every other port publish on 127.0.0.1. The listener's own
`address` decides the inference port's publication. To publish the Dashboard
beyond loopback, set `VLLM_SR_DASHBOARD_HOST_BIND=0.0.0.0` together with
`DASHBOARD_ADMIN_EMAIL` and `DASHBOARD_ADMIN_PASSWORD` (or an explicit
`DASHBOARD_ALLOW_OPEN_BOOTSTRAP=true`); `serve` refuses otherwise. On loopback,
the first visitor creates the administrator unless those variables were set
before `serve`. They don't reset existing users.
`VLLM_SR_OBSERVABILITY_HOST_BIND=0.0.0.0` publishes Jaeger, Prometheus and
Grafana beyond loopback; Grafana's login is `admin/admin` until someone changes
it. Inspect the actual bindings
with `docker ps` before opening access; a tunnel doesn't close a port that is
already public. `--minimal` starts no Dashboard and no observability stack.

Dashboard sessions and inference API credentials are separate. Hand access
details and credentials over through the user's chosen private channel.

## Stack identity

One host runs one stack per `VLLM_SR_STACK_NAME` (default `vllm-sr`; it names
the containers). For a second stack, set a different name and a
`VLLM_SR_PORT_OFFSET`, which moves every host port by the same amount, and use
both values for every later command of that stack. An inherited
`VLLM_SR_STATE_ROOT_DIR` can load another stack's active configuration: point
it at the directory that contains `.vllm-sr`, usually the config directory, or
leave it unset.

## Add or replace a backend

1. Check the backend's protocol, model name and required capabilities with a
   bounded direct request, keeping credentials in environment variables.
2. Connect the provider, its model card and a decision; a provider or a card
   alone doesn't make a model routable.
3. Plan and apply the change ([Configuration](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md)), then
   preview and probe the affected entrypoints. A new backend or listener
   topology can require a restart.

Size Router models apart from the generation backends: resident weights, the
memory for the intended context and concurrency, and disk. Encoder input
limits, candidate eligibility and the backend's context are separate; see
[token boundaries](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md#token-boundaries-and-public-errors).

## Upgrade and removal

- **Upgrade:** rerun the installer with the same channel (the dev channel
  installs over stable in place), then `vllm-sr serve --config config.yaml`,
  which pulls the matching images and restarts the stack on its active
  configuration.
- **Stop:** `vllm-sr stop` removes the containers and networks and keeps
  `config.yaml`, `.vllm-sr/`, `models/`, the data volumes and the images.
- **Remove**, only when the user asks, because it deletes their data: after
  `vllm-sr stop`, delete the stack's volumes (`docker volume ls --filter
  name=vllm-sr`), the config directory's `.vllm-sr/` and `models/`, the images
  (`docker image ls --filter 'reference=ghcr.io/vllm-project/semantic-router/*'`), the engine
  cache `~/.cache/vllm-sr`, and the CLI (`~/.local/share/vllm-sr` and
  `~/.local/bin/vllm-sr`).
