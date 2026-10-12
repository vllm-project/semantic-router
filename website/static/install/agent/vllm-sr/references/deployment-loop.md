# Deployment options

Use this reference with the [main skill](https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md) for GPU platforms,
Envoy, engine mode, Kubernetes, Dashboard access, a second stack, upgrades and
removal. Installed `vllm-sr serve --help` lists every option, grouped by where
it applies.

## GPU platforms

`--platform rocm` selects the `vllm-sr-rocm` image (about 20 GB on disk), passes
`/dev/kfd` and `/dev/dri` into the Router container, and runs the Router's own
models on the GPU by default. AMD Instinct MI300X and MI325X are validated.
`--platform cuda` selects `vllm-sr-cuda` and passes `--gpus all`; it needs
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
- The default decision model, model consumers and native grants determine
  which resources load. Inspect `vllm-sr instance models` for actual ready
  replicas; an exposed frontend or healthy process alone does not prove model
  readiness. Use a bounded inference request to verify a chosen model.
- `--platform auto` inspects the actual container host. If both CUDA and ROCm
  are available, select one explicitly. Existing visibility masks are honored;
  automatic discovery never guesses remote daemon hardware from the CLI host.

## Engine mode and replicas

`vllm-sr serve MODEL --engine` (or `-e`) starts the persistent instance frontend,
Dashboard and managed model pool with routing disabled. Without `-e`, every
invocation starts Router mode, including an existing Engine config. MODEL alone
does not choose Engine mode. Dashboard observes this startup choice and manages
model resources independently.

The frontend publishes `POST /v1/systemone`, its `/v1/decisions` alias, and
`GET /v1/systemone/models` according to `listeners[].systemone.models` and the
listener's API keys. Initial Engine bootstrap grants the chosen model on port
8899. Existing listener grants never widen when selecting another model.
Classify, embeddings, rerank and bundle endpoints belong to direct `vllm-srun`
workers and are not automatically exposed by this frontend. Use `vllm-sr stop`
with the same stack identity to stop the persistent stack.

MODEL replaces only the configured default judgment deployment's artifact.
Other model flags are optional: `--runtime-profile` preserves the configured
value unless supplied; `--revision` resolves a branch, tag or full commit once
before startup and records the immutable pin for every replica. Built-in
models use their release pin by default. Bare serve keeps the configured
default deployment, or starts with Vela 2.0 0.3B for a new config.

`-dp N` (`--data-parallel-size N`) writes canonical replicas. A new placement
uses the first N available GPUs; explicitly use `--device-ids 0` to colocate
all N workers on one host GPU, or `--device-ids 0,1` for two workers across two
GPUs. Device IDs do not change the visible mask. Existing masks map host IDs
to runtime ordinals, and existing configured placements scale in their saved
order. Kubernetes uses pod allocation ordinals in config, not host device IDs.
See the [model runtime Quickstart](https://vllm-sr.ai/docs/model-runtime/quickstart).

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
Add `--context` to name a kubectl context, `--platform rocm` or `cuda` for the
GPU image and resources derived from canonical placement (`amd.com/gpu` or
`nvidia.com/gpu`),
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
