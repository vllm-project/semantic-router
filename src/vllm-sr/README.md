# vLLM Semantic Router CLI

`vllm-sr` configures and runs the local vLLM Semantic Router stack. It can also
validate or migrate config files, deploy the router Helm chart, inspect virtual
models and Recipes, and send test requests.

Full documentation: <https://vllm-sr.ai/docs/installation/>

## Install

```bash
pip install vllm-sr
vllm-sr --version
```

For CLI development:

```bash
cd src/vllm-sr
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Serve a Decision 1.0 model

`vllm-sr decision serve MODEL` starts one catalog model as a standalone SystemOne
service. An installed Decision-enabled CLI release selects the image for the
detected backend automatically; `--image` is an advanced development override.
See the
[Decision Runtime guide](https://vllm-sr.ai/docs/installation/decision-runtime/overview)
for launch and API examples, or the
[image build guide](https://github.com/vllm-project/semantic-router/tree/main/src/vllm-sr/decision_runtime/image) for source-checkout testing.

Decision Runtime supports Kai, Lex, Eos, Sol, Nox, and Lux on qualified ROCm
hardware. Kai, Lex, and Eos can also run on Linux CPU. See the
[model and backend table](https://vllm-sr.ai/docs/installation/decision-runtime/models)
for current support and model IDs.

`POST /v1/systemone` accepts one state with Noul, Choice, or Score questions.
`POST /v1/systemone/batches` applies the same questions to several states; it
is a Decision extension, not an official SystemOne SDK method. Every request
names the model served on that port. See
[API examples and SDK usage](https://vllm-sr.ai/docs/installation/decision-runtime/api) and
[launch options](https://vllm-sr.ai/docs/installation/decision-runtime/parameters).

## Start a local stack

The general `vllm-sr serve` stack requires Docker or Podman on Linux, macOS,
or WSL2. This platform list does not apply to Decision Runtime, whose current
qualified backends are listed above. A native Windows Python environment can
run config and catalog commands, but it cannot run the local container stack.

```bash
# Start Router, Envoy, Dashboard, and observability.
vllm-sr serve

# Use Podman.
vllm-sr serve --runtime podman

# Check the stack and open the Dashboard.
vllm-sr status
vllm-sr dashboard
```

The Dashboard is available at <http://localhost:8700>. The routed
OpenAI-compatible listener uses the first port in `config.yaml` (`8899` in the
reference config).

For local `serve`, `listeners[].address` controls the host port publication.
Use `127.0.0.1` or `::1` for host-only access. Envoy listens on the container
bridge interface so both the published port and Dashboard can reach it; this
keeps the host loopback restriction, including after Dashboard config saves.
Standalone `config envoy` generation retains the configured listener address.

`vllm-sr serve` starts the routing stack. It does not start the physical LLM
backends referenced by `providers.models`; those endpoints must already be
running and reachable.

Useful lifecycle commands:

```bash
vllm-sr logs router
vllm-sr logs envoy
vllm-sr logs dashboard
vllm-sr stop
```

Add `--minimal` to run Router and Envoy without Dashboard or observability. Add
`--readonly` to keep Dashboard available without config editing.

Local startup waits up to 1800 seconds for readiness after containers start.
Use `--startup-timeout SECONDS` with a positive integer when model loading or
GPU compilation needs a different budget, for example
`vllm-sr serve --startup-timeout 7200`. This Docker-only option also covers
Dashboard readiness during first-run setup. If the wait expires, the CLI exits
with an error and leaves containers running for `vllm-sr status` and
`vllm-sr logs router`; use `vllm-sr stop` to stop them. Request inference
deadlines are configured separately.

## Test routing

`route preview` reports which signals, decision, algorithm, and plugins matched without
calling the selected model backend:

```bash
vllm-sr route preview --prompt "Explain inflation in plain English."
vllm-sr route preview --prompt "Explain inflation in plain English." --json
vllm-sr route preview \
  --model vllm-sr/mom-v1-blend \
  --prompt "Summarize this architecture plan." \
  --json
```

Use `--messages` for an OpenAI-style messages array and `--endpoint` when the
Router management API is not at `http://localhost:8080`:

```bash
vllm-sr route preview \
  --messages '[{"role":"user","content":"Explain inflation."}]' \
  --endpoint http://localhost:8080
```

`request chat` sends a real one-shot completion through the routed listener. It uses
`vllm-sr/auto` unless `--model` is set:

```bash
vllm-sr request chat "Hello"
vllm-sr request chat --model my-virtual-model --json "Hello"
vllm-sr request chat --base-url https://gateway.example.com "Hello"
```

`--base-url` must point to an OpenAI-compatible routed endpoint, such as an
ingress or port-forwarded gateway. It is not the Router management API used by
`route preview` and `storage vector-stores`.

## Choose a configuration

The CLI reads canonical v0.3 YAML with
`version/listeners/providers/routing/global`. Author a file directly, start
from a [maintained Recipe](../../config/recipes/README.md), or fork a bundled
virtual model.

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

Route policy lives in `routing.decisions[]`. For example, this decision
fragment defines a final static fallback; merge it into a complete config that
declares `local-model` under `providers.models` and `routing.modelCards`:

```yaml
routing:
  decisions:
    - name: local-fallback
      description: Handle requests that did not match an earlier decision.
      priority: 0
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: local-model
      algorithm:
        type: static
```

`vllm-sr init` was removed in v0.3. For older files or supported external
provider configs, use the explicit conversion commands:

```bash
vllm-sr config migrate --config old-config.yaml
vllm-sr config import \
  --from openclaw \
  --source openclaw.json \
  --target config.yaml
```

The current field reference is generated in the
[configuration guide](https://vllm-sr.ai/docs/installation/configuration/).
Focused examples live under [`config/fragments/`](../../config/fragments/).
Use those sources instead of copying plugin or algorithm schemas from this
package README.

Keep credentials out of YAML. Reference environment variables and authorize
Recipe-specific variables explicitly:

```bash
export PROVIDER_API_KEY=...
vllm-sr serve --config recipe.yaml --recipe-env PROVIDER_API_KEY
```

## Connect Models and build Mixture-of-Models

Run `vllm-sr serve`, then use Dashboard to connect provider Models, choose a
built-in or custom Recipe, and assign Models to its decisions. Dashboard keeps
provider connection details separate from reusable routing policy and exposes
the published names through `/v1/models`.

For source-controlled deployments, validate and serve one complete user-owned
configuration:

```bash
vllm-sr config validate --config my-models.yaml
vllm-sr serve --config my-models.yaml
```

## Evaluate single models and MoM

`vllm-sr benchmark` and Dashboard **Evaluation** use sr-bench 1.0. Both clients
share a durable worker, frozen datasets, run IDs, per-benchmark quality, token
buckets, costs, latency and wall time. The old evaluation command/API is removed.

```bash
vllm-sr benchmark catalog
vllm-sr benchmark setup --benchmark all
vllm-sr benchmark dataset prepare --benchmark mmlu-pro --profile quick
vllm-sr benchmark dataset preparations
vllm-sr benchmark plan --manifest candidate.json --output frozen.json
vllm-sr benchmark run --manifest frozen.json --detach
vllm-sr benchmark report RUN_ID
vllm-sr benchmark compare BASELINE_ID CANDIDATE_ID
```

Dataset preparation uses the shared service by default, matching **Evaluation →
Datasets → Prepare dataset** in Dashboard. The worker installs missing data
preparation dependencies, downloads the pinned source and publishes a frozen
dataset. It does not start a model, build a grading sandbox or run an evaluation.
Gated sources need their access approval and credentials in the worker environment.
The CLI waits and prints the manifest; `dataset prepare --no-wait` returns a job
for `dataset preparations PREPARATION_ID`. Closing either client leaves the job
running. `dataset options` lists sources, profiles and access requirements.

With `--url`, preparation writes to the selected worker's store. Local source
files and history options require explicit `dataset prepare --local`; this mode
cannot be combined with `--url` or `SR_BENCH_URL` and does not upload files.

The managed core worker survives Dashboard/config reloads. An image-only upgrade
replaces an idle worker while preserving its journal; active runs must finish or
be cancelled, and dataset preparations must finish first. Set `VLLM_SR_BENCH_PORT`
for both `serve` and `benchmark` when the default `8090 + stack port offset` host
port is occupied. The override is an
absolute loopback host port and does not change Dashboard's internal connection.
For optional code and
agent harnesses, prepare a dedicated worker and select it with `SR_BENCH_URL`;
this suppresses managed worker creation. Keep service/model credential values
in the worker environment. Register targets on its host with
`benchmark target register --file targets.json` for Dashboard selection.

Quick/dev sets enable bounded tuning; standard is disjoint holdout. Preview has
no capability score, and replay is a saved-answer estimate. Priced live runs are
required for a measured savings claim. Unknown usage is not zero, and spend
reservations are not a universal provider-enforced hard USD cap. See the
[sr-bench guide](../../website/docs/benchmarking/sr-bench.md) for setup, manifests,
all nine adapters, failure recovery, regrading and dev-only training export.

## Deploy to Kubernetes

The Kubernetes target installs or upgrades the Helm release:

```bash
vllm-sr serve \
  --target k8s \
  --profile dev \
  --namespace semantic-router \
  --config config.yaml

vllm-sr status --target k8s --namespace semantic-router
vllm-sr logs router --target k8s --namespace semantic-router -f
vllm-sr stop --target k8s --namespace semantic-router
```

Kubernetes requires a complete, non-empty config. The CLI does not merge local
Docker defaults or sample routes into it. Credential references are stored in
a release-scoped Secret, and literal credentials or credential-bearing URLs
are rejected.

`--platform amd` and `--platform nvidia` are local-container shortcuts. On
Kubernetes, select GPU images, resources, and device plugins through Helm
values, a deployment profile, or the operator.

See [Kubernetes installation](https://vllm-sr.ai/docs/installation/k8s/) for
gateway, profile, and production guidance.

## Inspect vector stores

`storage vector-stores` reads vector stores from the Router management API. It
does not create, modify, or delete stores.

```bash
vllm-sr storage vector-stores
vllm-sr storage vector-stores --endpoint http://router.example.com:8080
```

The Router must be running with a vector-store backend enabled. `--endpoint`
points to the management API, not the routed inference listener.

## Local ports and state

Default ports in the reference local stack are:

| Service | Port | Purpose |
| --- | ---: | --- |
| Dashboard | `8700` | Configuration, Playground, and embedded observability |
| Routed inference listener | `8899` | OpenAI-compatible model requests |
| Router management API | `8080` | Eval, config, replay, and vector-store APIs |
| Router metrics | `9190` | Prometheus metrics |
| Jaeger | `16686` | Trace UI |
| Prometheus | `9090` | Metrics storage and queries |

Listener and management ports can be changed in YAML. Local Dashboard data is
stored under `.vllm-sr/dashboard-data/` and survives `stop` unless that
workspace directory is removed.

To run independent stacks from multiple worktrees, use a distinct name and
port offset on every lifecycle command:

```bash
export VLLM_SR_STACK_NAME=lane-b
export VLLM_SR_PORT_OFFSET=200
vllm-sr serve
vllm-sr status
vllm-sr stop
```

## Troubleshooting

- `route preview` and `storage vector-stores` use the Router management API,
  normally port `8080`.
- `request chat` uses the routed inference listener from `config.yaml`, normally
  port `8899`.
- A healthy Router and Envoy do not prove that an external model backend can
  generate. Use Dashboard **Verify** or `chat` to test the backend path.
- If a lifecycle command reports that the stack is busy, let the active
  `serve` or `stop` finish and retry.
- Set `NO_COLOR=1` for plain CLI output. JSON modes keep stdout free of status
  messages so it can be consumed by scripts.

Run `vllm-sr COMMAND --help` for command-specific options. For installation,
security, configuration, and operations, use the
[website documentation](https://vllm-sr.ai/docs/).

## License

Apache 2.0
