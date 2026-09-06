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

Local `serve` requires Docker or Podman on Linux, macOS, or WSL2. A native
Windows Python environment can run config and catalog commands, but it cannot
run the local container stack.

## Start a local stack

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

## Test routing

`eval` reports which signals, decision, algorithm, and plugins matched without
calling the selected model backend:

```bash
vllm-sr eval --prompt "Explain inflation in plain English."
vllm-sr eval --prompt "Explain inflation in plain English." --json
vllm-sr eval \
  --model vllm-sr/mom-v1-blend \
  --prompt "Summarize this architecture plan." \
  --json
```

Use `--messages` for an OpenAI-style messages array and `--endpoint` when the
Router management API is not at `http://localhost:8080`:

```bash
vllm-sr eval \
  --messages '[{"role":"user","content":"Explain inflation."}]' \
  --endpoint http://localhost:8080
```

`chat` sends a real one-shot completion through the routed listener. It uses
`vllm-sr/auto` unless `--model` is set:

```bash
vllm-sr chat "Hello"
vllm-sr chat --model my-virtual-model --json "Hello"
vllm-sr chat --base-url https://gateway.example.com "Hello"
```

`--base-url` must point to an OpenAI-compatible routed endpoint, such as an
ingress or port-forwarded gateway. It is not the Router management API used by
`eval` and `rag list`.

## Choose a configuration

The CLI reads canonical v0.3 YAML with
`version/listeners/providers/routing/global`. Author a file directly, start
from a [maintained Recipe](../../config/recipes/README.md), or fork a bundled
virtual model.

```bash
vllm-sr validate --config config.yaml
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
vllm-sr validate --config my-models.yaml
vllm-sr serve --config my-models.yaml
```

To evaluate concurrently running baseline and candidate deployments from one
Dashboard, point `EVALUATION_DEPLOYMENTS_DIR` at the strict, read-only
`evaluation-deployments.v1` registry described in the
[Evaluation Plane guide](../../website/docs/benchmarking/evaluation-plane.md#address-baseline-and-candidate-deployments-together),
then use the same `vllm-sr serve` command. The CLI mounts that directory into
Dashboard only; Router and Envoy do not inherit it. Leaving the variable unset
preserves the current single-runtime behavior.

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

`rag list` reads vector stores created through the Router's OpenAI-compatible
Vector Stores API. It does not create, modify, or delete stores.

```bash
vllm-sr rag list
vllm-sr rag list --endpoint http://router.example.com:8080
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

- `eval` and `rag list` use the Router management API, normally port `8080`.
- `chat` uses the routed inference listener from `config.yaml`, normally port
  `8899`.
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
