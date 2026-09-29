# Development Guide

Use the repository's local image workflow for changes that affect Router or CLI
behavior. It builds the same service topology that contributors exercise in
local validation.

## Prerequisites

- Git
- GNU Make
- Docker or Podman
- Python 3.10 or newer for the CLI, tests, training, and simulator tools

The repository bootstrap target creates its Python environment and installs the
tooling used by the validation harness:

```bash
make harness-bootstrap
```

Individual subprojects may have additional requirements. Do not install a
repository-root `requirements.txt`; none exists. Use the dependency file or
package metadata beside the component you are changing.

## Build and run locally

```bash
make vllm-sr-dev
VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest \
  vllm-sr serve --image-pull-policy never
```

The build installs the editable `vllm-sr` CLI, builds Router and Dashboard
images tagged `latest`, and ensures the official Envoy image is available.
Set `VLLM_SR_IMAGE` explicitly because an editable CLI installation with a
stable package version defaults to that release's image tag. The CLI derives
the official Dashboard image with the same tag; `--image-pull-policy never`
prevents pulling missing images.

Useful lifecycle commands:

```bash
vllm-sr status
vllm-sr logs router
vllm-sr logs envoy -f
vllm-sr dashboard
vllm-sr stop
```

For ROCm-specific work:

```bash
make vllm-sr-dev VLLM_SR_PLATFORM=amd
VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:latest \
  vllm-sr serve --image-pull-policy never --platform amd
```

If you customize `DOCKER_TAG`, `DOCKER_REGISTRY`, or the Make image variables,
pass the actual built images to `serve` through `VLLM_SR_IMAGE` and, when needed,
`VLLM_SR_DASHBOARD_IMAGE`. The build's completion message prints a startup
command with the selected images.

## Select the right tests

Inspect the repository facts, then run the owning domains' checks:

```bash
make impact ENV=cpu CHANGED_FILES="path/one path/two"
make check CHANGED_FILES="path/one path/two"
```

Common targeted suites include:

```bash
# Router and native bindings
make test-semantic-router
make test-binding

# Classifiers
make test-category-classifier
make test-pii-classifier
make test-jailbreak-classifier

# Python CLI
make vllm-sr-test

# Fleet simulator
make vllm-sr-sim-test
```

Select integration or E2E explicitly when a change is visible through startup,
routing, an API, a deployment profile, or another live path:

```bash
make verify DOMAIN=<domain>
make verify PROFILE=<profile>
```

## Test backends

Running the provider mocker from source requires Python 3.11 or newer.

Use the shared provider mocker for deterministic protocol, routing and fault
checks. It serves OpenAI Chat Completions, Responses, Anthropic Messages and
image fixtures from one lightweight service:

```bash
make test-provider-mocker
make docker-run-provider-mocker
# Or run the service directly in its isolated Python environment:
make start-provider-mocker
```

`PROVIDER_MOCKER_IMAGE` selects an existing image for reuse. Without that setting,
the Docker target builds the local service. The mocker is maintained separately
from product releases. Its image tag identifies a content hash of the runtime
package, dependency lock, Dockerfile and `.dockerignore`; changing docs or tests
alone reuses the image. CI resolves that tag to an image digest and passes the
same artifact to all consumers. Only changes to those runtime inputs publish a
new helper image.

For a test that needs actual generation, the optional tiny-model runner uses
`Qwen/Qwen3-0.6B` and the upstream llama.cpp CPU server. It pins the image digest,
model revision and checksum; downloads the Q8_0 weights into an ignored cache;
and disables thinking. It does not build another inference image:

```bash
make tiny-model-smoke  # health, real text, SSE termination and stop sequences
make tiny-model-serve # foreground real backend on localhost:8000
```

The model smoke has bounded CPU, memory, context and output limits. Use a separate
terminal for the running backend and route requests through `vllm-sr serve` when
validating Router behavior. Protocol edge cases stay in the deterministic suite.

## Validate a local stack

The configured listener is the client-facing endpoint. For the setup generated
by an empty workspace it is `http://localhost:8899`:

```bash
curl -sS http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Use the virtual model name from your active configuration. `vllm-sr status`
shows the stack and published ports when you use a custom listener or port
offset.

## Debugging

- Inspect component logs with `vllm-sr logs <service>` before relying on
  container names.
- Set `RUST_LOG=debug` for native-library diagnostics.
- Set `SR_LOG_LEVEL=debug` for Router diagnostics.
- Run `vllm-sr config validate --config <file>` before debugging a configuration at
  runtime.
- See [Common Errors](/docs/troubleshooting/common-errors) and
  [Container Connectivity](/docs/troubleshooting/container-connectivity) for
  startup and network failures.
