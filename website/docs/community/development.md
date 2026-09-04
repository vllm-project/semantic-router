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
make agent-bootstrap
```

Individual subprojects may have additional requirements. Do not install a
repository-root `requirements.txt`; none exists. Use the dependency file or
package metadata beside the component you are changing.

## Build and run locally

```bash
make vllm-sr-dev
vllm-sr serve --image-pull-policy never
```

The build installs the editable `vllm-sr` CLI and creates local Router,
Dashboard, Envoy, and Fleet Sim images. `--image-pull-policy never` ensures the
run uses those local images.

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
vllm-sr serve --image-pull-policy never --platform amd
```

## Select the right tests

Start with the repository report for your changed files:

```bash
make agent-report ENV=cpu CHANGED_FILES="path/one,path/two"
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

Use the affected E2E selector when a change is visible through startup,
routing, an API, a deployment profile, or another live path:

```bash
make agent-e2e-affected CHANGED_FILES="path/one,path/two"
```

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
- Run `vllm-sr validate --config <file>` before debugging a configuration at
  runtime.
- See [Common Errors](/docs/troubleshooting/common-errors) and
  [Container Connectivity](/docs/troubleshooting/container-connectivity) for
  startup and network failures.
