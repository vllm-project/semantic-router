---
sidebar_position: 1
title: Quickstart
description: Install vLLM Semantic Router, start the local stack, configure a model, and send a first request.
---

# Quickstart

This guide starts the local Docker stack and sends a request through the Router.
The Router itself runs on CPU; the model backend can be local or remote.

## Requirements

- Python 3.10 or newer
- Docker
- Linux, macOS, or WSL2 on Windows

Native Windows Python can run configuration and validation commands, but the
local Docker serving workflow requires WSL2 or another Linux environment.

## Install

### One-line installer

On macOS or Linux:

```bash
curl -fsSL https://vllm-sr.ai/install.sh | \
  bash -s -- --channel dev
```

The installer creates an isolated CLI environment, adds a launcher under
`~/.local/bin`, prepares Docker, and starts `vllm-sr serve` unless you opt out.
The development channel resolves and pins the newest published `.dev` package
so the Quickstart follows current project capabilities. It prints the
Dashboard URL and, on a remote host, an SSH tunnel hint.

### Install with pip

```bash
python -m venv vsr
source vsr/bin/activate
VLLM_SR_DEV_VERSION="$(
  python -m pip index versions --pre vllm-sr 2>/dev/null |
  awk -F': ' '/^Available versions:/ {
    count = split($2, versions, ", ")
    for (i = 1; i <= count; i++) {
      if (versions[i] ~ /^[0-9]+([.][0-9]+)*[.]dev[0-9]+$/) {
        print versions[i]
        exit
      }
    }
  }'
)"
test -n "${VLLM_SR_DEV_VERSION}" || {
  echo "No published vllm-sr development package found" >&2
  exit 1
}
python -m pip install --upgrade "vllm-sr==${VLLM_SR_DEV_VERSION}"
vllm-sr --version
```

The resolver selects the first published `.dev` version from pip's ordered
version list and installs that exact package. A bare `--pre` is not sufficient
when the stable release has a higher PEP 440 precedence than development builds
with the same base version. For production, select and pin an explicit stable
version after reviewing its release notes and compatibility requirements.

## Open or start the local stack

The one-line installer starts the stack automatically; continue to the
Dashboard URL it prints. If you installed with pip or passed `--no-launch`, run
the following command from the directory where you want to keep `config.yaml`
and local runtime state:

```bash
vllm-sr serve
```

On the first run, an empty workspace starts the Dashboard in setup mode. Open
[http://localhost:8700](http://localhost:8700), then:

1. add one or more model endpoints;
2. choose a routing preset or a single-model baseline; and
3. activate the generated configuration.

Activation writes `config.yaml` and starts the inference listener. The local
stack exposes Envoy at `http://localhost:8899` by default.

:::tip[Want a local model?]
Follow [Local model with Ollama](ollama) if you want a simple local backend
without setting up vLLM or a GPU environment.
:::

## Send a request

After activation:

```bash
curl http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Use the model name shown by your active configuration. A virtual model resolves
to its recipe; a configured physical model name is sent directly to that
backend.

## Operate the stack

```bash
vllm-sr status
vllm-sr logs router
vllm-sr logs envoy -f
vllm-sr dashboard
vllm-sr stop
```

The Dashboard can show bounded Router, Envoy, and Dashboard logs for supported
local stacks. Log access is permission-controlled, and service output may
contain tenant or credential-adjacent data. Treat both Dashboard log access and
the local log directory as sensitive.

## Start from YAML

If you already have a complete canonical config:

```bash
vllm-sr validate --config config.yaml
vllm-sr serve --config config.yaml
```

Environment references such as `${MODEL_API_KEY}` are resolved from the launch
environment. Do not put literal credentials in a config that will be shared or
committed.

For older configs, use:

```bash
vllm-sr config migrate --config old-config.yaml
```

## Next

- [Choose a Deployment](deployment-options) for local, GPU, Kubernetes, and
  gateway options.
- [Configuration](configuration) for canonical YAML, recipes, and environment
  bindings.
- [Models, Entrypoints, and Serving](../tutorials/global/models-entrypoints-serving)
  for built-in models, backend binding, and the complete CLI workflow.
- [Why Semantic Routing](../overview/goals) for the design goals.
- [Routing Pipeline](../overview/signal-driven-decisions) for signals,
  decisions, algorithms, and plugins.
- [Troubleshooting](../troubleshooting/common-errors) when the stack does not
  become ready.

For support, open a
[GitHub issue](https://github.com/vllm-project/semantic-router/issues) or join
the `#semantic-router` channel in vLLM Slack.
