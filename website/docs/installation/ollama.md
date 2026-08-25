---
sidebar_position: 3
description: Step-by-step guide to serve a local model with Ollama and connect it to vLLM Semantic Router through the setup dashboard or YAML config.
---

# Configure models with Ollama

[Ollama](https://ollama.com/) is a simple way to run local LLMs without a full vLLM or GPU stack. Ollama exposes an OpenAI-compatible API on port `11434`, which Semantic Router can use as a model backend during first-run setup or in hand-authored YAML.

This guide walks through:

1. Installing Ollama and pulling a model on your host
2. Making the Ollama API reachable from Docker
3. Registering the model in the Semantic Router setup dashboard
4. Activating the config and sending a test request

:::tip
Semantic Router runs in Docker during `vllm-sr serve`. The name
`host.docker.internal` resolves the host from the container, but it does not
make a loopback-only Ollama server reachable. Complete the bind-address step
below before starting Semantic Router.
:::

## Prerequisites

- Semantic Router installed and runnable with [`vllm-sr serve`](/docs/installation) (Linux, macOS, or WSL2 with Docker)
- Ollama installed on the **same machine** that runs Docker
- Enough disk space for at least one model (for example, `llama3.2:3b` is about 2 GB)

## 1. Install Ollama

Install Ollama from [ollama.com/download](https://ollama.com/download) for your platform, then confirm the CLI is available:

```bash
ollama --version
```

On Linux you can also use the install script:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Ollama starts a background service automatically. It listens on
`http://127.0.0.1:11434` by default, which is reachable from the host but not
from a Docker container.

## 2. Make Ollama reachable from Docker

Set `OLLAMA_HOST=0.0.0.0:11434`, then restart Ollama. The exact way to set the
environment variable depends on how Ollama is installed.

On Linux with the standard systemd service:

```bash
sudo systemctl edit ollama.service
```

Add the following override, save it, and restart the service:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

On macOS, quit the Ollama application, set the launch environment, and reopen
the application:

```bash
launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
```

On Windows, quit Ollama, add the user environment variable `OLLAMA_HOST` with
the value `0.0.0.0:11434`, and restart Ollama from the Start menu. The
[Ollama server FAQ](https://docs.ollama.com/faq#how-do-i-configure-ollama-server)
has the current platform-specific steps.

For WSL, follow the Windows steps when Ollama is the Windows application, or
the Linux steps when the server itself runs inside the WSL distribution.

:::warning
Ollama's local API does not require authentication. Binding to `0.0.0.0` can
give other hosts access to model listing and generation. Restrict TCP port
`11434` to the container bridge, host gateway, or other trusted local sources,
and never publish it to an untrusted network.
:::

## 3. Pull a model

Pull a model tag from the [Ollama library](https://ollama.com/library). This example uses `llama3.2:3b`, a small general-purpose model that works well for local testing:

```bash
ollama pull llama3.2:3b
```

List locally available models:

```bash
ollama list
```

![Pull an Ollama model and confirm it is available locally](/img/installation/ollama/ollama-pull-and-list.png)

:::note
Use the **exact Ollama tag** (for example `llama3.2:3b`, `qwen2.5-coder:7b`) as the model name in Semantic Router. The router forwards that name to Ollama unchanged.
:::

## 4. Verify Ollama is serving

Before opening the Semantic Router dashboard, confirm Ollama responds on the host:

```bash
curl http://localhost:11434/v1/models
```

Send a quick chat completion:

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }'
```

![Verify the Ollama OpenAI-compatible API with curl](/img/installation/ollama/ollama-api-verify.png)

If either command fails, fix Ollama on the host before continuing. Semantic Router cannot reach a backend that is not already serving on port `11434`.

Then verify the address that the Router container will use:

```bash
docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  curlimages/curl:8.12.1 \
  http://host.docker.internal:11434/v1/models
```

If the host check succeeds but the container check fails, recheck
`OLLAMA_HOST` and the host firewall before continuing.

## 5. Connect the model in the Dashboard

Start Semantic Router (or use the instance already started by the installer):

```bash
vllm-sr serve
```

If `config.yaml` does not exist, the same command creates a secure managed
workspace and starts Router, Dashboard, PostgreSQL, and Valkey. Open
[http://localhost:8700](http://localhost:8700), create the first administrator,
then open **Build → Models**.

Register your Ollama model:

| Field | Value |
| --- | --- |
| **Model name** | Your Ollama tag, for example `llama3.2:3b` |
| **Provider** | **Local vLLM** |
| **Base URL or host** | `host.docker.internal:11434` |
| **Endpoint label** | `primary` (or any short label) |
| **Default** | Select this model if it is your only backend |

Why **Local vLLM** and not **OpenAI-compatible API**?

- Ollama serves an OpenAI-compatible surface at `/v1/chat/completions`.
- **Local vLLM** writes the host and protocol you enter as an `endpoint`
  backend reference, so enter `host.docker.internal:11434` explicitly.

Alternatively, choose **OpenAI-compatible API** and enter
`http://host.docker.internal:11434/v1`; that provider type writes a `base_url`.
Both paths use Ollama's OpenAI-compatible API.

Test the connection, then save the Model.

## 6. Publish a Mixture of Models

Open **Build → Mixture of Models**, choose a Recipe, and assign the Ollama
Model to every required decision. A single-model Recipe is the smallest useful
starting point; choose a richer Recipe when you add more backends.

Choose a stable API model name, review the topology, and publish the
Entrypoint. These are Router Management resources; the Dashboard does not write
or activate a replacement YAML file.

Envoy is already listening on port `8899`; the published Entrypoint becomes
available without restarting `serve`.

## 7. Test through Semantic Router

Send a request through the router proxy:

```bash
curl http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Hello from Semantic Router!"}]
  }'
```

If you kept the default single-model baseline, you can also use the auto-routing alias:

```bash
curl http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Hello from Semantic Router!"}]
  }'
```

A JSON chat completion response means Ollama is wired correctly.

## YAML configuration (advanced)

If you prefer to edit YAML directly instead of the dashboard, add a model entry like this:

```yaml
version: v0.3
providers:
  models:
    - name: llama3.2:3b
      provider_model_id: llama3.2:3b
      backend_refs:
        - provider: ollama
          endpoint: http://host.docker.internal:11434/v1
      control:
        retry:
          count: 1
          on: [unavailable]
routing:
  modelCards:
    - name: llama3.2:3b
      capabilities: [chat]
recipes:
  - name: default
    routing:
      decisions:
        - name: default-route
          description: Route every request.
          priority: 100
          rules:
            operator: AND
            conditions: []
entrypoints:
  - model_names: [ollama]
    recipe: default
    assignments:
      default-route:
        models:
          - model: llama3.2:3b
global:
  services:
    backend_dispatch:
      bind_address: 0.0.0.0
      port: 8180
      audience: vllm-sr.backend-dispatch
      capability_ttl: 30s
      max_request_body_bytes: 67108864
    backend_egress:
      policy_file: /app/config/backend-egress-policy.yaml
```

Validate and serve:

```bash
vllm-sr validate --config config.yaml
vllm-sr serve
```

## Troubleshooting

### Router cannot reach Ollama

- Use `host.docker.internal:11434` in config, not `localhost:11434`. Inside the router container, `localhost` refers to the container itself.
- Confirm Ollama is listening on a container-reachable address. The default
  `127.0.0.1:11434` binding is not sufficient; configure `OLLAMA_HOST` as shown
  above and restart Ollama.
- The local runtime adds a `host.docker.internal:host-gateway` mapping for
  Docker or Podman. This provides name resolution and routing, not a proxy for
  the host loopback interface. If connectivity still fails, see
  [Container connectivity](../troubleshooting/container-connectivity).
- Confirm Ollama responds on the host: `curl http://localhost:11434/v1/models`.

### Model not found or 404 from Ollama

- The **Model name** in Semantic Router must match the Ollama tag exactly (`llama3.2:3b`, not `llama3.2`).
- Run `ollama list` and pull the tag if it is missing: `ollama pull <tag>`.

### Slow first request

- Ollama loads models on demand. The first request after idle time may take longer while weights are loaded into memory.

### Reasoning models (Qwen3 and similar)

- Some reasoning models spend the full token budget on internal thinking when called through Ollama's OpenAI-compatible endpoint. For advanced local setups with Qwen3-style models, see [`bench/grounded_fusion/ollama_proxy.py`](https://github.com/vllm-project/semantic-router/blob/main/bench/grounded_fusion/ollama_proxy.py) in the repository.

## Next steps

- Add more backends and turn on semantic routing presets in the dashboard
- Read the [Configuration guide](configuration) for decisions, signals, and model cards
- See the [agentgateway homelab blog post](/blog/agentgateway-semantic-brain-homelab) for a multi-model setup that includes local Ollama
