---
title: Anthropic Claude Configuration
sidebar_label: Anthropic Models
---

# Anthropic Claude Configuration

This guide explains how to configure Anthropic Claude models as backend
inference providers. The semantic router accepts OpenAI-format requests and
automatically translates them to Anthropic's Messages API format, returning
responses in OpenAI format for seamless client compatibility.

## Environment Setup

### Setting the API Key

Anthropic API keys must be available as environment variables. Create a `.env`
file or export directly:

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxx
```

### Verifying the Environment

Before running the router, verify the API key is accessible:

```bash
# Should print your API key
echo $ANTHROPIC_API_KEY

# Test the key directly with Anthropic API
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-sonnet-4-5","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}'
```

## Basic Configuration

### Minimal Setup

Add Claude models to your `model_config` with `api_format: "anthropic"`:

```yaml
model_config:
  "claude-sonnet-4-5":
    api_format: "anthropic"
    access_key: "${ANTHROPIC_API_KEY}"

default_model: "claude-sonnet-4-5"
```

> The `${ANTHROPIC_API_KEY}` syntax references the environment variable. You can
> also hardcode the key directly (not recommended for security).

### Decision-Based Routing

Route requests to different Claude models based on classification:

```yaml
model_config:
  "claude-sonnet-4-5":
    api_format: "anthropic"
    access_key: "${ANTHROPIC_API_KEY}"
  "claude-3-haiku-20240307":
    api_format: "anthropic"
    access_key: "${ANTHROPIC_API_KEY}"

decisions:
  - name: "complex_queries"
    description: "Route complex tasks to Sonnet"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "math"
        - type: "domain"
          name: "computer_science"
    modelRefs:
      - model: "claude-sonnet-4-5"
        use_reasoning: false

  - name: "simple_queries"
    description: "Route simple tasks to Haiku"
    priority: 50
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "other"
    modelRefs:
      - model: "claude-3-haiku-20240307"
        use_reasoning: false

default_model: "claude-sonnet-4-5"
```

:::note
The `use_reasoning` field is required for all `modelRefs`. Set it to `false` for
Claude models as they don't use the reasoning family feature (which is designed
for models like Qwen3).
:::

## Hybrid Configuration

Mix Anthropic models with local vLLM endpoints:

```yaml
vllm_endpoints:
  - name: "local-gpu"
    address: "127.0.0.1"
    port: 8000
    weight: 1

model_config:
  # Local model via vLLM
  "Qwen/Qwen2.5-7B-Instruct":
    reasoning_family: "qwen3"
    preferred_endpoints: [ "local-gpu" ]

  # Cloud model via Anthropic API
  "claude-sonnet-4-5":
    api_format: "anthropic"
    access_key: "${ANTHROPIC_API_KEY}"

decisions:
  - name: "local_processing"
    description: "Use local model for general queries"
    priority: 50
    modelRefs:
      - model: "Qwen/Qwen2.5-7B-Instruct"
        use_reasoning: true

  - name: "cloud_processing"
    description: "Use Claude for complex queries"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "computer_science"
    modelRefs:
      - model: "claude-sonnet-4-5"
        use_reasoning: false
```

## Sending Requests

Once configured, send standard OpenAI-format requests:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "max_tokens": 1024
  }'
```

The router will:

1. Parse the OpenAI-format request
2. Convert it to Anthropic Messages API format
3. Call the Anthropic API
4. Convert the response back to OpenAI format
5. Return the response to the client

## Supported Parameters

The following OpenAI parameters are translated to Anthropic equivalents:

| OpenAI Parameter        | Anthropic Equivalent  | Notes                                     |
|-------------------------|-----------------------|-------------------------------------------|
| `model`                 | `model`               | Model name passed directly                |
| `messages`              | `messages` + `system` | System messages extracted separately      |
| `max_tokens`            | `max_tokens`          | Required by Anthropic (defaults to 4096)  |
| `max_completion_tokens` | `max_tokens`          | Alternative to max_tokens                 |
| `temperature`           | `temperature`         | 0.0 to 1.0                                |
| `top_p`                 | `top_p`               | Nucleus sampling                          |
| `stop`                  | `stop_sequences`      | Stop sequences                            |
| `stream`                | —                     | **Not supported** (see limitations below) |

## Current Limitations

### Streaming Not Supported

The Anthropic backend currently only supports non-streaming responses. If you
send
a request with `stream: true`, the router will return an error:

```json
{
  "error": {
    "message": "Streaming is not supported for Anthropic models. Please set stream=false in your request.",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

**Workaround:** Ensure all requests to Anthropic models use `stream: false` or
omit
the `stream` parameter entirely (defaults to `false`).

```bash
# Correct - non-streaming request
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

:::tip
If your application requires streaming responses, consider using a local vLLM
endpoint or an OpenAI-compatible API that supports streaming, and configure
decision-based routing to direct streaming-critical workloads to those backends.
:::
