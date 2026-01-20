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

## Quick Start with vllm-sr CLI

### Step 1: Set Your API Key

Export your Anthropic API key as an environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxx
```

### Step 2: Initialize Configuration

```bash
mkdir my-router && cd my-router
vllm-sr init
```

### Step 3: Configure Anthropic Models

Edit `config.yaml` with the minimal Anthropic setup:

```yaml
version: "1"

listeners:
  - name: http
    address: 0.0.0.0
    port: 8080
    timeout: 300s

providers:
  default_model: "claude-sonnet-4-5"
  models:
    - name: "claude-sonnet-4-5"
      api_format: "anthropic"
      access_key: "${ANTHROPIC_API_KEY}"
      endpoints: [ ]

decisions:
  - name: "default"
    description: "Default routing to Claude"
    priority: 50
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "other"
    modelRefs:
      - model: "claude-sonnet-4-5"
```

:::info
The `${ANTHROPIC_API_KEY}` syntax references the environment variable. The key
is expanded at runtime, keeping your credentials secure.
:::

### Step 4: Start the Router

```bash
vllm-sr serve config.yaml
```

### Hybrid Configuration (vLLM + Anthropic)

Mix Anthropic models with local vLLM endpoints for cost optimization:

```yaml
version: "1"

listeners:
  - name: http
    address: 0.0.0.0
    port: 8080
    timeout: 300s

providers:
  default_model: "Qwen/Qwen2.5-7B-Instruct"
  models:
    # Local model via vLLM for simple queries
    - name: "Qwen/Qwen2.5-7B-Instruct"
      endpoints:
        - name: "local-gpu"
          endpoint: "host.docker.internal:8000" # change this to your vLLM endpoint
          weight: 1

    # Cloud model via Anthropic API for complex queries
    - name: "claude-sonnet-4-5"
      api_format: "anthropic"
      access_key: "${ANTHROPIC_API_KEY}"
      endpoints: [ ]

decisions:
  - name: "simple_queries"
    description: "Use local model for simple queries"
    priority: 50
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "other"
    modelRefs:
      - model: "Qwen/Qwen2.5-7B-Instruct"

  - name: "complex_queries"
    description: "Use Claude for complex queries"
    priority: 100
    rules:
      operator: "OR"
      conditions:
        - type: "domain"
          name: "computer_science"
        - type: "domain"
          name: "math"
    modelRefs:
      - model: "claude-sonnet-4-5"
```

### Testing Hybrid Routing

**Simple query → Routes to local vLLM (Qwen):**

```bash
curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "input": "Where is Beijing?",
    "instructions": "You are a helpful assistant",
    "stream": false
  }'
```

Response (routed to `Qwen/Qwen2.5-7B-Instruct`):

```json
{
  "id": "resp_c3c0724bddedf0ca2e262e7d",
  "object": "response",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "status": "completed",
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Beijing is located in northern China. It is the capital city of the People's Republic of China..."
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 22,
    "output_tokens": 93,
    "total_tokens": 115
  }
}
```

**Complex query (computer science) → Routes to Anthropic (Claude):**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Explain the basic rules of Weiqi"}],
    "stream": false
  }'
```

Response (routed to `claude-sonnet-4-5`):

```json
{
  "id": "msg_01EGW1bYrBedQt8vw1uxyR3z",
  "object": "chat.completion",
  "model": "claude-sonnet-4-5",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "# Basic Rules of Weiqi (Go)\n\n## Objective\nControl more territory than your opponent..."
      }
    }
  ],
  "usage": {
    "completion_tokens": 337,
    "prompt_tokens": 21,
    "total_tokens": 358
  }
}
```

:::tip
Use `model: "auto"` to let the router automatically select the best model based
on your decision rules. Simple queries go to the cost-effective local model,
while complex queries are routed to Claude.
:::

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

## Response API Support

The router also supports the OpenAI Response API (`/v1/responses`) with
Anthropic
backends. Requests are translated to Chat Completions format, sent to Claude,
and
responses are converted back to Response API format.

### Enabling Response API

Add the `response_api` configuration to your config file:

```yaml
response_api:
  enabled: true
  store_backend: "memory"
  ttl_seconds: 86400
  max_responses: 1000
```

### Sending Response API Requests

```bash
curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5",
    "input": "What is the difference between threads and processes?",
    "instructions": "You are a helpful assistant"
  }'
```

### Example Response

```json
{
  "id": "resp_54a277277aa864ee6e18754d",
  "object": "response",
  "created_at": 1768803933,
  "model": "claude-sonnet-4-5",
  "status": "completed",
  "output": [
    {
      "type": "message",
      "id": "item_594ea35de68d22a8a3563cc7",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "# Threads vs Processes\n\n## **Process**\n- Independent execution unit with its own memory space..."
        }
      ],
      "status": "completed"
    }
  ],
  "output_text": "# Threads vs Processes...",
  "usage": {
    "input_tokens": 21,
    "output_tokens": 311,
    "total_tokens": 332
  },
  "instructions": "You are a helpful assistant"
}
```

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
