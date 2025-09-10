# Reasoning Routing Quickstart

This short guide shows how to enable and verify “reasoning routing” in the Semantic Router:
- Minimal config.yaml fields you need
- Example request/response (OpenAI-compatible)
- A comprehensive evaluation command you can run

Prerequisites
- A running OpenAI-compatible backend for your models (e.g., vLLM, OpenAI-compatible server)
- Envoy + the router (see Start the router section)

1) Minimal configuration
Put this in config/config.yaml (or merge into your existing config). It defines:
- Categories that require reasoning (e.g., math)
- Reasoning families for model syntax differences (DeepSeek/Qwen3 use chat_template_kwargs; GPT-OSS/GPT use reasoning_effort)
- Which concrete models use which reasoning family

```yaml
# vLLM endpoints that host your models
vllm_endpoints:
  - name: "endpoint1"
    address: "127.0.0.1"
    port: 8000
    models: ["deepseek-v3", "qwen3-7b", "openai/gpt-oss-20b"]
    weight: 1

# Reasoning family configurations (how to express reasoning for a family)
reasoning_families:
  deepseek:
    type: "chat_template_kwargs"
    parameter: "thinking"
  qwen3:
    type: "chat_template_kwargs"
    parameter: "enable_thinking"
  gpt-oss:
    type: "reasoning_effort"
    parameter: "reasoning_effort"
  gpt:
    type: "reasoning_effort"
    parameter: "reasoning_effort"

# Default effort used when a category doesn’t specify one
default_reasoning_effort: medium  # low | medium | high

# Map concrete model names to a reasoning family
model_config:
  "deepseek-v3":
    reasoning_family: "deepseek"
    preferred_endpoints: ["endpoint1"]
  "qwen3-7b":
    reasoning_family: "qwen3"
    preferred_endpoints: ["endpoint1"]
  "openai/gpt-oss-20b":
    reasoning_family: "gpt-oss"
    preferred_endpoints: ["endpoint1"]

# Categories: which kinds of queries require reasoning and at what effort
categories:
- name: math
  use_reasoning: true
  reasoning_effort: high  # overrides default_reasoning_effort
  reasoning_description: "Mathematical problems require step-by-step reasoning"
  model_scores:
  - model: openai/gpt-oss-20b
    score: 1.0
  - model: deepseek-v3
    score: 0.8
  - model: qwen3-7b
    score: 0.8

- name: general
  use_reasoning: false
  reasoning_description: "General chit-chat doesn’t need reasoning"
  model_scores:
  - model: qwen3-7b
    score: 1.0
  - model: deepseek-v3
    score: 0.8

# A safe default when no category is confidently selected
default_model: qwen3-7b
```

Notes
- Reasoning is controlled by categories.use_reasoning and optionally categories.reasoning_effort.
- A model only gets reasoning fields if it has a model_config.<MODEL>.reasoning_family that maps to a reasoning_families entry.
- DeepSeek/Qwen3: router sets chat_template_kwargs: { parameter: true } when reasoning is enabled.
- GPT/GPT-OSS: router sets reasoning_effort to the category/default effort when reasoning is enabled.

2) Start the router
Option A: Local build + Envoy
- Build and run the router
  - make build
  - make run-router
- Start Envoy (install func-e once with make prepare-envoy if needed)
  - func-e run --config-path config/envoy.yaml --component-log-level "ext_proc:trace,router:trace,http:trace"

Option B: Docker Compose
- docker compose up -d
  - Exposes Envoy at http://localhost:8801 (proxying /v1/* to backends via the router)

3) Send example requests
Math (reasoning should be ON and effort high)
```bash
curl -sS http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "system", "content": "You are a math teacher."},
      {"role": "user",   "content": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?"}
    ]
  }' | jq
```

General (reasoning should be OFF)
```bash
curl -sS http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": "Who are you?"}
    ]
  }' | jq
```

Example response (shape)
The exact fields depend on your backend. The router keeps the OpenAI-compatible shape and may add metadata.

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1726000000,
  "model": "openai/gpt-oss-20b",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "The derivative is 3x^2 + 4x - 5." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 85, "completion_tokens": 43, "total_tokens": 128 },
  "routing_metadata": {
    "category": "math",
    "selected_model": "openai/gpt-oss-20b",
    "reasoning_enabled": true,
    "reasoning_effort": "high"
  }
}
```

4) Run a comprehensive evaluation
You can benchmark the router vs a direct vLLM endpoint across categories using the included script. This runs a ReasoningBench based on MMLU-Pro and produces summaries and plots.

Quick start (router + vLLM):
```bash
SAMPLES_PER_CATEGORY=25 \
CONCURRENT_REQUESTS=4 \
ROUTER_MODELS="auto" \
VLLM_MODELS="openai/gpt-oss-20b" \
./bench/run_bench.sh
```

Router-only benchmark:
```bash
BENCHMARK_ROUTER_ONLY=true \
SAMPLES_PER_CATEGORY=25 \
CONCURRENT_REQUESTS=4 \
ROUTER_MODELS="auto" \
./bench/run_bench.sh
```

Direct invocation (advanced):
```bash
python bench/router_reason_bench.py \
  --run-router \
  --router-endpoint http://localhost:8801/v1 \
  --router-models auto \
  --run-vllm \
  --vllm-endpoint http://localhost:8000/v1 \
  --vllm-models openai/gpt-oss-20b \
  --samples-per-category 25 \
  --concurrent-requests 4 \
  --output-dir results/reasonbench
```

Tips
- If your math request doesn’t enable reasoning, confirm the classifier assigns the "math" category with sufficient confidence (see categories.threshold in your setup) and that the target model has a reasoning_family.
- For models without a reasoning_family, the router will not inject reasoning fields even when the category requires reasoning (this is by design to avoid invalid requests).
- You can override the effort per category via categories.reasoning_effort or set a global default via default_reasoning_effort.

