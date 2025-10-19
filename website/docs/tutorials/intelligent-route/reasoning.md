# Reasoning Routing

This short guide shows how to enable and verify “reasoning routing” in the Semantic Router:

- Minimal config.yaml fields you need
- Example request/response (OpenAI-compatible)
- A comprehensive evaluation command you can run

Prerequisites

- A running OpenAI-compatible backend for your models (e.g., vLLM or any OpenAI-compatible server). It must be reachable at the addresses you configure under vllm_endpoints (address:port).
- Envoy + the router (see Start the router section)

1) Minimal configuration
Put this in config/config.yaml (or merge into your existing config). It defines:

- Categories that require reasoning (e.g., math)
- Reasoning families for model syntax differences (DeepSeek/Qwen3 use chat_template_kwargs; GPT-OSS/GPT use reasoning_effort)
- Which concrete models use which reasoning family
- The classifier (required for category detection; without it, reasoning will not be enabled)

```yaml
# Category classifier (required for reasoning to trigger)
classifier:
  category_model:
    model_id: "models/category_classifier_modernbert-base_model"
    use_modernbert: true
    threshold: 0.6
    use_cpu: true
    category_mapping_path: "models/category_classifier_modernbert-base_model/category_mapping.json"

# vLLM endpoints that host your models
vllm_endpoints:
  - name: "endpoint1"
    address: "127.0.0.1"
    port: 8000
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
  "deepseek-v31":
    reasoning_family: "deepseek"
    preferred_endpoints: ["endpoint1"]
  "qwen3-30b":
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
  - model: deepseek-v31
    score: 0.8
  - model: qwen3-30b
    score: 0.8


# A safe default when no category is confidently selected
default_model: qwen3-30b
```

Notes

- Reasoning is controlled by categories.use_reasoning and optionally categories.reasoning_effort.
- A model only gets reasoning fields if it has a model_config.&lt;MODEL&gt;.reasoning_family that maps to a reasoning_families entry.
- DeepSeek/Qwen3 (chat_template_kwargs): the router injects chat_template_kwargs only when reasoning is enabled. When disabled, no chat_template_kwargs are added.
- GPT/GPT-OSS (reasoning_effort): when reasoning is enabled, the router sets reasoning_effort based on the category (fallback to default_reasoning_effort). When reasoning is disabled, if the request already contains reasoning_effort and the model’s family type is reasoning_effort, the router preserves the original value; otherwise it is absent.
- Category descriptions (for example, description and reasoning_description) are informational only today; they do not affect routing or classification.
- Categories must be from MMLU-Pro at the moment; avoid free-form categories like "general". If you want generic categories, consider opening an issue to map them to MMLU-Pro.

2) Start the router
Option A: Local build + Envoy

- Download classifier models and mappings (required)
  - make download-models
- Build and run the router
  - make build
  - make run-router
- Start Envoy (install func-e once with make prepare-envoy if needed)
  - func-e run --config-path config/envoy.yaml --component-log-level "ext_proc:trace,router:trace,http:trace"

Option B: Docker Compose

- docker compose up -d
  - Exposes Envoy at http://localhost:8801 (proxying /v1/* to backends via the router)

Note: Ensure your OpenAI-compatible backend is running and reachable (e.g., http://127.0.0.1:8000) so that vllm_endpoints address:port matches a live server. Without a running backend, routing will fail at the Envoy hop.

3) Send example requests
Math (reasoning should be ON and effort high)

```bash
curl -sS http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
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
    "model": "MoM",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user",   "content": "Who are you?"}
    ]
  }' | jq
```

Verify routing via response headers
The router does not inject routing metadata into the JSON body. Instead, inspect the response headers added by the router:

- X-Selected-Model
- X-GATEWAY-DESTINATION-ENDPOINT

Example:

```bash
curl -i http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [
      {"role": "system", "content": "You are a math teacher."},
      {"role": "user",   "content": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 7?"}
    ]
  }'
# In the response headers, look for:
#   X-Selected-Model: <your-selected-model>
#   X-GATEWAY-DESTINATION-ENDPOINT: <address:port>
```

4) Run a comprehensive evaluation
You can benchmark the router vs a direct vLLM endpoint across categories using the included script. This runs a ReasoningBench based on MMLU-Pro and produces summaries and plots.

Quick start (router + vLLM):

```bash
SAMPLES_PER_CATEGORY=25 \
CONCURRENT_REQUESTS=4 \
ROUTER_MODELS="MoM" \
VLLM_MODELS="openai/gpt-oss-20b" \
./bench/run_bench.sh
```

Router-only benchmark:

```bash
BENCHMARK_ROUTER_ONLY=true \
SAMPLES_PER_CATEGORY=25 \
CONCURRENT_REQUESTS=4 \
ROUTER_MODELS="MoM" \
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

- If your math request doesn’t enable reasoning, confirm the classifier assigns the "math" category with sufficient confidence (see classifier.category_model.threshold) and that the target model has a reasoning_family.
- For models without a reasoning_family, the router will not inject reasoning fields even when the category requires reasoning (this is by design to avoid invalid requests).
- You can override the effort per category via categories.reasoning_effort or set a global default via default_reasoning_effort.
- Ensure your OpenAI-compatible backend is reachable at the configured vllm_endpoints (address:port). If it’s not running, routing will fail even though the router and Envoy are up.
