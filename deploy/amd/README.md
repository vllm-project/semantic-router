# vLLM Semantic Router on AMD ROCm

This playbook documents the reference AMD profile for a single real ROCm vLLM backend that exposes multiple served-model aliases. The router selects one alias per decision and forwards that alias unchanged to the backend OpenAI request body.

## Overview

- Physical backend model: `Qwen/Qwen3.5-122B-A10B-FP8`
- Docker service name expected by the profile: `vllm:8000`
- Served-model aliases exposed by the backend:
  - `openai/gpt-oss-120b`
  - `Qwen/Qwen3.5-122B-A10B`
  - `Qwen/Qwen3.5-9B`
  - `Kimi-K2-Thinking`
  - `GLM-4.7`
  - `DeepSeek-V3.2`
- Reference routing profile: [config.yaml](./config.yaml)

The active AMD profile contains 20 routing decisions. Each decision has exactly one `modelRef`. Guardrail and PII examples remain in the YAML as commented templates, but they are not active in this reference profile.

## Installation

### Step 1: Start the AMD vLLM backend

Create the shared Docker network first, then start the single ROCm backend container:

```bash
sudo docker network create vllm-sr-network 2>/dev/null || true

sudo docker run -d \
  --name vllm \
  --network=vllm-sr-network \
  --restart unless-stopped \
  -p "${VLLM_PORT_122B:-8090}:8000" \
  -v "${VLLM_HF_CACHE:-/mnt/data/huggingface-cache}:/root/.cache/huggingface" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 32G \
  -v /data:/data \
  -v "$HOME:/myhome" \
  -w /myhome \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 \
  --entrypoint python3 \
  vllm/vllm-openai-rocm:v0.17.0 \
  -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-122B-A10B-FP8 \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --served-model-name openai/gpt-oss-120b Qwen/Qwen3.5-122B-A10B Qwen/Qwen3.5-9B Kimi-K2-Thinking GLM-4.7 DeepSeek-V3.2 \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --max-model-len 262144 \
    --language-model-only \
    --max-num-seqs 128 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.85
```

If the backend fails its startup memory check on a single MI300X, reduce `--gpu-memory-utilization` first. If it still does not fit, reduce `--max-model-len`.

Verify that the container is up and that all aliases are visible:

```bash
sudo docker ps --filter name=vllm
curl -s "http://localhost:${VLLM_PORT_122B:-8090}/v1/models"
```

The `/v1/models` response should include all six alias IDs listed above.

### Step 2: Install vLLM Semantic Router

```bash
sudo apt-get install python3.12-venv
python3 -m venv vsr
source vsr/bin/activate
pip3 install vllm-sr
```

### Step 3: Prepare the AMD routing profile

If you are running from this repository, use [config.yaml](./config.yaml) directly as the reference AMD profile. If you want a standalone copy, download it:

```bash
wget -O config.yaml https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/amd/config.yaml
```

### Step 4: Start vLLM Semantic Router

Use the canonical AMD local serve path:

```bash
vllm-sr serve --image-pull-policy never --platform amd
```

If you are using a local `config.yaml` in the current directory, the router will load it automatically on startup.

### Step 5: Access the dashboard

```text
http://<your-server-ip>:8700
```

You should see routing metrics, selected decision metadata, and selected model aliases for each request.

## Architecture

```text
Client
  |
  v
vLLM Semantic Router (:8899)
  |
  +-- signal evaluation
  |   - keyword
  |   - embedding
  |   - domain
  |   - language
  |   - fact_check
  |   - context
  |   - complexity
  |
  +-- decision selection
  |   - 20 active decisions
  |   - one modelRef per decision
  |
  +-- alias-forwarded OpenAI request
  |   - model: openai/gpt-oss-120b
  |   - model: Kimi-K2-Thinking
  |   - model: GLM-4.7
  |   - model: DeepSeek-V3.2
  |   - model: Qwen/Qwen3.5-122B-A10B
  |   - model: Qwen/Qwen3.5-9B
  |
  v
Single ROCm vLLM backend on vllm:8000
  |
  v
Qwen/Qwen3.5-122B-A10B-FP8
```

The router does not use `external_model_ids` in this profile. The selected alias is sent directly to the backend, and the backend accepts it because the container was started with matching `--served-model-name` entries.

## Alias Catalog

| Alias | Role in the profile |
|------|----------------------|
| `openai/gpt-oss-120b` | Default fallback plus creative, legal, business, and psychology routes |
| `Qwen/Qwen3.5-122B-A10B` | STEM reasoning routes for hard math, chemistry, biology, health, and physics |
| `Qwen/Qwen3.5-9B` | Lightweight alias for easy math and casual chat |
| `Kimi-K2-Thinking` | Deep reasoning routes |
| `GLM-4.7` | Fast Chinese/English QA and history |
| `DeepSeek-V3.2` | Engineering and coding routes |

## Active Routing Decisions

| Priority | Decision | Signals | Target alias | Reasoning |
|---------:|----------|---------|--------------|-----------|
| 160 | `creative_ideas` | `keyword:creative_keywords` + `fact_check:no_fact_check_needed` | `openai/gpt-oss-120b` | `high` |
| 150 | `hard_math_problems` | `domain:math` + `complexity:math_problem:hard` | `Qwen/Qwen3.5-122B-A10B` | `high` |
| 149 | `easy_math_problems` | `domain:math` or `complexity:math_problem:easy/medium` | `Qwen/Qwen3.5-9B` | `low` |
| 148 | `chemistry_problems` | `domain:chemistry` | `Qwen/Qwen3.5-122B-A10B` | `medium` |
| 147 | `biology_problems` | `domain:biology` | `Qwen/Qwen3.5-122B-A10B` | `medium` |
| 146 | `health_medical` | `domain:health` | `Qwen/Qwen3.5-122B-A10B` | `medium` |
| 145 | `physics_problems` | `domain:physics` | `Qwen/Qwen3.5-122B-A10B` | `medium` |
| 144 | `engineering_problems` | `domain:engineering` | `DeepSeek-V3.2` | `medium` |
| 143 | `law_legal` | `domain:law` | `openai/gpt-oss-120b` | `medium` |
| 142 | `business_economics` | `domain:business` or `domain:economics` | `openai/gpt-oss-120b` | `medium` |
| 141 | `complex_engineering` | `domain:computer science` + `embedding:deep_thinking_en` + `complexity:computer_science:hard` | `DeepSeek-V3.2` | `high` |
| 138 | `psychology_queries` | `domain:psychology` | `openai/gpt-oss-120b` | `medium` |
| 137 | `philosophy_queries` | `domain:philosophy` | `Kimi-K2-Thinking` | `high` |
| 136 | `history_queries` | `domain:history` | `GLM-4.7` | `medium` |
| 131 | `complex_reasoning` | `embedding:deep_thinking_zh` or `keyword:thinking_zh` | `Kimi-K2-Thinking` | `high` |
| 131 | `fast_coding` | `domain:computer science` or `complexity:computer_science:easy/medium` | `DeepSeek-V3.2` | `low` |
| 130 | `quick_question` | `embedding:fast_qa_zh` + `language:zh` + `context:short_context` | `GLM-4.7` | `off` |
| 120 | `fast_qa` | `embedding:fast_qa_en` + `language:en` + `context:short_context` | `GLM-4.7` | `off` |
| 110 | `deep_thinking` | `embedding:deep_thinking_en` or `keyword:thinking_en` or `context:long_context` | `Kimi-K2-Thinking` | `high` |
| 100 | `casual_chat` | `domain:other` or `language:en/zh` or `context:short_context` | `Qwen/Qwen3.5-9B` | `off` |

## Usage Examples

Test these queries in the dashboard playground at `http://<your-server-ip>:8700`:

### Example 1: Creative writing

```text
Write a bedtime story about a robot learning to paint.
```

- Expected decision: `creative_ideas`
- Expected model alias: `openai/gpt-oss-120b`

### Example 2: Hard math proof

```text
Prove that the square root of 2 is irrational using proof by contradiction.
```

- Expected decision: `hard_math_problems`
- Expected model alias: `Qwen/Qwen3.5-122B-A10B`

### Example 3: Easy arithmetic

```text
What is 15 + 27?
```

- Expected decision: `easy_math_problems`
- Expected model alias: `Qwen/Qwen3.5-9B`

### Example 4: Deep reasoning in Chinese

```text
请认真分析人工智能对未来社会治理的影响，并给出系统性的应对策略。
```

- Expected decision: `complex_reasoning`
- Expected model alias: `Kimi-K2-Thinking`

### Example 5: Fast English QA

```text
Who are you?
```

- Expected decision: `fast_qa`
- Expected model alias: `GLM-4.7`

### Example 6: Complex engineering design

```text
Design a distributed rate limiter using Redis and explain the failure modes.
```

- Expected decision: `complex_engineering`
- Expected model alias: `DeepSeek-V3.2`

## Validation Checklist

- `sudo docker ps --filter name=vllm` shows the single backend container as healthy.
- `curl -s "http://localhost:${VLLM_PORT_122B:-8090}/v1/models"` lists all six alias IDs.
- The router is started with `vllm-sr serve --image-pull-policy never --platform amd`.
- Requests hitting the playground show one selected decision and one selected alias per request.
- `deploy/amd/config.yaml` remains aligned with this document's alias catalog and decision table.

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [vLLM Semantic Router GitHub](https://github.com/vllm-project/semantic-router)
