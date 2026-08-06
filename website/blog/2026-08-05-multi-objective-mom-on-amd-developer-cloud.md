---
slug: multi-objective-mom-on-amd-developer-cloud
title: "Three ROCm Models, Five Routing Objectives on AMD Developer Cloud"
description: Deploy a three-tier open-weight Qwen model pool on AMD Instinct MI300X and route five isolated optimization objectives across real physical backends.
authors: [Xunzhuo]
tags: [amd, rocm, deployment, mixture-of-models, vllm, semantic-router]
image: /img/amd-deploy-0.png
---

<div align="center">

![AMD Developer Cloud and vLLM Semantic Router overview](/img/amd-deploy-0.png)

</div>

The first AMD Developer Cloud deployment guide showed how to put vLLM Semantic
Router in front of a balance-oriented ROCm backend. The maintained
`multi-objective` recipe takes the next step: clients choose the optimization
objective they want, while the router keeps each objective's signals,
projections, decisions, algorithms, and plugins isolated.

This guide deploys three physical ROCm models, exposes five honest local routing
lanes, and presents five stable Mixture-of-Models entrypoints. Requests now move
between models with materially different memory, latency, and quality profiles
instead of simulating those differences with aliases on one backend.

<!-- truncate -->

## From One Profile to Five Objectives

The balance recipe exposes one automatic routing policy. The multi-objective
recipe exposes five request-facing model IDs:

| Client model | Objective |
| --- | --- |
| `vllm-sr/mom-balanced-v1` | Balance quality, latency, cost, and load. |
| `vllm-sr/mom-flash-v1` | Prefer interactive latency and retain a bounded heavy lane. |
| `vllm-sr/mom-economy-v1` | Stay local and spend additional compute only when justified. |
| `vllm-sr/mom-frontier-v1` | Escalate from direct answers to confidence routing, ReMoM, Fusion, or Router Flow. |
| `vllm-sr/mom-private-v1` | Keep private or suspicious requests on local policy-compatible routes. |

An entrypoint selects one recipe before signal evaluation. Names inside that
recipe are local to the recipe, so a signal or decision in the privacy program
cannot accidentally activate a route in the speed program.

The request flow is:

```text
Client model
  -> entrypoint
  -> isolated recipe
  -> signals and projections
  -> decision and algorithm
  -> logical backend alias
  -> one of three physical ROCm models
```

## The Three-tier Model Pool

An 8×MI300X host has enough HBM to run independent single-GPU tiers while
leaving capacity for replicas, experiments, or a larger tensor-parallel model.
This deployment uses three GPUs:

| GPU | Physical model | Local lanes | Role |
| --- | --- | --- | --- |
| 0 | `Qwen/Qwen3.5-122B-A10B-FP8` | `local/qwen3.5-122b-frontier` | Highest-quality local synthesis and review. |
| 1 | `Qwen/Qwen3.5-9B` | `local/qwen3.5-9b-economy`, `local/qwen3.5-9b-private` | Lowest-cost interactive and isolated privacy traffic. |
| 2 | `Qwen/Qwen3.6-35B-A3B-FP8` | `local/qwen3.6-35b-flash`, `local/qwen3.6-35b-balanced` | Latest flash-class MoE for coding and general reasoning. |

Qwen3.6-35B-A3B activates roughly 3B parameters per token and has AMD Day-0
support in vLLM. It is the fast reasoning tier. Qwen3.5-9B remains useful because
small dense models have low scheduling and memory overhead. Qwen3.5-122B-A10B
provides the stronger escalation target without relying on a remote API.

Aliases are used only where two policy lanes share one physical checkpoint. The
names stay under the `local/` namespace and never impersonate a proprietary
vendor model. This allows privacy and balanced traffic to have independent
telemetry and pricing metadata while retaining an explicit physical mapping.

The smaller backends use a 32K serving limit to preserve predictable single-GPU
KV-cache capacity. The large backend retains its 262K serving limit. These are
deployment limits; they do not redefine each checkpoint's architectural maximum.

## Step 1: Start the ROCm Backends

Create the shared network:

```bash
sudo docker network create vllm-sr-network 2>/dev/null || true
```

Set the shared image and cache location:

```bash
export VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:latest}"
export VLLM_HF_CACHE="${VLLM_HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$VLLM_HF_CACHE"
```

Start the 122B frontier backend on GPU 0:

```bash
sudo docker run -d \
  --name vllm \
  --network vllm-sr-network \
  --restart unless-stopped \
  -p "127.0.0.1:${VLLM_PORT_122B:-8090}:8000" \
  -v "$VLLM_HF_CACHE:/root/.cache/huggingface" \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc host \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 32G \
  -e ROCR_VISIBLE_DEVICES=0 \
  -e VLLM_ROCM_USE_AITER=1 \
  --entrypoint python3 \
  "$VLLM_IMAGE" \
  -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-122B-A10B-FP8 \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name qwen/qwen3.5-rocm \
    --reasoning-parser qwen3 \
    --max-model-len 262144 \
    --language-model-only \
    --max-num-seqs 128 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.85
```

Start the 9B economy/private backend on GPU 1:

```bash
sudo docker run -d \
  --name vllm-qwen35-economy \
  --network vllm-sr-network \
  --restart unless-stopped \
  -p "127.0.0.1:${VLLM_PORT_9B:-8091}:8000" \
  -v "$VLLM_HF_CACHE:/root/.cache/huggingface" \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc host \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 32G \
  -e ROCR_VISIBLE_DEVICES=1 \
  -e VLLM_ROCM_USE_AITER=1 \
  --entrypoint python3 \
  "$VLLM_IMAGE" \
  -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name \
      local/qwen3.5-9b-economy \
      local/qwen3.5-9b-private \
    --reasoning-parser qwen3 \
    --max-model-len 32768 \
    --language-model-only \
    --max-num-seqs 128 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.50
```

Start the Qwen3.6 flash/balanced backend on GPU 2:

```bash
sudo docker run -d \
  --name vllm-qwen36-flash \
  --network vllm-sr-network \
  --restart unless-stopped \
  -p "127.0.0.1:${VLLM_PORT_35B:-8092}:8000" \
  -v "$VLLM_HF_CACHE:/root/.cache/huggingface" \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc host \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 32G \
  -e ROCR_VISIBLE_DEVICES=2 \
  -e VLLM_ROCM_USE_AITER=1 \
  --entrypoint python3 \
  "$VLLM_IMAGE" \
  -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-35B-A3B-FP8 \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name \
      local/qwen3.6-35b-flash \
      local/qwen3.6-35b-balanced \
    --reasoning-parser qwen3 \
    --max-model-len 32768 \
    --language-model-only \
    --max-num-seqs 128 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.60
```

The host ports are intentionally bound to loopback. Semantic Router reaches
`vllm:8000`, `vllm-qwen35-economy:8000`, and
`vllm-qwen36-flash:8000` on the shared Docker network.

Confirm that all physical tiers and aliases are present:

```bash
curl -sS http://127.0.0.1:8090/v1/models
curl -sS http://127.0.0.1:8091/v1/models
curl -sS http://127.0.0.1:8092/v1/models
```

## Step 2: Install and Start vLLM Semantic Router

Install the released CLI as an isolated `uv` tool:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv tool install --upgrade vllm-sr
```

Download the maintained recipe from `main` after the release containing this
recipe is published:

```bash
mkdir -p "$HOME/.config/vllm-sr/recipes"
curl -fsSLo "$HOME/.config/vllm-sr/recipes/multi-objective.yaml" \
  https://raw.githubusercontent.com/vllm-project/semantic-router/main/config/recipes/multi-objective/config.yaml
```

Validate and start the maintained recipe:

```bash
vllm-sr validate \
  --config "$HOME/.config/vllm-sr/recipes/multi-objective.yaml"

vllm-sr serve \
  --platform amd \
  --config "$HOME/.config/vllm-sr/recipes/multi-objective.yaml"
```

The maintained recipe points Looper back through
`vllm-sr-envoy-container:8899`. This is required for Fusion, ReMoM, and Router
Flow because their internal model calls must re-enter Envoy before provider
resolution sends each logical model to its physical backend.

For dashboard-first onboarding, import:

> `https://raw.githubusercontent.com/vllm-project/semantic-router/main/config/recipes/multi-objective/config.yaml`

The first visit presents the initial administrator registration flow. After
that account is created, public first-admin registration closes automatically.

## Step 3: Verify the Public Model Catalog

The router advertises entrypoints rather than physical backend aliases:

```bash
curl -sS http://127.0.0.1:8899/v1/models
```

Each record includes recipe metadata such as:

```json
{
  "id": "vllm-sr/mom-balanced-v1",
  "routing": {
    "resolution": "virtual",
    "selectable": true,
    "recipe": "balanced"
  }
}
```

## Step 4: Send Requests to Different Objectives

The client changes only the `model` field.

```bash
curl -sS http://127.0.0.1:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm-sr/mom-flash-v1",
    "messages": [
      {"role": "user", "content": "Summarize this incident in three bullets."}
    ],
    "max_tokens": 512
  }'
```

Use the frontier objective when the request benefits from bounded
multi-response orchestration:

```bash
curl -sS http://127.0.0.1:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vllm-sr/mom-frontier-v1",
    "messages": [
      {
        "role": "user",
        "content": "Compare several approaches, challenge the assumptions, and synthesize the strongest recommendation."
      }
    ],
    "max_tokens": 512
  }'
```

Response headers expose the selected recipe, decision, and logical model. Router
Replay persists the same recipe identity, so debugging and aggregate analysis
remain scoped to the objective the client selected.

## Step 5: Evaluate the Deployed Objectives

The installed CLI can evaluate objective selection without generating a
completion:

```bash
vllm-sr eval \
  --endpoint http://127.0.0.1:8080 \
  --model vllm-sr/mom-flash-v1 \
  --prompt "Summarize this incident in three bullets."

vllm-sr eval \
  --endpoint http://127.0.0.1:8080 \
  --model vllm-sr/mom-private-v1 \
  --prompt "Keep this customer record private: user@example.com."
```

The maintained
[`probes.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/multi-objective/probes.yaml)
contains 102 backend-independent cases used by repository CI and recipe
calibration. It checks:

- all 15 decisions
- multilingual and preference-conflict boundaries
- PII and jailbreak containment
- tool and multi-turn request shapes
- long-input behavior
- entrypoint, recipe, algorithm, plugin, and signal evidence

### Validated MI300X Run

The maintained commands were exercised on an 8×MI300X host with vLLM
`0.19.0+rocm721`. The final run verified:

- all three physical backends returned concurrent OpenAI-compatible completions
- all five public entrypoints generated non-empty final answers
- balanced selected the Qwen3.6 balanced lane
- flash and economy selected the 9B low-latency lane
- privacy selected the isolated 9B private alias with reasoning disabled
- frontier Fusion used all three physical tiers and the 122B judge
- the 102-probe suite matched all 15 decisions with 0 errors

The backend-independent routing evaluation completed at 32.229 requests per
second with p50 79.431 ms and p95 2062.777 ms. A representative generated
request took 0.183 seconds for economy, 0.336 seconds for flash, 0.637 seconds
for privacy, 3.590 seconds for balanced, and 44.654 seconds for three-model
frontier Fusion. These are validation observations from one host, not general
performance guarantees.

## Operating the Pool

A production rollout should:

1. replace the example operating-cost and quality scores with measured values
2. benchmark TTFT, TPOT, throughput, and error rates under representative load
3. configure listener API keys and management authentication
4. keep internal services private to the runtime network
5. persist Replay and state stores according to retention policy
6. rerun the probe suite after every model, threshold, or prompt change

Entrypoints define the client-visible objective, recipes isolate policy, and the
model catalog owns the explicit logical-to-physical execution contract.

## Next Steps

- Read the [multi-objective recipe](https://github.com/vllm-project/semantic-router/tree/main/config/recipes/multi-objective).
- Follow the [AMD ROCm installation guide](https://vllm-sr.ai/docs/installation/amd-rocm).
- Explore [entrypoints and recipes](https://vllm-sr.ai/docs/tutorials/global/entrypoints-and-recipes).
- Inspect Replay and Insights after exercising each entrypoint.
