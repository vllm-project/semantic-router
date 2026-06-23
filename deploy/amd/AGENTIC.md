# Agentic Intelligence on AMD ROCm

This playbook documents the agentic routing profile for the AMD ROCm
deployment path. It uses the same single vLLM backend pattern as
[README.md](README.md): one physical ROCm model exposes multiple served-model
aliases, and vLLM Semantic Router chooses among those aliases through signals,
projections, semantic decisions, and Router Learning adaptations.

## Profile

- Recipe: [agentic-saars.yaml](../recipes/agentic-saars.yaml)
- Backend service expected by the recipe: `vllm:8000`
- Physical backend model: `Qwen/Qwen3.6-35B-A3B`
- Served-model aliases:
  - `qwen/qwen3.6-rocm`
  - `google/gemini-2.5-flash-lite`
  - `google/gemini-3.1-pro`
  - `openai/gpt5.4`
  - `anthropic/claude-opus-4.6`

The profile does not call OpenRouter or any external provider. Every configured
model points at the same `vllm:8000` backend, so the aliases simulate a
heterogeneous fleet while preserving the AMD local deployment shape.

## Routing Intent

The recipe combines the useful ideas from the `balance` and `privacy` profiles
without copying either one directly:

| Route family | Model alias | Purpose |
| --- | --- | --- |
| `local_security_containment` | `qwen/qwen3.6-rocm` | Keep prompt-injection and jailbreak-like requests local. |
| `local_privacy_policy` | `qwen/qwen3.6-rocm` | Keep private code, internal docs, credentials, and PII-like requests local. |
| `domain_legal_health` | `anthropic/claude-opus-4.6` | Route legal, compliance, and health tasks to the high-care domain alias. |
| `domain_code_complex` | `google/gemini-3.1-pro` / `openai/gpt5.4` | Route hard architecture, systems, and coding work to stronger technical aliases. |
| `domain_code` | `google/gemini-2.5-flash-lite` / `google/gemini-3.1-pro` | Route medium software and repository work to code-capable aliases. |
| `domain_stem_research` | `google/gemini-3.1-pro` / `openai/gpt5.4` | Route math, science, and research synthesis to stronger reasoning aliases. |
| `domain_business` | `google/gemini-2.5-flash-lite` | Route business and product analysis to a medium-cost alias. |
| `complex_general` | `google/gemini-3.1-pro` / `openai/gpt5.4` | Route hard non-private planning, architecture, and synthesis work. |
| `medium_general` | `google/gemini-2.5-flash-lite` | Route medium difficulty explanations and follow-ups. |
| `simple_general` / `default_general` | `qwen/qwen3.6-rocm` | Keep simple or unmatched traffic on the cheapest local alias. |

Security and privacy routes have the highest priority and bypass Router
Learning, so sensitive traffic does not leave the local AMD backend even when it
is complex. Domain routes normally allow Router Learning, so an active tool loop
or protected session can keep the established model when continuity matters.

## Router Learning Policy

The recipe enables Router Learning once at the global router level:

```yaml
global:
  router:
    learning:
      enabled: true
      adaptation:
        enabled: true
        strategy: routing_sampling
        candidate_set: decision
      protection:
        enabled: true
        scope: conversation
```

Agentic continuity is expressed as adaptation plus protection, not as a
semantic route or decision algorithm. Each decision first chooses the model it
would normally use: simple tasks choose the local/simple alias, complex tasks
choose stronger aliases, privacy tasks choose the local AMD alias, and domain
tasks choose their domain aliases. Adaptation can propose a better model from
runtime experience, and protection decides whether to keep the current model
for continuity.

The default recipe uses `scope: conversation`. That protects one agent run:
turns with the same `x-conversation-id` can stay on the established model, while
a new `x-conversation-id` in the same `x-session-id` can route again.

For stricter applications, switch to `scope: session`:

```yaml
global:
  router:
    learning:
      protection:
        scope: session
```

`scope: session` protects the established model across conversations sharing
the same `x-session-id` until the session idles out or the matched decision
bypasses adaptation. Privacy and security decisions still bypass Router
Learning and stay local.

The policy enables:

- tool-loop hard lock, so tool results stay on the model that requested them
- context-portability hard lock, so provider-owned continuation state is not
  moved to another backend
- prefix-cache and handoff penalties, so warm sessions are not moved casually
- switch-history penalty, so long-horizon agents do not bounce between aliases
- configurable identity scope, so deployments can choose conversation-level or
  session-level protection

The aliases keep different example pricing values. That is intentional:
protection uses handoff cost and cached-input pricing when estimating
prefix-cache loss. Switching between cheap aliases is easier to
justify than switching into or out of expensive frontier aliases.

Hard policy decisions opt out with:

```yaml
adaptations:
  mode: bypass
```

## Run

Start the AMD vLLM backend with the Qwen3.6 single-card profile, making sure
the served-model aliases include all five names above:

```bash
sudo docker network create vllm-sr-network 2>/dev/null || true

sudo docker run -d \
  --name vllm \
  --network=vllm-sr-network \
  --restart unless-stopped \
  -p "${VLLM_PORT_QWEN36:-8090}:8000" \
  -v "${VLLM_HF_CACHE:-/mnt/data/huggingface-cache}:/root/.cache/huggingface" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 32G \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 \
  --entrypoint python3 \
  vllm/vllm-openai-rocm:latest \
  -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-35B-A3B \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --served-model-name qwen/qwen3.6-rocm google/gemini-2.5-flash-lite google/gemini-3.1-pro openai/gpt5.4 anthropic/claude-opus-4.6 \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --max-model-len 262144 \
    --language-model-only \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --prefix-caching-hash-algo sha256 \
    --kv-cache-dtype fp8 \
    --kv-cache-metrics \
    --kv-cache-metrics-sample 0.01 \
    --enable-logging-iteration-details \
    --gpu-memory-utilization 0.90
```

The checkpoint above is the non-FP8 `Qwen/Qwen3.6-35B-A3B` model. The
`--kv-cache-dtype fp8` flag only compresses the runtime KV cache to preserve
long-context headroom on a single MI300X.

`vllm/vllm-openai-rocm` is the official vLLM ROCm image for the
OpenAI-compatible API server. The image name does not imply an OpenAI-hosted
model; this recipe still serves the local Qwen checkpoint through vLLM.

The cache parameters are explicit so Router Learning experiments have visible
token-cache behavior:

- `--enable-prefix-caching` enables automatic prefix reuse across requests.
- `--prefix-caching-hash-algo sha256` uses collision-resistant prefix hashes.
- `--kv-cache-metrics` exposes KV-cache residency and reuse metrics.
- `--enable-logging-iteration-details` logs per-iteration context and
  generation token details.

Inspect startup capacity and live cache behavior with:

```bash
sudo docker logs vllm 2>&1 | \
  grep -Ei 'kv cache|prefix cache|maximum concurrency|iteration'

curl -s "http://localhost:${VLLM_PORT_QWEN36:-8090}/metrics" | \
  grep -E 'vllm:cache_config_info|vllm:kv_cache_usage_perc|prefix_cache|kv_cache'
```

Then import or serve the agentic profile:

```bash
vllm-sr serve --image-pull-policy never --platform amd \
  --config deploy/recipes/agentic-saars.yaml
```

For dashboard-first setup, import:

```text
deploy/recipes/agentic-saars.yaml
```

or use the raw GitHub URL once the recipe is published.

After merge, the recipe can be loaded from:

```text
https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/recipes/agentic-saars.yaml
```

## Model Choice

`Qwen/Qwen3.6-35B-A3B` is the default for this recipe because it is a newer
open-weight Qwen3.6 checkpoint and is tuned for agentic coding and thinking
preservation, which maps directly to the Router Learning goal of stable
multi-turn tool workflows. The MI300X 192 GB HBM class leaves enough headroom
for this non-FP8 model in the single-backend agentic profile.

`Qwen/Qwen3.6-27B` is a reasonable dense alternative if the workload
prioritizes dense coding behavior over MoE throughput. `Qwen/Qwen3.5-122B-A10B`
is still useful for deployments that are already validated on the older AMD
profile, but it is no longer the preferred agentic default here.

## Validation

Local config validation:

```bash
PYTHONPATH=src/vllm-sr python3 -m cli.main validate \
  --config deploy/recipes/agentic-saars.yaml
```

Conversation-scope behavior should be validated with:

- fresh routing matrix: simple, complex, privacy, and domain prompts reach the
  intended route family
- same-session multi-run: a new `x-conversation-id` can re-route inside the
  same `x-session-id`
- tool-loop stability: tool-result turns keep the established model and emit
  `x-vsr-learning-actions: protection=hold_current` with
  `x-vsr-learning-scopes: protection=conversation`

Session-scope behavior should be validated by changing only
`global.router.learning.protection.scope` to `session` and running a
same-session multi-run test:

- the first non-bypass run establishes the session model
- a later run with a different `x-conversation-id` stays on that model
- privacy or security runs still bypass and route to `qwen/qwen3.6-rocm`

The `x-vsr-learning-*` header family is the compact response surface for this
check. It reports active learning methods plus method-keyed action, reason, and
scope, for example:

```http
x-vsr-learning-methods: adaptation,protection
x-vsr-learning-actions: adaptation=keep_base,protection=hold_current
x-vsr-learning-scopes: protection=conversation
x-vsr-learning-reasons: adaptation=base_best,protection=cache_cost_high
```

Use `x-vsr-replay-id` to inspect full Router Replay diagnostics, including base
selected model, final selected model, cache warmth, memory token counters, and
candidate score traces.

Repo harness validation for this docs/example surface:

```bash
make agent-report ENV=amd \
  CHANGED_FILES="deploy/recipes/agentic-saars.yaml deploy/amd/AGENTIC.md"
make agent-lint \
  CHANGED_FILES="deploy/recipes/agentic-saars.yaml deploy/amd/AGENTIC.md"
```
