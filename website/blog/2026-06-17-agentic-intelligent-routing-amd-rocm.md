---
slug: agentic-intelligent-routing-amd-rocm
title: "Agentic Intelligent Routing on AMD ROCm"
description: Run Session-Aware Agentic Routing with vLLM Semantic Router on AMD ROCm for private work, tool-loop continuity, domain routing, and observable agent sessions.
authors: [Xunzhuo, haichen, andyluo]
tags: [amd, rocm, agentic, routing, vllm, semantic-router]
image: /img/agentic-saars-amd-rocm.png
---

<div align="center">

![SAAR on AMD ROCm routing flow](/img/agentic-saars-amd-rocm.png)

</div>

Agentic routing needs a stability policy. A single prompt can be routed by
topic, complexity, domain, or privacy, but an agent session also carries tool
calls, tool results, warm context, and continuation state. The AMD ROCm
agentic profile packages those concerns into one runnable recipe:
`deploy/recipes/agentic-saars.yaml`.

The recipe runs behind an OpenAI-compatible Envoy listener, uses vLLM Semantic
Router for signal extraction and decisioning, and targets an AMD ROCm vLLM
backend. Its goal is direct: simple tasks stay simple, complex tasks use
stronger lanes, private tasks stay local, domain tasks use domain lanes, and
multi-turn agent work stays stable through SAAR.

<!-- truncate -->

## Runtime Shape

The profile uses the same AMD deployment shape as the existing ROCm examples:
one local vLLM backend is exposed through multiple served-model aliases, and
the router selects the alias that matches the request policy.

```text
Client or agent
  -> Envoy listener on :8899
  -> vLLM Semantic Router
  -> AMD ROCm vLLM backend on vllm:8000
  -> Qwen/Qwen3.6-35B-A3B
```

In this recipe, the aliases are semantic lanes:

| Lane | Purpose |
| --- | --- |
| `qwen/qwen3.6-rocm` | Local AMD lane for private work, security containment, simple requests, and fallback traffic. |
| `google/gemini-2.5-flash-lite` | Medium lane for explanations, business analysis, and moderate coding work. |
| `google/gemini-3.1-pro` | Strong technical lane for complex code, architecture, STEM, and research synthesis. |
| `openai/gpt5.4` | Strong reasoning lane for hard non-private planning and long-horizon work. |
| `anthropic/claude-opus-4.6` | High-care domain lane for non-private legal, compliance, and health analysis. |

The aliases all point at the same AMD vLLM backend in the recipe. That keeps
the example local and reproducible while still exercising the router policy,
SAAR state, replay records, headers, and dashboard topology.

## What The Recipe Expresses

`agentic-saars.yaml` combines privacy-first policy, domain routing, difficulty
routing, and session-aware agent routing. The decision order is deliberate:
security and privacy routes win before agentic, domain, or complexity routes.

```text
local_security_containment
  -> local_privacy_policy
  -> agentic_session_route
  -> domain_legal_health
  -> domain_code_complex
  -> domain_code
  -> domain_stem_research
  -> domain_business
  -> complex_general
  -> medium_general
  -> simple_general
  -> default_general
```

The practical scenarios are:

| Scenario | Recipe behavior |
| --- | --- |
| Prompt-injection or jailbreak-like text | `local_security_containment` keeps it on the local AMD lane. |
| Private code, internal documents, credentials, PII, customer data, or employee data | `local_privacy_policy` keeps it on the local AMD lane. |
| Agent workflows with tool calls, tool results, multi-turn context, or execution plans | `agentic_session_route` uses SAAR when the request is not private. |
| Simple questions and short summaries | `simple_general` routes to the local simple lane. |
| Complex architecture, systems, coding, and planning | `domain_code_complex` or `complex_general` routes to stronger lanes. |
| Legal, compliance, and health analysis | `domain_legal_health` routes to the high-care domain lane. |
| Business, product, STEM, and research synthesis | Domain decisions route to the matching specialist lane. |
| Anything unmatched | `default_general` falls back to the local AMD lane. |

Privacy routing is content-sensitive. The recipe does not rely on users saying
"use the local model" or "do not send this to the cloud." Those phrases are
only weak auxiliary hints. The high-signal privacy path comes from PII,
credential, private-code, internal-document, customer-data, employee-data, and
similar markers. That means a prompt containing an API key, payroll row,
customer export, internal incident review, or proprietary repository snippet
can stay local even when the user never explicitly asks for local processing.

## SAAR On AMD

SAAR stands for Session-Aware Agentic Routing. In this recipe it appears as
the `session_aware` algorithm on `agentic_session_route`.

The route is only eligible for non-private agent traffic. Privacy and security
policy have already run before SAAR is considered, so a private coding session
stays local even if it is complex or tool-heavy.

The SAAR policy is tuned for stable agent loops:

- `tool_loop_hard_lock` keeps a tool result with the session that requested it.
- `context_portability_hard_lock` avoids moving non-portable continuation state.
- `decision_drift_reset` allows a real task change to re-evaluate the route.
- `prefix_cache_weight` treats warm context as useful session state.
- `handoff_penalty_weight` and `switch_history_weight` reduce oscillation.
- `remaining_turn_prior_weight` is conservative early in likely-long sessions.

This is the core of the AMD recipe: the router does not just classify a
single prompt. It tracks enough session context to keep an agent workflow
coherent while still enforcing local-first policy.

In a replay-backed validation run, SAAR produced both expected stability
actions:

| Case | Replay evidence |
| --- | --- |
| A simple turn followed by an agentic refactor plan | `session_action=switch`, from `qwen/qwen3.6-rocm` to `google/gemini-2.5-flash-lite`. |
| A tool-result turn after a complex tool request | `session_action=hard_lock`, keeping `google/gemini-3.1-pro` even though the fresh proposal was cheaper. |

Those actions are visible in router replay under `route_diagnostics` and
`session_policy`, not inferred from public response headers.

## Eval Snapshot

The profile was evaluated with `inferoa eval --profile=agentic-routing`
against the OpenAI-compatible router endpoint. The run covered fresh-session
routing, multi-run same-session trade-offs, privacy containment, and tool-loop
stability.

| Metric | Result |
| --- | --- |
| Checks passed | `105 / 105` |
| Acceptable routing runs | `11 / 11` |
| Questionable or unreasonable routing | `0` |
| Model turns | `12` |
| SAAR actions observed | `switch=1`, `hard_lock=1` |
| Prompt cache hit rate | `61.10%` |
| Prefix-cache discount, using recipe prices | `$0.003051776` |
| Auto cost vs always `google/gemini-3.1-pro` | `69.76%` lower |
| Auto cost vs always `openai/gpt5.4` | `87.91%` lower |

The route coverage matched the intended recipe:

| Scenario | Observed route |
| --- | --- |
| Simple math | `simple_math_fast_path -> qwen/qwen3.6-rocm` |
| Business analysis | `domain_business -> google/gemini-2.5-flash-lite` |
| Complex code or architecture | `domain_code_complex -> google/gemini-3.1-pro` |
| Privacy-sensitive input | `local_privacy_policy -> qwen/qwen3.6-rocm` |
| Agentic session trade-off | `agentic_session_route` with a replayed `switch` or `hard_lock` action |

## Run The Recipe

Start an AMD ROCm vLLM backend that serves the aliases used by the recipe.
The backend model is `Qwen/Qwen3.6-35B-A3B` and the expected in-network
endpoint is `vllm:8000`.

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

Those cache flags make the vLLM token-cache behavior visible during SAAR
testing:

| Flag | Why it is in the recipe |
| --- | --- |
| `--enable-prefix-caching` | Reuses matching request prefixes across agent turns. |
| `--prefix-caching-hash-algo sha256` | Uses collision-resistant prefix hashes. |
| `--kv-cache-metrics` | Exposes KV-cache usage, residency, and reuse metrics. |
| `--kv-cache-metrics-sample 0.01` | Samples 1% of cache blocks for low-overhead metrics. |
| `--enable-logging-iteration-details` | Logs per-iteration context and generation token counts. |

Check startup capacity and live token-cache details with:

```bash
sudo docker logs vllm 2>&1 | \
  grep -Ei 'kv cache|prefix cache|maximum concurrency|iteration'

curl -s "http://localhost:${VLLM_PORT_QWEN36:-8090}/metrics" | \
  grep -E 'vllm:cache_config_info|vllm:kv_cache_usage_perc|prefix_cache|kv_cache'
```

Then serve the router profile:

```bash
vllm-sr serve --image-pull-policy never --platform amd \
  --config deploy/recipes/agentic-saars.yaml
```

For dashboard-first setup, import the same recipe file:

```text
deploy/recipes/agentic-saars.yaml
```

## Validate The Scenarios

Use the OpenAI-compatible route through the router listener. The response
headers are the quickest way to confirm the selected decision and alias.
All validation requests below use `vllm-sr/auto`; the router then selects the
served-model alias.

Simple request:

```bash
curl -s -D /tmp/headers.txt http://localhost:8899/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Briefly define semantic routing in one sentence."}]
  }'

grep -i 'x-vsr-selected' /tmp/headers.txt
```

Expected route: `simple_general`.

Private request:

```bash
curl -s -D /tmp/headers.txt http://localhost:8899/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "This repository snippet includes a private API key sk-test-private-000. Identify the risk."}]
  }'
```

Expected route: `local_privacy_policy`.

Domain request:

```bash
curl -s -D /tmp/headers.txt http://localhost:8899/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Review this contract clause for indemnity, liability cap, and compliance exposure."}]
  }'
```

Expected route: `domain_legal_health`.

Agentic tool-loop request:

```bash
curl -s -D /tmp/headers.txt http://localhost:8899/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-session-id: demo-agent-session' \
  -d '{
    "model": "vllm-sr/auto",
    "tools": [{
      "type": "function",
      "function": {
        "name": "run_tests",
        "description": "Run the relevant validation command.",
        "parameters": {"type": "object", "properties": {}}
      }
    }],
    "messages": [{"role": "user", "content": "Plan and execute a migration with checkpoints, rollback, and validation."}]
  }'
```

Expected route: `agentic_session_route`. Follow-up tool-result turns on the
same `x-session-id` should preserve the agent loop; inspect
`x-vsr-selected-decision`, `x-vsr-selected-model`, and `x-vsr-session-phase`.

## What To Inspect

The profile enables replay on the routes where debugging matters most. In the
dashboard and response headers, inspect:

- matched decision and selected alias
- privacy and security policy outcomes
- agent session phase
- replay id and replay `route_diagnostics`
- SAAR `session_policy`, including previous, proposed, and selected models
- tool trace snippets
- replayed request and response samples
- topology path through signals, projections, decisions, and model selection

Those surfaces make the recipe useful for real agent development. You can
verify that private tasks stay local, simple tasks take the simple path,
domain tasks hit the expected specialist route, and tool-heavy agent sessions
remain stable under SAAR.

## Files

- AMD playbook: `deploy/amd/AGENTIC.md`
- Runnable recipe: `deploy/recipes/agentic-saars.yaml`
- Config validation:

```bash
PYTHONPATH=src/vllm-sr python3 -m cli.main validate \
  --config deploy/recipes/agentic-saars.yaml
```
