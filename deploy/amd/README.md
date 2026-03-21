# vLLM Semantic Router on AMD ROCm

This playbook documents the AMD reference profile for a single real ROCm vLLM backend that exposes multiple semantic served-model aliases. The router keeps replay and Insights signal-native: records show matched signals, selected decision, selected alias, and cost/savings, without inventing a separate runtime dimension schema.

## Overview

- Physical backend model: `Qwen/Qwen3.5-122B-A10B-FP8`
- Docker service name expected by the profile: `vllm:8000`
- Served-model aliases exposed by the backend:
  - `qwen/qwen3.5-rocm`
  - `google/gemini-2.5-flash-lite`
  - `google/gemini-3.1-pro`
  - `openai/gpt5.4`
  - `anthropic/claude-opus-4.6`
- Reference routing profile: [balance.yaml](../recipes/balance.yaml)
  - Canonical layout: `version/listeners/providers/routing/global`
  - `providers.defaults.default_model` points at the SIMPLE tier
  - `providers.models[].pricing` is example pricing for Insights cost comparison
  - `routing.decisions` uses tier-prefixed dual-layer families
  - `global.model_catalog.modules` only tightens learned-signal thresholds for conservative overlays

The active AMD profile contains 20 routing decisions:

- `simple_*`: lowest-cost FAQ and general fallback
- `medium_*`: low-to-mid-cost domain/scenario refinement
- `verified_*`: hard-to-hit factual overlays that still bias toward cheap models
- `feedback_*`: explicit follow-up recovery lanes with narrow correction/clarification cues
- `complex_*`: hard technical and STEM synthesis
- `reasoning_*`: high-reasoning escalation
- `premium_*`: one premium legal path only

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
    --served-model-name qwen/qwen3.5-rocm google/gemini-2.5-flash-lite google/gemini-3.1-pro openai/gpt5.4 anthropic/claude-opus-4.6 \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --max-model-len 262144 \
    --language-model-only \
    --max-num-seqs 128 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.85
```

### Step 2: Install vLLM Semantic Router

```bash
curl -fsSL https://vllm-semantic-router.com/install.sh | bash
```

### Step 3: Access the dashboard

If everything is working, the dashboard is available at:

```text
http://<your-server-ip>:8700
```

Complete onboarding and import the reference profile from remote:

> https://raw.githubusercontent.com/vllm-project/semantic-router/main/deploy/recipes/balance.yaml

Onboarding remote import can apply the full YAML directly. If you import the same file into the DSL editor, the routing surfaces decompile from `routing.modelCards`, `routing.signals`, and `routing.decisions`, while `providers` stays YAML-native.

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
  |   - fact_check
  |   - user_feedback
  |   - preference
  |   - language
  |   - context
  |   - complexity
  |   - domain
  |
  +-- dual-layer decision selection
  |   - tier by decision family prefix
  |   - domain/scene refinement inside the tier
  |
  +-- alias-forwarded OpenAI request
  |   - SIMPLE: qwen/qwen3.5-rocm
  |   - MEDIUM: google/gemini-2.5-flash-lite
  |   - COMPLEX: google/gemini-3.1-pro
  |   - REASONING: openai/gpt5.4
  |   - PREMIUM: anthropic/claude-opus-4.6
  |
  v
Single ROCm vLLM backend on vllm:8000
  |
  v
Qwen/Qwen3.5-122B-A10B-FP8
```

The runtime does not add a separate 15-dimension scorecard. Instead, the profile expresses those routing ideas through native vSR signals and then exposes the matched signals, chosen decision, and chosen alias in replay and Insights.

The `fact_check` and `user_feedback` lanes are intentionally conservative:

- they require both the learned signal and explicit lexical confirmation
- they keep most traffic on `qwen/qwen3.5-rocm`
- they do not add any extra route-local plugins beyond the profile's existing replay capture

## Alias Catalog

| Tier | Alias | Example pricing per 1M tokens | Role in the profile |
|------|-------|-------------------------------|---------------------|
| SIMPLE | `qwen/qwen3.5-rocm` | prompt `$0.00`, completion `$0.00` | Free self-hosted default alias for fast QA, broad fallback, and most low-cost traffic |
| MEDIUM | `google/gemini-2.5-flash-lite` | prompt `$0.01`, completion `$0.04` | Low-cost expressive route for creative and softer medium tasks |
| COMPLEX | `google/gemini-3.1-pro` | prompt `$0.48`, completion `$1.92` | Hard STEM and architecture design |
| REASONING | `openai/gpt5.4` | prompt `$1.20`, completion `$4.80` | Multi-step reasoning, proofs, and philosophy |
| PREMIUM | `anthropic/claude-opus-4.6` | prompt `$1.80`, completion `$7.20` | Reserved for legal and high-risk analysis |

Pricing is intentionally exaggerated for Insights demos so savings are easy to see. These values are not intended to mirror real vendor billing.

## Active Routing Decisions

| Priority | Decision | Tier | Key signals | Target alias |
|---------:|----------|------|-------------|--------------|
| 260 | `premium_legal` | PREMIUM | `domain:law` + `embedding:premium_legal_analysis` or `complexity:legal_risk:medium/hard` | `anthropic/claude-opus-4.6` |
| 250 | `reasoning_math` | REASONING | `domain:math` + hard math / proof markers | `openai/gpt5.4` |
| 245 | `reasoning_philosophy` | REASONING | `domain:philosophy` + reasoning markers | `openai/gpt5.4` |
| 240 | `complex_architecture` | COMPLEX | CS or engineering + architecture design + multi-step or long-context | `google/gemini-3.1-pro` |
| 235 | `complex_stem` | COMPLEX | physics/chemistry/biology/engineering + hard synthesis | `google/gemini-3.1-pro` |
| 232 | `feedback_wrong_answer_verified` | COMPLEX | `user_feedback:wrong_answer` + explicit correction markers + evidence/high-stakes cue | `google/gemini-3.1-pro` |
| 220 | `medium_code_general` | MEDIUM | code/domain/preference + easy or medium code complexity | `qwen/qwen3.5-rocm` |
| 216 | `verified_business` | MEDIUM | business/economics + `fact_check:needs_fact_check` + explicit verification/source cue | `google/gemini-2.5-flash-lite` |
| 215 | `medium_business` | MEDIUM | business/economics + business embedding or non-hard complexity | `qwen/qwen3.5-rocm` |
| 211 | `verified_history` | MEDIUM | `domain:history` + `fact_check:needs_fact_check` + explicit verification/source cue | `google/gemini-2.5-flash-lite` |
| 210 | `medium_history` | MEDIUM | `domain:history` + explanatory or non-hard complexity | `qwen/qwen3.5-rocm` |
| 205 | `medium_psychology` | MEDIUM | `domain:psychology` + explanatory or non-hard complexity | `qwen/qwen3.5-rocm` |
| 200 | `medium_creative` | MEDIUM | creative keyword / embedding / preference | `google/gemini-2.5-flash-lite` |
| 190 | `reasoning_general` | REASONING | deep analysis markers + hard complexity or long context | `openai/gpt5.4` |
| 185 | `feedback_need_clarification` | SIMPLE | `user_feedback:need_clarification` + explicit clarification markers | `qwen/qwen3.5-rocm` |
| 181 | `verified_fast_qa_zh` | SIMPLE | short Chinese FAQ + `fact_check:needs_fact_check` + explicit verification/source cue | `qwen/qwen3.5-rocm` |
| 180 | `simple_fast_qa_zh` | SIMPLE | `embedding:fast_qa_zh` + `language:zh` + `context:short_context` | `qwen/qwen3.5-rocm` |
| 176 | `verified_fast_qa_en` | SIMPLE | short English FAQ + `fact_check:needs_fact_check` + explicit verification/source cue | `qwen/qwen3.5-rocm` |
| 175 | `simple_fast_qa_en` | SIMPLE | `embedding:fast_qa_en` + `language:en` + `context:short_context` | `qwen/qwen3.5-rocm` |
| 170 | `simple_general` | SIMPLE | short-context fallback for general traffic | `qwen/qwen3.5-rocm` |

This ordering is intentional:

- specialized premium and hard-reasoning routes win first
- explicit correction recovery beats ordinary medium traffic, but only with strong confirmation cues
- factual overlays sit just above their cheap base routes instead of replacing them
- complex technical routes beat generic reasoning routes
- medium routes only accept easy or medium complexity
- simple routes remain the broad default landing zone

## How This Profile Maps `clawrouter` Ideas into vSR Signals

The profile does not expose the `clawrouter` ideas as runtime dimension fields. It maps them into existing signal families instead:

| Routing idea | vSR expression in this profile |
|--------------|--------------------------------|
| `token_count` | `context` via `short_context`, `medium_context`, `long_context` |
| `code_presence` | `keyword:code_request_markers` plus `domain:computer science` |
| `reasoning_markers` | `keyword:reasoning_request_markers` plus `complexity:*:hard` |
| `technical_terms` | `keyword:code_request_markers` / `architecture_markers` plus `embedding:code_general` / `architecture_design` |
| `creative_markers` | `keyword:creative_request_markers` plus `preference:creative_collaboration` |
| `simple_indicators` | `keyword:simple_request_markers` plus `context:short_context` |
| `multi_step_patterns` | `keyword:multi_step_markers` and `keyword:agentic_request_markers` |
| `question_complexity` | `complexity:general_reasoning`, `complexity:code_task`, `complexity:math_task`, `complexity:legal_risk` |
| `imperative_verbs` | `keyword:agentic_request_markers` plus `preference:agentic_execution` |
| `constraint_count` | `keyword:constraint_markers` and structured complexity rules |
| `output_format` | `keyword:output_format_markers` plus `preference:structured_delivery` |
| `reference_complexity` | `keyword:reference_heavy_markers` plus domain-specific embeddings |
| `negation_complexity` | `keyword:negation_markers` |
| `domain_specificity` | `domain:*` plus embeddings like `business_analysis` or `history_explainer` |
| `agentic_task` | `keyword:agentic_request_markers` plus `preference:agentic_execution` and hard complexity |

That keeps the profile aligned with vSR's native routing model:

- heuristic signals carry explicit lexical cues
- learned signals carry fuzzy, higher-value boundaries
- decision names carry the tier and scene semantics

## Usage Examples

Test these in the dashboard playground at `http://<your-server-ip>:8700`:

### Example 1: Cheapest English FAQ

```text
Who are you? Answer briefly.
```

- Expected decision: `simple_fast_qa_en`
- Expected alias: `qwen/qwen3.5-rocm`

### Example 2: Cheapest Chinese FAQ

```text
CPU 是什么意思？请简短回答。
```

- Expected decision: `simple_fast_qa_zh`
- Expected alias: `qwen/qwen3.5-rocm`

### Example 3: Low-cost coding route

```text
Debug this Python stack trace and suggest the most likely fix.
```

- Expected decision: `medium_code_general`
- Expected alias: `qwen/qwen3.5-rocm`

### Example 4: Verified Chinese FAQ overlay

```text
法国的首都是哪里？请给出处。
```

- Expected decision: `verified_fast_qa_zh`
- Expected alias: `qwen/qwen3.5-rocm`

### Example 5: Verified business overlay

```text
Compare two B2B SaaS pricing strategies and cite sources for the market-share claim.
```

- Expected decision: `verified_business`
- Expected alias: `google/gemini-2.5-flash-lite`

### Example 6: Creative mid-tier route

```text
Brainstorm three launch taglines for a premium tea brand.
```

- Expected decision: `medium_creative`
- Expected alias: `google/gemini-2.5-flash-lite`

### Example 7: Complex architecture route

```text
Design a distributed rate limiter for multi-region traffic and explain the failure modes.
```

- Expected decision: `complex_architecture`
- Expected alias: `google/gemini-3.1-pro`

### Example 8: Reasoning math route

```text
Prove rigorously that the square root of 2 is irrational.
```

- Expected decision: `reasoning_math`
- Expected alias: `openai/gpt5.4`

### Example 9: Premium legal route

```text
Analyze the indemnity and limitation-of-liability clauses in this SaaS contract and summarize the main legal risks.
```

- Expected decision: `premium_legal`
- Expected alias: `anthropic/claude-opus-4.6`

### Example 10: Wrong-answer recovery

```text
That's wrong. Cite the source and answer again.
```

- Expected decision: `feedback_wrong_answer_verified`
- Expected alias: `google/gemini-3.1-pro`

### Example 11: Clarification recovery

```text
Explain that more clearly and give one simple example.
```

- Expected decision: `feedback_need_clarification`
- Expected alias: `qwen/qwen3.5-rocm`

## Validation Checklist

- `sudo docker ps --filter name=vllm` shows the single backend container as healthy.
- `curl -s "http://localhost:${VLLM_PORT_122B:-8090}/v1/models"` lists all five tier-aware alias IDs.
- The router is started with `vllm-sr serve --image-pull-policy never --platform amd`.
- Requests hitting the playground show matched signals, one selected decision, one selected alias, and cost/savings in Insights.
- `deploy/recipes/balance.dsl` remains aligned with the maintained routing authoring story, and `deploy/recipes/balance.yaml` remains aligned with this document's alias catalog, decision table, and examples.

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [vLLM Semantic Router GitHub](https://github.com/vllm-project/semantic-router)
