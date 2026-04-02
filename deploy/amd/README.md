# vLLM Semantic Router on AMD ROCm

This playbook documents the AMD reference profile for a single real ROCm vLLM backend that exposes multiple semantic served-model aliases. The maintained `balance` profile is intentionally balance-first: it keeps a small number of real cost and risk lanes instead of treating every semantic niche as its own decision.

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
  - canonical authoring surface: [balance.dsl](../recipes/balance.dsl)
  - executable probe manifest: [balance.probes.yaml](../recipes/balance.probes.yaml)
  - `providers.defaults.default_model` points at the SIMPLE tier
  - `providers.models[].pricing` is example pricing for Insights cost comparison
  - `global.model_catalog.modules` can still tighten learned-signal thresholds without changing the routing-owned DSL surface

The active AMD profile contains 16 routing decisions:

- `simple_*` (2): cheapest factual and general fallback lanes
- `medium_*` (3): low-cost coding, explainer, and creative lanes
- `verified_*` (3): evidence-sensitive overlays layered above their cheaper base lanes
- `feedback_*` (2): explicit correction and clarification recovery lanes
- `complex_*` (2): hard technical and multi-step execution lanes
- `reasoning_*` (2): proof-grade math and deep general reasoning lanes
- `premium_*` (1): premium legal escalation only
- `fallback_*` (1): absolute terminal safety-net lane when no earlier decision matches

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

Onboarding remote import can apply the full YAML directly. If you import the same file into the DSL editor, the routing surfaces decompile from `routing.modelCards`, `routing.signals`, `routing.projections`, and `routing.decisions`, while `providers` and `global` stay YAML-native.

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
  |   - reask
  |   - language
  |   - context
  |   - structure
  |   - complexity
  |   - domain
  |
  +-- projection coordination
  |   - domain partition winner
  |   - intent partition winner
  |   - difficulty band
  |   - verification band
  |   - urgency band
  |
  +-- tiered decision selection
  |   - priority and tier choose one route
  |   - route rules combine raw signals with projection outputs
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

The runtime does not add a parallel “scorecard” schema. The profile expresses routing through native vSR signals and projections, then exposes the matched signals, chosen decision, and chosen alias in replay and Insights.

## Alias Catalog

| Tier | Alias | Example pricing per 1M tokens | Role in the profile |
|------|-------|-------------------------------|---------------------|
| SIMPLE | `qwen/qwen3.5-rocm` | prompt `$0.00`, completion `$0.00` | Free self-hosted default alias for fast QA, broad fallback, and most low-cost traffic |
| MEDIUM | `google/gemini-2.5-flash-lite` | prompt `$0.01`, completion `$0.04` | Low-cost expressive and verified explainer lane |
| COMPLEX | `google/gemini-3.1-pro` | prompt `$0.48`, completion `$1.92` | Hard technical, health, and verified correction lane |
| REASONING | `openai/gpt5.4` | prompt `$1.20`, completion `$4.80` | Proofs and deep general reasoning |
| PREMIUM | `anthropic/claude-opus-4.6` | prompt `$1.80`, completion `$7.20` | Reserved for legal and high-risk analysis |

Pricing is intentionally exaggerated for Insights demos so savings are easy to see. These values are not intended to mirror real vendor billing.

## Active Routing Decisions

| Priority | Decision | Alias | What it is for | Match sketch |
|---------:|----------|-------|----------------|--------------|
| 260 | `premium_legal` | `anthropic/claude-opus-4.6` | Highest-risk legal and compliance analysis | law or explicit legal-risk cues + premium legal embedding, verification overlay, or medium/hard `legal_risk` |
| 250 | `reasoning_math` | `openai/gpt5.4` | Proofs, derivations, and hard math | `domain:math` + `projection:balance_reasoning` or hard math complexity |
| 244 | `reasoning_specialist` | `openai/gpt5.4` | Philosophy and deep general reasoning | philosophy or reasoning/research cues + medium-or-higher reasoning band, excluding technical and correction overlays |
| 242 | `complex_agentic` | `google/gemini-3.1-pro` | Multi-step plans, migrations, and runbooks | agentic embedding / markers + workflow structure + medium-or-higher difficulty |
| 236 | `complex_technical` | `google/gemini-3.1-pro` | Systems design and specialist STEM synthesis | architecture or STEM cues + medium-or-higher difficulty, excluding fast-QA and workflow overlays |
| 232 | `feedback_wrong_answer_verified` | `google/gemini-3.1-pro` | Explicit correction on evidence-sensitive follow-ups | verified correction overlay + explicit correction evidence, excluding code-heavy recovery |
| 220 | `medium_code_general` | `qwen/qwen3.5-rocm` | Low-medium cost coding and bug triage | code markers / embedding + medium/complex band, plus urgent simple bug triage, excluding creative drafting |
| 218 | `verified_health` | `google/gemini-3.1-pro` | Evidence-sensitive health and medical guidance | `domain:health` + verification pressure + health guidance or medium+ band |
| 214 | `verified_explainer` | `google/gemini-2.5-flash-lite` | Evidence-sensitive business, history, and psychology explanation | explainer cues + verification pressure, excluding fast-QA and correction overlays |
| 212 | `feedback_need_clarification` | `qwen/qwen3.5-rocm` | Cheap clarification and single-turn re-ask lane | `projection:feedback_clarification_overlay`, excluding verified/correction/code overlays |
| 208 | `medium_explainer` | `qwen/qwen3.5-rocm` | Low-cost business, history, and psychology explanation | explainer cues + medium/complex band, or strong explainer embeddings in simple traffic, excluding verified overlays |
| 200 | `medium_creative` | `google/gemini-2.5-flash-lite` | Creative writing and interpersonal drafting | creative markers / embedding + simple or medium band |
| 184 | `verified_fast_qa` | `qwen/qwen3.5-rocm` | Short English or Chinese factual questions with verification | fast-QA embeddings or simple cue + short context + explicit verification cues |
| 180 | `simple_fast_qa` | `qwen/qwen3.5-rocm` | Cheapest short-context factual answers | fast-QA embeddings or simple cue + short context + simple band, excluding verification and urgency |
| 170 | `simple_general` | `qwen/qwen3.5-rocm` | Lowest-cost fallback for general traffic | short simple traffic or medium-context `domain:other` traffic, excluding explicit specialist and verification overlays |

This ordering is intentional:

- high-risk legal and proof-heavy reasoning win first
- projection-driven correction recovery beats ordinary verified or explainer traffic
- health and verified explainer overlays sit above their cheap base lanes
- complex technical and agentic routes beat generic medium lanes
- fast-QA stays cheap unless verification is explicit
- `simple_general` remains the broad fallback

## Signal Overview

The profile uses the standard vSR signal families directly under `routing.signals`:

| Signal family | Role in this profile | Representative names |
|---------------|----------------------|----------------------|
| `keywords` | explicit lexical confirmation for verification asks, legal risk, creative requests, coding cues, and feedback cues | `verification_markers`, `legal_risk_markers`, `code_request_markers`, `clarification_feedback_markers` |
| `embeddings` | learned intent and specialist boundaries | `fast_qa_en`, `architecture_design`, `business_analysis`, `premium_legal_analysis`, `agentic_workflows` |
| `fact_check` | evidence-sensitive detection that feeds verification pressure | `needs_fact_check` |
| `user_feedbacks` | weak explicit feedback evidence that feeds projection-driven overlays | `wrong_answer`, `need_clarification` |
| `reasks` | repeated same-question detection that strengthens clarification overlays | `likely_dissatisfied` |
| `language` | language-aware fast-QA detection | `en`, `zh` |
| `domains` | subject-area routing and partitioning | `law`, `math`, `history`, `health`, `computer science`, `other` |
| `context` | token-count bands for cheap fallback versus longer tasks | `short_context`, `medium_context`, `long_context` |
| `structure` | cheap workflow and urgency overlays | `ordered_workflow`, `numbered_steps`, `exclamation_emphasis` |
| `complexity` | reusable difficulty boundaries for general, code, math, legal, agentic, and evidence-heavy requests | `general_reasoning`, `code_task`, `math_task`, `legal_risk`, `agentic_delivery`, `evidence_synthesis` |

Notable profile-specific details:

- `context` bands are non-overlapping: `short_context` is `0-999`, `medium_context` is `1K-7999`, and `long_context` is `8K-256K`.
- `reask("likely_dissatisfied")` is intentionally narrow and only strengthens the clarification overlay; it is not a general-purpose escalation trigger.
- the profile no longer keeps emotion, preference, jailbreak, or PII signals in the routing-owned surface because they do not materially improve balance-driven model selection.
- `user_feedback` is not consumed directly by routes; it stays inside feedback projections so a single learned misfire does not steal short first-turn traffic.

## Projection Overview

The balance profile keeps only the projections that materially coordinate model selection:

| Projection | Purpose |
|------------|---------|
| `balance_domain_partition` | softmax-exclusive winner over the maintained domain set |
| `balance_intent_partition` | softmax-exclusive winner over the maintained intent embeddings |
| `difficulty_score` -> `difficulty_band` | maps traffic into `balance_simple`, `balance_medium`, `balance_complex`, and `balance_reasoning` |
| `verification_pressure` -> `verification_band` | marks `verification_required` traffic |
| `feedback_correction_pressure` -> `feedback_correction_band` | fuses explicit correction, verification, and anti-code evidence into `feedback_correction_verified` |
| `feedback_clarification_pressure` -> `feedback_clarification_band` | fuses clarification, `reask`, and anti-fast-QA evidence into `feedback_clarification_overlay` |
| `urgency_pressure` -> `urgency_band` | catches short urgent bug triage or other elevated-short-context requests |

The profile intentionally does not keep emotion-only projection bands. They added routing complexity without improving balance decisions.

## Calibration Loop

The stable examples are also maintained as machine-readable probes in [balance.probes.yaml](../recipes/balance.probes.yaml) for live `POST /api/v1/eval` calibration loops. The maintained suite currently covers the 15 calibrated non-fallback decisions with 54 probe variants, including a greeting guardrail that should stay on `simple_general` and multi-turn `messages` probes that exercise clarification and verified-correction follow-ups.

Run local validation first:

```bash
cd src/semantic-router
go run ./cmd/dsl validate ../../deploy/recipes/balance.dsl
```

Then run the repo-native routing calibration loop against a live router:

```bash
python3 tools/agent/scripts/router_calibration_loop.py run \
  --router-url http://<router-host>:8080 \
  --probes deploy/recipes/balance.probes.yaml \
  --yaml deploy/recipes/balance.yaml \
  --dsl deploy/recipes/balance.dsl
```

This produces versioned before / after artifacts under `.augment/router-loop/` and keeps the probe manifest, deployed YAML, and source DSL tied to the same run.
