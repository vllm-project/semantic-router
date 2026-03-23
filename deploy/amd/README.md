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

The active AMD profile contains 22 routing decisions:

- `simple_*` (3): lowest-cost FAQ and general fallback
- `medium_*` (5): low-to-mid-cost domain/scenario refinement
- `verified_*` (5): evidence-sensitive overlays layered just above their base routes
- `feedback_*` (2): explicit correction and clarification recovery lanes
- `complex_*` (3): hard technical, STEM, and agentic synthesis
- `reasoning_*` (3): high-reasoning escalation
- `premium_*` (1): one premium legal path only

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

Onboarding remote import can apply the full YAML directly. If you import the same file into the DSL editor, the routing surfaces decompile from `routing.modelCards`, `routing.signals`, `routing.projections`, and `routing.decisions`, while `providers` stays YAML-native.

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
  +-- projection coordination
  |   - domain partition winner
  |   - intent partition winner
  |   - difficulty band
  |   - verification band
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

| Priority | Decision | Alias | What it is for | Match sketch |
|---------:|----------|-------|----------------|--------------|
| 260 | `premium_legal` | `anthropic/claude-opus-4.6` | Highest-risk legal and compliance analysis | `domain:law` + `projection:verification_required` + premium legal embedding or hard legal-risk / hard routing band |
| 250 | `reasoning_math` | `openai/gpt5.4` | Proofs, derivations, and hard math | `domain:math` + `projection:balance_reasoning` |
| 245 | `reasoning_philosophy` | `openai/gpt5.4` | Philosophy prompts that need deep argumentation | `domain:philosophy` + `projection:balance_reasoning` |
| 243 | `complex_agentic` | `google/gemini-3.1-pro` | High-structure execution plans, migrations, and workflow orchestration | agentic embedding / preference / markers + `projection:balance_complex` or `projection:balance_reasoning`, excluding architecture markers |
| 240 | `complex_architecture` | `google/gemini-3.1-pro` | Complex systems and architecture design | CS or engineering + architecture embedding / markers + `projection:balance_complex` or `projection:balance_reasoning` |
| 235 | `complex_stem` | `google/gemini-3.1-pro` | Complex STEM synthesis outside dedicated math | STEM domain + STEM or research embedding, or high routing band |
| 232 | `feedback_wrong_answer_verified` | `google/gemini-3.1-pro` | Explicit correction on evidence-sensitive follow-ups | `user_feedback:wrong_answer` + correction markers + short/medium context + verification pressure or evidence-synthesis escalation |
| 220 | `medium_code_general` | `qwen/qwen3.5-rocm` | Low-medium cost coding, debugging, and technical Q&A | code domain / markers / embedding / coding preference + `projection:balance_medium` or `projection:balance_complex`, excluding agentic, architecture, and creative cues |
| 216 | `verified_business` | `google/gemini-2.5-flash-lite` | Evidence-sensitive business or economics requests | business/economics + `projection:verification_required` or hard evidence synthesis + business embedding or medium/complex routing band |
| 215 | `medium_business` | `qwen/qwen3.5-rocm` | Mid-tier business and economics analysis | business/economics + `embedding:business_analysis` + `projection:balance_medium` or `projection:balance_complex`, excluding verification overlay |
| 214 | `verified_health` | `google/gemini-3.1-pro` | Evidence-sensitive health and medical guidance | `domain:health` + `projection:verification_required` + health embedding or medium/complex/reasoning band |
| 211 | `verified_history` | `google/gemini-2.5-flash-lite` | Source-sensitive history explanation | `domain:history` + `projection:verification_required` or hard evidence synthesis + history embedding or medium/complex routing band |
| 210 | `medium_history` | `qwen/qwen3.5-rocm` | Mid-tier history explanation and comparison | `domain:history` + `embedding:history_explainer` + `projection:balance_medium` or `projection:balance_complex`, excluding verification overlay |
| 205 | `medium_psychology` | `qwen/qwen3.5-rocm` | Psychology and behavior queries with nuanced explanation | `domain:psychology` + `embedding:psychology_support` + `projection:balance_medium` or `projection:balance_complex` |
| 200 | `medium_creative` | `google/gemini-2.5-flash-lite` | Creative writing, copywriting, and ideation | creative markers / embedding / collaboration preference + `projection:balance_simple` or `projection:balance_medium` |
| 190 | `reasoning_general` | `openai/gpt5.4` | Non-specialist deep analysis and multi-step reasoning | reasoning / research / multi-step cues + `projection:balance_complex` or `projection:balance_reasoning`, excluding specialist embeddings and broad technical markers |
| 185 | `feedback_need_clarification` | `qwen/qwen3.5-rocm` | Cheap clarification follow-up lane | `user_feedback:need_clarification` + clarification markers + short/medium context |
| 181 | `verified_fast_qa_zh` | `qwen/qwen3.5-rocm` | Chinese short FAQ with explicit verification ask | `embedding:fast_qa_zh` + `language:zh` + `context:short_context` + simple/medium routing band + verification cue or fact-check pressure |
| 180 | `simple_fast_qa_zh` | `qwen/qwen3.5-rocm` | Cheapest Chinese factual / definitional answers | `embedding:fast_qa_zh` + `language:zh` + `context:short_context` + `projection:balance_simple`, excluding verification overlay |
| 176 | `verified_fast_qa_en` | `qwen/qwen3.5-rocm` | English short FAQ with explicit verification ask | `embedding:fast_qa_en` + `language:en` + `context:short_context` + simple/medium routing band + verification cue or fact-check pressure |
| 175 | `simple_fast_qa_en` | `qwen/qwen3.5-rocm` | Cheapest English factual / definitional answers | `embedding:fast_qa_en` + `language:en` + `context:short_context` + `projection:balance_simple`, excluding verification overlay |
| 170 | `simple_general` | `qwen/qwen3.5-rocm` | Lowest-cost fallback for non-specialized traffic | short simple traffic, or medium-context `domain:other` traffic with simple/medium band, excluding fast-QA embeddings |

This ordering is intentional:

- specialized premium and hard-reasoning routes win first
- explicit correction recovery beats ordinary medium traffic, but only with strong confirmation cues
- factual overlays sit just above their cheap base routes instead of replacing them
- complex technical routes beat generic reasoning routes
- medium routes only accept easy or medium complexity
- simple routes remain the broad default landing zone

## Signal Overview

The profile uses the standard vSR signal families directly under `routing.signals`:

| Signal family | Role in this profile | Representative names |
|---------------|----------------------|----------------------|
| `keywords` | explicit lexical confirmation for route style, verification asks, feedback cues, and task shape | `verification_markers`, `agentic_request_markers`, `architecture_markers`, `clarification_feedback_markers` |
| `embeddings` | learned intent and specialist boundaries | `fast_qa_en`, `architecture_design`, `business_analysis`, `premium_legal_analysis`, `reasoning_general_en` |
| `fact_check` | evidence-sensitive detection that feeds verification pressure | `needs_fact_check` |
| `user_feedbacks` | explicit correction or clarification overlays | `wrong_answer`, `need_clarification` |
| `preferences` | collaboration style and request framing | `coding_partner`, `creative_collaboration`, `agentic_execution` |
| `language` | language-specific fast-QA split | `en`, `zh` |
| `domains` | subject-area routing and partitioning | `law`, `math`, `history`, `health`, `computer science`, `other` |
| `context` | token-count bands for cheap fallback versus longer tasks | `short_context`, `medium_context`, `long_context` |
| `complexity` | easy / medium / hard difficulty for general, code, math, legal, agentic, and evidence-heavy requests | `general_reasoning`, `code_task`, `math_task`, `legal_risk`, `agentic_delivery`, `evidence_synthesis` |

Notable profile-specific signal details:

- `context` bands are non-overlapping: `short_context` is `0-999`, `medium_context` is `1K-7999`, and `long_context` is `8K-256K`.
- `complexity` signals are reusable across both route predicates and projection scores through sublevels such as `code_task:hard` or `evidence_synthesis:medium`.
- short lexical verification and correction cues are intentionally literal in this profile, so examples that say `verify this`, `answer with citations`, or Chinese `给出处` are more reliable than looser paraphrases.
- `jailbreak` and `pii` signals are still defined in the profile for safety surfaces, but they are not the primary routing predicates for the 22 active decisions.

## Projection Overview

The profile uses `routing.projections` as the coordination layer between raw signal detections and final route selection.

| Projection | Kind | Purpose | Outputs or members |
|------------|------|---------|--------------------|
| `balance_domain_partition` | partition | resolves one domain winner across the supported routing domains | biology, business, chemistry, computer science, economics, engineering, health, history, law, math, other, philosophy, physics, psychology |
| `balance_intent_partition` | partition | resolves one learned-intent winner across the maintained embedding lanes | `agentic_workflows`, `architecture_design`, `code_general`, `creative_tasks`, `fast_qa_en`, `fast_qa_zh`, `general_chat_fallback`, and related specialist embeddings |
| `difficulty_score` | score | blends context, keywords, embeddings, and complexity sublevels into one difficulty signal | source for the difficulty band mapping |
| `difficulty_band` | mapping | converts `difficulty_score` into reusable routing bands | `balance_simple`, `balance_medium`, `balance_complex`, `balance_reasoning` |
| `verification_pressure` | score | blends `fact_check`, verification cues, high-stakes domains, long-context pressure, and wrong-answer correction pressure | source for the verification mapping |
| `verification_band` | mapping | converts `verification_pressure` into verification routing outputs | `verification_standard`, `verification_required` |

In practice, the profile routes in two steps:

1. Raw signals fire under `routing.signals`.
2. Projections turn that raw evidence into named outputs such as `balance_complex` or `verification_required`.
3. Decisions combine ordinary signals with those projection outputs.

That lets the profile reuse one difficulty story and one verification story across many routes without repeating the same threshold logic inside every decision.

## Usage Examples

Test these in the dashboard playground at `http://<your-server-ip>:8700`:

The same stable examples are also maintained as machine-readable probes in [`balance.probes.yaml`](./balance.probes.yaml) for live `POST /api/v1/eval` calibration loops. The maintained suite currently covers all 22 decisions with 54 probe variants, so routing changes are checked against a small robustness set instead of one crafted prompt per route.

Each decision below includes every maintained probe variant from the manifest, so the README stays copy-pasteable for playground checks and aligned with the executable eval suite.

### `premium_legal`

Expected alias: `anthropic/claude-opus-4.6`

High-stakes legal analysis that should avoid generic business routing.

#### `contract_clause_analysis`

```text
Provide a legal analysis of the indemnity clause, liability cap, and compliance obligations in this contract.
```

#### `regulatory_risk_review`

```text
Assess the legal risk in this agreement by analyzing indemnification, limitation of liability, and the compliance duties each party assumes.
```

### `reasoning_math`

Expected alias: `openai/gpt5.4`

Pure mathematical reasoning that should not collapse into a generic reasoning lane.

#### `irrationality_proof`

```text
Prove rigorously that the square root of 2 is irrational.
```

#### `integer_ratio_proof`

```text
Give a formal proof that sqrt(2) cannot be expressed as a ratio of integers.
```

### `reasoning_philosophy`

Expected alias: `openai/gpt5.4`

Philosophy-domain reasoning with explicit argumentative depth.

#### `av_ethics`

```text
Compare utilitarianism and deontology, then argue which framework better handles autonomous-vehicle dilemmas.
```

#### `compatibilism_argument`

```text
Compare compatibilism and libertarian free will, then argue which view better explains moral responsibility.
```

### `complex_agentic`

Expected alias: `google/gemini-3.1-pro`

Multi-step agentic planning with execution phases, checkpoints, and rollback structure.

#### `migration_runbook`

```text
Plan a zero-downtime monolith-to-microservices migration with checkpoints, rollback steps, owners, and validation after each phase.
```

#### `platform_cutover_plan`

```text
Create a phased runbook for consolidating two internal platforms, with owners, dependencies, rollback criteria, and verification gates for every phase.
```

#### `incident_recovery_runbook`

```text
Create a phased incident-recovery runbook with owners, checkpoints, rollback criteria, and verification gates for each stage.
```

### `complex_architecture`

Expected alias: `google/gemini-3.1-pro`

System-design and architecture requests without strong workflow-orchestration cues.

#### `distributed_rate_limiter`

```text
Design the software architecture for a distributed rate limiter in a microservices platform, including service boundaries and consistency tradeoffs.
```

#### `multi_region_feature_flags`

```text
Design the architecture for a multi-region feature-flag service, including storage boundaries, cache strategy, and consistency tradeoffs.
```

### `complex_stem`

Expected alias: `google/gemini-3.1-pro`

Specialist STEM reasoning with technical explanation plus experiment design.

#### `battery_degradation`

```text
In electrochemistry terms, compare SEI growth, lithium plating, and cathode cracking as causes of lithium-ion battery degradation, then propose experiments to isolate the dominant mechanism.
```

#### `qubit_decoherence`

```text
Compare dielectric loss, flux noise, and quasiparticle poisoning as causes of superconducting-qubit decoherence, then propose experiments to isolate the dominant source.
```

### `feedback_wrong_answer_verified`

Expected alias: `google/gemini-3.1-pro`

Wrong-answer correction requests that also require verified, sourced answers.

#### `roman_republic_correction`

```text
This is wrong. Please correct the explanation of why the Roman Republic collapsed and cite reliable historical sources.
```

#### `earlier_answer_wrong`

```text
You got this wrong earlier; re-answer why the Roman Republic collapsed and support the correction with sources.
```

#### `meiji_correction`

```text
That is incorrect. Please correct the explanation of what caused the Meiji Restoration and support the correction with sources.
```

### `medium_code_general`

Expected alias: `qwen/qwen3.5-rocm`

Mid-tier coding help without architecture-heavy or agentic workflow cues.

#### `python_stack_trace`

```text
Debug this Python stack trace and suggest the most likely fix.
```

#### `failing_unit_test`

```text
A Java unit test is failing after a refactor; explain the most likely cause and suggest the first fix to try.
```

#### `integration_test_refactor`

```text
After a refactor, an integration test started failing in a Java codebase. Explain the most likely cause and the first code change to inspect.
```

### `verified_business`

Expected alias: `google/gemini-2.5-flash-lite`

Business analysis with explicit evidence or source requirements.

#### `pricing_strategy_evidence`

```text
Verify this claim with evidence: compare two B2B SaaS pricing strategies and cite sources for the market-share claim.
```

#### `churn_benchmark_sources`

```text
Compare enterprise SaaS churn benchmarks and verify the claim with sources before drawing a conclusion.
```

#### `retention_sources`

```text
Compare B2B SaaS retention benchmarks and support the answer with sources before recommending a pricing model.
```

### `medium_business`

Expected alias: `qwen/qwen3.5-rocm`

Business reasoning without explicit verification requirements.

#### `plg_vs_slg`

```text
Explain when a mid-market SaaS company should prefer product-led growth over sales-led growth, and outline the trade-offs.
```

#### `pricing_model_tradeoffs`

```text
Explain when a B2B software company should prefer usage-based pricing over seat-based pricing, and outline the trade-offs.
```

#### `annual_vs_monthly`

```text
Explain the trade-offs between annual and monthly pricing for a B2B SaaS product.
```

### `verified_health`

Expected alias: `google/gemini-3.1-pro`

Health-domain answers with explicit reliable-source requirements.

#### `iron_deficiency`

```text
What are the early symptoms of iron deficiency? Please cite reliable medical sources.
```

#### `sleep_apnea_sources`

```text
What are common early signs of sleep apnea? Answer with citations to reliable medical sources.
```

### `verified_history`

Expected alias: `google/gemini-2.5-flash-lite`

History explanations that explicitly demand citations or verification.

#### `roman_republic_with_citations`

```text
Verify the claim and answer with citations: why did the Roman Republic collapse?
```

#### `meiji_restoration_sources`

```text
Explain what caused the Meiji Restoration and support the answer with reputable historical sources.
```

### `medium_history`

Expected alias: `qwen/qwen3.5-rocm`

History explanations without explicit evidence or verification overlays.

#### `roman_republic_plain`

```text
In plain language, explain why the Roman Republic collapsed.
```

#### `ming_dynasty_plain`

```text
In plain language, explain why the Ming dynasty fell.
```

#### `meiji_plain`

```text
In plain language, explain what caused the Meiji Restoration.
```

### `medium_psychology`

Expected alias: `qwen/qwen3.5-rocm`

Psychology explanation and practical intervention lane.

#### `procrastination`

```text
Why do people procrastinate even when the task matters, and what interventions tend to help?
```

#### `confirmation_bias`

```text
Why do people fall into confirmation bias, and what strategies usually help reduce it?
```

#### `procrastination_important_work`

```text
Why do people procrastinate on important work, and what interventions usually help?
```

### `medium_creative`

Expected alias: `google/gemini-2.5-flash-lite`

Creative ideation without specialist routing or verification cues.

#### `tea_house_taglines`

```text
Invent three evocative taglines for a fictional tea house called Moonleaf.
```

#### `pottery_branding`

```text
Invent three concise names and taglines for a boutique pottery studio with a calm, modern feel.
```

### `reasoning_general`

Expected alias: `openai/gpt5.4`

Deep reasoning requests with explicit non-specialist framing.

#### `uncertainty_frameworks`

```text
Compare three general approaches to reasoning under uncertainty, without focusing on any specific domain.
```

#### `inference_modes`

```text
Compare deductive, inductive, and abductive reasoning in general terms, without anchoring the discussion to any one domain.
```

### `feedback_need_clarification`

Expected alias: `qwen/qwen3.5-rocm`

Clarification feedback overlays that ask for a simpler restatement.

#### `explain_more_clearly`

```text
Explain that more clearly and give one simple example.
```

#### `restate_simply`

```text
That was confusing. Restate it more simply and walk me through one concrete example.
```

#### `concrete_example`

```text
Please explain that more clearly and give one concrete example.
```

### `verified_fast_qa_zh`

Expected alias: `qwen/qwen3.5-rocm`

Short Chinese factual questions with explicit verification or source cues.

#### `paris_source`

```text
法国的首都是巴黎吗？给出处。
```

#### `jupiter_verify`

```text
太阳系最大的行星是木星吗？请核实并给来源。
```

#### `australia_capital_verify`

```text
澳大利亚的首都是悉尼还是堪培拉？请核实并给出处。
```

### `simple_fast_qa_zh`

Expected alias: `qwen/qwen3.5-rocm`

Short Chinese factual questions without verification overlays.

#### `water_formula`

```text
水的化学式是什么？请简短回答。
```

#### `months_in_year`

```text
一年有几个月？请简短回答。
```

### `verified_fast_qa_en`

Expected alias: `qwen/qwen3.5-rocm`

Short English factual questions with explicit verification or source requirements.

#### `australia_capital`

```text
Verify this with a source: Is the capital of Australia Sydney or Canberra?
```

#### `light_vs_sound`

```text
Verify with a source whether light travels faster than sound.
```

### `simple_fast_qa_en`

Expected alias: `qwen/qwen3.5-rocm`

Very short English factual or identity-style Q and A without verification cues.

#### `who_are_you`

```text
Who are you? Answer briefly.
```

#### `arithmetic`

```text
What is 2 + 2? Answer briefly.
```

### `simple_general`

Expected alias: `qwen/qwen3.5-rocm`

Short general explanations that should avoid fast-QA and specialist routes.

#### `induction_cooktops`

```text
In one short paragraph, explain how induction cooktops work for a home kitchen user.
```

#### `refrigerator_cooling`

```text
In one short paragraph, explain how a refrigerator keeps food cold.
```

#### `heat_pump_plain`

```text
In one short paragraph, explain how composting works for a first-time apartment resident.
```

## Validation Checklist

- `sudo docker ps --filter name=vllm` shows the single backend container as healthy.
- `curl -s "http://localhost:${VLLM_PORT_122B:-8090}/v1/models"` lists all five tier-aware alias IDs.
- The router is started with `vllm-sr serve --image-pull-policy never --platform amd`.
- Requests hitting the playground show matched signals, one selected decision, one selected alias, and cost/savings in Insights.
- `deploy/recipes/balance.dsl` remains aligned with the maintained routing authoring story, and `deploy/recipes/balance.yaml` remains aligned with this document's alias catalog, signal summary, projection summary, decision table, and examples.

## Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [vLLM Semantic Router GitHub](https://github.com/vllm-project/semantic-router)
