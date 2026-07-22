# Balance Router Recipe

This recipe is the recommended default general-purpose routing profile. It trades off capability, cost, latency, and safety by escalating from a free local lane through mid-tier and complex models, reserving premium aliases only for high-stakes or specialist work.

The maintained assets live in the parent `deploy/recipes/` directory:

- `balance.yaml`
- `balance.dsl`
- `balance.probes.yaml`

All five model aliases currently route through the shared `vllm:8000` backend as mock provider names rather than calling live external APIs. That keeps the recipe runnable in local dev without gated credentials while still demonstrating cost-aware routing economics.

## Design Goals

- Keep ordinary traffic on `qwen/qwen3.5-rocm`, the free local default lane.
- Escalate to `google/gemini-2.5-flash-lite` for low-cost verified explanation and correction overlays.
- Use `google/gemini-3.1-pro` for complex specialist, deep reasoning, and verified health work.
- Reserve `openai/gpt5.4` for formal math proofs and other narrow premium reasoning overlays.
- Reserve `anthropic/claude-opus-4.6` for high-stakes legal and compliance analysis.
- Route feedback-driven clarification cheaply and route evidence-sensitive corrections without over-escalating to the complex tier.
- Keep routing policy-driven through signals, projections, and decisions rather than user preference.
- Record every decision through `router_replay` on every maintained route.

Routing behavior is expressed in `routing.signals`, `routing.projections`, and `routing.decisions`. The paired `balance.dsl` file uses `DECISION_TREE` authoring sugar; `balance.yaml` is the canonical flat runtime representation consumed by the router.

## Model Assumptions

| Alias | Role | Backend |
|---|---|---|
| `qwen/qwen3.5-rocm` | Local default, coding, explainers, creative, fast QA, fallbacks | `vllm:8000` |
| `google/gemini-2.5-flash-lite` | Verified explainers and wrong-answer correction | `vllm:8000` |
| `google/gemini-3.1-pro` | Complex specialist, deep reasoning, verified health | `vllm:8000` |
| `openai/gpt5.4` | Formal math proofs | `vllm:8000` |
| `anthropic/claude-opus-4.6` | Premium legal and compliance analysis | `vllm:8000` |

Replace `providers.models[].backend_refs[]` and alias names when binding this recipe to your own model pool. The dashboard preset metadata in `dashboard/backend/handlers/presets.go` lists the same five aliases as the maintained `balance` preset.

## Route Order

Higher priority wins. The probe suite calibrates the 13 non-fallback lanes below; `casual_chat` remains an absolute safety net rather than a first-class calibration target.

| Priority | Decision | Target model | Reasoning | Purpose |
|---|---|---|---|---|
| `260` | `premium_legal` | `anthropic/claude-opus-4.6` | high | High-stakes legal and compliance analysis |
| `252` | `formal_math_proof` | `openai/gpt5.4` | high | Formal math proofs and derivations |
| `250` | `reasoning_deep` | `google/gemini-3.1-pro` | high | Deep philosophy and first-principles reasoning |
| `242` | `complex_specialist` | `google/gemini-3.1-pro` | high | Multi-step runbooks, systems design, specialist STEM |
| `232` | `feedback_wrong_answer_verified` | `google/gemini-2.5-flash-lite` | off | Wrong-answer correction with source requirements |
| `220` | `medium_code_general` | `qwen/qwen3.5-rocm` | medium | Mid-tier coding, debugging, and technical Q&A |
| `218` | `verified_health` | `google/gemini-3.1-pro` | medium | Health guidance with explicit source requirements |
| `214` | `verified_explainer` | `google/gemini-2.5-flash-lite` | off | Evidence-sensitive business, history, or psychology explanation |
| `212` | `feedback_need_clarification` | `qwen/qwen3.5-rocm` | off | Clarification follow-ups and simple restatements |
| `208` | `medium_explainer` | `qwen/qwen3.5-rocm` | medium | Business, history, and psychology explanation without verification overlays |
| `200` | `medium_creative` | `qwen/qwen3.5-rocm` | off | Creative ideation and interpersonal drafting |
| `184` | `fast_qa` | `qwen/qwen3.5-rocm` | off | Short factual questions in English or Chinese |
| `170` | `simple_general` | `qwen/qwen3.5-rocm` | off | Everyday explanations and lightweight drafting |
| `10` | `casual_chat` | `qwen/qwen3.5-rocm` | off | Final fallback when no earlier lane matches |

The route order is the core control surface. Premium legal and formal math win before general reasoning; feedback and verification overlays sit above ordinary explainers; fast QA and simple general traffic stay on the cheapest lane unless complexity or domain signals justify escalation.

## Cost Profile

Pricing is configured in `providers.models[].pricing`, which is the layer the router uses for cost accounting, replay cost snapshots, and savings calculations.

| Model | Prompt / 1M | Completion / 1M | Role |
|---|---|---|---|
| `qwen/qwen3.5-rocm` | `$0.00` | `$0.00` | Local default and cheapest lanes |
| `google/gemini-2.5-flash-lite` | `$0.01` | `$0.04` | Low-cost verified explanation and correction |
| `google/gemini-3.1-pro` | `$0.48` | `$1.92` | Complex specialist and deep reasoning |
| `openai/gpt5.4` | `$1.20` | `$4.80` | Formal math and premium reasoning overlay |
| `anthropic/claude-opus-4.6` | `$1.80` | `$7.20` | Premium legal and high-risk analysis |

These values are example prices for routing economics and Insights demos, not vendor billing quotes.

## Signal Strategy

### Complexity and balance projections

Complexity bands and `balance_*` projections (`balance_simple`, `balance_medium`, `balance_complex`, `balance_reasoning`) drive most escalation decisions. They combine domain classifiers, embedding similarity, structure signals, and context length.

### Domain and specialty lanes

- **Legal:** `law` domain, `legal_risk_markers`, and `premium_legal_analysis` embeddings route to `premium_legal`.
- **Formal math:** `math` domain plus reasoning markers, excluding verification and specialist overlays, route to `formal_math_proof`.
- **Deep reasoning:** philosophy domain, research synthesis embeddings, and general reasoning embeddings route to `reasoning_deep`.
- **Complex specialist:** agentic workflows, architecture design, multi-step structure markers, and STEM embeddings route to `complex_specialist`.
- **Health:** `health` domain with verification pressure routes to `verified_health`.

### Feedback overlays

- `feedback_correction_verified` combines wrong-answer feedback, correction markers, and reask signals for `feedback_wrong_answer_verified`.
- `feedback_clarification_overlay` routes cheap clarification follow-ups to `feedback_need_clarification`.

### Verification pressure

`verification_required`, `verification_markers`, `reference_heavy_markers`, and `needs_fact_check` steer evidence-sensitive traffic toward verified lanes instead of ordinary explainers.

### Fast-path guardrails

`fast_qa_en`, `fast_qa_zh`, `simple_request_markers`, and `short_context` keep short factual questions on the cheap lane when specialist overlays are absent.

## Audit Behavior

Every maintained route enables `router_replay` with:

- `enabled: true`
- `max_records: 100000`
- `max_body_bytes: 4096`

That keeps a route-level audit trail for every decision without letting callers pick a model by preference.

## Run Locally

Build the local dev image and serve with the balance recipe:

```bash
make vllm-sr-dev
vllm-sr serve --image-pull-policy never --config deploy/recipes/balance.yaml
```

For AMD platform work, use `make vllm-sr-dev VLLM_SR_PLATFORM=amd` and `vllm-sr serve --image-pull-policy never --platform amd --config deploy/recipes/balance.yaml`.

## Customization Points

- **Model bindings:** update `providers.models[]` and each decision's `modelRefs[]` to point at your backend pool.
- **Pricing:** tune `providers.models[].pricing` so Insights and replay cost snapshots reflect your deployment economics.
- **Escalation thresholds:** adjust `routing.projections` and complexity bands before changing decision priority order.
- **Probe coverage:** extend `balance.probes.yaml` when adding or retuning a decision lane; keep positive, negative, and boundary variants per decision.
- **DSL round-trip:** edit `balance.dsl` for reviewable tree-shaped policy, then regenerate or sync `balance.yaml` when the authoring shape matters.

## Maintained Test Queries

These samples come from `balance.probes.yaml`. The full suite includes paraphrase and multi-turn variants for calibration robustness.

### `premium_legal`

- `Provide a legal analysis of the indemnity clause, liability cap, and compliance obligations in this contract.`

### `formal_math_proof`

- `Prove rigorously that the square root of 2 is irrational.`

### `reasoning_deep`

- `Compare utilitarianism and deontology, then argue which framework better handles autonomous-vehicle dilemmas.`

### `complex_specialist`

- `Plan a zero-downtime monolith-to-microservices migration with checkpoints, rollback steps, owners, and validation after each phase.`

### `feedback_wrong_answer_verified`

- `This is wrong. Please correct the explanation of why the Roman Republic collapsed and cite reliable historical sources.`

### `medium_code_general`

- `Debug this Python stack trace and suggest the most likely fix.`

### `verified_health`

- `What are the early symptoms of iron deficiency? Please cite reliable medical sources.`

### `verified_explainer`

- `Verify this claim with evidence: compare two B2B SaaS pricing strategies and cite sources for the market-share claim.`

### `feedback_need_clarification`

- `Explain that more clearly and give one simple example.`

### `medium_explainer`

- `In plain language, explain why the Roman Republic collapsed.`

### `medium_creative`

- `Invent three evocative taglines for a fictional tea house called Moonleaf.`

### `fast_qa`

- `Verify this with a source: Is the capital of Australia Sydney or Canberra?`

### `simple_general`

- `In one short paragraph, explain how induction cooktops work for a home kitchen user.`

## Validation Commands

Repo-local contract check:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/src/semantic-router"
go test ./pkg/config -run TestMaintainedConfigAssetsUseCanonicalV03Contract -count=1
```

Primary routing validation should stay probe-driven against a live router, because the important correctness property for this recipe is end-to-end decision behavior rather than a synthetic unit assertion.

Remote probe evaluation:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
ROUTER_URL="http://<router-host>:8080"
cd "$REPO_ROOT"
python3 tools/agent/scripts/router_calibration_loop.py eval \
  --router-url "$ROUTER_URL" \
  --probes deploy/recipes/balance.probes.yaml
```

Durable deploy:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
ROUTER_URL="http://<router-host>:8080"
cd "$REPO_ROOT"
python3 tools/agent/scripts/router_calibration_loop.py deploy \
  --router-url "$ROUTER_URL" \
  --yaml deploy/recipes/balance.yaml \
  --dsl deploy/recipes/balance.dsl
```

Docs-only changes should still pass the repo lint gate:

```bash
make agent-lint CHANGED_FILES="deploy/recipes/balance/README.md,deploy/recipes/README.md"
```

## Runtime Note

Router configuration is managed via the `/config/router` APIs, with durable versions tracked under `/config/router/versions`. After a durable deploy, confirm the in-memory routing surface matches the maintained local recipe before running probes. See the privacy recipe README for the same runtime workflow note.

This is a runtime behavior note, not part of the recipe contract itself. The maintained recipe still lives in the YAML, DSL, and probe assets listed above.
