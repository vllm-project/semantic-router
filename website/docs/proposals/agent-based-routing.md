---
title: Agent-Based Routing
description: Design proposal for typed agent backends, single-agent selection, bounded handoff, and multi-agent composition under Epic #2994.
created: 2026-08-29
status: Proposal
---

> **Status:** Proposal · **Created:** 2026-08-29 · **Epic:** [#2994](https://github.com/vllm-project/semantic-router/issues/2994)

## Problem

Today the router selects **logical models** bound to inference endpoints. Agent
endpoints may also expose tools, memory, domain knowledge, state, permissions, and
a task contract. Treating both as the same opaque backend loses information needed
for safe selection, handoff, and collaboration.

The [Agent Routing recipe](../../../config/recipes/agent/README.md) routes agentic
**workloads to model lanes**. [Router Flow](./router-flow-workflows) orchestrates
bounded multi-**model** workflows. Neither provides a typed **agent backend**
contract, agent-aware selection, or router-native agent composition.

## Proposal

Extend the v0.3 configuration contract with an opt-in **agent target layer** that
preserves full compatibility with existing model backends. Recipes may declare mixed
candidate pools and choose among models and agents using the same decision, signal,
and trace surfaces.

This document defines the contract and phased delivery plan for maintainer review.
Implementation PRs should follow approval of this proposal, not precede it.

## Target kinds

| Kind | Role today | After this proposal |
| --- | --- | --- |
| `model` | Default. Logical model → `backend_refs` → inference endpoint. | Unchanged. |
| `agent` | Not expressible. | Logical agent → typed metadata + `backend_refs` → agent endpoint. |

A decision candidate references one kind explicitly. Model-only recipes require no
migration.

## Agent backend contract (draft)

Each agent entry in `providers.agents[]` describes a deployable agent target:

| Field group | Purpose |
| --- | --- |
| **Identity** | Stable `name`, `agent_id`, and semver `version` for catalog and trace. |
| **Access** | `backend_refs[]` using the same physical binding shape as models (`endpoint`, `base_url`, auth, health). |
| **Capabilities** | Declared skills the router may match against request or decision requirements (for example `code_edit`, `security_review`). |
| **Tools** | Supported tool identifiers for eligibility filtering. The router does not execute tools. |
| **I/O contract** | Expected request/response protocol surface (OpenAI-compatible default). |
| **Memory and state** | Expectations for session continuity: whether the agent owns memory, accepts external memory references, or is stateless. |
| **Trust boundary** | Scope label (`tenant_scoped`, `internal`, `external`) used by privacy and policy signals. |
| **Health and lifecycle** | Health probe path, deprecation, and availability metadata for selection and fallback. |

Illustrative shape (subject to review):

```yaml
providers:
  agents:
    - name: coding-agent-v1
      agent_id: acme/coding-agent
      version: "1.2.0"
      backend_refs:
        - base_url: https://agent.example/v1
          type: openai_compatible
      capabilities: [code_edit, test_run, repo_search]
      tools: [read_file, write_file, run_tests]
      trust_boundary: tenant_scoped
      health:
        path: /health
```

Routing catalog entries mirror `routing.modelCards` as `routing.agentCards` with
capability, cost, latency, and quality metadata used by selection.

Decision candidates use `targetRefs` (proposed name) alongside or instead of
`modelRefs`:

```yaml
routing:
  decisions:
    - name: coding_with_review
      rules: { ... }
      targetRefs:
        - kind: agent
          agent: coding-agent-v1
        - kind: agent
          agent: security-review-agent-v1
        - kind: model
          model: qwen/qwen3.6-rocm
      algorithm:
        type: agent_static   # deterministic baseline; name TBD
```

## Selection

**Phase 2 baseline (deterministic, before learned policies):**

1. Filter by required capabilities and tools (from decision rules or signals).
2. Apply domain, privacy, and trust constraints (reuse containment signals).
3. Prefer state continuity when `AgenticSessionContext` indicates an active agent
   or tool loop.
4. Tie-break on declared capacity, latency, and cost metadata.
5. Follow an explicit per-decision fallback chain.

Unsupported capability requirements fail explicitly unless the decision declares a
compatible degradation target.

## Handoff

Agent handoff carries a **gateway context envelope** (coordinated with
[#2546](https://github.com/vllm-project/semantic-router/issues/2546)) across
boundaries:

- provenance chain (which agent produced which artifact);
- permitted context slices and memory references;
- tool-result receipts and remaining tool authority;
- budget counters (tokens, time, cost);
- cancellation token.

The router validates the envelope before invoking the next participant. Handoff does
not rematch the semantic decision.

Session and tool-loop continuity guards from [Router Learning](./router-learning-memory-and-adaptations)
and [Model Execution Fallback](./model-execution-fallback) apply before an agent
switch mid-session.

## Bounded composition

Reuse the Router Flow execution substrate rather than building a parallel
orchestrator. Extend the worker pool from model-only `modelRefs` to kind-aware
`targetRefs`.

| Pattern | Router-native realization |
| --- | --- |
| Delegation | Planner assigns a subtask to one worker agent. |
| Sequential handoff | Static plan with envelope pass-through between agent steps. |
| Parallel collaboration | Independent agent steps with bounded concurrent execution. |
| Supervisor or judge | Dedicated judge step before synthesis. |
| Result synthesis | Final step aggregates bounded worker outputs. |

Composition limits (non-negotiable):

- maximum participants, depth, turns, tokens, time, and cost per plan;
- tool authority per step and per workflow;
- acyclic dependency graph;
- no widening beyond the decision's declared `targetRefs` pool;
- planner output is untrusted data validated before execution.

Proposed algorithm surface: extend `algorithm.type: workflows` with
`participant_kind: agent` or a sibling `agent_workflows` type. Final name is an open
question.

## Ownership

| Workgroup / layer | Responsibility |
| --- | --- |
| **Agentic & Context (#2987)** | Agent target semantics, selection objectives, handoff envelope usage, agent composition. |
| **MoM & Routing (#3037)** | Model pools and model-level collaboration only. Reuse looper primitives; do not conflate contracts. |
| **Data Plane & Networking** | Invocation, streaming, cancellation, health, transport. |
| **Evaluation & Quality** | Shared benchmark and regression contracts for agent selection and composition. |

## Phased delivery

Each phase is a separate implementation PR gated on maintainer review of the prior
phase.

| Phase | Deliverable | Epic completion criterion |
| --- | --- | --- |
| **0** | This proposal, execution plan, GitHub sub-issue decomposition | Research graduation gate |
| **1** | `providers.agents` schema, validation, docs; zero migration for model-only recipes | Agent backend contract |
| **2** | Deterministic agent selector, probes, bench harness | Single-agent selection baseline + eval |
| **3** | Handoff envelope integration with #2546, cancel, fallback, traces | Bounded handoff path |
| **4** | Agent-aware workflow extension, one composition baseline with measurable gain | Multi-agent composition baseline |
| **5** | Shadow, canary, promotion, rollback, unsupported-capability behavior | Lifecycle and observability review |

Phase 0 is proposal-only. Phases 1–5 are implementation tracks opened only after
contract agreement.

## Evaluation

Each phase ships reproducible evidence:

- **Phase 2:** probe suite and bench metrics for selection quality, latency, cost,
  and failure recovery versus declared baselines.
- **Phase 4:** same task set comparing best single agent versus one composition
  pattern; report collaboration gain, handoff loss, and trace completeness.
- **Phase 5:** shadow routing and offline promotion workflow reviewed by maintainers.

## Scope and non-goals

This proposal covers router-native agent target selection, handoff, and bounded
composition. It does **not**:

- build or host a general-purpose agent framework, tool platform, or memory system;
- implement internal reasoning or tools of individual agents;
- support arbitrary, unbounded DAGs, recursive delegation, or autonomous loops;
- replace agent runtimes, protocols, or external workflow orchestrators;
- duplicate model-only MoM selection and collaboration owned by #3037;
- require migration of existing model-only recipes.

## Open questions

- Final name for decision candidates: `targetRefs` versus extending `modelRefs` with
  `kind`.
- Whether agent composition extends `workflows` or introduces `agent_workflows`.
- Minimum envelope fields required for Phase 3 versus deferred to #2546.
- How agent pricing and usage metadata align with existing `ModelParams` and replay
  schema.
- Which trace fields are safe for user-facing responses during multi-agent plans.
- Graduation criteria to move the epic from `research` to scheduled delivery.

## References

- [Epic #2994: Enable agent-based routing and multi-agent composition](https://github.com/vllm-project/semantic-router/issues/2994)
- [Agentic & Context workgroup #2987](https://github.com/vllm-project/semantic-router/issues/2987)
- [Router Flow Workflows](./router-flow-workflows)
- [Model Execution Fallback](./model-execution-fallback)
- [Unified Config Contract v0.3](./unified-config-contract-v0-3)
- [Agent Routing recipe](../../../config/recipes/agent/README.md)
- [Trusted gateway context envelope #2546](https://github.com/vllm-project/semantic-router/issues/2546)
