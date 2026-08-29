# PL-0039: Agent-Based Routing (Epic #2994)

## Goal

Land a maintainer-reviewed agent backend contract and phased delivery plan before
any implementation PR. Epic #2994 extends the router from model selection to typed
agent selection, handoff, and bounded composition without becoming a general-purpose
agent framework.

## Scope

- Proposal-only PR: design document, proposals index, sidebar, and this execution
  plan.
- Decompose Epic #2994 into phased implementation tracks after proposal approval.
- Coordinate envelope and context boundaries with #2546, #2984, and #2973.

## Non-Goals

- Implementing config schema, selectors, handoff middleware, or workflow extensions
  in the proposal PR.
- Building agent runtimes, tool platforms, or memory systems.
- Duplicating MoM model collaboration owned by #3037.

## Exit Criteria

- [ ] Proposal merged at `website/docs/proposals/agent-based-routing.md`.
- [ ] Maintainers agree on draft contract fields and phased delivery order.
- [ ] GitHub sub-issues filed under #2994 for Phases 1–5.
- [ ] Open questions in the proposal resolved or explicitly deferred.

## Task List

- [x] `PROP-01` Draft agent-based routing proposal with contract and delivery plan.
- [x] `PROP-02` Add proposal to index and website sidebar.
- [x] `PROP-03` Add execution plan PL-0039.
- [ ] `PROP-04` Maintainer review and contract agreement on open questions.
- [ ] `PROP-05` File Phase 1–5 sub-issues under #2994.
- [ ] `PHASE-1` Agent provider schema and validation (blocked on PROP-04).
- [ ] `PHASE-2` Deterministic agent selector and eval suite.
- [ ] `PHASE-3` Handoff envelope with #2546 coordination.
- [ ] `PHASE-4` Agent-aware workflow composition baseline.
- [ ] `PHASE-5` Shadow, promotion, rollback, and observability.

## Next Action

Open proposal-only PR for maintainer review. Pause implementation until contract
fields, `targetRefs` naming, and composition algorithm surface are agreed.

## Operating Rules

- One phase per implementation PR; no multi-phase mega-PRs.
- Model-only recipes must keep working without migration through every phase.
- Reuse Router Flow bounded-worker patterns for composition; do not fork a second
  orchestrator.
- Update routing-surface catalog, CONFORMANCE.md, and public docs when a phase
  ships user-visible behavior.

## Related Docs

- [Agent-Based Routing proposal](../../../../website/docs/proposals/agent-based-routing.md)
- [Epic #2994](https://github.com/vllm-project/semantic-router/issues/2994)
- [Agentic & Context #2987](https://github.com/vllm-project/semantic-router/issues/2987)
- [Router Flow Workflows proposal](../../../../website/docs/proposals/router-flow-workflows.md)
- [Model Execution Fallback proposal](../../../../website/docs/proposals/model-execution-fallback.md)
