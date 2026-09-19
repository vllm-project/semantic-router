# PL-0042: Agentic Selection Facts (#3379)

## Goal

Accept bounded, versioned agent lineage and delegated-role facts from a configured
trusted request boundary, and let only validated facts contribute routing evidence
and narrow hard model eligibility. External input must never widen the configured
candidate set or weaken authorization, safety, residency, or budget policy.

This plan owns Phases 1 and 2 of [PL-0041](pl-0041-agent-aware-router-contracts.md).
Phase 3 (#3380 handoff envelopes) stays out of scope.

## Scope

- A versioned selection-facts envelope schema with deterministic validation for
  bounds, depth, cardinality, lifetime, and internal conflicts.
- A disabled-by-default config contract and trust policy at one existing request
  boundary, with explicit missing, malformed, expired, conflicting, untrusted, and
  partial-deployment behavior.
- Ingestion and trust enforcement at the extproc request boundary, including
  stripping the carrier before the request leaves the Router.
- A typed signal family projected only from accepted facts.
- Hard eligibility narrowing applied at every seam that produces candidate models,
  including Eval and Router Learning.
- Content-minimized Replay provenance for accepted and rejected facts, plus
  configuration, protocol, security, reload, and E2E coverage.

## Non-Goals

- Discovering, invoking, transporting, or orchestrating external agents.
- Persisting durable task state, transcripts, tool payloads, credentials, or
  hidden reasoning.
- Defining the #3380 handoff envelope lifecycle.
- Introducing `targetRefs`, agent inventory, or mixed candidate kinds; decisions
  remain `modelRefs`-only.
- Treating caller-provided facts as trusted without policy validation.

## Decisions To Confirm

Implementation proceeds on the recorded default so work is not blocked. Each item
must be confirmed with maintainers before the PR that depends on it merges.

- [ ] `CONFIRM-01` PL-0041 `PROP-04` is still unchecked while #3379 is labelled
  `accepted` and `in-progress`. Default: treat the merged proposal as the agreed
  contract and tick `PROP-04`/`PROP-05` in the first PR of this plan.
- [ ] `CONFIRM-02` Carrier is a configurable request header holding the versioned
  envelope, stripped before the upstream request. Default over a body extension so
  one contract serves every protocol codec.
- [ ] `CONFIRM-03` Trust is an operator-declared boundary: a configured marker set
  by the authenticated gateway and stripped from client requests. Cryptographic
  integrity verification is deferred behind a versioned policy field. Closes the
  proposal's open question on ingress authenticators.
- [ ] `CONFIRM-04` Signal family name and rule vocabulary. Default: a dedicated
  family separate from the existing untrusted `metadata` family, so the trust
  distinction stays visible in config, decisions, and Replay.
- [ ] `CONFIRM-05` Failure policy defaults: invalid or untrusted input degrades to
  ordinary routing with diagnostics, while valid facts that leave no eligible
  candidate fail closed rather than silently ignoring the facts. `CONFIRM-08`
  records the envelope-level detail.
- [ ] `CONFIRM-06` Relationship to
  [TD-054](../tech-debt/td-054-typed-request-capability-eligibility-gap.md).
  Default: this plan applies capability narrowing through one shared eligibility
  function and records the remaining TD-054 surface as still open, rather than
  adding a second parallel filter.
- [ ] `CONFIRM-07` Which provenance fields are safe on user-facing responses versus
  operator-only Replay. Default: Replay-only, with any response header gated behind
  the existing debug trigger.
- [ ] `CONFIRM-08` What happens when a presented envelope fails validation.
  Default: drop the whole envelope, route the request exactly as if no facts had
  been presented, and record the rejection reasons in Replay. No partial
  acceptance and no request failure, so this stays consistent with `CONFIRM-05`
  rather than contradicting it.

  Rationale: facts only ever narrow within `decisions[].modelRefs`, which the
  operator already configured and which is safe by construction. Dropping them
  returns the request to operator policy rather than letting it escape policy, so
  no rejection path can weaken authorization, safety, or residency on its own.
  The standing rule this rests on: caller-supplied facts must never be the sole
  enforcement point for a policy. Anything that must hold regardless of caller
  input belongs in config.

  All-or-nothing rather than per-field acceptance, because a partial-acceptance
  rule lets whoever populates the envelope shed an inconvenient constraint by
  malforming that one field while keeping the rest. Forward compatibility does not
  depend on partial acceptance: `encoding/json` already ignores unknown fields, so
  a newer gateway adding fields costs nothing, and rejections fire only on
  genuinely malformed values.

  The fail-closed case in `CONFIRM-05` is unaffected: valid facts that narrow the
  candidate set to empty still fail the request rather than being ignored. That is
  `TASK-06` behavior, not validation behavior.

  Deferred alternative, not implemented: classify each field as constraining,
  meaning it narrows selection, or evidential, meaning it only biases selection,
  and fail the request on a constraining rejection, optionally behind a `Strict`
  bound defaulting off. Revisit before `TASK-04` if operators need a stricter
  posture. Classification if revived: `trust_boundary.*`, `budget.*`, `version`,
  envelope-level `malformed` and `expired`, and `lineage.depth` over `MaxDepth`
  constraining; `delegated_role`, `task_phase`, and the lineage identifiers
  evidential; `required_capabilities` and `allowed_candidates` undecided.

- [ ] `CONFIRM-09` Three schema choices the merged proposal does not settle,
  decided in the validator and listed here so review can overturn any of them
  without rereading the code.

  `expires_at` is **required**. An envelope without an expiry has an unbounded
  lifetime, and bounding lifetime is one of the proposal's stated validation
  rules. The cost is that every gateway profile must set it. Alternative if that
  proves too strict: treat an absent expiry as `now + MaxLifetime`.

  `context_portability` is a **closed set**, `portable` or `sticky`, and any
  other value rejects the envelope. The proposal only ever shows `sticky`, so the
  vocabulary is invented here. Closed is the right posture for a field that gates
  mid-session model switching, because an unrecognized value cannot be acted on
  safely. These map onto the existing `HasNonPortableContext` flag consumed by
  `pkg/selection/session_aware.go` in `TASK-05`.

  A currency with no cost is `conflicting`, and a cost with no currency is
  `missing`. Neither is interpretable alone, and an uninterpretable budget must
  not silently contribute nothing while the rest of the envelope is applied.

  Also recorded from `TASK-02`: only `remaining_tokens` is evidenced by the
  proposal. The names and units of `remaining_time_ms` and `remaining_cost`, and
  the presence of `currency`, are extrapolated to match `modelpricing.Rates`.

## Exit Criteria

- Schema, trust, bounds, expiry, conflict, and fail/degrade behavior are versioned
  and deterministic.
- Only validated facts narrow hard eligibility or contribute routing evidence.
- External input cannot widen privileges or the configured candidate set at any
  candidate-producing seam.
- Replay explains accepted and rejected facts without sensitive content.
- Security and E2E tests cover authenticated, untrusted, malformed, stale, nested,
  and conflicting inputs.
- Routing is byte-identical when the contract is absent or disabled.

## Task List

- [ ] `TASK-01` Resolve the proposal's open questions into recorded decisions,
  align PL-0041 phase state, and index this plan.
- [x] `TASK-02` Add the versioned envelope schema and its validator as a standalone
  package with stable rejection reason codes and no runtime wiring. Landed as
  `src/semantic-router/pkg/agenticfacts`, pending review.
- [x] `TASK-03` Add the disabled-by-default config contract and trust policy,
  including canonical import/export, reference config, and required public docs.
  Landed in `pkg/config` (`agentic_facts.go`, canonical schema wiring, reference
  config, validator, docs), pending review.
- [x] `TASK-04` Ingest and validate the envelope at the request boundary, enforce
  trust, strip the carrier, and emit diagnostics without changing selection.
  Landed in `pkg/extproc` (`req_filter_agentic_facts.go`, wired into
  `handleRequestHeaders`), pending review. Both the carrier and trust-marker
  headers are stripped on every return path, including the skip-processing
  bypass, not only the normal routing path.
- [ ] `TASK-05` Project accepted facts into the typed signal family and wire it
  through the routing-surface catalog, validators, decision engine, and Replay
  signal state.
- [ ] `TASK-06` Narrow hard eligibility from accepted facts at every seam that
  produces candidates, with a subset property test and explicit empty-set behavior.
- [ ] `TASK-07` Emit content-minimized Replay provenance for accepted and rejected
  facts and for eligibility narrowing.
- [ ] `TASK-08` Add maintained E2E coverage for authenticated, untrusted, malformed,
  stale, nested-delegation, and conflicting-constraint requests.

## Next Action

Commit `TASK-04` on `feat/3379-agentic-facts-schema`, then start `TASK-05`.
`CONFIRM-02`, `CONFIRM-03`, `CONFIRM-08`, and `CONFIRM-09` are now implemented
and externally visible in `pkg/extproc`, not just recorded defaults, so raise
all four for maintainer ruling before the branch is complete rather than after.

## Operating Rules

- One commit per task, all tasks landing in a single pull request on
  `feat/3379-agentic-facts-schema`. Keep the commits separate and ordered so a
  reviewer can read the eligibility change without the config churn around it.
- Facts may narrow eligibility; they must never widen it or relax policy.
- Only validated facts leave the ingestion seam. The raw envelope is bounded,
  request-scoped, and never becomes general-purpose state.
- Diagnostics and provenance stay content-minimized: field names, reason codes, and
  counts, never field values or transcripts.
- The contract stays disabled by default, and absent or disabled config leaves
  existing routing unchanged.
- Update the routing-surface catalog, the repo-owned `config/` tree, and the public
  config docs in the same change that alters the schema.
- Behavior-visible tasks ship E2E updates in the same PR.

## Related Docs

- [Agent-Aware Router Contracts proposal](../../../../website/docs/proposals/agent-based-routing.md)
- [PL-0041: Agent-Aware Router Contracts (Epic #2994)](pl-0041-agent-aware-router-contracts.md)
- [TD-054: Typed request capability eligibility gap](../tech-debt/td-054-typed-request-capability-eligibility-gap.md)
- [Change surfaces](../change-surfaces.md)
- [Feature-complete checklist](../feature-complete-checklist.md)
- [Testing strategy](../testing-strategy.md)
- [Feature #3379](https://github.com/vllm-project/semantic-router/issues/3379)
- [Epic #2994](https://github.com/vllm-project/semantic-router/issues/2994)
