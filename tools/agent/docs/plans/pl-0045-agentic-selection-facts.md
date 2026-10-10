# PL-0045: Agentic Selection Facts (#3379)

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
  including Router Learning. Eval is recorded under `CONFIRM-06`.
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

  Rule vocabulary for `TASK-05`: text-equality rules only, matching on string
  fields such as `delegated_role` and `task_phase`, following the same rule
  shape already used by the `metadata` family. Numeric predicate rules
  (`gt`/`gte`/`lt`/`lte`) against the `Budget` fields (`remaining_tokens`,
  `remaining_time_ms`, `remaining_cost`) are explicitly deferred, not
  implemented in `TASK-05`.

  Rationale: of those three numeric fields, only `remaining_tokens` is
  evidenced by the merged proposal; `remaining_time_ms` and `remaining_cost`
  were extrapolated when `TASK-02` built the schema and their names and units
  are still open per `CONFIRM-09`. Building comparison-rule support against
  fields that may still be renamed or resized risks wasted work. Revisit once
  `CONFIRM-09` is settled with maintainers.
- [ ] `CONFIRM-05` Failure policy defaults: invalid or untrusted input degrades to
  ordinary routing with diagnostics, while valid facts that leave no eligible
  candidate fail closed rather than silently ignoring the facts. `CONFIRM-08`
  records the envelope-level detail.
- [ ] `CONFIRM-06` Where caller-declared capabilities are enforced. Default: in
  the strict candidate requirements that `main` already owns
  (`candidate_requirements.capabilities: declared`), rather than in a second,
  parallel filter. The caller's `required_capabilities` join the capabilities
  derived from the request in `validateModelDemand`, which every strict seam
  calls, so caller and request capabilities share one function and one
  vocabulary.

  An earlier version of this branch had its own filter in the legacy path. It
  was replaced after rebasing onto `main`, which had meanwhile added the strict
  path. Keeping both would have meant two filters with opposite rules for a
  model that declares no capabilities (legacy keeps it, strict excludes it),
  and a strict recipe would have silently ignored caller capabilities. The
  cost of this choice: caller capabilities have no effect unless the recipe
  opts in, and an opted-in recipe must declare capabilities on every model it
  can route to.

  A route action's destination is filtered like any other candidate. `main`'s
  strict path already checks the destination first and then the decision's
  `modelRefs`, so a destination that lacks a required capability falls back to
  a capable `modelRefs` entry, and with none the request fails closed. Neither
  outcome reaches a model the operator did not configure for that decision.
  The alternative was to exempt the destination, so that a caller could not
  turn an operator's explicit route, typically a prompt-attack route, into a
  refusal. That was rejected in favour of one consistent rule: facts narrow
  every candidate set, and an empty set fails closed. The cost is that a caller
  can make a route action refuse instead of answer; it still cannot make it
  answer from a different, unconfigured model.

  Eval is deliberately left fact-free. `SelectModelForEval` has no request
  context, and the eval API never reads the carrier header, so no envelope
  reaches it and there are no facts to apply. It passes an empty caller set
  explicitly, so the subset rule still holds at this seam. What Eval loses is
  accuracy, not safety: it previews every request as if no agent facts were
  presented, so an operator cannot preview the effect of capability rules.

  Closing that gap means letting the eval API accept an envelope in the request
  body, the way `IntentRequest.Metadata` already lets operators test metadata
  rules. That was not done here for two reasons. The bounds converter the
  validator needs is unexported inside `pkg/extproc` and would have to move
  somewhere `pkg/services` can reach. More importantly, facts are trusted in
  production only because a gateway sets the trust marker, and an eval request
  has no gateway, so accepting facts there creates a surface where any caller
  of the eval API supplies facts with no trust check. That is a deliberate
  decision a maintainer should make, and it belongs in its own task rather than
  inside an eligibility-narrowing step. The alternative considered and rejected
  was adding an `AgenticFacts` field to `EvalModelSelectionInput` that nothing
  populates, which would be dead plumbing of the kind that got `currency` and
  `allowed_candidates` removed.
- [ ] `CONFIRM-07` Which provenance fields are safe on user-facing responses versus
  operator-only Replay. Default: Replay-only, with any response header gated behind
  the existing debug trigger.

  `TASK-07` implemented the Replay-only default and added no response header.
  Replay stores a status and reason codes, never a value the caller sent.
  `root_invocation_id` is deliberately not recorded. It would let an operator
  find every request of one agent task, but it is a string the caller chooses
  and could contain anything. It can be added later without a migration,
  because `route_diagnostics` is one JSONB column.
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
  evidential; `required_capabilities` undecided.

- [ ] `CONFIRM-09` Schema choices the merged proposal does not settle, decided
  in the validator and listed here so review can overturn any of them without
  rereading the code.

  `allowed_candidates` was **removed** in `TASK-06`, along with its
  `max_candidates` bound. It was never in the merged proposal: the proposal's
  envelope example lists `required_capabilities` and no candidate list, and
  states only the principle that facts "may narrow eligibility." The field was
  added during `TASK-02` without a recorded reason.

  It was dropped rather than kept because it required the calling agent to know
  the operator's exact model names, which couples an external system to router
  configuration. Renaming a model would silently break every caller that named
  it, with the caller's constraint then matching nothing. `required_capabilities`
  has no such coupling: it uses symbolic tokens, and the operator maps them to
  models in the model cards. Capability narrowing alone is what the proposal
  actually asked for. Revisit only if a concrete use case appears that
  capabilities cannot express.

  `expires_at` is **required**. An envelope without an expiry has an unbounded
  lifetime, and bounding lifetime is one of the proposal's stated validation
  rules. The cost is that every gateway profile must set it. Alternative if that
  proves too strict: treat an absent expiry as `now + MaxLifetime`.

  `context_portability` is a **closed set**, `portable` or `sticky`, and any
  other value rejects the envelope. The proposal only ever shows `sticky`, so the
  vocabulary is invented here. Closed is the right posture for a field that gates
  mid-session model switching, because an unrecognized value cannot be acted on
  safely. These map onto the existing `HasNonPortableContext` flag consumed by
  `pkg/selection/session_aware.go`, wired in `TASK-06`, not `TASK-05`: `TASK-05`
  only projects accepted facts into the typed signal family without changing
  selection, per its own task description, so the hard-lock wiring belongs with
  the rest of eligibility narrowing.

  `budget.currency` was **removed** in `TASK-06`, together with the paired rule
  that made a cost without a currency `missing` and a currency without a cost
  `conflicting`. That rule existed only to keep the field coherent, so deleting
  the field deletes the rule.

  The reason is the same one that removed `allowed_candidates`: the proposal
  never mentions a currency. It was added in `TASK-02`, no code ever read it,
  and nothing in this plan was going to.

  An earlier version of this entry claimed that "only `remaining_tokens` is
  evidenced by the proposal." That was wrong, and the correction matters because
  it changes which fields are actually ours. The proposal's field table names
  "remaining **token, time, or cost** counters", so all three concepts are
  evidenced; only the field names `remaining_time_ms` and `remaining_cost`, and
  the choice of milliseconds, are decided here.

  Audit performed while removing these two fields, so review can see the whole
  surface at once. Every remaining envelope field maps to a proposal field
  group: lineage identifiers and depth, delegated role, task phase, the three
  budget counters, capability requirements, context portability, and the
  tenant, residency, and label of the trust boundary. `version` and
  `expires_at` are the only required fields. Ten fields are carried but read by
  no code yet, which is expected: the proposal assigns them uses that later
  phases cover, and this plan implements only what `TASK-05` and `TASK-06`
  need.

- [ ] `CONFIRM-10` Capability vocabulary. Default: `required_capabilities`
  accepts only canonical `llmprotocol` capability names (`text`, `tools`,
  `reasoning`, `structured_json`, `image_input`, and so on), the vocabulary
  strict candidate requirements match model cards against. An unknown name
  rejects the envelope as `required_capabilities:malformed` rather than being
  ignored, because ignoring it would tell the caller its request was narrowed
  when it was not. Model card aliases (`vision`, `tool_use`) and `chat` are
  refused too, so each capability has one spelling.

  This conflicts with the merged proposal's example envelope, which lists
  `required_capabilities: [code_review, structured_output]`. Neither name is in
  the vocabulary, so that envelope is rejected. The proposal was not edited
  here; a maintainer should decide whether to update its example or ask for a
  wider, operator-defined vocabulary.

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
- [x] `TASK-05` Project accepted facts into the typed signal family and wire it
  through the routing-surface catalog, validators, decision engine, and Replay
  signal state. Landed as the `agentic_facts` family across `pkg/config`
  (rule schema, catalog, validators, canonical import/export), `pkg/dsl`
  (compiler/decompiler), `pkg/classification` (evaluator, readiness, dispatch),
  `pkg/decision` (`SignalMatches`), `pkg/extproc` (request-facts plumbing and
  `VSRMatchedAgenticFacts`), `pkg/routerreplay` (`Signal.AgenticFacts`), and
  `pkg/services` (API matched/unmatched signal exposure), plus reference config,
  fragment, and tutorial docs. Pending review.

  Two findings worth reviewer attention. First, `Signals.AgenticFactsRules` uses
  the YAML key `agentic_facts_rules` internally because `Signals` inlines into
  `RouterConfig`, where `agentic_facts` is already taken by the `TASK-03` trust
  and bounds config; the canonical public surface keeps the clean
  `routing.signals.agentic_facts` spelling. Second, `hasEnvelopeRoutingFacts`
  now takes the request context and counts accepted facts, so a request with no
  prompt text but a valid envelope still reaches decision evaluation, matching
  the behavior request metadata already had.
- [x] `TASK-06` Narrow hard eligibility from accepted facts at every seam that
  produces candidates, with a subset property test and explicit empty-set
  behavior. Landed in `pkg/extproc`, pending review.

  Capabilities are enforced through `main`'s strict candidate requirements,
  per `CONFIRM-06`. `validateModelDemand` takes the caller's capability set as
  a parameter, so the compiler makes every call site state it, and adds it to
  the capabilities the request itself needs. The check only removes
  candidates; a property test confirms over 500 random pools and capability
  lists that the result is a subset of the configured `modelRefs` and of what
  the request alone allows. With no capable model the request fails closed with
  `main`'s `503` `no_eligible_model`, whose fixed message names no model and no
  capability.

  Applied at every strict seam: the decision prefilter, the selection context,
  the route-action destination and its `modelRefs` fallback, the learning
  candidate pool, the dispatch recheck, context overflow, and automatic output
  admission. Learning mattered most: two of its candidate sets draw from a
  wider pool than the matched decision, up to every model in the deployment, so
  it must apply caller capabilities itself. Each of the first five seams has its
  own test on a fresh context, because later stages narrow again and would
  otherwise hide a broken seam; passing an empty set at any one of them fails
  exactly that test. Eval is excluded, recorded under `CONFIRM-06`.

  The validator accepts only canonical capability names, recorded under
  `CONFIRM-10`.

  `context_portability: sticky` now reaches `nonPortableContextBinding` with
  its own reason, `agentic_facts_sticky`, so Replay can tell an agent's
  declaration apart from real provider-side state. The lock is bounded twice
  over: the operator must enable `ContextPortabilityHardLock`, and
  `SessionAwareSelector.Select` refuses to lock when the previous model is not
  in the candidate list, so sticky cannot reach a model outside the configured
  set.

  Also in this task, two envelope fields were removed after checking them
  against the merged proposal: `allowed_candidates` and `budget.currency`.
  Both are recorded in `CONFIRM-09`, along with a correction to an earlier
  claim about which budget counters the proposal evidences.
- [x] `TASK-07` Emit content-minimized Replay provenance for accepted and rejected
  facts and for eligibility narrowing. Landed in `pkg/routerreplay` and
  `pkg/extproc`, pending review.

  Replay records two fields in `route_diagnostics`: `agentic_facts_status`
  (`accepted` or `rejected`) and `agentic_facts_reasons` (one entry per
  rejection, as `field:reason`, or a bare reason code when the whole envelope
  failed, with repeats removed). Both are absent when the contract is disabled
  or no envelope was sent, so existing records do not change. The shape copies
  the existing Memory fields, `memory_status` and `memory_reason`, rather than
  adding a new nested type. Redaction follows the same rule as Memory: the
  status stays visible, and the reasons are cleared for readers without
  `replay.detail`. A test puts a marker string in every caller-supplied field
  and checks it never appears in the stored JSON, for both an accepted and a
  rejected envelope.

  Scope was cut to the minimum on purpose:

  - **Eligibility narrowing counts are not recorded.** The task text asks for
    them, but they already appear in the `decision_models_filtered` log event,
    and the matched rule names in `signals.agentic_facts` already show which
    rules fired. If they are added later, learning's counts must stay separate
    from the decision's counts: learning can filter a much larger list, up to
    every model in the deployment, so one shared number would be misleading.
  - **A capability refusal is recorded by `main`'s existing path.** It writes a
    record with lifecycle `failed`, terminal reason `selection_rejected`, and
    `agentic_facts_status: accepted`. No agentic-specific refusal field was
    added.

  Found and fixed while doing this task, in its own commit: `ingestAgenticFacts`
  checked the trust marker before checking whether an envelope was sent, so
  every ordinary request with the contract enabled was recorded as rejected
  with reason `untrusted`. Routing was never affected, but Replay would have
  shown almost every request as a rejected envelope. It now checks for an
  envelope first.
- [ ] `TASK-08` Add maintained E2E coverage for authenticated, untrusted, malformed,
  stale, nested-delegation, and conflicting-constraint requests. Tests are
  written and wired into the `routing-strategies` profile, but have not yet
  passed a run. Tick this when that profile passes in CI.

  Two test cases, one contract each, in `e2e/testcases/`:

  - `agentic-facts-routing` sends a reviewer envelope and checks the selected
    decision: no envelope, authenticated, untrusted, malformed, stale, nested
    within the depth bound, nested past it, a capability alias (`vision`), and
    conflicting lineage. Only the trusted and valid cases may reach the
    reviewer decision; every other case must still return 200 on the default
    decision, because a rejected envelope is ignored, not fatal.
  - `agentic-facts-eligibility` checks capability narrowing. A control case
    with no requirement selects the higher-quality model; requiring `tools`,
    which only the other model declares, selects that model instead; requiring
    `reasoning`, which no model declares, returns 503 with the fixed
    no-eligible-model message and no model name or caller value.

  The profile gains one recipe, `agentic-facts-policy`, reached through its own
  entrypoint and opted into `candidate_requirements.capabilities: declared`,
  and three models used only by that recipe: preferred `[text]`, alternate
  `[text, tools]`, and a default `[text]`. Strict mode excludes models that
  declare nothing, so the default decision cannot use the shared `base-model`.
  Shared model cards are untouched, so no other recipe or test in the profile
  sees a change. `context: known_limits` is not enabled, because it would
  require every request to send `max_tokens`.
  `agentic_facts` is enabled for the whole profile; the other tests send no
  envelope, so they are unaffected.

  `routing-strategies` was chosen because it runs on pull requests and already
  hosts `metadata-routing`. Replay is not checked end to end: the
  `router-replay` profile runs only on manual selection, and its test token
  lacks `replay.detail`, so reasons would arrive redacted. Replay is covered by
  the `TASK-07` unit tests instead.

  E2E cannot check two things, both covered elsewhere: that the carrier and
  trust headers are removed before the backend (the mock backend records only
  the request body; unit tests cover both removal paths), and that a gateway
  strips a client-supplied trust marker (a deployment responsibility under
  `CONFIRM-03`; the test plays the trusted gateway itself).

  Checked without a cluster: the profile config parses with `config.Parse`;
  loaded into a router, the reviewer decision selects the preferred model with
  no capability, the alternate with `tools`, and no model with `reasoning`, and
  the default decision selects the default model. A full local run did not complete: the router downloads about
  3 GB of embedding models at startup, and on the development machine the Kind
  cluster reached only about 0.1 MB/s, so the router could not become ready
  within its 60-minute startup limit. This is a local network limit, not a test
  failure; the first real result will come from CI.

  Follow-ups found during this task, recorded here and deliberately not changed
  because they alter shared CI or E2E infrastructure:

  - **CI does not re-run these tests on later changes.** `routing-strategies`
    is selected only when files under its `paths` in
    `tools/agent/test-domain-registry.yaml` change. This pull request selects it
    because it edits the profile's `values.yaml`, but a later pull request that
    touches only agentic facts code will not. A possible fix, needing a
    maintainer decision, is adding `e2e/testcases/agentic_facts_*.go`,
    `src/semantic-router/pkg/agenticfacts/**`, and
    `src/semantic-router/pkg/extproc/req_filter_agentic_*.go` to those paths,
    following the `istio` and `router-replay` profiles.
  - **`E2E_USE_WORKSPACE_MODELS=true` fails for every profile using the Helm
    chart.** The chart always mounts `models-volume` at `/app/models`
    (`deploy/helm/semantic-router/templates/deployment.yaml`), and the runner's
    overlay in `e2e/pkg/framework/runner_lifecycle.go` adds a second mount at
    the same path, so Kubernetes rejects the deployment with "mountPath must be
    unique". This predates this plan.

## Next Action

Open the pull request for `feat/3379-agentic-facts-schema` and read the
`routing-strategies` result in CI. If it passes, tick `TASK-08`. If it fails,
the per-case failure lines name the case and the decision or model it got.

After that, close out `TASK-01` and raise the two `TASK-08` follow-ups with
maintainers.

Every `CONFIRM` item except `CONFIRM-01` is now implemented and externally
visible in code rather than recorded as a default, so all of them need a
maintainer ruling before the branch is complete. Three carry more weight than
the rest because they were decided here rather than by the proposal:
`CONFIRM-06`, which moves capability enforcement onto the strict path and
applies it to a route action's destination; `CONFIRM-09`, which removed two
envelope fields; and `CONFIRM-10`, whose vocabulary rejects the proposal's own
example capability names.

`TASK-08` checks routing through response headers rather than Replay; the
reason is recorded in its task entry.

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
- [Change surfaces](../change-surfaces.md)
- [Testing strategy](../testing-strategy.md)
- [Feature #3379](https://github.com/vllm-project/semantic-router/issues/3379)
- [Epic #2994](https://github.com/vllm-project/semantic-router/issues/2994)
