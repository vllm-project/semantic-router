---
title: Agent Routing Protection Baseline
sidebar_position: 8
---

# Agent Routing Protection Baseline

This is the independently runnable baseline for
[issue #2338](https://github.com/vllm-project/semantic-router/issues/2338).
It tests current production protection and progress-gate behavior using
versioned single-request and multi-turn fixtures. It does not establish
measured model-quality gains or complete the issue's delegated-role coverage.

## Run

From the repository root, with Go installed:

```bash
make bench-agent-routing-protection
```

The target runs the learning-session tests of the router's `pkg/extproc`
package. It does not download model weights, start a router deployment, contact
model providers, or require an external state store. The report is written to
`.agent-harness/agent-routing-protection/report.json`.

The same tests can also run directly:

```bash
cd src/semantic-router
go test ./pkg/extproc -run '^TestRouterLearningSession' -count=1 -v
```

In CI:

- **Where.** The `Router Contracts` job runs these tests with
  `make test-core-unit` and `ROUTER_PROTECTION_REPORT` set, before it starts
  external services or downloads model weights. Their exit status gates the
  job.
- **The report.** The `agent-routing-protection` artifact contains the JSON
  report, including per-turn failures when a contract regresses.
- **On failure.** Upload is attempted even after failure; if setup fails
  before a report exists, the upload warns about the missing file.

The tests also run in the normal core `make test-semantic-router` gate. Any
per-turn model, sampling permission, hard-lock status, preflight reason, Replay
action/reason, or progress-gate verdict/application mismatch fails the gate.
Each run executes the corpus twice and compares report bytes.

## What runs

The corpus lives at
`src/semantic-router/pkg/extproc/testdata/router_learning_sessions.v2.yaml`:

- **The file.** The YAML accepts comments, rejects unknown fields and
  duplicate keys, and must contain exactly one document.
- **Scenarios** declare their scope and protection mode.
- **Steps** append semantic messages, and supply the already-eligible
  candidates, the upstream algorithm's proposal and scores, and any
  provider-state reference or cache-warmth input.
- **Expected results** are separate grading fields, and never supply routing
  state.
- **Progress-gate scenarios** also declare an explicit versioned calibration,
  and script content-minimal typed outcomes after dispatch. Those outcomes
  drive the production evidence evaluator; they don't directly select the
  expected route.

The runner uses production message fact extraction, protection preflight,
switch protection, Replay diagnostic conversion and session-memory writes:

- **Real state between turns.** The next turn reads the previous **actual**
  model selection, and memory is reset between scenarios and repetitions.
- **Scripted proposals, real protection.** The model-choice proposal is
  scripted; the protection decision isn't simulated or reimplemented in the
  runner.
- **Commits.** Accepted corpus steps explicitly commit the staged session
  decision, to model a successful dispatch. In the request pipeline, selection
  only stages ownership: provider preparation, credential resolution and final
  encoding must succeed before ownership is committed.
- **Rejections.** Cancellation and immediate rejections don't replace the
  previous owner, or increment its turn and switch counters.

A companion integration test sends the maintained tool-loop history through
the production Router Learning orchestration, with adaptation enabled. It
checks actual sampling invocations and final-model continuity, plus a bypass
control that samples during the same tool loop; it covers the in-process
orchestration, not HTTP or Envoy dispatch.

`selection_decision_paths_test.go` separately guards component-error
composition:

- observe protection preserves an applied adaptation proposal, even on policy
  rejection;
- preflight reads protection-scoped warm state;
- rescue respects score direction;
- cancellation covers selector shortcuts and successful returns;
- Eval preserves ambiguous-candidate rejection.

In-process dispatch tests cover late rejection, successful ownership
commitment, and an older failure arriving after a newer dispatch. None of
these checks invoke live model backends.

Covered contracts include:

- First-request baseline establishment.
- Active tool calls, tool results and immediate user follow-ups blocking switches.
- Completed tool exchanges releasing their historical lock.
- Provider-bound response state and release with portable history.
- Small score advantages being held and clear advantages allowing a switch.
- Warm-cache sampling suppression.
- Candidate-set changes preventing restoration of an ineligible previous model;
  a hard-bound continuation with an excluded owner is rejected, not rerouted.
- Session-scope continuity and conversation-scope isolation.
- Missing identity, observe mode and bypass mode retaining their current semantics.
- Cold-start and insufficient-evidence switch suppression.
- Environment failures and missing outcomes not counting as model regressions.
- Sustained regression permitting escalation and sustained recovery permitting downgrade.
- Observe-mode verdicts remaining non-enforcing.
- Cooldown and window-scoped oscillation suppression.
- A suppressed verdict not restoring a current model that has become ineligible.

The profile fixes a switch margin of 0.05, minimum turns of zero and stability
weight of zero. This isolates continuity locks and basic switch permission from
cache-cost and history-penalty tuning; it is not a production tuning recommendation.

## Read the report

The report identifies its schema and the SHA-256 of the exact corpus bytes:

- **Per step:** every proposed and final model, preflight reason, Replay
  action and reason, hard-lock result, scripted cache warmth and assertion
  failure.
- **Rejected steps** use `rejected: true`, an empty final model and a terminal
  rejection outcome. They don't write a new session owner, and aren't counted
  as model switches.

Its metrics:

- the contract pass rate and switches;
- blocked-switch violations, unsafe sampling violations and unnecessary
  switches;
- missed **scripted** opportunities;
- Replay explanation coverage, and progress-gate contract and explanation
  coverage;
- safe enforced-suppression application, and avoided holds of ineligible
  current models.

Every rate carries its count and denominator; an empty denominator is `null`,
not a successful zero-violation measurement. A scripted opportunity means the
fixture specifies a permitted, sufficiently advantageous proposal; it's not a
counterfactual estimate of real task quality.

:::caution
Quality benefit, billed cost, inference latency, measured cache savings and
statistical uncertainty are explicitly unavailable. A deterministic contract
corpus is not a sample of real agent sessions, and test execution time is not
model latency; those claims need paired task runs and measured outcomes.
:::

## Extend coverage

- **Keep scenarios short,** with stable IDs and incremental message
  histories.
- **Pair a blocked state with its released state,** so a policy that never
  switches can't pass.
- **Give exact expectations** for the final model, sampling permission,
  hard-lock status, preflight reason and Replay explanation.
- **Tag coverage.** Steps carry semantic coverage tags; validation rejects
  missing required capabilities and unknown tags.
- **No fixed count.** Adding or renaming scenarios doesn't require a fixed
  scenario count.
- **Keep scenarios isolated,** and reports deterministic.

The corpus carries an explicit missing-coverage list, copied into every
report:

- adaptation learning and delayed outcome delivery;
- rescue and failed dispatch or fallback paths;
- timeouts and idle expiry;
- upstream authorization, residency and budget eligibility;
- real model measurements;
- transport and external state stores.

Delegated roles and multi-arm shadow execution are added only when their
owning features land. These gaps are not passing results, so
[issue #2338](https://github.com/vllm-project/semantic-router/issues/2338) stays open for them.

## Remaining issue criteria

The corpus gates the current protection baseline. The rest of
[issue #2338](https://github.com/vllm-project/semantic-router/issues/2338) is split by the evidence it must produce:

| Issue criterion | Follow-up evidence |
| --- | --- |
| Multi-turn sessions and delegated roles | Add role facts and selected/final-role assertions when the owning contract lands. |
| Safe exploration and missed opportunities | Extend the deterministic progress-gate coverage with delayed outcomes, timeouts, failed switches, rescue evidence and failure/fallback provenance. |
| Hard protection constraints | Exercise upstream authorization, safety, residency, context, capability and budget conflicts as well as candidate eligibility. |
| Quality, cost, latency, cache and uncertainty | Run paired baseline/exploration tasks with actual model outcomes, repeated measurements and agreed acceptance thresholds. |
| PR and release validation | Keep deterministic protection checks in PR tests; integrate measured evidence into a separate release or scheduled benchmark gate. |

Existing live agent-task scripts can provide task examples for the measured
runner, but their completion rubrics are not protection assertions. Shared
scenario loading and measurement design belong in that follow-up.
