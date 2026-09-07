---
title: Agent Routing Protection Baseline
sidebar_position: 8
---

# Agent Routing Protection Baseline

This is the independently runnable baseline for
[issue #2338](https://github.com/vllm-project/semantic-router/issues/2338).
It tests current production protection behavior using versioned single-request
and multi-turn fixtures. It does not establish measured model-quality gains or
complete the issue's future session and delegated-role coverage.

## Run

From the repository root, with the normal Go, Rust and C build tools installed:

```bash
make bench-agent-routing-protection
```

The target builds the CPU native bindings required to link the router package.
It does not download model weights, start a router deployment, contact model
providers, or require an external state store. The report is written to
`.agent-harness/agent-routing-protection/report.json`.

After building native bindings, the targeted tests can also run directly:

```bash
cd src/semantic-router
go test ./pkg/extproc -run '^TestRouterLearningSession' -count=1 -v
```

CI runs `make bench-agent-routing-protection` explicitly in the core
`Test And Build` job after building CPU bindings and before starting external
services or downloading model weights. Its exit status gates the job. The
`agent-routing-protection` artifact contains the JSON report, including per-turn
failures when a contract regresses. Upload is attempted even after failure;
if setup fails before a report exists, the upload warns about the missing file.

The tests also run in the normal core `make test-semantic-router` gate. Any
per-turn model, sampling permission, hard-lock status, preflight reason, Replay action or reason mismatch fails the
gate. Each run executes the corpus twice and compares report bytes.

## What runs

The corpus lives at
`src/semantic-router/pkg/extproc/testdata/router_learning_sessions.v1.yaml`.
The YAML accepts comments, rejects unknown fields and duplicate keys, and must contain exactly one document. Each scenario declares its scope and protection mode. Each step appends semantic
messages and supplies the already-eligible candidates, the upstream algorithm's
proposal and scores, and any provider-state reference or cache-warmth input.
Expected results are separate grading fields and never supply routing state.

The runner uses production message fact extraction, protection preflight,
switch protection, Replay diagnostic conversion and session-memory writes.
The next turn reads the previous **actual** model selection. Memory is reset
between scenarios and repetitions. A companion integration test sends the maintained tool-loop history through the production Router Learning orchestration with adaptation enabled. It checks actual sampling invocations and final-model continuity, plus a bypass control that samples during the same tool loop. This covers the in-process orchestration, not HTTP or Envoy dispatch.

The model-choice proposal is scripted;
the protection decision is not simulated or reimplemented in the runner.

Covered contracts include:

- First-request baseline establishment.
- Active tool calls, tool results and immediate user follow-ups blocking switches.
- Completed tool exchanges releasing their historical lock.
- Provider-bound response state and release with portable history.
- Small score advantages being held and clear advantages allowing a switch.
- Warm-cache sampling suppression.
- Candidate-set changes preventing restoration of an ineligible previous model.
- Session-scope continuity and conversation-scope isolation.
- Missing identity, observe mode and bypass mode retaining their current semantics.

The profile fixes a switch margin of 0.05, minimum turns of zero and stability
weight of zero. This isolates continuity locks and basic switch permission from
cache-cost and history-penalty tuning; it is not a production tuning recommendation.

## Read the report

The report identifies its schema and the SHA-256 of the exact corpus bytes.
It contains every proposed/final model, preflight reason, Replay action/reason,
hard-lock result, scripted cache warmth and assertion failure.

Metrics include contract pass rate, switches, blocked-switch violations, unsafe
sampling violations, unnecessary switches, missed **scripted** opportunities and
Replay explanation coverage. Every rate carries its count and denominator; an
empty denominator is `null`, not a successful zero-violation measurement.
A scripted opportunity means the fixture specifies a permitted, sufficiently
advantageous proposal. It is not a counterfactual estimate of real task quality.

Quality benefit, billed cost, inference latency, measured cache savings and
statistical uncertainty are explicitly unavailable. A deterministic contract
corpus is not a sample of real agent sessions, and test execution time is not
model latency. Those claims require paired task runs and measured outcomes.

## Extend coverage

Add short scenarios with stable IDs and incremental message histories. Pair a
blocked state with its released state so a policy that never switches cannot
pass. Include exact expectations for the final model, sampling permission, hard-lock
status, preflight reason and Replay explanation. Steps carry semantic coverage
tags; validation rejects missing required capabilities and unknown tags. Adding
or renaming scenarios does not require a fixed scenario count. Preserve scenario isolation and deterministic reports.

The corpus carries an explicit missing-coverage list, copied into every report.
It includes adaptation learning and outcome delivery, rescue and failure paths,
timeouts, idle/cooldown boundaries, upstream authorization/residency/budget
eligibility, real model measurements, transport and external state stores.
Delegated roles, multi-arm shadow execution and future evidence-calibrated
session switching must be added only when their owning features land.

Keep issue #2338 open after this baseline: these gaps are not passing results.

## Remaining issue criteria

This PR gates the current protection baseline and leaves issue #2338 open.
Follow-up work is split by the evidence it must produce:

| Issue criterion | Follow-up evidence |
| --- | --- |
| Multi-turn sessions and delegated roles | Add role facts and selected/final-role assertions when the owning contract lands. |
| Safe exploration and missed opportunities | Add missing outcomes, sustained improvement/degradation, oscillating evidence, cooldown, timeouts, failed switches, and failure/fallback provenance. |
| Hard protection constraints | Exercise upstream authorization, safety, residency, context, capability and budget conflicts as well as candidate eligibility. |
| Quality, cost, latency, cache and uncertainty | Run paired baseline/exploration tasks with actual model outcomes, repeated measurements and agreed acceptance thresholds. |
| PR and release validation | Keep deterministic protection checks in PR tests; integrate measured evidence into a separate release or scheduled benchmark gate. |

Existing live agent-task scripts can provide task examples for the measured
runner, but their completion rubrics are not protection assertions. Shared
scenario loading and measurement design belong in that follow-up.
