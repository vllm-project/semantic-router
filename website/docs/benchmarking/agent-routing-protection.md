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

The tests also run in the normal core `make test-semantic-router` gate. Any
per-turn model, sampling permission, Replay action or reason mismatch fails the
gate. Each run executes the corpus twice and compares report bytes.

## What runs

The corpus lives at
`src/semantic-router/pkg/extproc/testdata/router_learning_sessions.v1.json`.
Each scenario declares its scope and protection mode. Each step appends semantic
messages and supplies the already-eligible candidates, the upstream algorithm's
proposal and scores, and any provider-state reference or cache-warmth input.
Expected results are separate grading fields and never supply routing state.

The runner uses production message fact extraction, protection preflight,
switch protection, Replay diagnostic conversion and session-memory writes.
The next turn reads the previous **actual** model selection. Memory is reset
between scenarios and repetitions. The model-choice proposal is scripted;
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
pass. Include exact expectations for the final model, sampling permission and
Replay explanation. Preserve scenario isolation and deterministic reports.

The corpus carries an explicit missing-coverage list, copied into every report.
It includes adaptation learning and outcome delivery, rescue and failure paths,
timeouts, idle/cooldown boundaries, upstream authorization/residency/budget
eligibility, real model measurements, transport and external state stores.
Delegated roles, multi-arm shadow execution and future evidence-calibrated
session switching must be added only when their owning features land.

Keep issue #2338 open after this baseline: these gaps are not passing results.
