# Session Token Budget

## Overview

Agent workflows can suffer a token-budget explosion: a single user request
triggers many more LLM calls than a chat request, and context grows as each turn
re-sends the full message history. Failed attempts and retry loops compound this,
spending tokens with no productive output.

`session_token_budget` lets the router enforce a **per-active-session token
budget at the dispatch layer**. Instead of a binary deny, an over-budget session
escalates through a **graduated response ladder**:

```text
shape tools  →  compress prompt  →  downgrade model  →  terminate
```

The router is the natural enforcement point: it sees the actual token flow across
all turns of a session, while an agent SDK sees only its own session and a billing
layer operates on aggregated cost with delay.

## What Problem Does It Solve?

Existing controls are insufficient for this concern:

- Provider/TPM rate limits (`ratelimit`) do a **binary** per-user/model deny on a
  sliding window — not a per-active-session cumulative budget, and with no
  graduated response.
- Per-workflow caps live in the agent SDK and per-tenant caps live in billing;
  neither has the router's cross-turn session trajectory.

`session_token_budget` fills that gap with one typed, auditable contract.

## How It Works

After session-transition fields are populated for a request, the router reads the
session's cumulative tokens (prompt + completion, accumulated across turns from
router session telemetry) and compares them to the configured budget:

```text
ratio = cumulative_tokens / budget_tokens
```

The `ratio` selects the highest triggered ladder stage from the configured
thresholds. Soft stages are cumulative (a downgrade-band session also has its
tools shaped and prompt compressed); `terminate` is exclusive and short-circuits
the request with an immediate response.

## Configuration

`session_token_budget` lives under `global.services`. It is **opt-in and
tri-state**: a complete no-op unless `enabled: true` **and** `budget_tokens > 0`.

```yaml
global:
  services:
    session_token_budget:
      enabled: true
      budget_tokens: 40000        # static per-session token ceiling
      thresholds:                 # ascending multipliers of budget_tokens
        shape_tools: 1.0          # ratio >= 1.0  → shape tools
        compress: 1.5             # ratio >= 1.5  → + compress prompt
        downgrade: 2.0            # ratio >= 2.0  → + downgrade model
        terminate: 3.0            # ratio >= 3.0  → terminate (429)
```

| Field | Meaning |
| --- | --- |
| `enabled` | Activates evaluation. Default `false`. |
| `budget_tokens` | Static per-session ceiling (the "expected budget"). Must be `> 0` to enforce. |
| `thresholds.*` | Over-budget ratio at which each stage fires. Any omitted (zero) field falls back to its default (`1.0 / 1.5 / 2.0 / 3.0`). Must be ascending. |

The static budget is the MVP. A per-`(domain, turn)` percentile budget model and a
system-prompt-fingerprint prior are planned follow-ups; until then, prefer a
conservative ceiling.

## Runtime Scope

This release lands the **config, evaluation, observability, and the `terminate`
stage**. When a session reaches the terminate threshold, the router returns an
immediate OpenAI-compatible `429` with an error of type `budget_exceeded`,
mirroring the fast-response path.

The soft ladder stages — `shape_tools`, `compress`, `downgrade` — are evaluated
and reported (headers/metrics) but their enforcement actions land in a follow-up.
A session in those bands proceeds to normal routing in this release.

## Response Headers

Emitted when budget evaluation ran (enforcement enabled and the request resolved
to a session):

| Header | Value |
| --- | --- |
| `x-vsr-budget-stage` | `none` / `shape_tools` / `compress` / `downgrade` / `terminate` |
| `x-vsr-budget-ratio` | cumulative / budget, two decimals (e.g. `2.40`) |
| `x-vsr-budget-exceeded` | `true` only on the terminate response |

## Metrics

| Metric | Type | Labels |
| --- | --- | --- |
| `vsr_session_budget_evaluations_total` | counter | `stage` |
| `vsr_session_budget_ratio` | histogram | — |

## Risk and Tuning

A legitimately long session (for example a complex SWE-bench task) could be
terminated prematurely. Mitigations:

- Enforcement is **opt-in**; start in observation by leaving a generous budget and
  watching `vsr_session_budget_ratio` before tightening.
- `terminate` is the **last** stage; softer stages aim to arrest growth first.
- A P90-based per-turn budget model (vs. a flat ceiling) is the planned durable
  fix for false positives.
