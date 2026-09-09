# PL-0043: Evidence-Calibrated Session Switch Gate

## Goal

Session-level model switches are gated on calibrated, bounded recent-window
progress evidence so that multi-turn conversations do not thrash between
models on single noisy turns. Implements
[issue #3377](https://github.com/vllm-project/semantic-router/issues/3377),
closing the evidence-calibrated session switch gate gap.

## Approach

1. Build a typed, content-minimal recent-outcome window per session
   (count + TTL bounded, event-time ordered, persisted with the shared session
   store).
2. Derive calibrated progress evidence from the window: regression/recovery
   streaks, progress trend, attributable coverage, cold start.
3. Gate switch proposals on that evidence: consecutive-regression thresholds,
   cooldown, oscillation guard, and hard-constraint short-circuit. The gate can
   only suppress a switch; hard eligibility and budget checks stay authoritative.
4. Wire the gate into the protection flow (main path and rescue path alike) and
   re-check candidate eligibility before committing a suppression.
5. Record every verdict in Router Replay (`switch_gate` section) so switches
   and suppressions are explainable after the fact.

## Status

Core implementation and replay coverage are in place. The configured window
bounds (`window_size`, `window_ttl_seconds`) drive the evidence window, cooldown
measures the last real model change (`last_switch_at`), the oscillation guard
counts switches inside the evidence window (`switch_timestamps`), and the
response capture plus outcome ingest views of one turn merge into a single
fact keyed by request ID. The gate ships disabled and defaults to `observe`
mode; threshold calibration (TASK-10) still gates any `enforce` default.

## Tasks

- [x] TASK-01: typed `TurnOutcome` facts + bounded recent-window store
      (`pkg/sessiontelemetry`).
- [x] TASK-02/03: response-side capture + outcome-ingest mirroring into the
      same window.
- [x] TASK-04: `EvaluateProgressEvidence` pure function (`pkg/selection`).
- [x] TASK-05: `EvaluateSwitchGate` pure function with window-scoped cooldown
      (`last_switch_at`) and oscillation inputs (`switch_timestamps`).
- [x] TASK-06: main and rescue paths are wired with replay traces; runtime
      state aligned (`last_switch_at`, `switch_timestamps`, window policy).
- [x] TASK-07: config, validator, and reference config; `window_size` and
      `window_ttl_seconds` drive the storage/read policy.
- [x] TASK-08: documentation (this plan, protection tutorial).
- [x] TASK-09: replay regression coverage plus internal live-traffic verification
      of observe and enforce suppression arcs; upstream cluster E2E is optional
      follow-up.
- [ ] TASK-10: calibrate threshold defaults on a held-out multi-turn
      evaluation set before `mode: enforce` can become a default. Calibration
      data is an external, versioned input.

## Non-Goals

- Storing raw conversation content for routing purposes.
- Model or tool calls from inside the gate.
- Replacing the learning outcome/protection contracts; learning consumes the
  same failure provenance through the existing ingest path.
- Defaulting to `enforce` before calibration (TASK-10).

## Operating Rules

- Content-minimal facts only: enums and scalars, never prompt or response text.
- Gate logic is pure and deterministic (no IO, no ambient clock in
  `pkg/selection` evaluation functions).
- Hard constraints stay authoritative before and after the gate.

## Related Docs

- [PL-0040: MoM Routing Hardening](pl-0040-mom-routing-hardening.md)
- Protection tutorial: `website/docs/tutorials/learning/protection.md`
