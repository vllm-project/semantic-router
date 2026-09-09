# PL-0043: Evidence-Calibrated Session Switch Gate

## Goal

Gate session-level model switches on calibrated, bounded recent-window
trajectory evidence so that multi-turn conversations do not thrash between
models on single noisy turns. Implements
[issue #3377](https://github.com/vllm-project/semantic-router/issues/3377) and
closes the evidence-calibrated session switch gate gap tracked in
[issue #3377](https://github.com/vllm-project/semantic-router/issues/3377).

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

M1-M5 implemented (TASK-01..TASK-09). The gate ships disabled and defaults to
`observe` mode, so uncalibrated thresholds cannot change behavior.

## Tasks

- [x] TASK-01: typed `TurnOutcome` facts + bounded recent-window store
      (`pkg/sessiontelemetry`).
- [x] TASK-02/03: response-side capture + outcome-ingest mirroring into the
      same window.
- [x] TASK-04: `EvaluateProgressEvidence` pure function (`pkg/selection`).
- [x] TASK-05: `EvaluateSwitchGate` pure function (`pkg/selection`).
- [x] TASK-06: gate wiring into the protection flow (main + rescue paths) with
      replay traces (`pkg/extproc`).
- [x] TASK-07: `progress_gate` tuning config + validator + reference config.
- [x] TASK-08: documentation (this plan, protection tutorial).
- [ ] TASK-09: live-traffic verification — done informally on an internal
      GPU host (observe and enforce arcs); upstream E2E testcase pending.
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
