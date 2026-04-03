# Balance Recipe Simplification Execution Plan

## Goal

- Reduce the maintained `balance` routing profile to the smallest balance-first decision set that still preserves high-risk escalation, hard-capability escalation, and cheap fallback coverage.
- Remove or demote routing features that act like taxonomy or style instrumentation instead of real model-boundary signals.
- Keep the maintained DSL, YAML, probe manifest, README, and validator expectations aligned so the profile is locally valid and live-calibratable from repository state alone.

## Scope

- `deploy/recipes/{balance.dsl,balance.yaml}`
- `deploy/recipes/balance.probes.yaml`
- `deploy/amd/README.md`
- `src/semantic-router/pkg/dsl/maintained_asset_roundtrip_test.go`
- any narrow DSL validator or CLI seams already touched in the same balance-maintenance loop when they materially affect balance validation behavior

## Exit Criteria

- The maintained `balance` profile expresses only balance-relevant lanes instead of taxonomy-style over-segmentation.
- The recipe has a simplified decision inventory with merged sibling lanes where the model tier does not justify separate routes.
- Obvious unused or weak routing features are removed or demoted, and any new signal that remains is actually consumed by a route or projection.
- `balance.dsl` and `balance.yaml` stay in sync.
- Static `sr-dsl validate` on `balance.dsl` reports no issues by default.
- The maintained balance tests pass.
- The live routing calibration loop against the owned router URL produces a versioned before / after report for the simplified profile.

## Task List

- [x] `BAL001` Create and index the durable execution plan for the balance simplification loop.
- [ ] `BAL002` Collapse the balance decision tree into a balance-first lane set, including merged reasoning, technical, explainer, and fast-QA routes.
- [ ] `BAL003` Remove or demote non-essential signals and projections, add the narrow `reask` signal, and ensure every retained balance-owned feature is actually consumed.
- [ ] `BAL004` Regenerate the maintained YAML and rewrite the executable probe manifest and AMD README to match the simplified contract.
- [ ] `BAL005` Tighten maintained balance tests and local validation until the simplified profile is statically clean.
- [ ] `BAL006` Run the repo-native routing calibration loop against the live router endpoint and capture the final report artifacts.

## Current Loop

- Loop status: opened on 2026-04-01.
- Completed in this loop:
  - created and indexed this execution plan so the balance simplification work can resume from the repository alone
  - audited the current maintained balance recipe, probe manifest, maintained tests, and AMD README
  - separated validator noise from recipe-design issues in the old `no mutual exclusion guard` warnings
  - confirmed the current 21-decision profile still mixes balance routing with taxonomy-style specialization
  - identified the first-pass simplification target: merge sibling decisions that share the same model tier and do not represent distinct cost or risk boundaries
- Next loop focus:
  - execute `BAL002` and `BAL003` together by rewriting the canonical DSL surface first, then regenerate YAML/probes/docs from that source of truth

## Decision Log

- 2026-04-01: keep this as a file-backed execution plan because the work spans maintained examples, tests, validation semantics, and live router calibration across multiple loops.
- 2026-04-01: treat `balance` as a cost-routing profile first, not as a taxonomy showcase; signals or decisions that do not materially improve model selection should be merged, removed, or demoted.
- 2026-04-01: prefer removing unused emotional/style projections and route-local preference clutter over adding more narrowly tuned features.
- 2026-04-01: add `reask` only if it is consumed by a maintained route; new signals do not belong in the profile as dead inventory.

## Follow-up Debt / ADR Links

- [Plans README](README.md)
- [Tech Debt README](../tech-debt/README.md)
