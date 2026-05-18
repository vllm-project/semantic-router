# Session-Aware Model-Switch Offline Weight Updates

## Goal

- Train offline weight updates for session-aware model-switch policies so learned tuning can build on logged routing evidence without becoming a runtime dependency.
- Provide a versioning and rollout contract so learned weights can be shadowed, compared, and reverted independently from the runtime code path.
- Compose with the existing shadow-mode `model_switch_gate` rather than replacing stable heuristic fallbacks.

## Scope

- `src/semantic-router/pkg/selection/offline_dataset.go` — offline dataset contract
- `src/semantic-router/pkg/selection/offline_updater.go` — batch weight updater
- `src/semantic-router/pkg/selection/policy_version.go` — versioned weight storage and rollout
- `src/semantic-router/pkg/selection/offline_metrics.go` — observability for offline operations
- `src/semantic-router/pkg/selection/offline_test.go` — unit tests
- `src/semantic-router/pkg/routerreplay/store/store.go` — data source (read-only)
- `src/semantic-router/pkg/observability/metrics/session_telemetry_metrics.go` — existing telemetry

## Architecture

### Data Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  RouterReplay    │────▶│ OfflineDataset   │────▶│  OfflineUpdater   │
│  Store (records) │     │ Builder          │     │  (batch replay)   │
└─────────────────┘     └──────────────────┘     └───────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  RLDrivenSel.   │◀────│ PolicyVersion    │◀────│ PolicyVersionStore│
│  (runtime)      │     │ (active weights) │     │ (versioned disk)  │
└─────────────────┘     └──────────────────┘     └───────────────────┘
                                ▲
                                │ shadow comparison
                                ▼
                        ┌──────────────────┐
                        │ ModelSwitchGate   │
                        │ (shadow/enforce)  │
                        └──────────────────┘
```

### Offline Dataset Contract

The `OfflineDatasetRecord` captures:

1. **Session telemetry** — session_id, turn_index, user_id, decision_name
2. **Transition logs** — previous_model, cache_warmth, net_switch_advantage, handoff_penalty
3. **Lookup table evidence** — quality_gap and handoff penalties from existing lookup tables
4. **Feedback signals** — explicit pairwise feedback, implicit detection, response status

Records are assembled by an `OfflineDatasetBuilder` that reads from the RouterReplay store
and joins transition evidence from model_switch_gate audit logs.

### Policy Versioning

Policy versions follow a lifecycle: `candidate → shadow → active → retired`.

- **candidate**: Produced by offline updater, not yet evaluated.
- **shadow**: Being compared against the active version in real traffic (model_switch_gate shadow mode).
- **active**: Drives real routing decisions via the RLDrivenSelector.
- **retired**: Kept for audit and rollback.

A `PolicyManifest` tracks which version is active and which is shadowed. Operators
can promote, shadow, or revert versions independently via the PolicyVersionStore API.

### Composition with ModelSwitchGate Shadow Mode

The model_switch_gate already supports `shadow` and `enforce` modes. Offline weight
updates compose as follows:

1. Offline updater produces a **candidate** policy version from historical data.
2. Operator promotes candidate to **shadow** via `PolicyVersionStore.Shadow(id)`.
3. At request time, the RLDrivenSelector uses the **active** weights for real decisions.
4. A shadow evaluator simultaneously scores with the **shadow** weights and records
   `ShadowComparison` events (agreement rate, score deltas).
5. `ShadowComparisonTotal` and `ShadowComparisonScoreDelta` metrics let operators
   monitor divergence in dashboards before promotion.
6. When satisfied, operator promotes shadow to active via `PolicyVersionStore.Activate(id)`.
7. The previous active version is automatically retired but remains on disk for rollback.

This ensures:

- Rule-based and heuristic fallbacks are never removed — they remain the final authority
  when the model_switch_gate is in `enforce` mode.
- Learned weights are always auditable and reversible.
- No training or external services are required at request time.

### Not Goals

- Mandatory online learning at request time.
- Removing rule-based or heuristic fallback paths.
- Shipping a black-box policy with no inspection hooks.
- Making external training services a runtime dependency.

## Exit Criteria

- [x] `OWU001` Define offline dataset contract types sourced from session telemetry, transition logs, lookup tables, and feedback signals.
- [x] `OWU002` Implement versioned policy weight storage with shadow/compare/revert lifecycle.
- [x] `OWU003` Implement offline batch updater that replays historical records and produces candidate weights.
- [x] `OWU004` Add Prometheus metrics for offline update runs, policy activations, and shadow comparisons.
- [x] `OWU005` Add unit tests for dataset contract, versioned storage, offline updater, and version comparison.
- [x] `OWU006` Document composition with shadow-mode model_switch_gate in this workstream plan.

## Task List

- [x] `OWU001` Define offline dataset contract types sourced from session telemetry, transition logs, lookup tables, and feedback signals.
- [x] `OWU002` Implement versioned policy weight storage with shadow/compare/revert lifecycle.
- [x] `OWU003` Implement offline batch updater that replays historical records and produces candidate weights.
- [x] `OWU004` Add Prometheus metrics for offline update runs, policy activations, and shadow comparisons.
- [x] `OWU005` Add unit tests for dataset contract, versioned storage, offline updater, and version comparison.
- [x] `OWU006` Document composition with shadow-mode model_switch_gate in this workstream plan.
- [ ] `OWU007` Implement `OfflineDatasetBuilder` that reads from RouterReplay store and joins transition evidence.
- [ ] `OWU008` Wire policy version loading into `RLDrivenSelector` initialization path.
- [ ] `OWU009` Add shadow evaluator that runs shadow-version scoring in parallel at request time.
- [ ] `OWU010` Add E2E test for offline weight update → shadow → promote flow.

## Current Loop

- Loop opened on 2026-05-10 from issue #1750.
- Completed in this loop:
  - defined the offline dataset contract (`OfflineDatasetRecord`, `OfflineOutcome`, `OfflineTransitionEvidence`)
  - implemented `PolicyVersionStore` with full lifecycle management (candidate/shadow/active/retired)
  - implemented `OfflineUpdater` with parent-weight blending, transition penalties, and min-observation thresholds
  - added Prometheus metrics for offline operations and shadow comparisons
  - added unit tests covering all offline subsystem components
  - documented architecture and composition with model_switch_gate shadow mode

## Decision Log

- 2026-05-10: Chose Beta distribution accumulation (same as online `UpdateFeedback`) for offline replay rather than introducing a separate reward model. Keeps the weight format identical across online and offline paths, simplifying version promotion.
- 2026-05-10: Added `MinRecordsPerModel` threshold so models with sparse offline evidence keep their parent-version priors unchanged, preventing noisy updates from small samples.
- 2026-05-10: Policy versions are file-backed JSON rather than database-backed to keep the offline path zero-dependency and operator-friendly for inspection.

## Follow-up Debt / ADR Links

- `OfflineDatasetBuilder` implementation depends on RouterReplay store query capabilities (session-range queries). May need a store extension for windowed reads.
- Shadow evaluator integration into the ExtProc hot path must respect latency budgets. Consider async recording or bounded channel.
- No ADR required yet — this is additive infrastructure that does not change the runtime decision contract.

## Validation

```bash
make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/selection/offline_dataset.go,src/semantic-router/pkg/selection/offline_updater.go,src/semantic-router/pkg/selection/policy_version.go,src/semantic-router/pkg/selection/offline_metrics.go,src/semantic-router/pkg/selection/offline_test.go,docs/agent/plans/pl-0030-session-aware-model-switch-workstream.md"
make test-semantic-router
```

## References

- Umbrella: #1513
- This issue: #1750
- Related feedback work: #1512
- Model switch gate: `src/semantic-router/pkg/selection/model_switch_gate.go`
- RL-driven selector: `src/semantic-router/pkg/selection/rl_driven.go`
- Session telemetry: `src/semantic-router/pkg/observability/metrics/session_telemetry_metrics.go`
