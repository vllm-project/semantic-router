# PL-0039: Evaluation Plane

## Goal

Replace the legacy dashboard evaluation feature with a versioned Evaluation
Plane that measures routing recipes, model pools, combined system behavior,
agentic and multimodal workloads, online preferences, safety, and capacity with
reproducible evidence and actionable reports.

## Scope

- Define versioned workload, policy, pool, environment, trace, metric, gate,
  artifact, and report contracts.
- Provide a durable run lifecycle, immutable run snapshots, progress events,
  cancellation, comparison, and content-addressed artifacts.
- Execute deterministic replay and live HTTP evaluation tracks through a
  benchmark catalog with declared capabilities and evidence levels.
- Aggregate quality, routing, cost, latency, reliability, safety, preference,
  agent continuity, multimodal, and capacity metrics into explicit gates.
- Replace the dashboard evaluation API and frontend instead of retaining the
  superseded task/dataset workflow.
- Add repository-native tests and end-to-end profiles, including an AMD live
  validation receipt stored outside the tracked source tree.

## Non-Goals

- Moving logical model selection or policy execution out of the Router.
- Moving physical upstream transport or placement ownership out of Envoy and
  the serving layer.
- Treating synthetic or mocked runs as production evidence.
- Persisting target credentials, private infrastructure identifiers, raw
  private prompts, or model outputs in tracked source artifacts.
- Maintaining API or UI compatibility with the removed dashboard evaluation
  implementation.

## Exit Criteria

- The legacy evaluation models, runners, handlers, API routes, hooks, and UI
  components no longer participate in the dashboard build or runtime.
- A run snapshot records its schema version, benchmark revisions, routing
  policy, model pool, environment, seed, budgets, and redaction policy.
- The pipeline can trigger and report routing, model-pool, end-to-end,
  agentic, multimodal, preference, safety, and capacity tracks.
- Reports include per-case evidence, aggregate metrics, confidence or sample
  coverage, gate dispositions, cost ledgers, comparisons, and architecture
  recommendations.
- Unit, integration, frontend, API, and repository-selected gates pass.
- A real AMD end-to-end run completes against the reviewed source and records
  reproducible local evidence without exposing private infrastructure.

## Task List

- [x] `TASK-01` Freeze the Evaluation Plane contract and benchmark catalog.
- [x] `TASK-02` Implement durable run, event, artifact, metric, gate, and report storage.
- [ ] `TASK-03` Implement replay and live executors for every evaluation track.
  All-track deterministic replay and bounded generic live execution are present.
  Qualified direct-arm live execution and repository-native benchmark adapters
  remain tracked by TD-049 and TD-050.
- [x] `TASK-04` Replace the dashboard API and remove the legacy backend implementation.
- [x] `TASK-05` Replace the dashboard evaluation page and remove legacy frontend code.
- [x] `TASK-06` Add deterministic fixtures, unit tests, integration tests, and E2E gates.
- [x] `TASK-07` Run local repository gates and repair all affected failures.
- [ ] `TASK-08` Complete AMD live validation and record ignored evidence artifacts.
  The dedicated AMD-hosted Docker/Kind control-plane campaign now exercises
  all eight tracks with the deterministic E0 fixture, report/artifact policy,
  comparison guards, cancellation, and restart recovery. Model-backed E1-E5
  execution remains pending explicit approval of the pinned serving plan; E0
  evidence is never treated as promotion evidence.
- [x] `TASK-09` Reconcile architecture findings, docs, and any durable debt before handoff.

## Next Action

After explicit approval of the pinned model, GPU, port, and image plan, run the
model-backed AMD campaign for `TASK-08`. Then close the qualified direct-arm
and native-adapter gaps in TD-049 and TD-050 before claiming E1-E5 parity with
upstream benchmark leaderboards.

## Operating Rules

- Keep evaluation in the replaceable control plane; preserve Router, Agent,
  Envoy, and serving-layer ownership boundaries.
- Pin code, benchmark, model, image, configuration, and dataset revisions in
  evidence; never use mutable tags as regression proof.
- Keep raw and private evidence under ignored task-specific directories.
- Run the smallest selected gate first and fix failures before expanding.
- Remove superseded implementations instead of creating parallel legacy paths.
- Mark unsupported evidence as unavailable or not applicable, never as a pass.

## Related Docs

- [Architecture guardrails](../architecture-guardrails.md)
- [Feature-complete checklist](../feature-complete-checklist.md)
- [Testing strategy](../testing-strategy.md)
- [Online evaluation assignment evidence gap](../tech-debt/td-048-online-evaluation-assignment-evidence-gap.md)
- [Router Flow Evaluation Campaign](pl-0037-router-flow-eval-campaign.md)
- [Dashboard Modeling Experience](pl-0038-dashboard-modeling-experience.md)
