# PL-0039: Evaluation Plane

## Goal

Ship one greenfield Evaluation Plane that turns pinned workloads, routing
recipes, model pools, runtime execution, and production evidence into
reproducible reports and fail-closed promotion decisions.

## Scope

- One versioned contract for run manifests, execution plans, records, metrics,
  gates, reports, comparisons, artifacts, execution attestations, and Campaigns.
- Eight tracks: routing, model pool, routing plus pool, agentic, multimodal,
  preference, safety, and capacity.
- Deterministic fixture replay, exact-pinned normalized benchmark replay,
  normalized live execution, and bounded runtime probes.
- Thirteen benchmark research descriptors, eleven executable safe-export
  normalizers, and two diagnostic-only source entries with explicit blockers;
  one canonical thirteen-item research inventory keeps these descriptors
  separate from additional installed adapters and publishes an honest
  readiness/native-parity boundary for every benchmark.
- Per-track evidence levels, weakest-track run level, G0-G9 gate dispositions,
  strict unavailable semantics, paired statistics, and architecture feedback.
- A server-owned Promotion Campaign whose change-profile catalog declares one
  typed evidence slot per G2-G9 gate, including controlled-live G3 and
  reference-to-fresh-live G5 pairs.
- A Dashboard experience for overview, experiment creation, run operations,
  reports, comparison, and Campaign decisions.
- Repository gates plus a fresh AMD end-to-end environment with retained runs,
  reports, restart evidence, and browser acceptance.

## Non-Goals

- Moving logical routing policy out of the Router.
- Moving physical transport, placement, queueing, or replica ownership out of
  Envoy and the serving/fleet layer.
- Moving tool, role, workspace, or external-side-effect ownership into the
  Router.
- Owning production traffic assignment or preference ingestion. The Evaluation
  Plane consumes one authenticated, sealed assignment/exposure ledger through
  its registered online-evidence provider and fails closed when that evidence
  is absent.
- Executing arbitrary upstream benchmark code or claiming native leaderboard
  parity from normalized imports. Maintained first-party adapters and live
  executors are the only executable benchmark boundary.
- Treating the Dashboard worker as a general-purpose, multi-tenant sandbox for
  untrusted code. It executes only registered first-party Evaluation
  executors; public reports remain typed and server-owned.
- Treating fixtures, HTTP reachability, source pins, checksums, or aggregate
  proxy metrics as promotion evidence.
- Persisting credentials, private prompts or outputs, target origins, or
  private infrastructure identifiers in public artifacts.
- Providing more than one active API, store, report, executor-resolution, or
  attestation contract for the same operation.

## Exit Criteria

- The Evaluation build and runtime contain only the current contract and
  current UI workflow.
- A fresh store accepts complete current bundles and quarantines unknown,
  incomplete, malformed, non-private, or symlinked evidence.
- The fresh-store lifecycle policy is `evaluation-lifecycle-policy.v2`; older
  unpublished Evaluation state fails closed and must not be silently migrated.
- Campaigns carry private owner, quota, audit, retention, hold, and deletion
  metadata, and expired Campaign collection releases their run references.
- Every run freezes suite/executor/source, policy, binding, pool, workload,
  target, topology, environment, code, seed, budgets, redaction identities, and
  the typed latency/error/throughput/scaling SLO whenever live Capacity is selected;
  planning requires at least two concurrency levels so scaling is observed.
- Every live target is one immutable Recipe-scoped Mixture-of-Models. The same
  hidden-label cohort can execute Router diagnostics, a complete direct-arm
  matrix, and routed Entrypoint outcomes, with server-owned receipts and
  server-reduced Recipe, pool, and joint conclusions.
- Every target declares its accepted executors per mode; the planner activates
  only a target-approved suite/mode/track/executor combination and exactly one
  executor cohort per run.
- Reports expose per-track evidence strength, complete coverage, rich metrics,
  uncertainty, three cost ledgers, G0-G9, failure evidence, lineage, and
  actionable architecture findings.
- Every published metric carries a validated estimator/version, analysis and
  cluster unit, weighting, missingness, and exclusion contract; the server
  rejects reports that omit or forge this analysis provenance.
- Normalized benchmark imports are sealed as exploratory E0 evidence. Exact
  source pins and parser re-derivation authenticate the parser and submitted
  bytes, not an upstream native benchmark run; imports cannot publish a
  qualified method or promotion gate.
- G4 is published only by the registered `declared-shift.server-live.v1`
  method: the server binds an exact source revision, immutable parser-verified
  CAS inputs, complete native source/target pairs, broker receipts, execution
  attestation, and its own pair/slice reduction. Imports and replay remain E0.
  G2, G6, G8, and G9 have reachable live suites and strict server-owned
  ledger reducers, but remain data-dependent whenever the target does not
  configure the corresponding production endpoint or the sealed window is
  incomplete. A live Capacity run is rejected before creation without its SLO or
  fewer than two concurrency levels;
  G7 is then decided only by server-reduced SLO headroom, never a proxy metric.
- Campaigns bind the catalog-declared G2-G9 slots rather than a fixed run-role
  bundle. Single-run, controlled-pair, and fidelity-pair bindings are distinct;
  every candidate anchor shares one candidate-subject digest and the required
  evidence level, executor, mode, track, attestation, private receipt, deletion
  protection, and restart validation are enforced per slot.
- Every Evaluation page, control, dialog, error/loading/empty state, keyboard
  path, deep link, and responsive layout passes frontend acceptance.
- Controlled-pair admission is single-flight across services, fails closed
  when another process owns the filesystem store, and cannot publish, reserve,
  launch, or return success after caller or service cancellation.
- Unit, race, integration, frontend, repository-selected, and CI gates pass.
- A fresh AMD deployment runs and retains representative replay, Recipe,
  model-pool, joint, multimodal, capacity, comparison, and Campaign evidence;
  restart and browser access are verified without exposing private
  infrastructure.

## Task List

- [x] `TASK-01` Freeze the singular Evaluation contracts, executor registry,
  eight-track catalog, metric reducers, G0-G9 policy, and report shape.
- [x] `TASK-02` Implement atomic run/evidence storage, content addressing,
  server anchors, execution attestations, bounded ledger indexing, quarantine,
  restart recovery, and referenced-run deletion protection.
- [x] `TASK-03` Implement the fail-closed worker sandbox and Go-owned live
  request broker with exact transcript receipts.
- [x] `TASK-04` Research and pin all thirteen benchmark sources; implement the
  eleven safe normalizers, explicit diagnostic-only blockers, trusted
  re-normalization, and per-track E0 diagnostic receipts. Upstream repositories
  are immutable research inputs, never executable Dashboard workers: promotion
  evidence comes only from maintained first-party broker protocols whose exact
  requests, outcomes, graders, and runtime snapshots are independently sealed.
- [x] `TASK-05` Implement deterministic replay, normalized replay/live, bounded
  runtime Recipe diagnostics, dense direct-arm model-pool execution, routed
  joint/multimodal/capacity execution, the current typed Capacity SLO contract
  and attested headroom reducer, hidden grading, metric/gate reduction,
  comparison, and architecture feedback. Recipe and model-pool conclusions are
  manifest-frozen, broker-attested, and server-reduced; worker-authored
  aggregates cannot enter the decision report.
- [x] `TASK-06` Implement the catalog-driven Promotion Campaign v2 with typed
  per-gate bindings, exact controlled-live G3 statistics, qualified
  reference-to-fresh-live G5 fidelity, and strict production-only G8/G9
  boundaries.
- [x] `TASK-07` Complete the Evaluation Dashboard information architecture,
  responsive visual system, strict decoders, all controls/dialogs, and
  Evaluation-specific unit and browser coverage.
- [x] `TASK-08` Reconcile the integrated implementation and documentation, and
  complete the local source review without running formatter, lint, test,
  build, or runtime gates on the implementation workstation.
- [ ] `TASK-09` Build and deploy the reviewed source through the canonical AMD
  local-image workflow into a fresh state root; run every affected formatter,
  lint, unit, race, integration, build, and browser gate there; execute and
  retain the complete representative run matrix and verify restart plus browser
  behavior.
- [ ] `TASK-10` Publish final reproducibility receipts, refresh the Evaluation
  Atlas, complete review, and keep required CI green.

## Next Action

Finish the integrated source review, mirror that exact state to a dynamically
discovered AMD validation host, and run the selected gates there before
creating the fresh evidence root. Execute the representative live and replay
matrix, retain its reports, and exercise lifecycle recovery during restart and
multi-principal validation. Do not represent data-dependent production gates
as completed evidence.

## Operating Rules

- Use per-track evidence strength and the weakest selected track for the run
  headline.
- Keep missing, failed, unavailable, and not-applicable observations distinct
  through plan, record, metric, gate, API, and UI layers.
- Pin code, benchmark, dataset, model, image, configuration, grader, price, and
  environment identities in evidence.
- Keep visible cases physically separate from hidden grading.
- Count failures and missing cells in coverage; require complete ledgers for
  cost-per-success claims.
- Let source receipts qualify a method only. Gate outcomes belong to typed
  server reductions or server-attested Campaign statistics.
- Keep private evidence under task-specific ignored directories and public
  reports free of target addresses, credentials, raw prompts, raw outputs, and
  host inventory.
- Run the smallest selected gate first, repair it, and expand only after it is
  green.
- Admit new capabilities only through versioned normalizer, executor, target,
  load, online-evidence, or reducer registries with exact tests.
- Resolve execution from the manifest-frozen executor identity and target
  capability contract; never infer an executor from a suite class or a live
  track from a built-in target ID. Server-owned ledger features exist only when
  their exact endpoint contract is configured.

## Related Docs

- [Evaluation Plane](../../../../website/docs/benchmarking/evaluation-plane.md)
- [Architecture guardrails](../architecture-guardrails.md)
- [Feature-complete checklist](../feature-complete-checklist.md)
- [Testing strategy](../testing-strategy.md)
- [Router Flow Evaluation Campaign](pl-0037-router-flow-eval-campaign.md)
- [Dashboard Modeling Experience](pl-0038-dashboard-modeling-experience.md)
