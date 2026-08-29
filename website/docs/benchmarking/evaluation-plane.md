---
title: Evaluation Plane
---

# Evaluation Plane

The Evaluation Plane answers a broader question than a model benchmark: does a
particular routing recipe, model pool, binding, and runtime environment improve
the system outcome for a frozen workload?

It provides one versioned run contract for deterministic replay and live
execution. The same run bundle can be created from the CLI, triggered through
the Dashboard, inspected in CI, and retained as deployment evidence.

## Current implementation and claim boundary

The first implementation deliberately separates a working evaluation control
plane from claims that still need runtime or benchmark-specific attestation:

- the legacy Dashboard evaluation database, runner, handlers, hooks, types, and
  task/dataset UI are removed;
- the new Dashboard owns versioned catalog, run lifecycle, cancellation, event
  replay, reports, comparisons, access control, immutable manifests, private
  evidence import, and server-anchored typed reports;
- the Python engine executes all eight track contracts with deterministic
  fixtures and can replay strictly normalized external suites;
- all 13 registered benchmark code revisions, plus the separate
  CodeRouterBench and xRouteBench dataset revisions, are exact pinned and
  rejected when dirty or mismatched;
- built-in fixtures, generic live probes, and operator-supplied normalized
  suites currently remain E0. They validate plumbing and diagnostics, not an
  upstream leaderboard or promotion claim;
- generic live targets expose only routing, multimodal, and capacity diagnostic
  probes. Model-pool and joint live evaluation require an attested direct-arm
  target; agentic, online preference, and hard-safety live evaluation require
  their own qualified execution contracts.

This fail-closed boundary is intentional. A successful HTTP call, clean Git
checkout, schema-valid bundle, or worker-authored checksum is not promoted into
stronger scientific evidence by inference.

## What it evaluates

An evaluation separates related questions so a good result cannot hide a weak
component.

| Track | Question | Primary evidence |
|-------|----------|------------------|
| Routing | Did the recipe choose an eligible, useful logical model? | route accuracy, coverage, fallback, oracle regret, decision latency |
| Model pool | Does the pool contain complementary arms for this workload? | per-arm outcome matrix, best single, pool oracle, unique wins, marginal contribution |
| Routing + pool | Does routing realize the pool's available value? | realized utility, normalized regret, quality-cost Pareto, reliability |
| Agentic | Does behavior remain useful across a trajectory? | task success, tool correctness, continuity, recovery, token and latency budget |
| Multimodal | Does the system admit, route, and execute each modality correctly? | capability coverage, route quality, grounding, privacy, media and latency cost |
| Preference | Do paired or logged outcomes favor the candidate? | win/tie rate, response coverage, propensity and effective sample size when available |
| Safety | Does the candidate preserve required policy behavior? | false positive/negative rates, hard violations, observed rate and confidence bound |
| Capacity | Where is the stable operating envelope? | throughput, TTFT and end-to-end percentiles, timeout/error rate, headroom and TCO |

The routing and model-pool tracks should be read together. A strong pool with a
weak router leaves utility unrealized. A strong router cannot compensate for a
pool with no capable arm, poor coverage, or correlated failures.

## Capability and claim boundary

The track catalog describes the questions understood by the Evaluation Plane;
the selected target describes which of those questions it can actually answer.
The Dashboard only permits a suite when every requested track is advertised by
that target. Missing evidence is reported as `unavailable`, never synthesized
from another track.

| Execution source | What it proves | What it does not prove |
|------------------|----------------|------------------------|
| Built-in `evaluation-smoke` fixture | contract, executor, metric, gate, artifact, and report plumbing across all eight track schemas | qualified E1-E5 evidence, live model quality, deployable pool quality, trajectory success, production safety, capacity, or user preference |
| Generic active runtime target | E0 route reachability, one bounded multimodal probe, and bounded concurrency diagnostics | direct-arm model-pool value, joint regret, qualified safety, trajectory, user preference, or promotion-grade capacity |
| Imported normalized suite | strict source pins, normalized schema, private-label separation, deterministic replay, metrics, gates, and lineage at E0 | source-to-row derivation or an upstream result until adapter execution and native metric/grader parity are attested |

The smoke fixture intentionally contains only a small deterministic workload.
Its all-track report validates the harness and must not be used as a model,
recipe, pool, safety, or capacity leaderboard. A future qualified live run must
bind server-owned model arms, use an evaluation-only direct-arm capability, and
correlate the logical route with the executed immutable model/runtime revision
before it can report model-pool or joint realized-value evidence.

## Evidence model

Every run freezes the following inputs before execution:

- visible workload cases and a physically separate grading artifact;
- policy or recipe revision, evaluated policy instance, and binding;
- model-pool and model-arm definitions;
- target and run environment, including configuration and code digests;
- tracks, graders, seed, repeats, sample and concurrency budgets;
- metric and gate definitions;
- redaction policy and secret environment references.

The Dashboard computes the canonical `manifest_digest` rather than trusting a
worker or browser value. The digest, gate-contract version, suite revisions,
and immutable source revision are revalidated when a pending run starts and
again when its report bundle is sealed. A worker cannot silently switch the
suite, source, or release-gate semantics after creation.

The browser never receives the hidden grading artifact. Target URLs are also
server-owned: the Dashboard submits a catalog `target_id`, not an arbitrary
endpoint. Credentials are environment references and are not persisted as
literal values in a manifest.

Evidence is labeled by strength:

| Level | Meaning |
|-------|---------|
| E0 | Structure and reachability: schema, DSL, references, capability slots, cycles, and deterministic digests |
| E1 | Signal evidence: discrimination, calibration where meaningful, missing/error rate, latency, OOD degradation, and paired invariance |
| E2 | Projection evidence: purity, coverage, overlap, downstream correlation, boundary churn, and contribution completeness |
| E3 | Decision evidence: expected-decision accuracy, branch/default behavior, collision and priority handling, hard-policy checks, and router-only latency |
| E4 | Algorithm evidence: feasible-oracle recall, realized utility, regret, model utilization, exploration, baselines, seed variance, and OOD |
| E5 | End-to-end evidence: task or trajectory outcome, reliability, privacy/safety, live latency, full cost, capacity, and cost per success |

A higher level is not automatically a better experiment. It means the report
contains a deeper downstream observation, not that every lower-level contract
passed. A synthetic fixture validates plumbing but does not earn E1-E5 product
evidence merely because it populates fields with deterministic values.

## Pinned benchmark adapters

External benchmark code and data stay in an ignored source cache. An adapter
must verify a clean Git checkout at its exact code and optional dataset commit,
then normalize source evidence into the Evaluation Plane IR. The source tree is
never executed from a browser request, and upstream headline results are not
copied into a vLLM-SR report.

| Adapter family | Registered benchmarks | Normalized evidence |
|----------------|-----------------------|---------------------|
| Prediction and preference | RouterArena; RouteJudge / ORBIT | blind decisions, response pairs, votes, exposure, propensity |
| Dense outcome matrix | CodeRouterBench; LLMRouterBench; RouterEval; RouterBench; MMR-Bench | case × arm outcomes, splits, graders, prices, pool and budget sweeps |
| Scenario and trajectory | xRouteBench; TwinRouterBench | session state, modality, trajectory prefix, step action, terminal result |
| Executable reliability and privacy | AceBench; continuity-bench | workspace/tool evidence, egress/privacy ledger, exact-step fault manifest, recovery |
| Composite and budget actions | FusionFactory / LLMFusionBench; R2-Router | subset/topology/synthesis actions and model × output-budget curves |

The canonical suite contract physically separates visible cases from hidden
grading labels and supports dense outcomes, pairwise preferences, trajectory
steps, perturbation pairs, fault events, media/license manifests, and compound
actions. Each installed suite records its redistribution policy and an evidence
ceiling; unsupported claims remain unavailable.

### What each registered benchmark contributes

| Benchmark | Native design | Evaluation Plane contribution | Qualification needed above E0 |
|-----------|---------------|-------------------------------|--------------------------------|
| RouterArena | blind query-to-model prediction file, task graders, cost and perturbation scoring | routing decisions, quality/cost frontier, oracle and robustness pairs | prediction export parity, grader/price snapshots, perturbation semantics |
| RouteJudge / ORBIT | budget-conditioned router recommendations followed by anonymous pairwise votes | preference pairs, exposure, participation, cost-preference frontier | assignment mechanism, propensity, missing-vote policy, calibrated offline/live parity |
| CodeRouterBench | ordered coding stream with dense arm outcomes and verified-history adaptation | prequential routing, regret, memory state, agentic OOD evidence | sequence/split freeze, no-future-leakage proof, sandbox and grader parity |
| LLMRouterBench | large frozen dense query × model outcome matrix | common algorithm comparison, budget gain, Pareto distance | exact matrix/split/grader/price reproduction and freshness policy |
| RouterEval | sampled pools from 3 to research-scale model counts | pool-size factorials, entropy, collapse, relative references | pool seed/metadata parity and separate deployability/capacity evidence |
| RouterBench | dense outcomes with model, cascade, and over-generation policies | no-information convex hull, quality/cost and AIQ baselines | matrix and budget-sweep parity with refreshed prices/models |
| xRouteBench / LLMRouter | single-turn, session, personalization, and multimodal scenarios | scenario/session state, modality, preference, hidden-call cost | media/license manifest, session grouping, complete hidden-call ledger |
| TwinRouterBench | routing at an agent trajectory prefix or step | downgrade/escalation decisions, step and terminal trajectory outcomes | static-label parity plus reproducible live SWE trajectory execution |
| MMR-Bench | multimodal query routing over dense MLLM outcomes | typed media cases, capability masks, quality/cost multimodal routing | media lineage, modality-specific graders, privacy and robustness slices |
| AceBench | executable agent tasks with utility, cost, and privacy boundaries | workspace/tool evidence, privacy ledger, edge/cloud policy | isolated sandbox, egress attestation, side-effect and privacy parity |
| continuity-bench | deterministic provider-failure session protocol | exact-step faults, fallback continuity, recovery and CPR-style metrics | real fault injection, state-transfer attestation, repeated seeds |
| FusionFactory / LLMFusionBench | subset, topology, and synthesis actions over multiple model outputs | compound action graph, hidden call accounting, synthesis outcome | topology/action normalization, full token/cost ledger, judge parity |
| R2-Router | model plus output-budget action over quality curves | adaptive compute budget, model × budget regret and frontier | curve extraction, stop/output accounting, budget-conditioned parity |

The registry captures these designs, but does not execute upstream code from a
browser. Maintained native normalizers and parity receipts are tracked as an
explicit architecture gap rather than being simulated by the common executor.

## Run workflow

An Evaluation Plane run follows one lifecycle:

```text
catalog target + frozen definition
              |
              v
       immutable run manifest
              |
              v
 plan cells -> execute cases -> collect typed evidence
              |
              v
 aggregate metrics -> evaluate gates -> render report
              |
              v
 content-addressed artifacts + lineage + checksums
```

The Dashboard exposes the maintained built-in/runtime lifecycle under
**Evaluation**:

1. Choose a maintained suite and a server-configured target.
2. Select a change profile, replay or live mode, tracks, sample limit,
   concurrency, and seed.
3. Start the run and follow its durable event stream.
4. Inspect track coverage, metrics, gates, failures, cost ledgers, and lineage.
5. Compare a candidate with a fixed baseline before making a promotion
   decision.

Use the deterministic `evaluation-smoke` suite first. It exercises all tracks
and the evidence pipeline without making a model-quality claim. Its current
built-in revision contains four cases. Move to a live or imported pinned suite
only after the fixture contract passes, and select only the tracks exposed by
that target's catalog entry.

The CLI exposes the same catalog, validation, run, report, comparison, and gate
operations under `vllm-sr eval`. `vllm-sr eval benchmarks` lists the pinned
adapter contracts; `vllm-sr eval verify-source --adapter <id> --source-root
<ignored-cache>` creates a read-only pin receipt. An operator can then use
`suite-install --request <manifest> --bundle <normalized-bundle> --source-root
<ignored-cache>`, followed by `suite-list`, `suite-show`, and `run --suite-store
<private-store>`. Installation always reruns source verification; it ignores a
caller-supplied verification receipt. Imported suites are CLI/operator-only in
this version and remain E0 until adapter execution attestation exists. Run
`vllm-sr eval --help` for the installed version's exact options.

## Run bundle

The default local store is `.vllm-sr/evaluation-store/`. Final artifacts are
immutable and content addressed. Mutable status and the run index use atomic
updates.

```text
.vllm-sr/evaluation-store/
  objects/sha256/<digest>
  runs/<run-id>/
    run-manifest.json
    status.json
    events.jsonl
    control-events.jsonl
    records.jsonl
    routing-traces.jsonl
    metrics.json
    gates.json
    report.json
    report.md
    report.html
    lineage.json
    provenance.json
    failure-cases.jsonl
    failure-summary.json
    checksums.sha256
    private-checksums.sha256
  index/runs.json
```

`control-events.jsonl` is the Dashboard-owned lifecycle stream;
`events.jsonl` is worker evidence. `records.jsonl`, grading cases, failure cases,
lineage, worker Markdown/HTML, the private checksum receipt, and the durable
manifest are private. The public artifact endpoint permits only server-verified
structured metrics, gates, provenance, aggregate failure summary, bounded
routing/capacity diagnostics, and their public checksum receipt. Dashboard
reports are rendered from a strictly parsed and server-anchored `report.json`,
not trusted worker-authored markup.

All authenticated Evaluation API responses, including errors, byte-range
artifact downloads, and event streams, carry `Cache-Control: private,
no-store`. Event streams use durable numeric IDs and `Last-Event-ID`; browser
reconnects replay only events after that cursor and suppress duplicate IDs.
Before publication, the Dashboard strictly validates bounded records, unique
case/attempt/arm keys, finite metric values, artifact receipts, lineage, and
the aggregate failure summary. E0 fixture evidence cannot satisfy a promotion
gate merely because the bundle is structurally complete.

Keep raw private workloads and outputs in an ignored store. A public report
should include aggregate evidence and reproducible provenance without exposing
private prompts, outputs, target addresses, credentials, or infrastructure
identifiers.

## Reading a report

Read the report in this order:

1. **Coverage.** Confirm how many planned cases produced usable evidence.
   Failure and unavailable cases remain in the denominator.
2. **Required gates.** A failed required gate blocks the claim. An unavailable
   gate means the run lacks the evidence to decide; it is not a pass.
3. **Primary quality and safety metrics.** Check confidence intervals and
   slices, not only the aggregate.
4. **Routing and pool decomposition.** Compare realized utility with best
   single and pool-oracle baselines.
5. **Latency, reliability, and three cost ledgers.** Runtime cost, evaluation
   overhead, and capacity/TCO answer different questions and must not be added
   without an explicit model.
6. **Failure cases and lineage.** Verify the candidate, environment, workload,
   and grader are the intended frozen revisions.

Candidate comparison is also fail-closed. It rejects self-comparison, a
mismatched `baseline_run_id`, workload/benchmark/seed drift, and treatment
factors that do not match the selected change profile. Aggregate point deltas
can diagnose a clear regression, but cannot pass a promotion: that requires a
case-aligned paired delta interval under a registered statistical comparison
contract.

The default gate disposition is explicit:

- `required`: must pass for the report claim;
- `advisory`: reported but does not block;
- `not_applicable`: the claim does not apply to this run;
- `waived`: deliberately waived with a recorded rationale.

Gate verdicts distinguish `pass`, `fail`, `unavailable`, `not_applicable`, and
`waived`. Do not collapse them into a boolean.

### G0-G9 release semantics

| Gate | Required evidence |
|------|-------------------|
| G0 Reproducibility | frozen inputs, seeds, failures, digests, and unbroken lineage |
| G1 Static correctness | strict schemas, conformance, references, reachability, coverage, deterministic replayability |
| G2 Hard policy | static enforcement plus dynamic `0/N`, one-sided bound, slice/fault coverage; any observed violation fails |
| G3 Offline value | paired incumbent and no-information baseline, frontier position, pool-normalized regret, router overhead |
| G4 Robustness / OOD | invariant and expected-change pairs, temporal/source/language/domain/modality slices, contamination audit |
| G5 Live fidelity | paired replay-to-live gap, fresh outputs, complete timeout/retry/failure accounting, calibrated grader |
| G6 Reliability / trajectory | terminal success, continuity, recovery, multi-seed stability, state isolation, idempotent side effects |
| G7 Cost / latency / capacity | three cost ledgers, load profile, tail latency, saturation, SLO crossing, headroom, retry amplification |
| G8 Shadow / canary | assignment/exposure counts, divergence, hard checks, risk budget, stop and rollback evidence |
| G9 Online preference | participation, propensity, effective sample size, confidence, segments, and preference-cost-latency frontier |

Before execution, the run selects a versioned `change_profile` such as recipe,
selector, model-pool, runtime-capacity, agent-multimodal, or online adaptation.
That profile marks every gate required, advisory, or not applicable. Missing
evidence for a required gate makes the promotion verdict unavailable; it never
becomes a pass by omitting the gate. A waiver cannot be self-issued by the
Evaluation Plane.

## Diagnosing recipe and pool design

Use evaluation outcomes to decide which architecture surface to change.

| Finding | Likely problem | First design action |
|---------|----------------|---------------------|
| Low route accuracy and a large pool-oracle gap | recipe signals, decisions, selector, or binding | inspect slices and decision trace; revise recipe before adding models |
| High route accuracy but low live quality | labels or route proxy do not match task utility | change grader/workload or optimize generation policy |
| Best single is close to pool oracle | pool has little complementarity | remove redundant arms or add a capability gap, then rerun the dense matrix |
| Pool oracle is high but realized utility is low | router cannot exploit pool | improve policy features, calibration, eligibility, or exploration |
| Agent success drops while single-turn quality holds | trajectory continuity or recovery | evaluate decision points, tool loop protection, state portability, and retry policy |
| Multimodal route succeeds but generation fails | serving capability or payload transport | separate modality admission/routing from backend generation support |
| Safety observes zero violations on a small sample | insufficient risk evidence | report `0/N` with a one-sided bound and expand adversarial coverage |
| Quality holds but capacity gates fail | serving placement or queueing | tune serving and placement; do not change the logical recipe without evidence |

This matrix keeps ownership clear: the Router selects a logical model, the
serving layer owns physical placement and transport, and the Evaluation Plane
measures both without moving those responsibilities.

## Online preference limits

Offline paired preference can compare frozen responses. Unbiased online
evaluation additionally needs the assignment mechanism, exposed alternatives,
behavior probability or propensity, protection/fallback action, and the action
that actually executed. Without those fields, production A/B, interleaving,
off-policy estimation, and adaptation gates must be reported as unavailable.

The Evaluation Plane never writes preference results directly to
`/v1/router/outcomes`. Online outcomes remain authenticated, replay-linked
runtime feedback with their own idempotency and policy controls.

## Promotion rule

An evaluation report recommends; it does not deploy. Promotion should require:

- the intended evidence level for the claim;
- all required gates passing;
- no unexplained coverage loss or environment drift;
- an explicit baseline and rollback plan;
- reviewed artifacts and checksums.

Capacity or production claims require a live, pinned environment. Synthetic
fixture results remain regression evidence only.

## Architecture gaps exposed by the pipeline

The evaluation work itself identified contracts that should be added to the
broader system instead of hidden in benchmark code:

- an evaluation-only, short-lived direct-arm execution capability with
  executed-model/runtime attestation and routed-call correlation;
- a versioned load campaign contract with warm-up, arrival process, duration,
  repetitions, resource snapshot, SLO, saturation, and headroom;
- native, sandboxed source-to-IR adapters with transformation receipts and
  upstream metric/grader parity;
- a separate assignment/exposure/propensity ledger for shadow, canary, online
  preference, and off-policy learning;
- a dedicated least-privilege evaluation worker boundary, separate from
  Dashboard deployment-management authority;
- private case-aligned comparison statistics, retention/GC, crash durability,
  evidence ownership, destructive-operation authorization, and lifecycle
  audit.

These gaps are durable debt with explicit exit criteria. Until each contract is
implemented, its dependent gate remains `unavailable`; the Evaluation Plane
does not compensate by inventing labels, endpoints, propensities, or evidence
levels.
