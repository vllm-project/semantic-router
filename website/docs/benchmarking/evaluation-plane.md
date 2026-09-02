---
title: Evaluation Plane
---

# Evaluation Plane

The Evaluation Plane measures whether a frozen routing recipe, logical model
pool, binding, workload, and runtime environment improve the system outcome. It
does not reduce evaluation to “did the router name the expected model?” A
useful decision must also be feasible, execute the intended arm, preserve
safety and privacy, stay within cost and latency budgets, and improve the final
task or trajectory outcome.

The implementation is a greenfield control-plane subsystem with one current
run contract, one executor registry, one durable bundle layout, one report
shape, and one server-attestation revision. CLI, Dashboard, comparison, and
Campaign workflows consume the same evidence model.

Live evaluation is subject-bound. The catalog publishes one target for each
request-reachable Mixture-of-Models Recipe, never a generic runtime target. A
run freezes that Mixture's Entrypoint aliases, Recipe and decision boundaries,
logical model arms, provider fallback, support models, prices, and
recipe/pool/binding/topology digests before any request is sent.

## System boundary

Evaluation observes the complete path while preserving runtime ownership:

```text
workload + session/tool/media state + policy constraints
                            |
                            v
              signals -> projections -> decision
                            |
                            v
       logical action: model(s), budget, selector/looper, fallback
                            |
                            v
       serving execution: endpoint, queue, retry, physical replica
                            |
                            v
 quality + success + cost + latency + safety + privacy + preference
```

- The Router owns logical Entrypoint resolution, Recipe execution, logical
  model selection, selector/looper policy, generation budget, and logical
  fallback constraints.
- Agent and product control planes own tools, roles, workspace mutation, and
  external side effects.
- Envoy and the serving/fleet layer own transport, physical placement,
  replicas, queues, and capacity.
- The Evaluation Plane correlates these facts without moving their ownership
  into the Router request path.

“Fleet” in this ownership description is conceptual. The retired Fleet
Dashboard is not part of Evaluation: the `/fleet-sim` routes, navigation, API
surface, and startup sidecar dependency have been removed. The standalone
`src/fleet-sim` research package, its documentation, and release tooling remain
independent and are not started by the Dashboard stack.

## Eight evaluation tracks

Every selected track has its own evidence level, coverage, metrics, and gates.
The run-level evidence level is the weakest selected track, so one strong track
cannot promote another track that lacks evidence.

| Track          | Question                                                                   | Current metric surface                                                                                                                                                                      |
| -------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Routing        | Did the Recipe choose a useful eligible logical arm?                       | coverage, accuracy, abstention, fallback, execution success, selected-arm count and entropy, route latency p50/p95                                                                          |
| Model pool     | Does the pool contain useful, learnable, and dependable complementarity?   | arm count, best single, pool oracle, oracle gain, unique wins, selection coverage/entropy, quality dominance, quality-cost Pareto dominance, per-arm and worst-arm reliability, pairwise failure overlap, all-arm failure rate |
| Routing + pool | How much of the pool ceiling did routing realize?                          | realized quality, oracle regret, normalized regret, oracle-capture ratio, reliability, complete runtime cost per success, latency p95                                                       |
| Agentic        | Does routing remain useful over a trajectory?                              | terminal success, task score, invalid-tool rate, trajectory length, privacy exposures per trajectory, complete runtime cost per successful trajectory                                       |
| Multimodal     | Were media capability, routing, execution, grounding, and privacy correct? | overall and per-modality support/quality, privacy violations                                                                                                                                |
| Preference     | Does qualified feedback favor the candidate?                               | agreement, propensity coverage, effective sample size and ratio, self-normalized IPS agreement                                                                                              |
| Safety         | Were hard policy and blocking decisions correct?                           | violations/case, violated-case rate and one-sided 95% upper bound, block accuracy, false-negative and false-positive rates                                                                  |
| Capacity       | Where is the observed stable envelope?                                     | per-level and aggregate throughput, p95/p99, success/error, scaling efficiency, observed saturation, stable concurrency upper bound, cost per success, and frozen-SLO headroom              |

Cost reducers fail closed on incomplete ledgers. For example, a missing arm or
trajectory cost does not produce an artificially low cost-per-success number.
A live run that selects Capacity must declare a versioned SLO before it can be
created and must request concurrency of at least two, so adjacent-level scaling
is measured rather than implicitly accepted. The manifest freezes required
concurrency, maximum p95 latency, maximum error rate, minimum throughput, and
minimum adjacent-level throughput scaling.
The worker reduces real load observations into a monotonic qualified envelope;
the server recomputes the profile and rejects altered level decisions,
saturation, headroom, or verdicts. Replay capacity remains diagnostic and
carries no SLO.

## Runnable execution sources

The catalog separates “understands this question” from “can answer it for this
target.” Every target declares `accepted_executors` for each of its modes. The
planner admits only a suite whose mode, tracks, and executor are accepted by
that target, and every run contains exactly one executor cohort.

| Source                             | Current capability                                                                                                                         | Scientific boundary                                                                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `evaluation-smoke`                 | deterministic four-case replay across all eight track schemas through `fixture-replay.v1`                                                  | E0 vertical-slice diagnostics only                                                                                                  |
| `live-mom-core`                    | the same immutable 64-case hidden-label cohort through `mom-cohort-replay.v1` or `live-runtime.v1`, with routing, a dense case-by-frozen-arm matrix, and routed outcomes | replay is E0; a complete server-attested live run can seal routing E3, model-pool E4, and joint E5 (run-level E3). Those levels do not by themselves satisfy G3, which requires a server-controlled pair |
| `live-agent-tasks`                 | complete `evaluation-agent-task-ledger.v1` evidence with `evaluation-agent-task-attempt.v1` repeated-task trajectories observed by an external production agent runtime, including grading, privacy, cost, and real-tool execution receipts bound to the exact Mixture | agentic E5 task-quality evidence after server validation and reduction; the evaluation worker does not execute tools, `benchmark_parity_claim` remains `none`, and this method has no Campaign gate and never qualifies G6 |
| `live-fault-recovery`              | complete brokered exact-step fault ledger with paired baseline/treatment receipts, repeated seeds, state, side effects, retry, and latency | E5 only after the server re-reduces at least 20 pairs across at least 5 seeds; Continuity labeled failover is diagnostic only        |
| `live-multimodal`                  | bounded eligible non-text requests through the active runtime                                                                              | E0 media transport and response diagnostics                                                                                         |
| `live-hard-policy`                 | Router-owned policy/config proof plus dynamic attack/block observations that exactly cover required rule/enforcement-point pairs           | G2 is data-dependent until the explicit endpoint supplies one complete sealed live window                                           |
| `live-production-experiment`       | consumes an explicitly configured external ledger of sealed randomized policy-arm assignments/exposures, rollout controls, and optional preference outcomes | G8 uses operational safety receipts and risk UCB; G9 additionally requires complete propensity-qualified target/reference outcomes; vLLM-SR does not create or operate the experiment |
| `live-capacity`                    | short repeated closed-loop load levels through the active Entrypoint for checking load execution, telemetry, and report generation | E0 diagnostic only; it does not qualify G7 or support a release capacity decision |
| Installed normalized suite, replay | exact pinned source export normalized into typed private artifacts and replayed by `normalized-suite-replay.v1`                            | per-track source-bound evidence only when the trusted installer re-derives it                                                       |
| Installed normalized suite, live   | visible cases executed against the active runtime by `normalized-suite-live.v1`                                                            | live execution receives no replay qualification; it must earn server-owned live evidence                                            |

Executor compatibility is registry data, not a target-ID branch. A target
provider declares accepted executor identities per mode; the run manifest
freezes the selected identity for every suite, and staging resolves that exact
identity before it admits work. Multiple executor implementations may share a
suite class without changing or ambiguating an existing run.

Track availability is likewise derived from each Mixture target's advertised
features and per-track requirements. Routing requires its frozen Recipe,
topology, Router evaluation API, and Envoy. Model-pool and joint require Envoy,
topology, and at least two executable frozen arms; they do not depend on the
Router diagnostics API. Multimodal additionally requires a non-text arm.
Agentic task quality and fault recovery use separate explicit ledgers and a
suite cannot substitute one for the other. Preference
requires a production experiment ledger, safety requires a hard-policy ledger,
and capacity requires Envoy. The state model is intentionally precise:
unsupported suite/target/executor combinations are rejected during planning;
malformed or missing required artifacts fail execution or sealing; an executed
cell with no observation is recorded as `unavailable`; a valid report without
the typed proof for a gate leaves that gate `unavailable`; an observed eligible
regression is `fail`; and a gate excluded by the selected change profile is
`not_applicable`. Catalog methods report only `configured` or `data_required`;
qualification comes from sealed run evidence, never from catalog presence.

### Configure production evidence services

`vllm-sr serve` forwards Evaluation configuration to the Dashboard container
from the host environment. Configuration contains only canonical origins,
bounded timeouts, and environment-variable names. Credential values are passed
as inherited container environment entries and never rendered into Docker
arguments, run manifests, catalog responses, reports, or logs.

| Dashboard environment variable | Contract |
| --- | --- |
| `EVALUATION_ROUTER_API_KEY_ENV` | Name of a dedicated Router bearer-token environment variable. The token must be declared under `global.services.management_api.auth.tokens`, its role must include `classify.invoke`, and it must not be the Dashboard Recipe-management credential. |
| `EVALUATION_ENVOY_API_KEY_ENV` | Name of the credential used only for brokered Envoy model discovery and chat calls. |
| `EVALUATION_AGENT_TASK_LEDGER_URL`, `_API_KEY_ENV`, `_TIMEOUT` | Exact canonical origin, independent credential reference, and Go duration for the sealed provider-observed agent-task ledger. |
| `EVALUATION_FAULT_RECOVERY_LEDGER_URL`, `_API_KEY_ENV`, `_TIMEOUT` | Exact canonical origin, independent credential reference, and Go duration for the sealed recovery ledger. |
| `EVALUATION_HARD_POLICY_LEDGER_URL`, `_API_KEY_ENV`, `_TIMEOUT` | Exact canonical origin, independent credential reference, and Go duration for the sealed policy ledger. |
| `EVALUATION_PRODUCTION_EXPERIMENT_LEDGER_URL`, `_API_KEY_ENV`, `_TIMEOUT` | Exact canonical origin, independent credential reference, and Go duration for the sealed assignment/exposure/outcome ledger. |

Each ledger is either entirely absent or configured with both an origin and an
API-key environment reference. Its timeout defaults to `30s` when configured
and must be at most `10m`. Router, Envoy, and ledger credential references are
pairwise distinct; every ledger origin is distinct from Router, Envoy, and the
other ledgers. Origins are exact `http(s)://host[:port]` values with no user
information, path, query, fragment, whitespace, or trailing slash. Invalid or
partial configuration fails closed before Evaluation routes become available.

With Router bearer authentication enabled, omitting
`EVALUATION_ROUTER_API_KEY_ENV` keeps model-pool, joint, multimodal, and
capacity work available through Envoy where their own requirements are met,
but removes routing evaluation from the target. Supplying the dedicated token
restores `router.evaluate`; the Go broker resolves its value server-side and
adds `Authorization` only to the exact frozen Router origin.
The Python worker carries only the `SecretRef` identity and never resolves its
environment value or constructs an authorization header. Consequently,
standalone `vllm-sr eval run` accepts unauthenticated targets but fails closed
when a manifest references credentials without the Dashboard broker.

### Address baseline and candidate deployments together

Set `EVALUATION_DEPLOYMENTS_DIR` when one Dashboard Evaluation service must
address multiple simultaneously running Mixture-of-Models deployments. The
canonical local `vllm-sr serve` path mounts this host directory read-only at
`/app/evaluation-deployments` in the Dashboard container only. Router and Envoy
receive neither that mount nor the environment variable. With the variable
unset, the current single-runtime target and its existing zero-configuration
behavior are unchanged.

The directory contains a strict `registry.json` and the referenced Router YAML
files:

```json
{
  "schema_version": "evaluation-deployments.v1",
  "deployments": [
    {
      "id": "baseline",
      "name": "Baseline",
      "description": "Current production deployment",
      "config_file": "baseline/config.yaml",
      "router_origin": "http://baseline-router:8080",
      "envoy_origin": "http://baseline-envoy:8899"
    },
    {
      "id": "candidate",
      "name": "Candidate",
      "description": "Candidate deployment",
      "config_file": "candidate/config.yaml",
      "router_origin": "http://candidate-router:8080",
      "envoy_origin": "http://candidate-envoy:8899"
    }
  ]
}
```

The schema rejects unknown fields, an empty deployment list, duplicate IDs or
resulting target IDs, non-canonical origins, absolute/traversing config paths,
and any symlink in the registry, config, or host mount path. Each config is
parsed through the same Mixture snapshot loader as the default runtime. The
server derives the config digest from its exact bytes and derives the Recipe,
selector, adaptation, binding, pool, and topology identities from the parsed
content. `registry.json` cannot contain credentials or ledger endpoints; the
existing global environment-only SecretRefs remain the only credential and
typed-ledger authority.

Catalog target IDs are deployment-scoped (`<deployment>--<mixture-id>`), while
the embedded Mixture ID and Recipe name remain the shared logical experiment
subject. Only the safe deployment name is projected into the authenticated
catalog. Origins, config paths, and SecretRefs remain private in the frozen run
manifest and broker. This lets a controlled pair bind one baseline target and
one candidate target without conflating their network address with the Recipe
treatment.

## Mixture-of-Models replay and live evaluation

The Mixture workspace links directly to Evaluation with the exact Entrypoint
selected. The browser submits only the catalog `target_id`. It cannot inject an
origin, Recipe, or model list. The server resolves and freezes the current
target into both the run and manifest; start-time validation fails closed when
the Recipe, pool, binding, topology, aliases, decisions, or executor admission
has drifted.

In zero-configuration single-runtime mode, the target ID remains the Mixture
ID. In a deployment registry, the target ID is deployment-scoped while the
embedded Mixture ID and Recipe name stay stable across baseline and candidate;
immutable digests distinguish their revisions. The server derives a canonical,
orthogonal factor graph:

| Factor | Exact executable identity |
| --- | --- |
| Recipe | non-classifier signals, decision rules/priority/plugins, and routing strategy; excludes candidates, selector algorithms, selector projections, and decision-local adaptations |
| Selector policy | classifier signals, projections, and exact per-decision algorithm configuration |
| Selector | selector-policy digest plus every selector-only support model's logical name, one-way provider-model identity, provider/config digest, declared runtime revision when available, and one-way backend-topology digest |
| Adaptation | exact per-decision online adaptation and protection configuration |
| Binding | Entrypoint/aliases/fallback plus exact decision and candidate-iteration model references; excludes selector identity |
| Pool | sorted candidate-arm executable identities, capabilities, modalities, prices, config digests, and declared runtime revisions |
| Environment | Router/Envoy and method-ledger origins/credential references plus candidate-pool serving topology; selector backend topology is not duplicated here |

This decomposition prevents the same change from being labeled as a Recipe,
selector, or online-adaptation experiment. Paired comparison enforces the exact
profile contract, including the required primary delta:

| Change profile | Required primary delta | Permitted dependent delta | Frozen factors |
| --- | --- | --- | --- |
| `schema_adapter` | source-code revision | none | Recipe, selector, adaptation, binding, pool, environment |
| `recipe` | Recipe | none | source code, selector, adaptation, binding, pool, environment |
| `selector` | selector | none | source code, Recipe, adaptation, binding, pool, environment |
| `model_pool` | pool | candidate binding and candidate-pool topology | source code, Recipe, selector, adaptation, runtime origins, credentials, and method ledgers |
| `runtime_capacity` | environment | none | source code, Recipe, selector, adaptation, binding, pool |
| `online_adaptation` | adaptation | none | source code, Recipe, selector, binding, pool, environment |

`agent_multimodal` does not yet have one independent server-owned treatment
digest. It remains valid for individual diagnostic runs, but paired comparison
fails closed instead of accepting generic Recipe/binding drift under a second
profile name. A future paired contract must freeze an explicit trajectory,
tool/state, media-admission, and modality-execution factor before enabling that
profile.

`live-mom-core` has one immutable revision and one 64-case workload identity in
both replay and live modes. Replay uses `mom-cohort-replay.v1` to produce a
deterministic frozen-target counterfactual; live uses `live-runtime.v1` to call
the target. Both modes grade routing from the same complete dense-pool oracle.
Replay remains E0. A complete live run with broker receipts, execution
attestation, and server reduction can seal routing E3, model-pool E4, and joint
E5, with run-level evidence E3 because the run reports its weakest selected
track. These levels alone do not pass G3; that gate additionally requires a
server-controlled baseline/candidate pair. The replay executor binds every
deterministic choice and outcome to the complete visible and hidden-grading
case snapshot, not only the case ID, so changed case content cannot silently
reuse an old pseudo-outcome.

One cohort produces three complementary observations:

1. **Recipe routing:** `POST /api/v1/eval?trace=true` evaluates the exact frozen
   Entrypoint and records the Recipe, decision, selection method, selection
   status, selected logical arm, fallback state, and trace digest.
2. **Model pool:** every case is sent directly to every frozen logical arm.
   The complete `case × arm` matrix measures per-arm quality/reliability/cost,
   best single, pool oracle, complementarity, unique wins, dominance, Pareto
   structure, and correlated failure.
3. **Routed system:** every case is sent through the frozen Entrypoint. The
   selected arm, response quality, token-derived frozen price, reliability,
   latency, oracle capture, and regret measure the realized routing-plus-pool
   system.

The worker receives visible prompts but not hidden labels. Its network sandbox
can request only `models.list`, `router.evaluate`, `arm-chat.completions`, or
`routed-chat.completions` for the exact manifest track/case/attempt. The Go
broker owns origins and credentials, verifies every virtual alias and Recipe,
confines direct calls to the frozen arm, confines routed selections to the
frozen decision boundary or explicit fallback, and writes one receipt per
ordinary observation. The server joins hidden labels after execution,
recomputes response quality and frozen-price cost, requires the dense pool
matrix, and derives a routing oracle from complete same-cohort arm outcomes
when no native route label exists. Missing matrix cells never become a zero or
a fabricated route failure.

## Benchmark research and adapter inventory

The benchmark registry pins 13 descriptors from the Intelligent Routing
Landscape. Their source inventory is 15 external checkouts: 13 code
repositories plus the separate CodeRouterBench and xRouteBench dataset
repositories. The research descriptor records each benchmark's native
ambition. The normalizer registry separately contains 13 descriptors for the
smaller evidence surface that can be safely derived from those pins today.

Eleven normalizers are executable. The RouteJudge/ORBIT and RouterEval
descriptors are diagnostic-only because their pinned checkouts do not expose
the required safe per-case JSON/CSV export; the Evaluation Plane does not
execute upstream code or deserialize their pickle artifacts.

| Benchmark                      | Exact source pin                                                                                   | Native design and emphasis                                                                                | Current safe normalized schema       | Executable tracks and strict limit                                                                                                                |
| ------------------------------ | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| RouterArena                    | `fda4c53bcf9a979fd9c6f6bb6b713d6ab08ff43e`                                                         | blind query-to-model predictions, task grading, cost, optimality, perturbation robustness, router latency | `routerarena.predictions-and-robustness.v2` | routing, model-pool, joint; import/replay remains E0. A parser-verified complete perturbation corpus can be re-executed live against the frozen Mixture and qualify only `declared-shift.server-live.v1` routing G4 |
| RouteJudge / ORBIT             | `494810de2605f69737e72b55baf6e60c95c6dec0`                                                         | budget-conditioned recommendations followed by anonymous preference votes                                 | `routejudge-orbit.unavailable.v1`    | diagnostic-only; the pin has neither RouteJudge exposure/vote records nor a safe per-case export                                                  |
| CodeRouterBench                | `e43839edb0d5d0a9feec2f7078019406ab4d64bd` plus dataset `e567d89bdd569c9c74ffc7c7118e50d15e46b886` | ordered coding stream, dense arm outcomes, verified-history adaptation, agentic OOD                       | `coderouterbench.id-results.v1`      | routing, model-pool, joint; ID task identity and dense results only, excluding the sandboxed OOD agent stream                                     |
| LLMRouterBench                 | `c77cb0506949d8f959e97967d2fefca0e8ff1b05`                                                         | large dense query-by-model matrix, budget gains, oracle gap, Pareto distance                              | `llmrouterbench.result-documents.v1` | model-pool; aligned result documents only, without a learned-router decision or native frontier reducer                                           |
| RouterEval                     | `bf94b49cc9f8b37181715a7309e1b70ff5308942`                                                         | model-pool scaling from small to research-scale pools, relative references and entropy                    | `routereval.unavailable.v1`          | diagnostic-only; per-case pool evidence is loaded only through pickle in the pin                                                                  |
| RouterBench                    | `cc67d1008bd8f3cf1e8040cc3ba4034d31b93c0c`                                                         | dense outcomes, model/cascade/over-generation actions, Zero Router convex hull and AIQ                    | `routerbench.wide-csv.v1`            | model-pool; converted wide matrix only, not cascade, over-generation, AIQ, or hull parity                                                         |
| xRouteBench / LLMRouter        | `da3430baaea672743c3957457b0c76faba19876e` plus dataset `ea4b6e1b29d9a734f55f0a637baf326bad6aa681` | single-turn, session, personalization, caption-first multimodal, multi-agent scenarios                    | `xroutebench.standardized-csv.v1`    | model-pool; one standardized scenario CSV at a time, with no inferred session, preference, media, or hidden-call ledger                           |
| TwinRouterBench                | `7cbb0deac8f697b5faa8489c309560e53d2ef088`                                                         | route at an agent trajectory prefix/step, static tier labels plus dynamic SWE execution                   | `twinrouterbench.static-summary.v1`  | agentic; static prefix summary only, explicitly not a dynamic SWE sandbox result                                                                  |
| MMR-Bench                      | `83c8308427a3597213fdba298c098da887b8b01b`                                                         | dense multimodal-model outcomes and quality-normalized cost curves                                        | `mmrbench.merged-csv.v1`             | model-pool, multimodal; media bytes are hashed, native normalized cost is not reported as USD, and native AUC/capability-mask claims are excluded |
| AceBench                       | `9a17bc2c7ee3fab9ca023036b82a81898512a001`                                                         | executable edge/cloud agent tasks, task utility, cost, privacy boundary                                   | `acebench.run-summary.v1`            | agentic; terminal summary only, without a complete tool, egress, side-effect, or privacy-exposure ledger                                          |
| continuity-bench               | `5b7e7f82027c5b983435057ddc4d7115b7e9a97b`                                                         | deterministic failover protocol, context preservation and latency overhead                                | `continuitybench.labeled-failover.v3` | agentic diagnostic; labeled failure observations are not a real timeout, 429/5xx, retry-after, partial-stream, network, or provider fault and cannot decide G6 |
| FusionFactory / LLMFusionBench | `ef62645a48b9e2167201047da047854415e2bc89`                                                         | choose model subsets, collaboration topology, reasoning thoughts, and synthesis                           | `fusionfactory.aligned-csv.v1`       | model-pool; aligned base/reasoning outcomes only, without graph topology, synthesis, or hidden judge calls                                        |
| R2-Router / R2-Bench           | `b0b2291aeee08feb4bedbd199ab014ec60d0004f`                                                         | joint model and output-budget action over quality curves                                                  | `r2bench.model-budget-csv.v1`        | model-pool; fixed 15-budget safe long form, without loading predictors or claiming deployment-curve/capacity parity                               |

These limitations are part of the adapter contract. A normalized field is
qualified only when it is parsed into a typed record and consumed by a current
reducer. A paper metric listed in research metadata is not executable evidence.

### What the benchmark set teaches vLLM-SR

- RouterArena makes blind prediction files, hidden grading, perturbation pairs,
  price snapshots, and router overhead explicit.
- RouteJudge separates preference collection, exposure, assignment, and
  propensity from correctness and safety.
- CodeRouterBench shows that adaptive routing must freeze stream order,
  feedback timing, memory state, and no-future-leakage rules.
- LLMRouterBench and RouterEval show why pool size, pool construction, split,
  and seed must be factorial treatment axes rather than hidden constants.
- RouterBench requires best-single, no-information frontier, and oracle
  baselines before attributing value to query-aware routing.
- xRouteBench shows that session, personalization, modality, and hidden calls
  need explicit state and cost records; caption-first routing is not native
  multimodal execution.
- TwinRouterBench makes the decision point a trajectory prefix and demonstrates
  why static step labels and dynamic task completion must remain separate.
- MMR-Bench requires media lineage, modality capability masks, and
  modality-specific quality/cost slices.
- AceBench makes privacy, egress, tools, and side effects trajectory-level hard
  constraints.
- continuity-bench separates availability from continuity and motivates real
  fault injection, repeated seeds, and state-transfer evidence.
- FusionFactory expands the action from one arm to a subset/topology/synthesis
  graph and requires every hidden call in the ledger.
- R2-Router expands the action to model plus budget and requires observed
  output length and budget compliance, not only a configured cap.

## Installing pinned benchmark imports

External checkouts and native exports stay in ignored private directories. A
parser-verified import follows one reproducible path:

```bash
vllm-sr eval benchmarks
vllm-sr eval normalizers
vllm-sr eval verify-source \
  --adapter <adapter-id> \
  --source-root <ignored-source-root>
vllm-sr eval suite-normalize \
  --adapter <adapter-id> \
  --suite-id <suite-id> \
  --source-root <ignored-source-root> \
  --export-root <frozen-native-export> \
  --output <new-normalized-output>
vllm-sr eval suite-install \
  --request <new-normalized-output>/request.json \
  --bundle <new-normalized-output>/bundle \
  --source-root <ignored-source-root> \
  --export-root <frozen-native-export> \
  --suite-store <private-suite-store>
```

The installer verifies the exact clean code and optional dataset pins, reruns
the registered parser against the supplied export, requires the request and
artifact bytes to match, validates visible/grading separation and record
coverage, and seals the source, artifact-set, manifest, and replay-executor
digests. This proves parser determinism for the supplied bytes. It does **not**
prove that the upstream benchmark, dataset generator, or native scoring command
produced those bytes.

Every normalized import is therefore explicitly exploratory E0 evidence. A
user-provided normalized bundle is schema-validated but cannot claim parser
verification. Neither import origin can declare a qualification expectation or
self-publish a gate. When a registered parser proves a complete, unique
source/target perturbation corpus, `normalized-suite-live.v1` may execute those
exact cases through the server broker. Only complete successful receipts plus a
server attestation and independent pair/slice reduction can then publish
the registered `declared-shift.server-live.v1` evidence source at E4. That
narrow G4 statement concerns the
current Mixture on the exact pinned corpus; it does not claim that upstream code
ran or that an upstream leaderboard was reproduced.

## Evidence levels

| Level | Observation depth                                                                                                 |
| ----- | ----------------------------------------------------------------------------------------------------------------- |
| E0    | contract, identity, reference, deterministic digest, execution reachability, and diagnostic plumbing              |
| E1    | signal discrimination, missing/error behavior, latency, calibration where meaningful, and paired invariance       |
| E2    | projection purity, coverage, overlap, downstream correlation, boundary churn, and contribution completeness       |
| E3    | expected decision, default/priority/collision behavior, hard-policy decision, and Router-only latency             |
| E4    | feasible oracle, realized utility, regret, pool use, baselines, seed variance, robustness, and OOD                |
| E5    | final task or trajectory outcome, live reliability, privacy/safety, complete cost, capacity, and cost per success |

Live records cannot declare their own level. Each live executor owns an
immutable registry that binds an exact evidence source ID to allowed tracks, a
typed payload, broker-receipt cardinality, required attestations, and a maximum
level. Unknown sources, source/track mismatches, missing typed facts, missing or
mis-scoped receipts, incomplete sealed ledgers, and claims above either the
source or executor ceiling all resolve to E0. The built-in live registry admits
only these source contracts: routing diagnostic E3, model-pool outcome E4,
routed joint outcome E5, provider agent-task ledger E5, fault-recovery ledger
E5, hard-policy ledger E4, production experiment ledger E5, and closed-loop
capacity E5. The normalized-live executor separately admits exact multimodal
outcomes and declared-shift routing evidence at E4.

Evidence strength is not a score. E5 can fail and E0 can be structurally
perfect. A run with routing E3 and model-pool E4 reports both track levels and a
run level of E3.

## G0-G9 gate semantics

Gate disposition is `required`, `advisory`, or `not_applicable`.
Gate verdict is `pass`, `fail`, `unavailable`, or `not_applicable`.
`unavailable` means the declared method lacks the typed evidence required to
decide; it is never rendered as a pass.

`live-agent-task.v1` is deliberately not a gate method. A complete
`evaluation-agent-task-ledger.v1` must contain at least 20 distinct tasks with
at least two `evaluation-agent-task-attempt.v1` attempts per task and at least
one actually executed tool. It must bind every attempt, trajectory,
grader/privacy/tool receipt, exact snapshot and pricing evidence,
token-priced model cost, and the full Mixture snapshot. Reports publish attempt
success and its one-sided 95% lower bound, all-repetitions task reliability and
its bound, mean score and steps, invalid-tool rate, privacy exposures, total
cost, and cost per success. The method earns agentic E5 with
`benchmark_parity_claim=none`; G6 remains exclusively the injected-fault
continuity method below.

| Gate                         | Required decision evidence                                                                                                                                | Current strict decision path                                                                                                                                                                                                                                                                                                                       |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| G0 Reproducibility           | immutable manifest/snapshots, records, lineage, digests, failures                                                                                         | server validates and anchors the exact sealed bundle                                                                                                                                                                                                                                                                                               |
| G1 Static correctness        | strict schema, references, plan coverage, deterministic identities                                                                                        | server validates current contract and complete plan accounting                                                                                                                                                                                                                                                                                     |
| G2 Hard policy               | Router-owned policy/config/runtime snapshot proof plus complete typed dynamic decisions that exactly cover required `(rule, enforcement point)` pairs     | `live-hard-policy` reads only an explicitly configured `hard_policy_ledger`; the server validates the retained response and re-reduces proof/observation coverage, block accuracy, and violations. Without that production window the method is data-required, not implementation-missing |
| G3 Offline value             | server-controlled AB/BA baseline/candidate live pair, complete dense pool, routed outcomes, absolute candidate safeguards, and paired non-inferiority | single-run G3 is never a promotion claim. Campaign v2 requires at least 20 complete case clusters and jointly enforces candidate normalized-regret upper bound `<= 0.25`, paired regret-delta upper bound `<= 0.05`, routed lift over the best fixed-arm no-information frontier `>= 0.05`, joint reliability `>= 0.80`, all-arm failure `<= 0.20`, quality non-inferiority, worst-arm reliability, every shared arm's failure non-inferiority, and candidate-only arm reliability `>= 0.80` |
| G4 Declared-shift robustness | registered parser, exact pinned source and perturbation CAS, complete unique source/target pairs, broker receipts, live execution attestation, and server-owned pair/slice reduction | `declared-shift.server-live.v1` can publish routing E4 only from `normalized-suite-live.v1` over the exact frozen corpus. Import and replay stay E0; the claim is limited to the declared relations and slices and does not assert upstream runner parity, generic OOD, or contamination coverage |
| G5 Live fidelity             | unchanged candidate subject, qualified live reference, later fresh attested live run, exact case cohort, and complete failure accounting | `evaluation-campaign-fidelity.v2` reports the one-sided 95% exact-binomial lower bound on exact decision/outcome fidelity. The bound must be `>= 0.95`; fewer than 59 aligned cases are unavailable because even an all-match cohort cannot prove the threshold |
| G6 Live fault recovery       | real exact-step injected fault, paired baseline/treatment receipts, terminal and state outcomes, duplicate side effects, retry/latency limits, repeated seeds | `live-fault-recovery` consumes only a configured production fault ledger. The server requires the full sealed window, at least 20 independent pairs and 5 seeds, and a one-sided 95% recovery lower bound of at least `0.8`; Continuity labeled-failover exports remain E0 diagnostics, do not prove real injected faults, and leave G6 unavailable |
| G7 Cost / latency / capacity | frozen typed SLO, at least two measured load levels, at least three independent measurement clusters per level, tail latency, worst-cluster error budget, cross-cluster error stability, minimum throughput, adjacent-level scaling, saturation, and headroom | `capacity.slo-envelope.v1` computes each cluster error rate and one-sided 95% Wilson upper bound independently, takes the worst cluster at each load level, requires the error-rate range across clusters to be at most 5%, and lets the server re-attest every field before accepting `capacity.slo_headroom >= 0` |
| G8 Shadow / canary           | complete production assignment/exposure ledger, minimum cohort, policy-arm support, SRM, risk budget, stop rule, and rollback readiness                  | `live-production-experiment` validates the full sealed window with at least 20 assignments and a frozen risk budget no greater than `0.2`. G8 compares a one-sided 95% risk-rate upper bound with that budget; a triggered stop fails the candidate even when rollback succeeds. Controlled paired runs remain diagnostics |
| G9 Online preference         | complete assignment/exposure/outcome cross-binding, explicit target/reference policies, propensities, support, ESS, segments, SNIPS lift and 95% interval | the same production ledger becomes G9-eligible only with full outcomes, ESS at least `10`, effective-sample ratio at least `0.5`, and at least 5 observations per declared segment. The server estimates target and reference SNIPS on the common randomized window; the lift lower bound must meet a non-negative frozen minimum. Missing outcomes are unavailable; eligible regression fails |

For G7, the release envelope ends at the first measured load level at or above
the required concurrency. Higher levels remain visible as saturation evidence,
but an expected failure above the required envelope does not overturn an
already-qualified service objective.

G4's registered live publisher deliberately reuses only the exact pinned
perturbation corpus; it does not execute untrusted upstream code. Parser-verified
imports remain useful E0 diagnostics and cannot self-promote. G2, G6, G8, and G9
are live-only and consume server-brokered ledgers
from explicit target endpoint contracts; default runtimes do not advertise
those tracks. Capacity similarly advertises `capacity.slo-envelope.v1`, but
only the frozen per-run load protocol, SLO, records, and server reduction can
publish G7. Method registration alone never publishes a gate boolean.

## Run lifecycle

```text
catalog + change profile + suite/target/mode/tracks + budgets
                              |
                              v
                 immutable run manifest
                              |
                              v
          resolve one execution plan and executor
                              |
                              v
         visible cases -> execution -> hidden grading
                              |
                              v
 typed records -> metrics/statistics -> complete G0-G9 set
                              |
                              v
      sealed artifacts + server anchor + public report
```

Creation freezes the canonical run UUID, suites and revisions, the single
suite-executor cohort, target snapshot, change profile, selected tracks, seed,
sample/concurrency budgets, policy/binding/pool/workload/environment digests,
model arms, backend topology, code revision, redaction policy, and credentials
by environment reference. The server recomputes the manifest digest and
revalidates target executor admission when a pending run starts. Report sealing
then requires every plan, record, and attestation identity to match that frozen
cohort.

The fixed Python worker first writes `report.json` as the strict
`worker-report-draft` wire contract. That untrusted draft deliberately omits
the server attestation revision, per-track sealed evidence levels,
controlled-pair membership, method reductions, and routing Recipe reduction.
The Dashboard rejects those fields if a worker tries to claim them, derives
them from durable records and server-owned execution evidence, and atomically
replaces the draft with the sealed public report. Standalone CLI reports and
comparisons remain explicitly unsealed drafts (`WorkerReportDraft` and
`StandaloneComparison`); they are useful for local diagnostics but are not a
Dashboard publication attestation.

Visible cases are selected by deterministic seed only after target capability
eligibility. Hidden grading remains private and joins after execution. Failed,
timed-out, and unavailable plan cells remain in coverage denominators.

### Lifecycle governance

Every run and Campaign bundle is published atomically with private lifecycle metadata that
binds a server-derived owner principal, the active policy revision, a retention
class, an optional evidence hold, and the creating audit decision. Raw user
identities and email addresses are not stored in public resource or lifecycle
responses. Campaign creation also requires every bound run to have the same
owner unless an administrator performs the operation. Start, cancel, delete,
hold, release, and retention changes require the resource owner or an
administrator. Protected, held, baseline-referenced, and Campaign-referenced
evidence cannot be deleted or collected.

The local store enforces bounded per-owner run/Campaign/byte quotas, a total
physical byte quota, and an audit byte bound. `GET /api/evaluation/v1/lifecycle/usage`
returns deterministic chargeable, reserved, physical, and audit usage; a
non-administrator sees only its own pseudonymous owner entry. Collection is an
administrator-only two-step protocol: first `POST
/api/evaluation/v1/lifecycle/collection` with `{"apply":false}`, then repeat
with `{"apply":true,"plan_digest":"sha256:..."}`. Apply recomputes and binds
the exact eligible run, Campaign, status, policy, evidence-reference, and byte
snapshot; a stale plan fails without deleting anything. Expired Campaigns are
removed before their newly unpinned run evidence in the same ordered plan.

Lifecycle decisions enter an immutable, bounded, hash-chained active segment
covering create, start, cancel, hold, release, retention, delete, and garbage
collection, including denied authorization and protection decisions. At the
configured bound, the store atomically seals the segment head and sequence into
a cryptographic checkpoint, preserves creation bindings for every live run and
Campaign, and removes the compacted event bodies. Startup validates the
checkpoint anchor, any active suffix, and every live resource binding; global
policy, checkpoint, chain, or binding corruption fails closed. This local
contract intentionally does not provide a queryable archive of compacted
per-event bodies. The built-in local store therefore does not satisfy a
long-term event-level retention requirement; such deployments need a separately
designed external immutable archive sink before enabling this lifecycle store.
Run-local lifecycle corruption quarantines that run and blocks usage,
collection, or scientific decisions that require a complete ledger. The active
`evaluation-lifecycle-policy.v2` / `evaluation-lifecycle-policy.2026-09-01`
contract is fresh-store only: unpublished v1 intermediate state must be removed,
and there is intentionally no legacy migration path.

## Promotion Campaign

A run comparison remains diagnostic. `evaluation-campaign.v2` instead composes
one immutable promotion decision from the selected change profile's catalog
slots. There is no fixed role bundle and no browser-owned applicability matrix.

| Slot | Binding | Exact evidence contract |
| ---- | ------- | ----------------------- |
| G2 | one run | live safety E3 or stronger, conclusive server-owned hard-policy receipt |
| G3 | controlled pair | `live-runtime.v1`; complete routing E3, model-pool E4, joint E5; paired receipt is E5 |
| G4 | one run | `normalized-suite-live.v1`, routing E4, `declared-shift.server-live.v1` |
| G5 | fidelity pair | qualified live reference plus later fresh live run; `normalized-suite-live.v1` or `live-runtime.v1` |
| G6 | one run | `live-fault-recovery` agentic E5, conclusive fault-recovery receipt |
| G7 | one run | live capacity E5, conclusive frozen SLO envelope |
| G8 | one run | live preference E5 from a production assignment/exposure control window |
| G9 | one run | live preference E5 from a propensity-qualified production outcome window |

Each profile declares every G2-G9 slot as `required`, `advisory`, or
`not_applicable`. Required bindings cannot be omitted; `not_applicable` bindings
cannot be supplied. Every run ID is single-use across the Campaign. Each
evidence anchor binds the slot, gate, role, run, manifest, public report,
private receipt, optional execution attestation, and candidate-subject digest.
All candidate anchors must identify the same exact code/config, Recipe,
selector/adaptation/binding/pool and model/support-arm subject even though each
gate may use a different suite or workload.

G3 is causal only when its two runs were created through the server-owned
controlled-pair endpoint. The server resolves simultaneously addressable
deployment-scoped targets with the same logical Mixture ID and Recipe name,
requires distinct Router and Envoy origins, resolves private credentials, then
interleaves every shared
case/track/attempt/operation coordinate in deterministic AB/BA blocks. The
second request begins only after the first finishes. Pair receipts bind session,
protocol, coordinate/block digest, variant manifest, order, timing and load
coordinates. `evaluation-campaign-paired-live.v3` records baseline and candidate
target IDs separately, and each arm's report, provenance, and execution
attestation must bind its own exact target. Independent post-hoc runs, a shared
deployment target, or shared Router/Envoy origins are rejected.

The G3 reducer clusters by independent case and requires at least 20 complete
clusters with two-sided 95% intervals. It derives the no-information frontier
as the best fixed candidate arm over the complete dense case-by-arm matrix—not
as zero and not as a caller-provided score. Passing requires all frozen
boundaries: candidate normalized regret `<= 0.25`, paired regret delta `<=
0.05`, routed lift over that fixed-arm frontier `>= 0.05`, joint reliability
`>= 0.80`, all-arm failure `<= 0.20`, and per-track quality non-inferiority with
a `0.05` margin. The full-pool worst-arm statistic is recomputed inside each
case bootstrap. Shared arms must meet a `0.02` failure-risk margin and an
absolute `0.80` reliability floor; candidate-only arms must meet the same
absolute floor. Baseline-only arms are disclosed but cannot excuse candidate
risk. Missing cells, zero-quality or all-failed oracles, duplicate coordinates,
and inconclusive intervals cannot pass.

G5 separately binds an unchanged candidate to a qualified live reference and a
later fresh, attested live execution of the exact case cohort. Its public
observation is the one-sided 95% Clopper-Pearson lower bound on exact
decision/outcome fidelity, with threshold `0.95`. A cohort below 59 is
unavailable; 59/59 is the smallest all-success cohort that can prove the
threshold. Replay is not a G5 Campaign source. For the generic profiles, G5
binds joint E5 evidence. The `agent_multimodal` profile is the explicit
exception: G3 is `not_applicable`, and G5 binds multimodal E4 evidence from
`normalized-suite-live.v1`.

Gate ownership stays narrow: controlled-pair failure/latency observations are
diagnostic and never substitute for production G8; G8/G9 require genuine
assignment, exposure and outcome windows. Campaign publication and run
publication/deletion share one coordinator. Referenced or baseline-dependent
runs cannot be deleted, and restart validation rejects dangling anchors,
contract drift, digest drift, or corrupted private evidence.

## Statistics and report reading order

The statistics contract is claim-driven:

- case-aligned comparisons are paired; aggregate point deltas are descriptive;
- registered paired statistics use independent case-clustered analysis units,
  an explicit non-inferiority margin, at least 20 units, and a two-sided 95%
  interval; two repeated or identical observations cannot qualify a release;
- proportions carry sample count and intervals where supported;
- clustered sessions/trajectories are not treated as independent turns;
- missing cells remain visible and are never converted to zero-quality
  successes;
- cost, latency, quality, and safety retain separate axes;
- runtime cost, evaluation overhead, and capacity/TCO remain separate ledgers;
- deterministic ordering and canonical floating-point reduction keep report
  digests reproducible.

Read a report in this order:

1. run and per-track evidence levels;
2. required gate verdicts and the exact missing evidence rationale;
3. plan coverage, failures, and unavailable cells;
4. quality, safety, preference, and per-slice metrics with sample counts and
   intervals;
5. best single, pool oracle, realized value, regret, per-arm/worst-arm
   reliability, pool availability, and failure overlap;
6. latency, reliability, and all three cost ledgers;
7. benchmark, policy, binding, pool, environment, target, code, and grader
   lineage;
8. failure cases and architecture feedback.

E0 reports intentionally omit promotion headline metrics. A mixed run whose
weakest track is E0 can still show architecture findings from its stronger
sealed tracks, but it cannot publish a run-level promotion summary.

## From evaluation to Recipe and pool design

Evaluation is an architecture feedback loop, not a scoreboard:

| Observation                                                  | Likely cause                                                                                 | Next controlled treatment                                                                    |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| low routing accuracy or coverage                             | signal/projection/decision topology, candidate eligibility, selector calibration             | hold pool and environment fixed; inspect decision traces and slices; change one Recipe layer |
| high route accuracy but weak final quality                   | expected-decision label is a poor utility proxy, generation policy drift, or grader mismatch | keep Recipe fixed; rerun task outcome and grader calibration                                 |
| best single is close to pool oracle                          | little useful pool complementarity                                                           | remove quality-dominated arms or add a capability-gap arm; rerun the dense matrix            |
| high oracle, low oracle capture, large normalized regret     | valuable pool that the selector cannot exploit                                               | improve features/calibration/exploration while freezing pool and workload                    |
| one arm or worst-arm reliability regresses while all-arm failure stays low | a degraded member is masked by healthier arms; pool availability alone is insufficient       | inspect the per-arm paired interval, repair or remove the regressed arm, and rerun the same dense cohort |
| high pairwise failure Jaccard or high all-arm failure rate   | correlated failures or missing capability                                                    | diversify failure domains/capabilities; do not merely add a similar arm                      |
| Pareto-dominated arms                                        | operational cost without quality gain                                                        | validate hard capability need, then prune and rerun capacity                                 |
| agentic single-step quality holds but terminal success drops | switching, state continuity, tool validity, recovery, or delayed credit                      | evaluate decision points and full trajectories separately                                    |
| modality admission passes but quality fails                  | backend capability, media transport, grounding, or modality grader                           | isolate admission, routing, execution, and grading phases                                    |
| low propensity coverage or ESS                               | online logging/assignment support mismatch                                                   | fix exposure policy and support overlap before interpreting preference                       |
| safety false negatives or any hard violation                 | policy enforcement or blocker coverage                                                       | block promotion immediately; expand adversarial and slice coverage                           |
| throughput rises but tail latency/error crosses the SLO      | saturation, queueing, retry/fallback amplification                                           | tune serving placement/runtime; keep logical Recipe attribution separate                     |

### Recipe experiment design

Freeze `workload × policy instance × binding × pool × environment × budget ×
seed`. Evaluate structure and reachability first, then signal/projection quality,
decision behavior, fixed-pool algorithm value, and finally live end-to-end
outcomes. Change one treatment factor per comparison. Priority collisions,
default branches, language/domain/modality slices, missing signals,
out-of-distribution data, and expected-invariant/expected-change pairs all
belong in the workload contract.

### Model-pool experiment design

Build a dense case-by-arm core before judging a router. Report best single,
cheapest successful arm, per-case oracle, marginal contribution, unique wins,
quality dominance, quality-cost Pareto dominance, pairwise failure overlap,
capability coverage, and pool-size sweeps. A large research pool is not a
deployable pool until context, modality, tool, trust-domain, availability,
failure-domain, and capacity constraints are validated.

### Joint experiment design

Use a factorial design only when the policy/binding meaning remains identifiable
across pools. Always publish pool-independent routing behavior,
pool-normalized regret/oracle capture, and end-to-end quality/cost/reliability.
Never attribute an improvement to the selector when the pool, budget, grader,
price snapshot, or environment changed at the same time.

## Durable evidence and trust boundary

The Dashboard store is private and rooted at its configured Evaluation data
directory:

```text
evaluation-root/
  objects/sha256/<digest>
  attestations/<run-id>.json
  campaigns/<campaign-id>/
    campaign.json
    lifecycle.json
  suites/
    objects/{visible,grading,metadata}/sha256/<digest>
    manifests/sha256/<digest>
    index/<suite-id>.json
  runs/<run-id>/
    run-manifest.json
    status.json
    control-events.jsonl
    records.jsonl
    routing-traces.jsonl       # private request-level evidence
    metrics.json
    gates.json
    report.json
    lineage.json
    provenance.json
    failure-summary.json
    capacity-profile.json
    checksums.sha256
    private-checksums.sha256
    report-anchor.json
```

The exact per-run set varies only where the current contract marks an artifact
optional. Final evidence is immutable and content addressed. Mutable status
uses atomic replacement. Run creation publishes status, manifest, and initial
event as one directory boundary. Worker evidence import, canonical object
publication, execution attestation, report anchoring, deletion, and restart
recovery share one publication coordinator so no reader observes a partial
decision state.

The trust boundary includes:

- strict JSON decoding across evaluation contracts; worker and public reports,
  method declarations, and typed evidence inputs additionally reject duplicate
  object keys; IDs, record counts, file sizes, and event streams are bounded;
- private regular-file and directory checks with symlink rejection;
- a networkless worker sandbox that fails closed when Linux Landlock or seccomp
  cannot be installed;
- a Go-owned broker for allowlisted method/origin/path, credentials, body
  limits, redirects, model IDs, and inline media;
- an exact server transcript and live execution attestation bound to manifest,
  target, policy, pool, binding, topology, and timing;
- independent server reductions for owned metrics/costs/coverage and current
  typed gates;
- `evaluation-server-attestation.v2` on the report and server anchor;
- public artifact allowlisting, secret-pattern rejection, and
  `Cache-Control: private, no-store` on every authenticated response;
- durable numeric SSE IDs with `Last-Event-ID` replay and duplicate
  suppression;
- bounded ledger pagination, quarantine warnings, and decision blocking when
  the durable ledger is incomplete;
- referenced-run deletion protection and restart validation for Campaigns.

Worker drafts, records, grading labels, routing traces, private lineage, prompts,
outputs, target origins, credentials, and infrastructure identifiers are never
public artifacts.

## Dashboard experience contract

The Evaluation page contract defines five consistent workspaces:

- **Overview**: capability, catalog method readiness, latest sealed evidence,
  and incomplete-ledger warnings;
- **New experiment**: change profile, suite/target/mode/track/executor
  eligibility, baseline lock, seed/sample/concurrency budgets, and explicit
  create/start intent; selecting a suite from another accepted executor cohort
  explicitly replaces the prior cohort;
- **Runs**: paginated durable ledger, filters, status, progress, timeline,
  cancel/delete actions, and an inspector that distinguishes execution from
  observation state;
- **Reports**: decision boundary, track coverage, metrics, gates, costs,
  failures, provenance, architecture feedback, and safe artifacts;
- **Compare**: same-cohort run-pair diagnostics plus the Promotion Campaign
  builder and sealed Campaign decision.

Each control must expose loading, empty, permission-denied, error, retry,
disabled, keyboard, focus, and responsive states where applicable. Destructive
actions use the shared Dashboard dialog treatment and typed confirmation. URL
state supports direct links to views, runs, reports, comparisons, and Campaigns.
The contract forbids turning a catalog capability, successful HTTP request, or
missing metric into a positive readiness state.

## API and CLI

The Dashboard exposes only the current resources:

- `GET /api/evaluation/v1/catalog`
- `GET|POST /api/evaluation/v1/runs`
- `GET|DELETE /api/evaluation/v1/runs/{id}`
- `POST /api/evaluation/v1/runs/{id}/start|cancel`
- `GET /api/evaluation/v1/runs/{id}/events`
- `GET /api/evaluation/v1/runs/{id}/report`
- `GET /api/evaluation/v1/runs/{id}/artifacts/{artifact-id}`
- `GET|POST /api/evaluation/v1/runs/{id}/lifecycle`
- `GET /api/evaluation/v1/compare?baseline_run_id=...&candidate_run_id=...`
- `POST /api/evaluation/v1/controlled-pairs`
- `GET|DELETE /api/evaluation/v1/controlled-pairs/{id}`
- `POST /api/evaluation/v1/controlled-pairs/{id}/cancel`
- `GET /api/evaluation/v1/lifecycle/usage`
- `POST /api/evaluation/v1/lifecycle/collection`
- `POST /api/evaluation/v1/campaign-readiness`
- `POST /api/evaluation/v1/campaigns`
- `GET|DELETE /api/evaluation/v1/campaigns/{id}`
- `GET /api/evaluation/v1/campaigns/{id}/decision`
- `GET|POST /api/evaluation/v1/campaigns/{id}/lifecycle`

The CLI surface under `vllm-sr eval` provides catalog, benchmark/normalizer
inventory, source verification, suite normalization/install/list/show,
manifest validation, execution, local worker-draft inspection, comparison, and
gate checks.
`vllm-sr eval --help` is the exact command reference for the installed build.

## Scale and extension admission

The current orchestrator remains small by admitting extensions through narrow
versioned registries:

- benchmark normalizers declare a closed native-export schema, required
  artifacts, exact parser, metric mappings, limitations, source pins, and
  parity tests;
- executors declare supported modes/tracks and consume one typed
  `EvaluationInputs` boundary;
- target providers declare server-owned origins, credentials by reference,
  Entrypoints, per-mode accepted executors, direct-arm execution/correlation,
  model/runtime revision, and evidence ceiling;
- load providers declare arrival process, warm-up, levels, duration,
  repetitions, resource observations, SLOs, saturation, and headroom;
- online providers declare assignment, exposure, propensity, support overlap,
  risk budget, stop, and rollback ledgers;
- reducers declare typed input records, exact metric/gate ownership, and
  cross-language golden tests.

An extension enters planning only after its immutable identity, capability
contract, trusted derivation, bounded inputs/outputs, evidence ceiling, and
tests are registered. An unsupported capability is rejected during planning;
an admitted run or Campaign uses `unavailable` only for an executed cell with
no observation or a gate whose valid report lacks its required typed proof. The
core does not infer observations or gate verdicts.

### Canonical catalog resources

The Python package owns the canonical metric-analysis catalog and research
benchmark inventory under `src/vllm-sr/cli/evaluation/golden/`. The Go service
and browser keep generated byte-identical mirrors so they can validate the
same contracts without loading Python. Do not edit those mirrors directly.
After changing either canonical JSON file, synchronize both mirror families:

```bash
python3 tools/ci/sync_evaluation_catalogs.py
```

`make dashboard-check` runs the corresponding `--check` mode and rejects a
missing or stale mirror.

## Promotion rule

An Evaluation report recommends; it does not deploy. Promotion requires the
intended per-track evidence levels, all required gates passing, complete
coverage, an exact baseline, unchanged comparison factors, reviewed checksums
and lineage, and an explicit rollout/rollback owner. Capacity and production
claims require a pinned live environment. Online claims additionally require
production assignment/exposure evidence. E0 fixtures, replays, and live
diagnostic sources that do not earn a server-owned evidence level are useful
for regression and diagnosis, but never promotion evidence.
