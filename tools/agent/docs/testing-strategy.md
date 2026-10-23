# Testing strategy

Use `make check` for changed-source checks and `make verify` for the product
boundary you changed. CI uses the same ownership registry to select the required
verification inventory.

```bash
make impact CHANGED_FILES="path/one path/two"
make check CHANGED_FILES="path/one path/two"
make verify DOMAIN=<domain>
make verify PROFILE=<profile>
make ci-full
```

`make check` runs the matching domains' local checks. `make ci-full` runs the
local pre-commit and core baseline; published-model inference and deployment
suites still require their explicit commands below. An ordinary core pass does
not qualify every model or deployment.

An explicit `BASE_REF` or `--base-ref` must resolve to a commit. An invalid value
fails the check rather than silently selecting a different comparison.

Local checks and CI include both paths of a rename, so moving a file cannot drop
checks for its former domain.

## Read a CI run

The PR, main, weekly, and explicit qualification entrypoints share one verification plan:

- **Plan** selects contracts from the changed paths, or the full CPU inventory.
  It records the source revision, selection reasons, required cases, and build
  dependencies in the `ci-plan` artifact.
- **Quality** checks source formatting, static rules, trusted security checks,
  and generated contracts. Every static check can start immediately.
- **Artifacts** builds selected container images, or acquires an already
  qualified provider-mocker image by immutable digest.
  Artifact production and verification are separate responsibilities.
- **Tests** executes the selected unit, integration, and end-to-end contracts.
  Contracts are grouped as Components, Runtime, Conformance, Integration,
  E2E, Packages, and Performance. Models and features are cases within these
  contracts. A build blocks only its consumers; execution workers are selected
  by environment, resource requirements, and state isolation.
- **Gate** compares the planned inventory with the actual results and consumed
  artifacts. A trusted status publisher exposes the branch-protection check
  **PR Gate**, including an explicit waiting state for committer authorization.

Display categories, execution workers, and verification identities are separate.
Catalog IDs identify results even when a display name changes. Job names describe
the execution boundary; model names and feature cases appear in its report.
Component checks default to one contract per job for independent dependencies,
parallel feedback and targeted retries. The catalog offers an explicit
`shared-components` policy (at most four compatible contracts) when measured
setup cost justifies sharing. Every contract retains its inventory, logs,
receipt and failure status. The model runtime declares its own setup
environment. A failed contract does not suppress independent siblings. Each
contract has a cumulative timeout; child processes are terminated before the
next contract starts.

Platform and Kubernetes contracts share workers only when their execution
target, runtime, device, platform, setup environment, batch policy, image
dependencies and resource class are compatible. The `integration` policy runs
at most three contracts within its declared time budget; `exclusive` keeps a
contract alone. Vela Omni explicitly selects exclusive execution. Batch limits
and setup allowances live in the catalog, and total job deadlines cannot exceed
360 minutes. Each Kubernetes profile
creates and destroys its own cluster. Teardown failure stops the worker, since
its remaining contracts could otherwise run against contaminated state.
Heavy model profiles remain isolated. Recipes use three balanced standalone
shards and a separate built-in MoM shard; all authored sources remain required.

Local deployment contracts appear under **Local Stack**. The existing serving,
routing, and lifecycle tests and the stateful memory tests remain separate
execution shards: they deliberately stop the stack or its database. Their common
deployment entrypoint is `vllm-sr serve`; memory is a capability exercised through
that entrypoint, not a separate deployment environment.

Unit tests exercise module logic. Integration tests exercise component
boundaries. End-to-end tests enter through a user-facing path and check its
result. Preview tests validate signal and decision behavior; routed-request
E2E also proves the request reached the expected backend. A browser test using
a controlled API validates UI behavior, not a live Router deployment.

Features such as memory and session-aware routing can have checks at several
levels. Recipes and model checkpoints are test inputs. The runner platform,
actual device, inference runtime, service dependencies, and artifact identities
are separate fields. For example, the model runtime on CPU is an inference
combination; Kind supplies a cluster, and `vllm-sr serve` starts a local stack.

Selected checks must finish successfully. Missing results, skipped required
cases, duplicate IDs, missing shards, a different source revision, or a wrong
artifact or device fail the Gate. Cases requiring optional checkpoints or
external services are declared before execution; they cannot become a pass by
calling `Skip`. The core profile's explicit exclusions are documented in
[`core_test_profiles.json`](../../ci/core_test_profiles.json). Raw framework
reports remain available alongside the common receipts.

## Product layers and execution policy

The catalog declares both `layer` (what a test proves) and `execution` (when its
hosted worker may run). These are separate from display categories and framework
case IDs. All configured workers are GitHub-hosted; no self-hosted runner or GPU
availability is assumed.

| Layer | Boundary and value | Default execution |
| --- | --- | --- |
| Source | Formatting, generated contracts, trusted static security, CI harness | Automatic affected checks |
| Router | Deterministic routing logic, parser/selector parity, protocol/tool units | Automatic affected checks |
| Runtime | Real model answers, image calibration, training/artifact tools, databases | Affected qualification after committer authorization |
| Product | CLI/package/dashboard units; live serve/stop, memory, recipe behavior | Units automatic; live stacks and recipes approved |
| Integrations | E2E framework units; routed requests through Envoy AI Gateway, standalone and one Operator deployment | Units automatic; live deployments approved |
| Performance | Paired benchmark execution and comparable model identity | Approved; numerical trends are advisory on hosted CPUs |

A new external platform profile is a separate maintenance commitment. Router
behavior shared by gateways belongs in the existing ExtProc contract; a new
platform profile must justify its distinct assertion and public support claim.
Storage backend semantics belong in their storage contracts, while Operator
qualification owns configuration conversion, reconciliation and a routed request.

A layer is not a dependency rule: a configuration change can require checks in
several layers. `domains.yaml` remains the only changed-path selector. Tests do
not become optional merely because their filenames contain `test`.

| Entry | Qualification |
| --- | --- |
| PR | Automatic affected checks; required expensive checks wait for `/ci run`. PR Gate stays pending with **Waiting for committer approval** while authorization is the only missing requirement. |
| PR with `/ci run full` | The complete versioned CPU inventory, including baseline stress cases. |
| Main push | Automatic affected source, deterministic router and product checks. No image or CLI publication. Reviewed PRs and the merge queue own affected live qualification. |
| Main dispatch with `publish=true` | Full CPU qualification and a complete commit-addressed development image/CLI set; the CLI publishes only after its images succeed. |
| Weekly Qualification | Monday full CPU qualification and nightly-channel image/Helm publication. The historical `nightly-build.yml` filename and channel remain compatible. |
| Release tag | Artifact/version pipeline: build and publish images, Helm, and Python with aligned versions. It does not invoke another full CPU suite; the catalog retains a release qualification profile for explicit use. |

The trusted publisher updates one bot comment per PR with the deferred
contracts, image work, and authorization command. It publishes **PR Gate** as
pending while required work awaits authorization, failure for actual test or
evidence failures, and success only for complete passing evidence. Waiting
does not reserve a runner. A genuine failure takes precedence when other work
also awaits authorization. Contributors without repository write permission
do not need to authorize CI and cannot approve these runs.
The short **CI Execution** job still summarizes the work that ran; its success
alone does not satisfy the separately published PR Gate.

A committer with current repository write, maintain, or admin permission can
comment `/ci run` on a PR to authorize all affected work. `/ci run models`,
`storage`, `product`, `e2e`, `operator`, or `performance` authorizes only that
scope; multiple scopes may be listed, and valid approvals for the same revision
accumulate. A partial approval does not waive the other required contracts or
image builds. `/ci run full` adds the full CPU inventory.

Command feedback uses 👀 for receipt, 🚀 after the rerun API succeeds, and 😕
with an actionable bot reply for rejected requests or a trigger that could not
be confirmed. Feedback starts when the command-handler job gets a runner. A
rocket confirms only that CI was triggered; **PR Gate** owns the test outcome.
Retries update the same request's bot reply instead of adding duplicate errors.

The command handler runs trusted main code and freezes the PR head, base, merge
revision, and original PR workflow run. It rechecks the comment and permissions
before rerunning that original event. Edits, withdrawn permissions, and changed
revisions invalidate approval. This preserves the required PR check context.
A standalone workflow dispatch cannot replace the PR Gate. Verified Mergify
queue PRs automatically qualify affected changes at the queue merge revision;
a human-created branch with the same prefix gets no exemption.

The publisher is installed on main. During its introduction, branches based on
the old entrypoint retain the strict legacy PR Gate; missing qualification
still fails that legacy check. Update such a branch after the publisher lands
to receive the pending state and bot guidance. Bootstrap does not waive
required evidence or authorize expensive work.

Before the command handler first lands on main, a committer can use the explicit
bootstrap form `/ci run full <head-sha> <base-sha> <merge-sha>` (or `affected`)
and rerun that original PR workflow. The three full SHAs, unedited comment,
current permission, and run creation time are verified; the short command does
not use this path.

For an isolated investigation, dispatch **CI** (`ci.yml`) with a catalog ID such
as `platform.models-cpu`. This produces the same evidence but does not satisfy
a different PR run. Documentation changes avoid unrelated model downloads;
calibration image fixtures remain executable inputs even under documentation
paths. Missing or skipped results from an authorized required check, a wrong
source revision, or stale artifacts fail qualification. Required work that
has not been authorized keeps PR Gate pending and cannot produce success.

Scheduled runs select this explicit inventory; an absent comparison base does
not mean every optional profile changed. Real PR changes still add affected
profiles beyond the full baseline.

The fixed CPU product inventory contains 31 contracts. The original restructuring
expanded it to 40 physical PR jobs, including planning, images, and result gates;
a representative core change selected six automatic jobs and nine after
affected approval. These historical counts precede the separate status
publisher; its control jobs are accounted for in the
[cost audit](ci-job-cost-audit.md#approval-status-publisher-follow-up).
Full qualification remains exceptional. Five authoring contracts (`training`, `onnx-artifacts`,
`bench-tools`, `learning-tools`, `soak-tools`) remain available on affected paths
or explicit manual selection. They do not run unconditionally in full product
qualification. Shared component runner, bootstrap, planner, and catalog changes
select all ten component contracts; each still has an independent job. Real-model/runtime qualification remains
in full. Envoy AI Gateway is the maintained external ExtProc integration;
standalone qualifies the Router's own listener. Agentgateway, AIBrix, Istio,
llm-d and Dynamo retain experimental deployment examples but no runnable E2E
profile or manual CI target. Shared signal and corpus-loader changes still
select their retained integration consumers. Counts vary with the actual diff;
the [cost audit](ci-job-cost-audit.md) records the expanded plan and does not
equate job counts with runner-minute savings.

The full CPU inventory lives in
[`verification_catalog.yaml`](../../ci/verification_catalog.yaml). See the
[workflow and job audit](ci-workflow-audit.md) for the evidence behind consolidation,
retirement, and retained coverage. The [testcase audit](ci-testcase-audit.md)
traces core assertions, replacement coverage, and remaining weak test oracles.

## Reproduce the affected boundary

| Contract | Local command | Required evidence |
| --- | --- | --- |
| Core | `make test-and-build-local` | Router logic, registered Go tools, selector parity, and local core checks. |
| Go tools | `make go-tools-test` | CLI, classifier operating-point, fusion evaluation, image calibration, and model-selection tests. |
| Dashboard | `make dashboard-check`; `make dashboard-test-wasm`; `make dashboard-test-e2e-evaluation` | Frontend and backend tests, compiled WASM behavior, and browser acceptance. |
| Model runtime | `make model-runtime-test` | Families, engines, scheduling, the HTTP contract, and golden answers on CPU. |
| Published models | `make test-models` | Classifier, cache, and Omni real-model tests serving the pinned Vela 1.0 packages through a managed model runtime, and the pinned Vela 2.0 0.3B and 0.8B served on CPU answering the Router's questions in one decisions call per stage, within tolerance of their recorded answers, without a skip. |
| Image-routing conformance | `make verify-image-routing-calibration` | Prepared Nano identity, every authored scored image, prototype provenance, frozen threshold/validation assertions, and multimodal profile package tests. |
| Local Stack serving contracts | `make vllm-sr-test-integration` | Live `serve`/`stop`, mounts, environment, pull policy, request behavior, and service isolation. |
| Local Stack memory contracts | `USE_DETERMINISTIC_MEMORY_EMBEDDINGS=0 make memory-test-integration` | Vela embedding, persistent retrieval, injection, user isolation, and persistence failure behavior. |
| Kubernetes | `make verify PROFILE=envoy-ai-gateway` | Deployment and routed requests through the selected profile. |
| E2E framework units | `make test-e2e-unit` | Collected Go helper and profile assertions, without starting a cluster. |
| Recipes | `make recipe-conformance-static`; `make recipe-conformance-live-cpu-all` | Authored contracts and live CPU Preview probes for all maintained sources. |
| Performance | `make perf-check PERF_BASE_REF=<base-commit>` | Complete benchmark inventory and a paired comparison using identical model artifacts. |

Use the prerequisites in the corresponding reusable workflow when reproducing
its job. Model contracts need the model runtime (`make model-runtime-install`).
Storage integration requires its selected services. In required CI, missing
dependencies fail the suite.

Kubernetes baseline inventory comes from `e2e/pkg/testmatrix/testcases.go`.
Regular runs exclude the two declared stress cases; full runs include them.
Case counts are discovered from the registry, rather than duplicated here.
Operator integration uses one memory-backed deployment and sends a real routed
request after reconciliation. Other storage configurations are protected by
source conversion tests, while storage contracts own data semantics. Resource
readiness alone is insufficient.

Standalone has seven Kubernetes cases covering chart defaults, actual provider
requests, model inventory, backend fallback, Router-origin errors, decision-model
propagation and configuration rollout. The CLI also starts standalone explicitly
and proves no Envoy container is present. These cases do not currently qualify
standalone SSE; the streaming profile exercises the maintained Envoy ExtProc path.

E2E framework units run once in the component executor, independently of the
selected Kind profiles. Their Go inventory and execution report must agree;
missing or skipped tests fail. The soak client and multimodal profile package
retain their dedicated `soak-tools` and image-calibration owners.

Core discovers external Go tools from
[`go-tools.mk`](../../make/go-tools.mk), including sources outside the Router Go
module.

Dashboard runs both Go tests in the JavaScript/WASM runtime and Node
assertions against the compiled compiler. Their discovered cases and actual
results must agree, just like the other required suites.

## Models and artifacts

The model runtime serves every published model: Domain, Guard, PII, FactCheck,
Feedback, Modality, Safety, Hazard, Halu grounding, Embedding, Reranker, Omni
Nano/Mini, and the decision models. Its golden answers pin each family's outputs,
and the Published Models contract and the E2E profiles keep the deployed semantic
assertions, including spans, input budgets, modalities, and dimensions; a common
runner does not reduce them to health checks. The runtime resolves immutable Hub
revisions from its registry and verifies every loaded file. The runner requires
every selected case and rejects skipped or missing results.

The runtime verifies the pinned Omni releases file by file. CI downloads each
once and shares it between compatible runtime and conformance contracts. Image
conformance uses the pinned Nano release and retains raw scores, threshold
checks, prototype protocol and validation results, and fixture provenance.
It does not replace deployed image extraction and routing tests. Reference-model
parity and maximum-context qualifications that require explicit inputs remain
separately declared; a shorter input acceptance test cannot qualify full context.

`make test-models` provisions its own prerequisites, and benchmarks let the
runtime download each pinned model on first start. The Vela 2.0 tests compare
the Router's fused decisions calls with answers recorded on CPU
(`src/semantic-router/pkg/classification/testdata/vela2_published_answers.json`);
after an intended change of the questions, their fusion or a pin,
`make record-vela2-answers` records them again. Product startup instead
provisions the models the active configuration references: the model runtime
downloads each model it serves, and the Router provisions label maps and the Omni
bundles. That is not the full test inventory. A pre-existing local model
cache does not activate optional inference during ordinary core tests.

A build identity includes its source revision, platform, and build inputs.
CI builds each selected image once, checks its content digest when loading it,
and reuses that read-only artifact across suites. If the exact provider fixture
has not yet been published by an explicit Main publication, PR and weekly runs
build and qualify their own candidate; registry access failures still fail.
Containers, databases, networks, and volumes are not shared between suites.
Image publication promotes the qualified OCI archive. CLI publication uploads
the wheel and source distribution already checked from an isolated installation.
Release metadata is applied before qualification.

CPU success does not qualify CUDA or ROCm. Accelerator image builds are reported
as builds. Device qualification requires actual inference on the declared
hardware, with its own supported verification entry and evidence.

## Execution targets and hardware extensions

Contracts, placement and batching have separate owners:

| Concern | Owner | Meaning |
| --- | --- | --- |
| Verification contract | [verification catalog](../../ci/verification_catalog.yaml) and the test framework | Required cases, runtime/device contract, inputs and acceptance |
| Execution target | [target registry](../../ci/execution_targets.yaml) and its [adapter](../../ci/execution_targets.py) | Reviewed runner selection, host platform and supported executors/runtime/device combinations |
| Batch policy | Catalog `batching` policies and each contract's `batch_policy` | Independent job or compatible shared job, count/time bounds and setup allowance |

Only `linux-amd64-cpu` is enabled today. It selects GitHub-hosted
`ubuntu-24.04` through the `cpu-v1` adapter. Source, component and heavy reusable
workflows read resolved placement from their planned contracts. The target ID,
registry digest, adapter, runner and platform enter the contract digest and
receipt; the Gate checks them again. Workflows bind the observed runner
environment from `runner.environment`, and receipts compare the actual host
platform. Local harness checks are allowed but recorded as local; local
receipts cannot qualify a GitHub Actions run.

The CPU adapter establishes host placement. It does **not** attest a model's
accelerator by copying a planned `device` string into the result. Existing
published-model launchers explicitly select CPU, and CPU Kubernetes profiles
use their declared fixtures. Some E2E receipt metadata still describes the
authored CPU configuration rather than independently observing every loaded
model. Consequently, the adapter rejects GPU device declarations; moving a job
to a GPU-labeled runner does not create GPU qualification.

A future self-hosted CPU target can reuse this adapter once its Linux platform,
dependencies, isolation and runner access are established. Register the reviewed
runner placement and update the relevant contract references; assertion code
and receipt identity need not be redesigned. Shared self-hosted machines must
not inherit disposable-runner cleanup: the disk-reclamation action runs only
on GitHub-hosted runners. No self-hosted target is enabled by this change.

A CUDA or ROCm target requires a concrete adapter before enabling its registry
entry. That adapter must observe the actual accelerator and device architecture,
runtime/engine version, model revision or model digest, and inference output;
the model-runtime model cards and response metadata already expose useful
identity fields. It must own a nonempty required inventory for that hardware and
fail on missing or skipped required cases. Existing GPU tests have different
requirements: ROCm fast-path tests require `gfx942`, whereas CUDA graph tests
require CUDA. `make model-runtime-test-gpu` alone can skip unavailable hardware
and is not a qualification adapter. CPU fixture tests exclude both `gpu` and
`prod_weights` and remain automatic source validation.

Use distinct verification IDs when qualifying the same semantic contract on
multiple targets so one CPU receipt cannot satisfy a GPU requirement. Keep
disabled or unsupported targets out of required CPU inventory; unknown targets,
unsupported adapters and mismatched placement fail planning or receipt checks.

## Diagnose a failure

Start with the plan and the failing verification's receipt, then its original
framework log. Check the source and artifact identities before comparing runs.
A missing model, database, or report is an incomplete verification, even if the
remaining cases passed. Correct the dependency or inventory problem and rerun
the affected verification; do not remove a required case to make the Gate green.

For Preview failures, inspect the per-probe signals, decisions, and latency in
the conformance report. The built-in MoM inventory includes all five entrypoints
and 347 probes. Performance CI reports numerical regressions as warnings;
benchmark execution, complete inventory, and matching model identities remain
required. The explicit local `make perf-check` still fails on allocation
regressions. Go allocation metrics do not measure native memory or GPU memory;
see [`perf/README.md`](../../../perf/README.md).

## Add a verification or platform

1. Update changed-path ownership in
   [`domains.yaml`](../domains.yaml). This is the only changed-path registry.
2. Add or extend a verification in
   [`verification_catalog.yaml`](../../ci/verification_catalog.yaml). Declare
   its product layer, automatic/approved policy, logical category, executor,
   inventory, runtime/device, execution target, services and build dependencies.
   Choose a batch policy independently; the default component policy is one
   contract per job. Kubernetes profiles can declare execution target, runtime,
   device, setup environment and batch policy in the domain registry. Keep
   concrete case discovery in its testing framework.
3. Extend the executor only when the new check has different execution needs.
   Emit actual case outcomes and consumed artifact identities. Add negative
   coverage proving that missing cases or dependencies cannot pass.
4. If it is a required public CPU contract, include it in the versioned full
   inventory and support mapping. Update the documentation with its command.
5. Run `make harness-check` and the affected local verification. PR, main,
   weekly, and release then use the same plan without four separate selectors.

Paper compilation and fixed-data learning reports are not product gates.
Their local tools remain available; changes to learning report code still run
its parser and report tests. Production session-aware routing and selector
parity remain required core contracts.
