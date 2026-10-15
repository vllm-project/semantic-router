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

The PR, main, nightly, and release entrypoints share one verification plan:

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
  artifacts. The branch-protection check remains **PR Gate**.

Display categories, execution workers, and verification identities are separate.
Catalog IDs identify results even when a display name changes. Job names describe
the execution boundary; model names and feature cases appear in its report.
Compatible component checks use one of three setup classes: **CLI**,
**Model Tools**, and **Router Tools**. Each selected contract runs in its own
worker with its own test inventory, logs, and result. A failed contract leaves
other independent checks running, but fails its worker and the Gate.

Platform contracts run in separate workers matched to their runtime, device,
target platform, and execution mode. Each keeps its own prepared models and
required receipt. Kubernetes profiles also run in separate workers with their
own images, clusters, state, and evidence. A profile failure does not suppress
independent profiles. Adding a model or feature does not automatically add a
job.

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

## When checks run

| Entry | Qualification |
| --- | --- |
| PR | Required quality checks and the contracts affected by the change. |
| PR with `ci/full` | The versioned full CPU product inventory. |
| Main | Affected contracts at the actual merge commit; only changed development artifacts are published. |
| Nightly | Full CPU qualification, including baseline stress cases. |
| Release | Full CPU qualification, compatible release performance comparison, and final package/version checks. |

A test-file change selects the boundary that test exercises. It is not
inherently a unit-only change. Shared configuration, defaults, or inference
changes can therefore select model integration and live recipes. Documentation
changes do not start unrelated model downloads or runtime stacks. Image assets
named by the calibration manifest are executable test inputs: changing them
selects the image-calibration contract even under a documentation directory.

For a manual qualification, dispatch **CI** (`ci.yml`),
select the branch or tag, and enter a catalog verification ID such as
`platform.image-calibration-cpu` or `platform.models-cpu`. It uses the
same plan, compatible build dependencies, required receipt, and Gate. Normal PR checks
calibrate the plan's merge-tree source SHA; reports must name that exact clean
commit. Full CPU, nightly, and release qualification also include this contract.

The full CPU inventory is declared in
[`verification_catalog.yaml`](../../ci/verification_catalog.yaml) and mapped to
the public deployment support matrix. A required contract that is missing or
unexecuted blocks qualification. Manual and experimental environments are
listed separately. A previous run on another commit cannot fill a missing
result in the current run.

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

Kubernetes baseline inventory comes from the test registry. Regular runs select
all 36 non-stress cases; full runs select all 38 cases, including the two stress
cases. Operator integration sends a real routed request after reconciliation;
resource readiness alone is insufficient.

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
and reuses that read-only artifact across suites.
Containers, databases, networks, and volumes are not shared between suites.
Image publication promotes the qualified OCI archive. CLI publication uploads
the wheel and source distribution already checked from an isolated installation.
Release metadata is applied before qualification.

CPU success does not qualify CUDA or ROCm. Accelerator image builds are reported
as builds. Device qualification requires actual inference on the declared
hardware, with its own supported verification entry and evidence.

## Diagnose a failure

Start with the plan and the failing verification's receipt, then its original
framework log. Check the source and artifact identities before comparing runs.
A missing model, database, or report is an incomplete verification, even if the
remaining cases passed. Correct the dependency or inventory problem and rerun
the affected verification; do not remove a required case to make the Gate green.

For Preview failures, inspect the per-probe signals, decisions, and latency in
the conformance report. The built-in MoM inventory includes all five entrypoints
and 315 probes. Performance CI reports numerical regressions as warnings;
benchmark execution, complete inventory, and matching model identities remain
required. The explicit local `make perf-check` still fails on allocation
regressions. Go allocation metrics do not measure native memory or GPU memory;
see [`perf/README.md`](../../../perf/README.md).

## Add a verification or platform

1. Update changed-path ownership in
   [`domains.yaml`](../domains.yaml). This is the only changed-path registry.
2. Add or extend a verification in
   [`verification_catalog.yaml`](../../ci/verification_catalog.yaml). Declare
   its product boundary, logical category, executor, inventory, runtime/device,
   services, and build dependencies. Kubernetes profiles also declare runtime,
   device, and resource class in the domain registry. Compatible cases join
   an existing worker; keep concrete case discovery in its testing framework.
3. Extend the executor only when the new check has different execution needs.
   Emit actual case outcomes and consumed artifact identities. Add negative
   coverage proving that missing cases or dependencies cannot pass.
4. If it is a required public CPU contract, include it in the versioned full
   inventory and support mapping. Update the documentation with its command.
5. Run `make harness-check` and the affected local verification. PR, main,
   nightly, and release then use the same plan without four separate selectors.

Paper compilation and fixed-data learning reports are not product gates.
Their local tools remain available; changes to learning report code still run
its parser and report tests. Production session-aware routing and selector
parity remain required core contracts.
