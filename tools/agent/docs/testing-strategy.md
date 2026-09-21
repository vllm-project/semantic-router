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

A failed Clippy invocation fails the changed-file check even when its diagnostics
point outside the edited files.

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
  and generated contracts. Checks that need native libraries wait for that
  artifact; other static checks can start immediately.
- **Artifacts** builds selected container images and native libraries, or
  acquires an already qualified provider-mocker image by immutable digest.
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
Compatible component checks share three workers: **CLI and Fleet**, **Model
Tools**, and **Router Tools**. Each selected contract keeps its own test
inventory, logs, and result. A failed contract leaves other independent checks
running, but fails its worker and the Gate.

Compatible runtime contracts share a worker for their runtime, device, target
platform, and execution mode. ORT model regression and image conformance can
reuse the same prepared model while retaining separate required receipts.
Kubernetes profiles run in bounded shards with compatible artifact and resource
requirements. Images are loaded once per worker, but profiles have separate
clusters, state, and evidence. A profile failure does not suppress independent
profiles; a cleanup failure fails the remaining work instead of running it in
contaminated state. Adding a model or feature does not automatically add a job.

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
are separate fields. For example, OpenVINO/CPU is an inference combination;
Kind supplies a cluster, and `vllm-sr serve` starts a local stack.

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
`native.image-calibration-cpu` or `native.ort-cpu`. It uses the
same plan, compatible build dependencies, required receipt, and Gate. Normal PR checks
calibrate the plan's merge-tree source SHA; reports must name that exact clean
commit. Full CPU, nightly, and release qualification also include this contract.

The full CPU inventory is declared in
[`verification_catalog.yaml`](../../ci/verification_catalog.yaml) and mapped to
the public deployment support matrix. A required contract that is missing or
unexecuted blocks qualification. Manual and experimental environments are
listed separately. A previous run on another commit cannot fill a missing
result in the current run.

The native matrix also runs a separate **RISC-V QEMU** contract: Candle classifier
parity, owned binding fixtures, and router diagnostics on `linux/riscv64` binaries
emulated on a Linux AMD64 runner. Its receipt names both the target and the host.
This proves the emulated instruction-set path; it does not qualify physical
RISC-V hardware or its performance and never loads the shared AMD64 libraries.

## Reproduce the affected boundary

| Contract | Local command | Required evidence |
| --- | --- | --- |
| Core | `make test-and-build-local` | Router logic, registered Go tools, native interfaces, selector parity, and local core checks. |
| Go tools | `make go-tools-test` | CLI, classifier operating-point, fusion evaluation, image calibration, and offline model-compatibility tests. |
| Dashboard | `make dashboard-check`; `make dashboard-test-wasm`; `make dashboard-test-e2e-evaluation` | Frontend and backend tests, compiled WASM behavior, and browser acceptance. |
| Native fixtures | `CI=true make test-owned-native` | Real libraries with small tensors: ownership, isolation, cleanup, and assembly. |
| Published Candle models | `make test-models MODEL_TEST_PROVIDER=candle` | Ten Vela families, real Halu grounding contracts, and legacy multimodal binding compatibility cases. |
| Published ORT models | `make test-models MODEL_TEST_PROVIDER=ort` | Ten Vela families, prepared Omni Nano/Mini contracts, classifier integration, and implicit/explicit execution defaults. |
| RISC-V QEMU | `make test-riscv-qemu` | Host/target classifier parity, owned fixtures, target ELF identity, and live router diagnostics under emulation. |
| OpenVINO runtime | `make verify-openvino-binding` | Owned-handle lifetime and token-budget checks under the race detector, plus pinned Vela Domain and Embedding tokenization and CPU inference. |
| Image-routing conformance | `make verify-image-routing-calibration` | Prepared Nano identity, every authored scored image, prototype provenance, frozen threshold/validation assertions, and multimodal profile package tests. |
| Local Stack serving contracts | `make vllm-sr-test-integration` | Live `serve`/`stop`, mounts, environment, pull policy, request behavior, and service isolation. |
| Local Stack memory contracts | `USE_DETERMINISTIC_MEMORY_EMBEDDINGS=0 make memory-test-integration` | Vela embedding, persistent retrieval, injection, user isolation, and persistence failure behavior. |
| Kubernetes | `make verify PROFILE=envoy-ai-gateway` | Deployment and routed requests through the selected profile. |
| E2E framework units | `make test-e2e-unit` | Collected Go helper and profile assertions, without starting a cluster. |
| Recipes | `make recipe-conformance-static`; `make recipe-conformance-live-cpu-all` | Authored contracts and live CPU Preview probes for all maintained sources. |
| Performance | `make perf-check PERF_BASE_REF=<base-commit>` | Complete benchmark inventory and a paired comparison using identical model artifacts. |

Use the prerequisites in the corresponding reusable workflow when reproducing
its job. ORT needs a compatible `ORT_DYLIB_PATH`; OpenVINO dependencies are in
`openvino-binding/requirements-test.txt`. Storage integration requires its
selected services. In required CI, missing dependencies fail the suite.

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
module. For the separate compatibility CLI test that needs a pinned checkpoint,
run `make test-modelcompat-native CANDLE_MODEL_PATH=/path/to/fixture`. The default
Core suite tests the offline command without downloading that checkpoint.

Dashboard runs both Go tests in the JavaScript/WASM runtime and Node
assertions against the compiled compiler. Their discovered cases and actual
results must agree, just like the other required suites.

## Models and artifacts

Published inference retains Domain, Guard, PII, FactCheck, Feedback, Modality,
Safety, Hazard, Embedding, and Reranker. The supported runtime inventory also
owns Candle Halu grounding and ORT Omni Nano/Mini contracts. Task adapters retain
their semantic assertions, including spans, input budgets, modalities, dimensions,
and ownership; a common runner does not reduce them to health checks. The
downloader resolves immutable revisions and validates the required format. The
runner requires every selected case and rejects skipped or missing results.
Candle's pinned legacy multimodal compatibility checkpoint remains distinct.

Prepared Omni bundles include a source identity, file hashes, and export parity.
Preparation is shared by compatible runtime and conformance contracts. Image
conformance uses the prepared Nano bundle and retains raw scores, threshold
checks, prototype protocol and validation results, and fixture provenance.
It does not replace deployed image extraction and routing tests. Reference-model
parity and maximum-context qualifications that require explicit inputs remain
separately declared; a shorter input acceptance test cannot qualify full context.

```bash
make download-models-test MODEL_TEST_PROVIDER=candle
make download-models-test MODEL_TEST_PROVIDER=ort
make download-models-perf
```

`make test-models` provisions its own prerequisites. Product startup and
`make download-models` instead provision models referenced by the active
configuration. They are not the full test inventory. A pre-existing local model
cache does not activate optional inference during ordinary core tests.

A build identity includes its source revision, platform, and build inputs.
CI builds each selected image and CPU native library artifact once, checks its
content digest when loading it, and reuses that read-only artifact across suites.
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
and 315 probes. For performance failures, distinguish allocation gates from
timing observations on shared runners. Go allocation metrics do not measure
native memory or GPU memory; see [`perf/README.md`](../../../perf/README.md).

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
