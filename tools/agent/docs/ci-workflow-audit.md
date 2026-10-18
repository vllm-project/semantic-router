# CI workflow value audit

This audit records the CI restructuring reviewed on 2026-10-18. It covers all
36 workflow files, their executable jobs, and the boundaries those jobs actually
check. The audit baseline is `1c629c80b383d7e53e01f266430d0fa41c5d9bf6`;
the decisions below describe this change, not the previous default behavior.
The ongoing execution policy is documented in [testing-strategy.md](testing-strategy.md).

The main cost problem was repeated environment preparation and broad automatic
qualification, especially image builds. Counting workflow files alone obscures
that problem: a reusable publisher, a lightweight lifecycle command, and a
seven-cluster integration matrix have very different costs and purposes.

## Observed cost and limits of the sample

The original audit sampled 60 completed workflow runs before this change:

| Caller | Completed runs sampled | Observed runner-minutes |
| --- | ---: | ---: |
| Pull Request | 40 | 1,238 |
| Main | 15 | 2,268 |
| Nightly | 5 | 3,751 |
| Total | 60 | 7,257 |

These are sums of job execution durations, not workflow elapsed time, billed
cost, or a controlled before/after experiment. Different changes selected
different inventories, and failed runs did not necessarily execute every job.
The sampled Main runs spent approximately 1,491 runner-minutes on image
construction/acquisition: **65.7% of their 2,268 minutes**. Nightly E2E execution
accounted for approximately 2,002 of 3,751 minutes, or 53.4%.

This supports reducing automatic image and deployment work before removing
small source checks. It does not establish a promised percentage saving.
Batching removes repeated checkout, dependency installation, image loading, and
test-binary compilation; the actual test bodies still run, and serial execution
can increase the elapsed time of an individual worker.

## Execution decisions

- **Pull requests:** run the affected automatic source/component checks first.
  Expensive runtime, service, image, and deployment qualification requires a
  committer's approval. A deferred required check or image remains required;
  the Gate fails until the planned evidence exists.
- **Manual approval:** a new `/ci run` comment authorizes the affected heavy
  work for the current PR revision and reruns the original Pull Request
  workflow. The trusted controller checks write/maintain/admin permission and
  binds approval to the PR head, base, and merge revision. New revisions require
  fresh approval. `/ci run full` requests the complete CPU inventory; narrower
  scopes are available in [ci_policy.py](../../ci/ci_policy.py). A separate
  manually dispatched verification is useful for diagnosis but does not replace
  the original PR Gate.
- **Main pushes:** run source/component checks selected by the automatic policy.
  They do not automatically publish the development CLI or production images.
  `Main` dispatched with `publish: true` runs complete qualification and
  publishes the six production images, Helm chart, and development CLI. The
  CLI is published only after its commit-addressed images exist. The six-image
  publication set is dashboard, operator, operator-bundle, and the CPU, CUDA,
  and ROCm router images; qualification can also need fixture images.
- **Weekly qualification:** the existing `nightly-build.yml` entrypoint now runs
  every Monday at 02:00 UTC, or on explicit dispatch. It uses the versioned full
  CPU inventory and retains successful-run image/Helm publication. An absent
  diff base does not make every repository file an affected product change.
  Existing `nightly` profile and publication names remain compatibility identifiers.
- **Release and maintenance:** keep explicit artifact/version publication and
  repository maintenance separate from per-PR product qualification. The
  release workflow is an artifact/version pipeline; it does not itself invoke
  another complete CPU test run.

Selection is owned by [domains.yaml](../domains.yaml),
[verification_catalog.yaml](../../ci/verification_catalog.yaml), and
[ci_plan.py](../../ci/ci_plan.py). This audit does not replace those executable
registries. There is no additional `merge_group` trigger in the audited workflow
inventory; a queued PR's original required checks must not be confused with a
separate queue-wide full run.

## Physical workers and preserved evidence

The scheduled planner previously treated every tracked file as changed when
there was no diff base. That expanded the plan to 63 contracts and 39 E2E
profiles, including optional deployments outside the versioned full CPU
inventory. Without an explicit comparison base or supplied changed paths,
weekly, release qualification, and Main publication now select the versioned
baseline: 39 contracts and 15 E2E profiles after retiring CK and adding the
model-runtime and benchmark-tool contracts. The release publication workflow
does not itself invoke that test-plan mode. Experimental profiles remain
available through affected PR changes and explicit manual selection. A full PR
run and explicit Main publication still retain affected-path additions.

Counts below compare physical execution of that versioned full CPU inventory.
They are execution-worker counts, not counts of assertions or a claim that the
old scheduled plan contained only 15 E2E profiles.

| Boundary | Before batching | After batching | What remains independent |
| --- | ---: | ---: | --- |
| Component contracts | 10 | 5 | All 10 discovery inventories, failures, logs, and source-bound receipts |
| Kubernetes E2E profiles | 15 | 7 | All 15 profiles, clusters, teardown, and receipts |
| Platform contracts | 2 | 1 | Model and image-routing reports and receipts |
| Recipe workflow, including inventory/results | 10 | 6 | All 8 live CPU recipes and every authored probe |

[execution_batches.py](../../ci/execution_batches.py) groups only compatible
workers. Component batches contain at most four contracts; model-runtime engine
dependencies remain separate from training/export tools. Each component has a
45-minute deadline including setup, and subprocess groups are retired before
the next contract. E2E/platform batches contain at most three contracts and
obey a combined execution budget; heavy model profiles remain isolated.
An E2E teardown failure stops the remaining profiles in that worker instead of
reusing an untrusted cluster. Ordinary independent failures still preserve
successful sibling evidence.

Recipe grouping uses three balanced standalone shards and a separate built-in
`mom-v1` shard. Current standalone variant weights are 129, 87, and 100; the
built-in shard has 315. Inventory and result aggregation remain separate jobs.
The report adapter reconstructs the same source-aware matrix and rejects a
missing image receipt or missing required probe. None of these reductions
means runner-minutes should fall in the same proportion as worker count.

## Workflow-by-workflow decisions

Job IDs below come from the YAML, including reusable caller jobs. A caller job
does not represent another machine in addition to its called workflow's jobs.
“Automatic” means selected by the changed-path policy; it does not mean that
every PR runs that row. “Approved/full” means affected PR work after approval,
or complete weekly/explicit publication qualification.

### Planning and entrypoints

| Workflow | Jobs | Actual value | Decision and execution time |
| --- | --- | --- | --- |
| [pr.yml](../../../.github/workflows/pr.yml) | `ci`, `pr-gate` | One PR execution entry and the stable branch-protection result. | Retain. Cheap work starts automatically; missing approved evidence keeps PR Gate red. No duplicate comment-triggered test pipeline. |
| [manual-ci.yml](../../../.github/workflows/manual-ci.yml) | `authorize`, `rerun` | Validates a comment and frozen revision, then reruns its original PR run. It executes trusted controller code. | Retain as the approval entrypoint; run only for `/ci run` comments. The two jobs separate authorization from Actions write permission. |
| [main.yml](../../../.github/workflows/main.yml) | `ci`, `images`, `helm`, `pypi` | Merged-source checks and explicit coherent development publication. | Change push behavior to source/component checks only. Publication jobs run after `publish: true` qualification; preserve CLI/image revision coupling. |
| [nightly-build.yml](../../../.github/workflows/nightly-build.yml) | `ci`, `docker`, `helm` | Broad detection outside the affected-path selection and promotion after successful qualification. | Change daily execution to Monday weekly execution; retain manual dispatch and full coverage. Keep the filename/profile for existing consumers. |
| [ci-changes.yml](../../../.github/workflows/ci-changes.yml) | `plan` | Resolves source/base, changed paths, approval, selected contracts, image dependencies, and display matrices. | Retain one planner. Approval changes execution eligibility without deleting required inventory. Scheduled full qualification uses the versioned inventory without treating all tracked files as affected changes. |
| [ci.yml](../../../.github/workflows/ci.yml) | `plan`; `image-router`, `image-dashboard`, `image-operator`, `image-fixtures`, `image-distribution`; `quality`, `generated`, `security`, `core`, `storage`, `dashboard`, `operator`, `local`, `recipes`, `platform`, `performance`, `package`, `tools`, `e2e-router`, `e2e-fixtures`, `e2e-dashboard`; `gate` | Connects producers to their actual consumers and reconciles independent receipts. | Retain the shared dispatcher. Merge compatible physical workers, not all jobs into an opaque shell script. Source checks start without waiting for images. Direct single-verification dispatch remains diagnostic. |

### Source and product checks

| Workflow | Jobs | Actual value | Decision and execution time |
| --- | --- | --- | --- |
| [pre-commit.yml](../../../.github/workflows/pre-commit.yml) | `quality` | Source lint/format checks, generated website references, CLI-reference tests, translation coverage, and an affected website build. | Retain automatic checks. Fix the pre-commit cache prefix, share Go checksum caches, skip Go setup for docs-only work, and avoid a second generated-reference pass inside the website build. |
| [check-generated.yml](../../../.github/workflows/check-generated.yml) | `generated` | Checks canonical configuration schema, OpenAPI, CRD documentation, runtime client, and decision catalog drift. | Retain automatic source-contract checking and align Go caching. Keep separate from Quality in this change: it has a different generator/toolchain boundary. |
| [security-scan.yml](../../../.github/workflows/security-scan.yml) | `security` | Tests trusted AST scanner behavior, scans candidate source using trusted policy, and scans rendered first-party Helm resources. | Retain affected automatic, weekly Monday 04:17 UTC, and manual runs. Remove the redundant full-plus-diff AST pass and advisory legacy regex scan. Keep SARIF collection plus the distinct HIGH/CRITICAL IaC gate. |
| [test-and-build.yml](../../../.github/workflows/test-and-build.yml) | `test-and-build` | Helm offline rendering, Go module/vet checks, discovered core tests, and trained-selector parity. | Retain automatic core contracts. Move the real development-launcher smoke test to Local Stack, its deployment owner, so Core does not build a second router image. The legacy filename does not imply a remaining general image-build job. |
| [dashboard-test.yml](../../../.github/workflows/dashboard-test.yml) | `dashboard-test` | Frontend/backend checks, WASM tests, controlled Evaluation browser tests, and build qualification with one dependency setup. | Retain affected automatic application coverage. Its controlled API/browser tests are not substitutes for the separately approved live dashboard E2E profile. |
| [test-tools.yml](../../../.github/workflows/test-tools.yml) | `tests`, expanded from component batches | CLI, harness, provider simulator, routing/calibration authoring, benchmark tools, E2E framework, soak-client/proxy, training, ONNX, and model-runtime contracts. | Consolidate 10 selected contracts into 5 compatible workers in a full run. Keep cheap contracts automatic; training/ONNX/runtime require approval. Remove the CK rewrite contract because the corresponding production surface was retired. |
| [package-check.yml](../../../.github/workflows/package-check.yml) | `package` | Builds and inspects the final wheel/sdist, resources, installed CLI behavior, and source/version metadata; supplies the exact distribution for promotion. | Retain automatic affected packaging qualification and publication qualification. It catches packaging-only regressions that source tests cannot. Do not replace it with `twine check` alone. |
| [test-storage.yml](../../../.github/workflows/test-storage.yml) | `storage` | Actual Redis, Valkey, Milvus, Qdrant, and PostgreSQL-backed contracts, with required inventory and skip rejection. | Move expensive service setup behind approval/full execution. Retain integration assertions; unit mocks do not establish storage behavior. |
| [test-local.yml](../../../.github/workflows/test-local.yml) | `integration`, once for each selected local contract | CLI serve/routing/lifecycle, the development launcher routed smoke request, and stateful memory integration. | Approved/full. Keep `local.cli` and `local.memory` separate because they stop shared services or databases; share immutable candidate images rather than rebuild in Core. |
| [test-platform.yml](../../../.github/workflows/test-platform.yml) | `contracts`, expanded from compatible batches | Published-model CPU inference and image-routing conformance with real runtime evidence. | Approved/full. Merge the two compatible workers while preserving independent reports, deadlines, failures, and model-cache sharing. |
| [recipe-conformance.yml](../../../.github/workflows/recipe-conformance.yml) | `inventory`, `live-cpu` matrix, `report` | Validates maintained assets and authored Preview acceptance/probe coverage across standalone and built-in latest sources. | Approved/full. Reduce 8 live workers to 4 through weighted grouping; keep inventory/report, serial long-context handling, and all probes. It does not replace a routed upstream-request assertion. |
| [integration-test-k8s.yml](../../../.github/workflows/integration-test-k8s.yml) | `integration-test`, expanded from E2E batches | Actual Kubernetes profiles consume qualified images, run isolated cases, and publish complete framework evidence. | Approved/full. Reduce 15 profile workers to 7 bounded compatible workers; set up Kind/load images/build the runner once per batch. Preserve the separate model-heavy worker and per-profile teardown. |
| [operator-ci.yml](../../../.github/workflows/operator-ci.yml) | `checks`, `bundle-validate`, `integration-test` matrix, `result` | Unit/generated/OLM contracts plus Operator reconciliation and an observed routed request. | Approved/full. Preserve current 1-variant affected and 7-variant full behavior in this change; evaluate the matrix separately below. Full expansion is 10 jobs, not 7. |
| [performance-test.yml](../../../.github/workflows/performance-test.yml) | `component-benchmarks` | Checks measurement contracts and runs real component/model/Looper benchmarks against baselines. | Approved/full. Repair benchmark compilation and retain evidence completeness. Numerical regressions remain advisory; shared hosted timing is not a reliable automatic PR veto. Execution failure or absent inventory still fails the contract. |

### Build and publication

| Workflow | Jobs | Actual value | Decision and execution time |
| --- | --- | --- | --- |
| [build-artifacts.yml](../../../.github/workflows/build-artifacts.yml) | `image` matrix, `published` | Builds/seals OCI candidates once, or acquires the qualified provider fixture by digest. Artifacts carry source and build identity. | Retain the producer boundary. Gate expensive affected PR images on approval and stop routine Main image publication. Keep per-image build caches and exact artifact handoff; cache hits are not qualification evidence. |
| [docker-publish.yml](../../../.github/workflows/docker-publish.yml) | `publish` matrix | Promotes the already-built OCI candidate with `skopeo`; checks published Dashboard images outside release mode. | Retain only as a gated publisher for explicit Main publication, weekly, and release. This is already promotion without rebuilding, so deleting it would not remove a second compilation. |
| [helm-publish.yml](../../../.github/workflows/helm-publish.yml) | `package`, `publish` | Versions/packages/renders a chart and promotes an existing package when supplied. | Retain publication/build-only modes. A validated Helm package and deployed Kubernetes behavior answer different questions. |
| [pypi-publish.yml](../../../.github/workflows/pypi-publish.yml) | `build`, `testpypi`, `pypi` | Builds when needed, verifies promoted distribution identity, and publishes to the selected package channel. | Retain explicit TestPyPI dispatch, gated development publication, and tag release publication. Reuse the qualified wheel/sdist; do not create a second unchecked development build. |
| [release.yml](../../../.github/workflows/release.yml) | `validate`, `images`, `helm-build`, `python-build`, `docker`, `helm`, `pypi`, `release-notes` | Cross-surface versions and immutable catalog snapshots; build all artifacts before publishing tags/packages and release notes. | Retain tag-triggered publication and manual build-only rehearsal. Do not count these publication dependencies as another full CPU test suite or use them to excuse missing PR qualification. |
| [precommit-publish.yml](../../../.github/workflows/precommit-publish.yml) | `validate`, `publish` | Builds the contributor tooling image on a narrow Dockerfile/linter-version/workflow change; publishes amd64/arm64 for the local pre-commit container target. | Retain narrow path triggers. This is a real developer-tool consumer, not an unused replacement for the hosted Quality job. It is an exception to the production-image Main publication policy. |

### Repository operations and reports

These workflows do not provide product test evidence. Their write permissions,
event triggers, and maintenance schedules are reasons to keep them apart from
untrusted PR test execution, even when they are small.

| Workflow | Jobs | Actual value | Decision and execution time |
| --- | --- | --- | --- |
| [maintenance.yml](../../../.github/workflows/maintenance.yml) | `board`, `stale`, `translation`, `leaderboard`, `unassign`, `flaky` | Existing dispatcher for independent maintenance tasks. | Retain. Board/stale/unassign/flaky run daily at 08:00 UTC; translation/leaderboard run Monday at 01:00 UTC. Manual dispatch chooses a task or all. No additional per-PR invocation. |
| [maintainer-board.yml](../../../.github/workflows/maintainer-board.yml) | `maintainer-board` | Produces issue/PR triage artifacts from bounded inventories. | Retain daily/manual reporting through Maintenance; it is not a merge correctness check. |
| [stale.yml](../../../.github/workflows/stale.yml) | `stale` | Marks inactivity at 30 days and closes after a further 30, with accepted/review/release exemptions. | Retain daily lifecycle operation. Do not merge its mutation policy into a test Gate. |
| [unassign-inactive-assignees.yml](../../../.github/workflows/unassign-inactive-assignees.yml) | `unassign-inactive` | Warns inactive assignees and releases ownership after a grace period, with exemptions and dry-run support. | Retain daily/manual. This is different from closing stale issues: accepted work can need reassignment without closure. |
| [check-translation-staleness.yml](../../../.github/workflows/check-translation-staleness.yml) | `check` | Updates unambiguous translation-status flags and opens a bot PR. | Retain weekly/manual maintenance. It is different from automatic translation coverage checks in Quality. |
| [contributor-leaderboard.yml](../../../.github/workflows/contributor-leaderboard.yml) | `update-leaderboard` | Refreshes public contributor data and updates its bot PR. | Retain weekly/manual; keep out of merge qualification. Follow-up: remove full website `npm ci` after preserving the generator's built-in Node/local-module dependency boundary. That setup is unnecessary for the current generator. |
| [flaky-tests.yml](../../../.github/workflows/flaky-tests.yml) | `report` | Reads recent result artifacts and separates merged-code failures from PR retry observations; it does not rerun E2E. | Retain daily aggregation through Maintenance. After Main becomes light and full runs become weekly, report sample counts/absence explicitly; a 14-day window may contain only two broad weekly runs. Lack of observations is not proof of stability. |
| [community.yml](../../../.github/workflows/community.yml) | `title-format`, `linked-issue`, `issue-lifecycle`, `issue-kind`, `issue-acceptance`, `issue-manager` | Admission/title checks, linked accepted issue validation, and issue lifecycle/commands with different write needs. | Retain event-driven operations. Combining every branch into one privileged job would obscure permission and event boundaries for little measured runtime benefit. |
| [community-labels.yml](../../../.github/workflows/community-labels.yml) | `pull-request-state`, `pull-request-sweep`, `issue-sweep` | Trusted post-check PR reconciliation and hourly repair of labels missed by events. | Retain current event/hourly behavior in this change. Measure API/runtime cost before reducing the repair frequency; these jobs are not redundant product tests. |
| [netlify-preview.yml](../../../.github/workflows/netlify-preview.yml) | `authorize`, `build`, `publish` | Authorizes `/netlify`, builds the requested commit, then publishes a preview under a separate permission boundary. | Retain explicit requested previews. A deployed preview has different value from the automatic website build; do not make it an unconditional PR job. |

## Operator: what the seven configurations prove

The full plan selects `memory`, `redis`, `milvus`, `qdrant`, `hybrid`, `mmbert`,
and `complexity` in [ci_plan.py](../../ci/ci_plan.py). An affected non-full
Operator run selects only `memory`. The YAML's `cache-backend` name therefore
covers both storage backends and model/configuration variants; it is not seven
independent cache implementations.

Every matrix entry creates a fresh Kind cluster, installs CRDs and the operator,
loads the same qualified request-path images, applies a different SemanticRouter
configuration, waits for reconciliation/readiness, and sends the same nonce
request. [check_operator_request.py](../../ci/check_operator_request.py) checks
the selected `operator_ci_route`, selected model, upstream response path, and
the provider's echoed nonce. This is a valuable real request assertion, not
merely `kubectl get` succeeding. However, it sends no repeated cache request and
uses no feature-specific expected route. The receipt aggregator verifies every
selected variant ran that common assertion; it does not manufacture additional
feature assertions from the variant names.

| Variant | Distinct configuration exercised | Current evidence limit | Value decision |
| --- | --- | --- | --- |
| `memory` | Basic reconciled deployment, CPU embedding configuration, in-memory cache configuration | No cache-hit or eviction assertion | Keep as the minimal affected Operator live smoke. It establishes the deploy-to-request path. |
| `redis` | Redis service, password Secret reference, index/connection configuration | No hit, TTL, or reconnect assertion | Retain a storage-specific integration boundary; add an observable repeated request before calling it cache correctness coverage. |
| `milvus` | Milvus service, authentication Secret, collection/index configuration | No persistence or query-result assertion | Retain distinct rendered/authentication/startup coverage; cache data behavior belongs in explicit assertions or Storage integration. |
| `qdrant` | Qdrant service and connection/collection configuration | No hit or persistence assertion | Retain distinct backend wiring; avoid claiming it duplicates the Storage suite's data assertions. |
| `hybrid` | HNSW plus Milvus configuration with different capacity/index settings | No hot/cold tier or failover assertion | Conditional value: keep full coverage for now; strengthen a hybrid behavior assertion or move the configuration combination into render/unit tests before removing its live cluster. |
| `mmbert` | Memory cache with HNSW/TTL settings; `memory` already uses the same mmbert embedding family | No extra model-behavior assertion | Strongest candidate for removing a dedicated cluster. Preserve configuration generation tests and the existing memory live path first. This change does not silently remove it. |
| `complexity` | Complexity candidate/rule configuration plus memory cache | The unconditional common route does not demonstrate complexity-dependent selection | Keep configuration compatibility coverage for now. Either test hard/easy inputs with distinct expected decisions or move this sample to render/unit coverage and retire its independent live worker. |

The first safe follow-up is to separate `checks` (unit/race, manifests/generated
code) from the approved deployment contract so cheap Operator source failures
can surface automatically. `bundle-validate` protects a real OLM distribution
surface: `make bundle` regenerates manifests and validates the bundle. It does
repeat part of `checks`' generation/setup, so sharing that preparation is worth
considering, while retaining both generated-drift and OLM-validation outcomes.

Do not blindly collapse seven configurations into one long cluster session.
Service/Secret isolation, cleanup after failed reconciliation, and independent
request receipts must survive. A smaller live matrix should follow explicit
coverage decisions: keep one basic deployment and necessary distinct backend
wiring, move redundant configuration combinations to fast render tests, and
give remaining feature variants a feature-specific assertion. This change
keeps the existing Operator implementation and full inventory intact.

## Acceptance and subsequent measurement

The restructuring must preserve source/artifact identity, required case
inventories, failure propagation, and stale-approval rejection. Worker-count
tests are insufficient on their own: the regression tests also exercise failed
siblings, missing evidence, timeouts, leftover subprocesses, and failed cluster
teardown. Performance compilation and the Istio prerequisite path receive
specific correctness fixes rather than being hidden by moving them to manual
execution.

After comparable workloads have run, measure runner-minutes by phase, queue and
end-to-end latency, setup versus test time, retry rates, and approved/deferred
inventory. Keep representative failed runs in that comparison. Use those
results to decide the next reductions, especially Operator variants, repeated
tool installation, and maintenance polling; do not infer savings from fewer
green check names alone.
