# CI job counts and cost audit

This is the quantitative companion to [the workflow audit](ci-workflow-audit.md).
The before revision is `1c629c80b383d7e53e01f266430d0fa41c5d9bf6`; after means
the restructuring merged in [PR #5223](https://github.com/vllm-project/semantic-router/pull/5223).
Counts expand the actual
matrices at each revision. They are not a comparison of check-name counts or a
measured after-change performance result. The later approval status publisher
is accounted for separately below; it does not change this historical baseline.

## Counting rules

A physical job is one selected `runs-on` job after expanding its matrix. A
reusable-workflow caller contributes **zero additional machines**. A skipped
job contributes zero allocations. An image built for several platforms inside
one Buildx job still counts as one job; that does not make its CPU cost equal
to a small source check. A provider fixture acquired by digest is one job,
just as a candidate fixture build is one job.

The scenario counts assume all selected prerequisites succeed and publication
is permitted on the canonical repository. They describe the complete execution
budget, not a prediction of how many jobs survive an early failure. The PR
control budget is three jobs: `ci-changes.plan`, `ci.gate`, and
`pr.pr-gate`. Main and scheduled qualification have two control jobs.
Publisher jobs are counted separately from qualification.

The exact inputs below are:

- Documentation: `README.md`.
- Router core: `src/semantic-router/pkg/extproc/processor_req_body.go`.
- CLI: `src/vllm-sr/cli/commands/serve_options.py`.
- Full CPU: the versioned full inventory with no additional affected profiles.
- Old scheduled run: no comparison base, so the old CLI passes every tracked
  path to `make_plan`. The new CLI supplies no changed paths in this situation.
- Explicit Main publication: no comparison base, `publish: true`.

These are representative inputs, not universal counts for every core or CLI
change. `/ci run full` also retains profiles selected by the real diff; a PR
changing an optional integration can exceed the versioned baseline. Registry
availability can add a provider-fixture **publisher** to Main if the qualified
fixture is missing; the producer allocation remains one. Counts below assume
that qualified fixture exists.

## Before and after expansion

| Scenario | Before selected contracts | After selected / required contracts | Before physical jobs | After physical jobs | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Documentation PR | 1 | 1 / 1 | 4 | 4 | Quality plus three control jobs; no runtime saving claimed. |
| Core PR, automatic run | 5 | 3 / 5 | 10 | 6 | Two E2E contracts and their images await approval. The Gate fails while required evidence is deferred. |
| Same core PR, approved affected run | 5 | 5 / 5 | 10 | 9 | Two images remain; two compatible E2E profiles use one worker. |
| Versioned full CPU PR, approved | 38 | 31 / 31 | 64 | 40 | Keep maintained product boundaries; retire four gateway qualifications, use one Operator deployment, and select five authoring contracts only when affected or requested. |
| No-base daily Nightly versus weekly qualification | 64 | 31 / 31 | 100 | 49 | Qualification is 92 versus 41; eight success-only image/Helm publication jobs are included in each total. |
| Main push, core input | 5 | 3 / 3 | 14 | 5 | Before: four image producers and three image publishers. After: automatic source/component evidence only. |
| Main push, CLI input | 5 | 4 / 4 | 17 | 6 | Before: five image producers, four image publishers, one CLI publisher and live Local Stack qualification. These eleven jobs move out of routine pushes. |
| Explicit Main publication | Not available | 31 / 31 | Not available | 50 | Full qualification, six image publishers, two Helm jobs, one development CLI publisher. |

Approval reruns the original PR workflow. The automatic six-job core attempt
and the approved nine-job attempt can both consume resources: **6 + 9 is 15
job allocations across two attempts**, not a nine-job lifetime budget. The
benefit is avoiding expensive work on unapproved revisions, with a cost from
repeating cheap checks after approval. Failed-job reruns can have a smaller
actual allocation, but should be measured rather than assumed. The successful
`manual-ci.yml` approval controller adds two separate hosted jobs (`authorize`
and `rerun`), so this complete approval transaction can allocate **17 jobs**.
Those controller jobs are outside the original PR workflow totals above.
Command reactions and rejection replies execute inside these existing jobs;
they add API calls but no additional job allocations.

The old full inventory contained nine component contracts. Ten component
contracts remain available after retiring CK and adding benchmark tools and
model-runtime qualification. Every selected component runs in its own job.
Five authoring contracts (`training`, `onnx-artifacts`, `bench-tools`,
`learning-tools`, `soak-tools`) now run only when affected or explicitly selected;
they are absent from the fixed product inventory. Full CPU component jobs are
therefore **9 → 5**, from scope selection, not batching. The complete full
contract inventory is **38 → 31**, including the new `operator.source` contract
and retiring the four required gateway integrations. Dynamo also leaves the
available manual inventory; it was never in this fixed full baseline.

Shared component runner, bootstrap, planner, and catalog changes explicitly
select all ten components through the changed-path registry. Python observation
changes select its eight component consumers. These changes still exercise the
authoring tests; `/ci run full` does not discard affected additions. This CI
refactor's audited diff selects **47 contracts and 48 physical jobs**, including
all ten components and 22 E2E profiles. That exceptional diff is not the fixed
31-contract, 40-job product baseline. Shared signal, corpus-loader, Responses
and failover changes select their affected integration consumers.

The same changed paths through the old planner's full entrypoint select
44 contracts, 19 E2E profiles and 70 physical jobs. The new registry repairs
seven missing affected-profile mappings: complexity, conversation, long-context,
learned model selection, preference, production stack and structure. It also
retires four previously required gateway profiles. Therefore **70 → 48 jobs**
is a same-input allocation comparison with a changed coverage mix; it is not a
controlled same-inventory cost measurement.

For a separate inventory-aligned comparison, explicitly selecting the current
22 E2E profiles in the old planner produces 22 workers versus nine now. Holding
these profile identities fixed removes **18 worker allocations through setup
sharing**: 13 E2E, four recipe workers and one platform worker. The complete old
execution shape with this profile inventory has 73 jobs versus 48 now; the
remaining seven reductions are six retired Operator live variants and obsolete
CK. The new Operator source contract replaces an embedded source job, so logical
contract identity is not unchanged. Retired weak testcase corpora also differ;
profile identity does not imply identical request workload. These are planned
allocations, not measured runner-minute savings.

| This refactor's approved full plan | Contracts | E2E profiles | E2E workers | Physical jobs |
| --- | ---: | ---: | ---: | ---: |
| Old selector, same changed paths | 44 | 19 | 19 | 70 |
| Old execution shape, explicitly aligned profiles | 47 | 22 | 22 | 73 |
| Current selector and execution shape | 47 | 22 | 9 | 48 |

## Approval status publisher follow-up

The later approval UX publishes **PR Gate** from
[`ci-status.yml`](../../../.github/workflows/ci-status.yml), whose `publish` job
runs when a PR run enters `in_progress` and when it completes. These events add
at most two short publisher jobs per PR attempt, outside that attempt's
qualification workflow. Explicit status-repair dispatches are separate runs.
The pending **Waiting for committer approval** check itself occupies
no runner. The publisher updates one bot comment with the remaining scope and
the `/ci run` command; actual failures remain failures and complete passing
evidence is required for success.

The existing `pr.pr-gate` runner job remains. Once the publisher is available,
it is named **CI Execution** and summarizes the executed qualification; the
publisher owns the required **PR Gate** check. During bootstrap, older
entrypoints retain the strict legacy PR Gate until the branch is updated.
The historical 40-job full baseline and 6 + 9 + 2 approval example above remain
evidence for the original execution shape. A later full attempt has those 40
qualification allocations plus at most two publisher allocations from the
normal events. This separate control cost does not rewrite the historical
baseline as 42 jobs. No publisher runner-minute or billing saving has been
measured.

## Every physical job in the versioned full CPU PR

Each row expands a real workflow job or a clearly enumerated matrix. The
before column sums to 64 and the after column to 40.

| Physical job or matrix | Before | After | Scope and reason for the decision |
| --- | ---: | ---: | --- |
| `ci-changes.plan`, `ci.gate`, `pr.pr-gate` | 3 | 3 | Resolve the immutable contract, validate evidence, expose the required PR result. Small control jobs remain. |
| `build-artifacts.image` / `published` | 5 | 5 | Dashboard, operator, operator-bundle, provider-mocker, CPU router. Build once and hand off exact image identity. |
| `pre-commit.quality` | 1 | 1 | Changed-source lint/format plus reference contracts. Cache installation; do not hide real source errors. |
| `check-generated.generated` | 1 | 1 | Canonical generated artifacts and drift checks. Keep distinct from source formatting. |
| `security-scan.security` | 1 | 1 | Retained AST and rendered infrastructure checks; remove redundant/noisy scanning steps without inventing a saved job. |
| `test-and-build.test-and-build` | 1 | 1 | Router Go/race contracts and selector parity. Keep automatic; installed CLI smoke belongs in the CLI integration contract. |
| `test-storage.storage` | 1 | 1 | Real storage services and their data assertions; approval/full qualification. |
| `dashboard-test.dashboard-test` | 1 | 1 | Frontend/backend application contracts. Keep an independent product boundary. |
| Operator source and deployment jobs | 10 | 4 | Old source + bundle + seven live variants + results. New combined source job + image-bundle inspection + one memory-backed deployment + results. Storage engines retain separate storage contracts. |
| `test-local.integration`, matrix `local.cli`, `local.memory` | 2 | 2 | Separate local product boundaries; preserve real lifecycle/routed-request assertions. |
| `recipe-conformance.inventory`, `live-cpu`, `report` | 10 | 6 | Inventory + eight recipe jobs + report becomes inventory + four source-aware live shards + report. |
| `performance-test.component-benchmarks` | 1 | 1 | Real benchmark execution and complete measurement evidence; numerical regressions remain advisory on shared hosted hardware. |
| `package-check.package` | 1 | 1 | Installed wheel and source/package identity; distinct from unit tests. |
| `test-tools.tests`, component matrix | 9 | 5 | Full selects CLI, model-runtime, E2E framework, CI harness, and provider simulator. Five authoring contracts remain affected/manual, each in its own job. Retire obsolete CK. |
| `test-platform.contracts`, platform matrix | 2 | 1 | Share the same runtime/model preparation for two CPU contracts; retain separate reports. |
| `integration-test-k8s.integration-test`, E2E matrix | 15 | 6 | Retire four external gateway integrations; batch the remaining 11 profiles with fresh clusters and independent receipts. |

For scheduled/publication plans the image set adds CUDA and ROCm router images.
The old no-base scheduled plan additionally selected the model-runtime fixture,
all 39 E2E profiles, and both component contracts omitted by its versioned full
inventory. This explains its eight image producers and eleven component jobs;
it was not just the old 38-contract baseline with a different cron expression.
The new scheduled plan has seven image producers, and optional profiles still
run when affected or explicitly selected.

### Component mapping

The ten available logical contracts each have a separate physical worker when
selected. Five belong to fixed full qualification:

| Job contract | Fixed full | Dependency and rerun reason |
| --- | --- | --- |
| `cli-unit` | Yes | Editable CLI/bench extras; historically the longest component job. |
| `model-runtime` | Yes | Runtime engine extras and CPU Torch; deterministic tiny-fixture tests are automatic, while real published-model checks remain approved. |
| `onnx-artifacts` | No | Real CPU ONNX inference and external-weight artifact behavior; qualify changes to artifact tooling. |
| `training` | No | Broader pinned training/export stack, including CPU Torch; qualify changes to model authoring. |
| `bench-tools` | No | Benchmark-client contracts; broad dependency ranges must not contaminate another contract's environment. |
| `e2e-unit` | Yes | Go framework unit/integration contracts without a live Kubernetes profile. |
| `harness-tools` | Yes | Registry, planner, receipt, authorization, and workflow invariants. |
| `learning-tools` | No | Research parsing plus calibration authoring tests. |
| `mock-provider` | Yes | Protocol simulator contracts with a dedicated provider-mocker venv. |
| `soak-tools` | No | Go soak client plus Python fault-proxy assertions. |

The original singleton component set was `cli-unit`, `ck-rewrite`, `e2e-unit`,
`harness-tools`, `learning-tools`, `mock-provider`, `onnx-artifacts`, `soak-tools`,
and `training`. CK's production rewrite surface was already retired. Its
historical job took only 2.97 runner-minutes across five Nightly runs, so removing
it is a maintenance/correctness decision, not the principal cost saving.

Shared setup is not hermetic test isolation. Killing subprocess groups and
using separate report directories does not roll back `pip install`, files, or
Python package versions. ONNX and training currently pin compatible ONNX/ORT
versions, but their larger environments and rerun granularity remain independent.
The reviewed policy defaults to one component contract per job; an explicit
shared policy supports future opt-in grouping only when its dependency and
measurement case is established. No current component opts into that grouping.

### E2E and platform mapping

The versioned E2E inventory changes from 15 profiles to 11 in six jobs.
Agentgateway, AIBrix, Istio, and llm-d deployments leave maintained qualification;
Envoy AI Gateway remains the representative third-party ExtProc gateway,
beside standalone and the focused streaming profile. This is an explicit
support-scope reduction, not equivalent cross-gateway coverage. Dynamo loses
its optional manual profile and does not affect this fixed baseline.

| Job | Exact profiles |
| --- | --- |
| Model-heavy singleton | `vela-omni` |
| Dashboard lane | `dashboard` |
| Provider fixture batch 1 | `envoy-ai-gateway`, `local-classifier-backend`, `routing-errors` |
| Provider fixture batch 2 | `standalone`, `streaming`, `vela-halu` |
| Provider fixture batch 3 | `vela-shield` |
| Router-only batch | `multimodal-routing`, `remote-embedding` |

Each job has at most three profiles and at most 270 minutes of declared profile
budget. The model-heavy profile remains alone. Compatible placement, runtime,
device, platform, image dependencies, and resource class must match.
[run_e2e_batch.py](../../ci/run_e2e_batch.py) creates a fresh uniquely named
cluster and state directories for every profile. It retires process groups and
refuses to continue after unsuccessful teardown. Batching shares machine and
image preparation, not a live application deployment between profiles.

The platform batch contains `platform.image-calibration-cpu` and
`platform.models-cpu`: two 120-minute contract budgets, one 240-minute batch.
It shares model files/runtime installation, but preserves separate evidence
and process deadlines. This can take longer to finish than two concurrent jobs.

### Recipe mapping

| Live shard | Exact source and recipes | Variant weight |
| --- | --- | ---: |
| `standalone-0` | Standalone `multi-objective` | 129 |
| `standalone-1` | Standalone `balance`, `knowledge` | 87 |
| `standalone-2` | Standalone `agent`, `feedback`, `privacy`, `accuracy` | 100 |
| `built-in-latest-mom-v1-0` to `-15` | Built-in latest `mom-v1`, one probe shard each | 19–32 (347 total) |

All eight recipes remain. Weights count authored probe variants, not measured
seconds. The heavy built-in recipe remains alone: its historical median job
time was 35.02 minutes, versus 4.85–7.48 minutes for standalone recipe jobs in
the sampled complete runs. With Vela 2.0 0.8B as its decision model, its
`evaluation.shards: 16` splits the probes into 16 jobs of similar estimated
read cost, and the report merges them. Inventory and report aggregation add two
jobs.

## Historical resource evidence

The preserved sample covers 2026-10-14 through 2026-10-18 and contains 60
completed runs, including failures and cancellations, across different source
revisions. The static job comparison pins the baseline SHA above; this historical
cost sample is not a controlled run of that single revision. One of the 40 PR runs
was cancelled before any job allocation; the job CSV therefore represents 39
PR runs. Executed counts below exclude skipped/zero-duration entries.

| Caller | Runs | Executed job records | Sum of runner-minutes |
| --- | ---: | ---: | ---: |
| Pull Request | 40 | 338 | 1,237.92 |
| Main | 15 | 221 | 2,267.98 |
| Nightly | 5 | 393 | 3,751.33 |
| Total | 60 | 952 | 7,257.23 |

Main image construction/acquisition consumed **1,491.10 minutes (65.7%)**.
Across all callers, ROCm image construction alone consumed **1,047.20 minutes
in 20 jobs**. The five Nightly runs consumed 2,002.07 minutes in the broad E2E
category, which includes Kubernetes, local, and Operator jobs. Calling that
entire figure Kubernetes-profile time would overstate its scope. All five
Nightly runs failed; publication jobs did not execute, and one run stopped
much earlier. These are observed costs, not successful-run budgets.

### Measured work retired after the first successful revision

The [successful full run at `7c18a160`](https://github.com/vllm-project/semantic-router/actions/runs/39120541417)
preceded the gateway and Operator scope reduction. It allocated 53 jobs; it is
not validation of the later 48-job plan. Its API job timestamps show the four
Operator deployments now removed consumed the following time:

| Removed Operator deployment | Completed job time |
| --- | ---: |
| [Milvus](https://github.com/vllm-project/semantic-router/actions/runs/39120541417/job/117656582749) | 6m47s |
| [Hybrid](https://github.com/vllm-project/semantic-router/actions/runs/39120541417/job/117656582788) | 6m25s |
| [Redis](https://github.com/vllm-project/semantic-router/actions/runs/39120541417/job/117656582877) | 6m43s |
| [Qdrant](https://github.com/vllm-project/semantic-router/actions/runs/39120541417/job/117656582918) | 6m03s |
| Total | 25m58s |

The same run's raw E2E reports record separate profile intervals:

| Retired gateway profile | Cases | Profile interval |
| --- | ---: | ---: |
| agentgateway | 4 | 4m03.38s |
| AIBrix | 2 | 6m28.40s |
| Istio | 3 | 6m17.15s |
| llm-d | 3 | 3m57.70s |
| Total | 12 | 20m46.63s |

The Operator figures include each job's setup; the gateway intervals exclude
shared worker preparation and include the profile's own deployment work. The
12 gateway case durations themselves sum to only 32.69 seconds. All passed on
their first attempt in that run. These are measurements of work in the previous
implementation that is now removed. They must not be subtracted from that run's
total or combined into a promised after-change saving: batch composition,
remaining setup, cache state, scheduling and the code under test also change.

### Measured setup within one old run

The public [October 18 Nightly run](https://github.com/vllm-project/semantic-router/actions/runs/39089835076)
provides step timestamps. “Setup” below is the sum of completed step durations
before the main execute/probe step, including checkout, disk preparation,
tool/dependency setup, image loading, and binary construction. It excludes
queue time and post-job cache saving. “Execution step” can itself contain
further dependency setup, cluster creation, test discovery, and teardown;
it is not pure assertion time.

| Layer | Jobs with execution steps | Setup total | Median setup per job | Execution-step total | Complete job total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Components | 11 | 5.08 min | 0.45 min | 31.75 min | 37.73 min |
| Kubernetes profiles | 39 | 189.20 min | 4.72 min | 231.03 min | 424.28 min |
| Platform | 2 | 5.88 min | 2.94 min | 19.68 min | 28.02 min |
| Recipe live workers | 8 | 31.90 min | 3.62 min | 41.62 min | 76.47 min |

This evidence favors reducing repeated E2E and recipe preparation before
aggressively packing small component jobs. It does not imply that all setup
minutes disappear: each remaining batch still pays setup, caches vary, and the
new qualification inventory differs.

For example, the [old router-profile worker](https://github.com/vllm-project/semantic-router/actions/runs/39089835076/job/117568367680)
spent setup time before exercising its profile; the same reusable workflow
repeated that preparation for every profile. The
[old platform worker](https://github.com/vllm-project/semantic-router/actions/runs/39089835076/job/117564394253)
separately installed the runtime and prepared disk/cache state. The reduction
in their setup **instances** is known; an after-change duration saving is not
yet measured.

Quality keeps one job and Generated keeps one job. Their pip/pre-commit/tool
caches and removal of duplicate reference generation reduce work inside jobs;
they do not reduce the physical count. Similarly, removing the leaderboard's
full website `npm ci` saves an unnecessary installation in its weekly
maintenance job, with no measured minute claim. Image publication promotes
existing candidates with `skopeo`; it was already not a second compilation.

### Calculated repeated image-artifact transfers

The same historical run's artifact API reported these archive sizes without
requiring any artifact download: CPU router **1,194,853,479 bytes**, provider
fixture **101,232,180 bytes**, and Dashboard **308,545,737 bytes**.
[load-ci-images](../../../.github/actions/load-ci-images/action.yml) downloads
the selected immutable artifacts separately on each worker.

Holding those artifact sizes fixed, the original 15 versioned E2E profiles
use 15 CPU-router + 9 provider + 1 Dashboard downloads. The remaining 11
profiles would use 11 + 8 + 1 without batching, and use 6 + 4 + 1 with batching.
The eight recipe workers become four CPU-router downloads. This yields:

| Consumers | Before calculated archive payload | After calculated archive payload | Reduction and cause |
| --- | ---: | ---: | --- |
| Original 15 versus retained 11 E2E profiles | 17.83 GiB | 7.34 GiB | 10.49 GiB: 4.55 from retiring integrations, 5.94 from batching retained profiles. |
| Same 8 live recipes | 8.90 GiB | 4.45 GiB | 4.45 GiB from batching. |
| These two consumer groups | 26.73 GiB | 11.79 GiB | 14.94 GiB with both scope and execution changes. |

The follow-up change from five Operator deployments to one removes four
consumers of the Operator, CPU-router, and provider archives. At the same
historical sizes, including the **196,515,397-byte** Operator archive, this
removes another calculated **5.56 GiB** of repeated payload. It is a reduced
live-deployment scope, not setup sharing or measured network savings.

These are calculated artifact payloads at fixed historical sizes, not measured
after-change network traffic. They exclude other consumers, transport overhead,
retries, model/cache transfers, and decompressed Docker storage. The producers
still upload the same candidates once. Artifact retention GB-hours, peak runner
disk use, CPU-hours, and billing dollars were not measured. Reducing repeated
consumer downloads does not by itself reduce retained artifact size.

## What can be claimed about savings

The versioned full PR budget falls from 64 to 40 physical jobs: 24 fewer
allocations, or 37.5%. The scheduled successful-run budget falls from 100 to 49:
51 fewer allocations. These are structural counts, **not runner-minute or
wall-clock reductions**.

Changing daily scheduled qualification to weekly removes six of seven scheduled
launches. If the same run cost were held constant, that cadence alone would
reduce scheduled runner use by 85.7%. The actual change also alters inventory,
batching, caches, and failure paths, so the historic average must not be
presented as measured future usage. With no manual dispatches and all
prerequisites succeeding, the planned job-launch budget is 700 versus 49 per
week; that combined count also must not be translated into a cost percentage.

Routine Main source/component qualification remains. Expensive qualification
and publication move to an explicit decision; they are not free when that
decision is exercised. A manually published commit needs the complete image
set so its development CLI's commit-addressed defaults remain usable.

Batching exchanges less repeated preparation for less parallelism and coarser
reruns. If one profile fails, rerunning its Actions job reruns its successful
batch siblings. A two- or three-profile batch can therefore increase both
feedback latency and retry work. Independent receipts improve diagnosis; they
do not make GitHub's job retry contract-granular. A separately dispatched
single contract is diagnostic evidence, not a substitute for the original
required PR Gate.

The evidence supports the bounded E2E/recipe/platform groups, not unbounded
serial suites. Keep long model/CLI work independent. All component contracts
retain independent jobs after reviewing their smaller shared setup benefit and
dependency/rerun costs. The Operator now exercises one supported memory-backed
deployment; deleted external-storage topologies are not silently merged into
that case. Do not add unrelated component packages just to reduce the UI count.
Further splits should follow observed queue/critical-path and rerun costs;
collect at least setup/execution/post durations, approved-attempt counts,
cache-hit state, and source-bound inventory for comparable runs first.

## Reproducing the count

Run each revision's own [ci_plan.py](../../ci/ci_plan.py),
[verification catalog](../../ci/verification_catalog.yaml), changed-path
registry, and recipe planner. Load the revisions in separate Python processes;
module caches must not mix before/after registries. No build, model download,
registry publication, or test execution is needed to call `make_plan`.

For the old scheduled scenario, pass `git ls-tree -r --name-only <before-sha>`
as the path list. For the new scheduled scenario, pass an empty list, matching
the CLI's no-base behavior. For approved PRs pass
`require_approval=True` and an authorization object with `authorized: true`
and `scopes: [affected]`; use `full: true` for the full PR. Old `make_plan` has
no approval or publication parameters. For explicit Main publication use
`profile="main", publish=True` with no paths. Use a full source SHA in all
plans. These are planner simulations, not authorization files for live CI.

For each plan, sum:

1. Two control jobs, plus one PR result job when `profile == "pr"`.
2. One producer/acquisition job per `images` entry.
3. One job per selected ordinary executor contract, including each local
   contract. Count `operator.source` in its source workflow.
4. The lengths of `component_batches`, `platform_batches`, and `e2e_batches`.
5. Operator runtime jobs: the old non-matrix jobs plus its actual
   `integration_matrix` length; the current workflow has one memory-backed
   deployment, one image-bundle inspection, and one result job. Read each
   revision's workflow instead of assuming the old matrix still exists.
6. Recipe inventory + live source-matrix length + report. Before uses
   `source_matrix_payload(inventories, None, root)`; after uses three shards
   per source. Obtain inventories with `discover_source_inventories` and
   `cpu_inventory(discover_inventory(root))`.
7. For Main/scheduled successful paths: one job per `publish_images` entry,
   two Helm jobs if requested, and one prebuilt development CLI publisher if
   requested. Reusable caller nodes add zero.

The matrices above document every member needed to cross-check that sum.
Historical runner-minutes are `sum(seconds) / 60` from executed job records;
step timing uses `completed_at - started_at` for the defined setup interval.
Keep failed/cancelled work and zero-job runs explicit when comparing samples.
