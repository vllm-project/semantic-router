# Decision paired performance release gate

The protected ROCm job must generate `report.json` and its `raw/` directory in
the same run that builds the candidate image. Qualify them with:

```bash
python3 tools/ci/decision_perf_release_gate.py validate \
  --report "$REPORT_DIR/report.json" --source-sha "$GITHUB_SHA" \
  --qualification-receipt "$QUALIFICATION_DIR/qualification.json" \
  --candidate-ref "$CANDIDATE_REF" --owner "$GITHUB_REPOSITORY_OWNER" \
  --run-id "$GITHUB_RUN_ID" --run-attempt "$GITHUB_RUN_ATTEMPT"
```

The performance report schema is `decision-paired-release-v6`. It requires
live old process and full-snapshot proof for all six models; v4/v5 receipts
cannot qualify.

The gate expects exactly six Decision model IDs, each measured at 32 questions
and 1 state, 8 questions and 8 states, and 32 questions and 32 states. Every
shape needs concurrency 1, 8, and 32. Each concurrency cell needs three
alternating old/new throughput rounds with at least 32 successful workflows
per arm per round. The raw benchmark receipt, untimed parity result, and
workflow log for every shape travel with the report. Their SHA256 hashes and
path confinement are checked again after the CI artifact handoff.
The report also hashes the preflight and formal audit logs, HTTP samples, and
metrics records. The protected run ID and attempt prevent a prior attempt at
the same commit from being reused in the qualification and package jobs.
The gate checks the maximum simultaneous HTTP calls across the entire low-load
phase and each throughput round. It requires concurrency 1, 8, and 32 cells
to run in that order without overlap for each shape. Metric captures retain
their producer order and cumulative counters and histogram buckets cannot
decrease between rounds, cells, or shapes for the same model service. Work
between measured waves may increase those cumulative values.
Within each model service, repeating a `/metrics` response hash must reproduce
the same parsed counters and histogram. Warmup and low-load phases follow the
tracked serial old/new schedule and finish before throughput waves; none may
add hidden load to a measured arm.

Both arms must use the same model revision and hardware/network scope. The
new runtime image source, benchmark source, and checked out source must match
the current commit. The gate verifies zero failed workflows, no parity
mismatches, matching decision counts, recomputed throughput and latency
percentiles, correct alternating wave timing, and positive physical batch
counter deltas. High-load cells must show actual batching above one row per
physical batch. Every concurrency 8 or 32 cell, including the single-state
shape, must retain at least 95% of its old-arm decisions/sec throughput.

The throughput win is predeclared over all six models and both multi-state
shapes: 8 questions/8 states and 32 questions/32 states, each at concurrency
32. For each of these 12 fixed cells, divide the new-arm decisions/sec by the
old-arm decisions/sec. The equal-weight geometric mean of all 12 ratios must
be at least 1.05. Each model's geometric mean of its two ratios must be at
least 1.00, so one model's gain cannot conceal another model's regression.
An individual model may be flat. The qualification summary publishes the
overall mean, each model's mean, and both shape ratios for every model; no
best-performing shape is selected after measurement. Three old/new alternating
rounds per cell limit order effects. The 0.95 cell floor is a fixed allowance
for measurement variation, not a statistical confidence bound; all cell
measurements remain in the hashed report for review.

The measurement compares synthetic
HTTP workflows, not task accuracy. Multi-state results compare the old
single-state fanout with the new batch protocol, so their wire requests are
different. The report labels that scope rather than attributing every gain to
the GPU kernel.

The code does not pin model-file versions. Each measurement records the
old/new model revision and artifact identity used for that run and checks
they agree. A new weight revision can be qualified without changing the gate.

The protected job must have a trusted old serving baseline and an immutable
new candidate image available on the same ROCm host. If either is unavailable,
the report cannot be produced and release stays closed. Publishing a previous
run's JSON cannot qualify a different source commit.

## Same-run producer input

`decision_perf_release_producer.py` takes a protected JSON file supplied by
the release environment, the current run's ROCm qualification receipt, and
the candidate image digest. It requires six live old/new service pairs on the
same host and selected ROCm device. Each endpoint must be bound through a
declared running container to `127.0.0.1`. The new image must match a registry
reference pinned with `@sha256:`; the protected old image may instead use an
exact local Docker `sha256:` image ID when redistribution is unavailable. The
producer checks the candidate image's clean source label, exact `drun` command
and environment, live container-init process, and artifact/scheduler status,
and requires the loopback API/metrics URL to map to the process's port 8000.
It then performs a separate untimed
preflight before each tracked semantic benchmark. An old preview service may
declare its audited model-ID and response adapter. For code baked into the old
image, `old.core_source_kind` would be `baked`, but the current producer
rejects that path: an image label alone is only an identity declaration. A
future trusted protected provisioner must separately attest the immutable
image ID, entrypoint, process, and executed source tree before baked-core
release qualification may be enabled. A directly executed mounted core is also
ineligible because it does not attest the live loaded artifact. The required
reviewed HTTP adapter uses `old.core_source_kind: mounted_adapter`: its adapter
and core are distinct read-only mounts checked against
`old_adapter_source_sha256` and `old_core_source_sha256` from this run's
protected configuration. A third read-only tree mount holds the complete
historical model snapshot, with co-located data and model Python. PID 1 must
execute the mounted adapter. Exactly these three old mounts are allowed; the
adapter must be a single file mount at its executed path. The producer hashes
every typed bind mount before and after measurement. Each host source must be
an absolute canonical path with no symlink ancestor, and the protected runner
must expose the live container PID under host `/proc` so the producer can match
the source device/inode to the actual bind mount before and after hashing.
This prevents a retargeted host path from being hashed instead of the running
service's mounted object. It independently reopens
the candidate-qualified data-only artifact, then verifies the old core's
binding inventory, complete
snapshot roster, manifest, and every bound file by size and SHA256. For each
selected qualified data file it compares the old path, size, and freshly
computed SHA256, reconstructs the selected-data content ID, and requires
exact equality with qualification. It parses metadata but never imports or
executes model-repository Python. The full snapshot has its own tree digest;
it is not misrepresented as a data-only materialization. A local-only old
image must remain present on the protected runner for later audit; its
content-addressed ID does not make the old source redistributable.
The core mount's `BINDINGS.json` entry keyed by `old_model_id` (or `model_id`)
must declare `repo`, `revision`, `manifest_file`, `manifest_sha256`, and a
complete `files` map from relative path to `{ "bytes": ..., "sha256": ... }`.
Every regular snapshot file must be bound, including co-located model code,
and each self-manifest entry must agree with that binding. Unselected
historical source files remain in the full snapshot and are verified, not
copied into the candidate's data-only cache.

The protected file has `schema_version: decision-paired-baseline-v2`, a
public-safe `hardware` label, one canonical `gpu_device` index,
`gpu_exclusivity: dedicated_gpu_no_unrelated_compute`, and exactly six
`models` entries. Both containers must expose that exact
`ROCR_VISIBLE_DEVICES` index and the same `/dev/kfd` and `/dev/dri` device
mapping on the protected host. Each entry
contains `model_id`, `revision`, `artifact_content_id`,
`artifact_metadata_sha256`, `artifact_manifest_sha256`,
`old_core_source_sha256`, `old_physical_batch_size`,
`new_physical_batch_size`, and an
`old_arm_overlay` digest or `none`. The producer reads the running candidate
container's exact `--max-batch` launch argument and requires it to match this
per-model value. The report and raw benchmark receipt record both old and new
declared physical batch sizes. The old value is a protected baseline
declaration, not an independently inferred old runtime setting. The gate uses
the new value as the occupancy
capacity when reconciling telemetry. The current untuned policy admits at
most B8. B16/B32 need separately produced, artifact-bound kernel-profile
coverage and numeric, HBM, and performance A/B qualification before that
policy can open; a protected config assertion is not a substitute for those
measurements.
Optional `old_model_id`, `old_response_mode` (`decision_v1` or
`legacy_preview`), and `old_token_env` select the audited old adapter. The
`old` and `new` objects each contain `url`, `container_id`, and `image_ref`;
`new.metrics_url` is required and `old.metrics_url` is optional. All URLs use
explicit loopback ports. The producer verifies each metrics port belongs to
the same inspected container as its API port. The new runtime serves metrics
on its API listener, so its two URLs must have the same origin. Every old bind
mount must appear in `old.mounts` with `destination`, `kind` (`file` or
`tree`), and `sha256`; writable, untyped, or undeclared mounts fail. A file
digest is SHA256 of its bytes, and its declared kind prevents it from standing
in for a tree. A tree digest is SHA256 of the domain separator
`decision-mounted-tree-v2` and a null byte, then each sorted entry's type and
relative POSIX path as eight-byte big-endian length-prefixed fields. Each
directory appends a zero content length; each file appends its eight-byte
content length followed by its exact bytes. Symlinks and special files fail.
The old delimiter-concatenated v1 tree digest is not accepted because two
different trees could have identical input streams. `old_arm_overlay` is
`none` or `sha256:` plus a digest found in that mount inventory. The artifact
metadata digest must match the locally verified content-addressed artifact
receipt.
`old.artifact_mount_destination` must name the full-snapshot tree mount. Set
`old.artifact_layout: full_snapshot_selected_data_v1` and
`old.artifact_locator: {"kind":"argument","flag":"--artifact-root"}`;
Docker's declared launch must pass that mount as the exact flag value, and the
adapter must derive the same root from its resident loaded engine object.
Nested or overlapping old mounts are rejected so a later bind mount cannot
shadow the verified artifact tree. Keep endpoint, container, and credential details only
in the protected file and environment.
For `mounted_adapter`, use
`old.adapter_locator: {"kind":"command_path","path":"/adapter/server.py"}`
and `old.core_locator:
{"kind":"python_import","module":"old_core","path":"/core/old_core.py"}`.
Both paths must stay inside their respective, non-overlapping mounts. The
adapter path must be the executable or the script immediately following a
Python interpreter in both Docker's declared launch and the live container-init
command line. A static mount of the core that the adapter never imports cannot
qualify. Set `old.api_container_port` to the old service's actual container
listener port; the producer requires its sole `127.0.0.1` host binding to match
the declared `/v1/systemone` URL. The attestation URL uses this same host port.
The
adapter must expose `GET /api/decision-baseline-attestation?challenge=<64 hex>`
on the same loopback listener as `/v1/systemone`. Redirects, non-200 status,
and non-JSON media types fail. Its bounded JSON response
must echo the challenge, report container PID 1 and its `/proc/1/stat` start
ticks, the adapter path and SHA256, the actual imported module name/file and
core mount digest, and the loaded model's ID, revision, artifact root and
content ID. The protected provisioner must review the adapter's source and
ensure these values are derived from the live imported module and loaded model
object rather than copied from request or configuration. The producer checks
the response against the host-side PID, mount digests, and qualified artifact,
then repeats the challenge after measurement. Before archiving the proof, the
producer replaces container-internal path and private module strings with their
SHA256 digests. It verifies path containment against the protected mount
declarations; the gate checks that both observations match the declared path
digests. The report also carries the declared full-snapshot tree digest and
two matching pre/post snapshot verifications that bind it to the selected-data
content ID. Both redacted process observations and their file digest travel
with the report. The gate requires these fields for every model; older
performance receipts without live process proof fail validation. An adapter
without this endpoint cannot qualify.
The old preview protocol does not guarantee an artifact-status endpoint, so
the protected provisioner must also establish that the old process actually
loaded the declared revision and artifact bytes. A matching read-only tree and
launch declaration narrow that trust boundary but are not loaded-artifact
attestation from HTTP alone. Even a live adapter report depends on its reviewed
implementation: external hashing cannot prove tensors resident in GPU memory,
and a privileged host operator can transiently alter bind sources between
pre/post checks. The pinned image, interpreter, imported libraries, historical
loader, reviewed adapter, protected-runner isolation, and loaded-object
derivation remain trusted. Source/mount hashes and semantic parity alone
cannot prove weight identity. If any old binding revision differs from
qualification, the producer fails closed; full-snapshot layout does not waive
same-revision qualification for all six models.
The adapter may add the attestation route, but its reviewed serving path must
retain the historical loader, admission, evaluation, and response behavior;
the producer's hash and nonce checks cannot establish that behavioral
equivalence on their own.

The performance report names the candidate digest. Both protected validation
steps independently revalidate the ROCm qualification receipt and join its
candidate digest and all six qualified model revisions/artifact content IDs to
the paired report before promotion. The gate also reconciles timed sample
requests and outcomes with workflow spans and summary counters, formal and
preflight audit case IDs/results with the receipt, and per-round metric
captures with physical-batch totals. Updating a raw file and its hash cannot
turn contradictory evidence into a passing report.

```bash
python3 tools/ci/decision_perf_release_producer.py \
  --baseline-config "$PROTECTED_BASELINE_CONFIG" \
  --qualification-receipt "$QUALIFICATION_DIR/qualification.json" \
  --candidate-ref "$CANDIDATE_REF" --owner "$GITHUB_REPOSITORY_OWNER" \
  --source-sha "$GITHUB_SHA" --run-id "$GITHUB_RUN_ID" \
  --run-attempt "$GITHUB_RUN_ATTEMPT" --output-dir "$REPORT_DIR"
```

The protected workflow invokes this producer after the current run's ROCm
candidate and six-model receipt exist. The protected runner must provision
the six live old/new service pairs and expose its current-run baseline JSON at
the private path named by `DECISION_PAIRED_BASELINE_CONFIG_PATH`. That path and
the service details are not repository artifacts. Until the runner, attested
old baseline, and provisioner actually exist, leave
`DECISION_RUNTIME_RELEASE_ENABLED` unset. A missing or stale baseline blocks
the qualified distribution, and Decision-enabled stable tags cannot fall back
to the ordinary prebuilt wheel.

GPU exclusivity is a protected runner prerequisite: the runner operator must
reserve the selected GPU and ensure no unrelated compute is active during the
timed windows. This producer verifies device visibility and live containers,
but does not enumerate all GPU processes. The protected file may declare
`gpu_clock_policy` as `unobserved`, `protected_fixed`, or `default_dynamic`;
the report labels it as a protected policy, not a measured clock trace. Both
service containers remain running through each model's alternating waves;
their actual HBM residency is not observed.

`c1`, `c8`, and `c32` mean maximum client in-flight HTTP requests per arm.
Each old wave submits one HTTP call per state; each new wave submits one batch
call per logical workflow. The same 32 logical workflows and case order are
used per arm per round, but submissions and server admission are not paired
across arms. Every timed sample carries a `logical_schedule_sha256` over the
ordered logical case IDs, shape, concurrency, phase, and round; the gate
recomputes it from paired workflow records. This identifies equal logical
work, not a common scheduled-arrival epoch or matching admission times. Until
the harness can provide such a shared timed-arrival trace, the release gate
accepts only an equal-total-work high-load decisions/sec gain; it does not
accept or report p95 latency improvement as an alternative win. Three
throughput rounds alternate old/new wave order. The new
runtime must attest scheduler concurrency 4 and queue 32; the physical batch
size is recorded per model from the running launch. The gate records this arrival scope and checks the resulting raw
workflow intervals, successes, throughput, and physical batch counters.
Each new-arm throughput round must account for one row preparation per
successful workflow and one physical row per successful decision. The gate
checks the configured per-model batch capacity against every round, requires
the full cumulative physical-batch histogram to count exactly the observed
forwards, and rejects a histogram whose bucket distribution cannot account
for those rows. It also sweeps the raw HTTP sample intervals separately for
each arm, phase, round, and cell: peak in-flight calls must not exceed the
declared c1/c8/c32 client concurrency. A hashed sample file alone does not
attest the client arrival or GPU scheduler behavior; these reconciliations
only rule out contradictory report evidence.
