# Training control-plane contract

The `semantic-router.training/v1` contract describes the planned shared management
surface for selector training and neural fine-tuning. This change supplies resource
and message definitions, OpenAPI, and executable examples. It does not register
HTTP routes, persist runs, dispatch workers, or change the current ML Pipeline UI.

## Sources and consumers

- `src/semantic-router/pkg/trainingcontract/`: canonical Go resources, boundary
  validation, generated JSON Schema, and the authored OpenAPI operation contract.
- `dashboard/frontend/src/generated/trainingContract.ts`: generated Console types.
- `src/training/control_plane/contracts.py`: Python structural validation using the
  same JSON Schema, without a second set of request or status models.
- `src/semantic-router/pkg/trainingcontract/testdata/`: shared selector and neural
  examples covering submission, completed RunGraph, worker exchanges, artifacts,
  evaluation, qualification, and a binding proposal. IDs and digests are examples,
  not downloadable artifacts or evidence of a real training result.

Run `make training-contract-generate` after changing Go types.
`make training-contract-check` checks generated drift, Go semantics, Python schema
and provenance compatibility. The Console fixture test runs with
`npm run test:unit -- src/utils/trainingContract.test.ts` in `dashboard/frontend`.
No trainer, model download, GPU, or running management process is needed.

## Resources and ownership

| Resource | Meaning |
| --- | --- |
| DataAsset | Named data collection with a target contract |
| DataSnapshot | Immutable bytes, profile, source metadata and preprocessing |
| Experiment | Related runs targeting the same output contract |
| TrainingRun | Frozen snapshot, trainer version, parameters and task plan |
| RunTask / Attempt | Logical step / one execution of that step |
| RunGraph | Run and tasks with their complete attempt history |
| Artifact / ArtifactVariant | Logical output / particular format and files |
| Evaluation | Measurements referencing an existing variant and snapshot |
| Qualification | Runtime compatibility evidence referencing an existing variant |
| BindingProposal | Reviewable variant and qualification reference; no activation |

Go management assigns resource IDs and owns public state. A file reference carries
an opaque server-owned handle, a SHA-256 digest and its byte count. Upload requests
contain bytes; they never choose server paths. The later storage implementation
must resolve handles and enforce ownership; a syntactically valid handle is not
proof of access. Downloads use file handles, not worker-returned paths.

There is no snapshot update operation. Its ID permanently identifies the bytes,
profile, source metadata and preprocessing captured at creation. Different bytes
or metadata require a new snapshot. Runs freeze their spec when submitted;
changing a snapshot or trainer requires another run, not retry. Persistence and
cross-resource checks enforce these rules in subsequent changes.

Dataset profiles describe the data and output shape. The optional pinned
`base_model` belongs to the frozen run spec, so one classifier snapshot can be
used to compare different base models without creating another dataset identity.

`source` is optional for user uploads. When present, it describes a pinned
repository revision. `parameters` contains JSON values, including strings, arrays
and booleans. The capability planner owns trainer-specific parameter validation;
this package does not enumerate trainers or implement a parameter framework.

## Typed profiles and provenance

Profiles use `target_contract` as their discriminator:

| Target | Required profile |
| --- | --- |
| `selector.model-choice/v1` | Candidate models and observation field names |
| `signal.label-scores/v1` | Ordered label mapping |
| `signal.spans/v1` | Span labels and Unicode code-point offsets |

These are distinct profiles, not interchangeable dataset layouts. The selector
profile identifies the observation fields; detailed observation/objective/feature
semantics belong to the selector foundation. The span profile defines a wire
shape without claiming an executor supports it. The Go validation functions check
profile choice, class order and task dependencies at the management boundary.
Python checks message structure. Go remains authoritative for resource existence,
profile compatibility and state transitions.

The common provenance envelope links an artifact to its run, task, attempt,
snapshot and trainer. An optional `manifest_bundle` file reference attaches the
existing Router Model manifests. This envelope does not replace or relax those
manifests: license, splits, preprocessing, code revision, dependencies, seed,
metrics and artifact identity retain their existing validation requirements.

`validate_classifier_provenance` applies the existing full bundle validator and
binds its label mapping to the classifier profile and its base model to the run. A classifier
qualification must use that validation before publishing compatibility evidence;
merely accepting a `WorkerResult` structurally does not qualify an artifact.
Existing classifier workflows remain unchanged.

## Management API

`training-v1.openapi.yaml` in the contract package defines operations under
`/api/training/v1` and references the generated schema directly. This is the
contract for future handlers, not a claim these endpoints already exist.

- Upload bytes; create and retrieve assets, snapshots, experiments and proposals.
- Validate a run spec; submit a run; list an experiment's runs; retrieve RunGraph.
- Read durable events, request cancellation or retry, and compare runs.
- Retrieve artifacts, variants, evaluations, qualification evidence and file bytes.

Authenticated requests use existing Dashboard permissions and session/CSRF rules.
Bodies contain no owner field; management derives ownership from authentication.
All resource references must resolve within the caller's accessible resources.

Creation returns `201`; submission, cancellation and retry return `202` with the
current RunGraph. A validation request returns `200` with `valid: true` or a `400`
APIError. Errors carry `code` and `message`: invalid input is `400`, unauthenticated
is `401`, forbidden is `403`, missing resources are `404`, state or idempotency
conflicts are `409`, and unexpected server failures are `500`.

Submission idempotency keys are scoped to the authenticated owner. Repeating a
key with the same decoded spec returns the existing run; a different spec is a
conflict. Cancellation while cancelling/cancelled returns the current graph.
Retry is accepted only for failed/cancelled runs: keep the run/task IDs, frozen
spec and successful outputs, retain attempt history, and assign new attempt IDs
to work that needs another execution. Concurrent retries cannot create duplicate
attempts. Retry of an active or successful run is a conflict.

Events are ordered by sequence. `after` is an exclusive cursor scoped to the run;
`next_after` is the final returned sequence, or the input cursor for an empty page.
Comparison returns evaluations with their run identities, not an automatically
ranked winner. The service must reject incompatible target/profile comparisons;
metrics retain their method and snapshot references so consumers can judge whether
the measurements are comparable.

## Lifecycle and worker boundary

Run transitions are:

```text
pending -> running -> succeeded | failed
pending | running -> cancelling -> cancelled
failed | cancelled -> pending  (explicit retry)
```

`skipped` is a task-only dependency outcome. Run status is derived by management,
not copied from a worker. A cancellation request records intent; the run becomes
cancelled after executing tasks have settled. A completed task may retain its
successful result if completion raced with cancellation.

Workers receive a versioned execution request: run/task/attempt identities,
executor/trainer versions, JSON parameters, the frozen snapshot and resolved input
variants. They do not receive the whole graph or decide dependency scheduling.
Submit is idempotent by attempt ID. Inspect and cancel use that same attempt ID;
the worker's job handle is an opaque acknowledgement. A lost submit response must
not require a new attempt ID to discover the existing execution.

Workers report running, succeeded, failed or cancelled. Only successful results
may publish outputs. Go validates results and assigns output resource IDs, then
makes the task outcome and its artifacts visible together. Later evaluation and
qualification tasks reference the published variant IDs; they do not copy files
or recreate the training artifact. Final protocol-invalid results must become
failed attempts with diagnostics rather than remaining running indefinitely.

The later lifecycle implementation persists attempt identity before dispatch and
reconciles that identity after restart. These are protocol requirements, not a
scheduler, retry policy or persistence implementation in this package.
