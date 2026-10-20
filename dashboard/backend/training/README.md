# Durable training management

This package persists training resources, run/task attempts, events and published
outputs. Dashboard exposes management operations under `/api/training/v2`,
independently of the legacy `ML_PIPELINE_ENABLED` flag. See the
[shared contract](../../../src/training/README.md#training-control-plane-contract)
for resource definitions and the
[OpenAPI reference](../../../src/semantic-router/pkg/trainingcontract/training-v2.openapi.yaml)
for routes, request limits, validation, idempotency, cancellation and retry.

New runs remain `pending` until a coordinator dispatches work. Capability planning,
worker scheduling and Console training flows are not connected yet. Qualification
publication requires a provenance-validating adapter; qualification and
binding-proposal HTTP routes are not registered.

## Persistence and access

SQLite metadata uses `DASHBOARD_WORKFLOW_DB_PATH` (`--workflow-db`), defaulting to
`./data/workflow.sqlite`. Uploaded and produced bytes live in `training-files/`
beside that database. Persist both locations across Dashboard restarts.

Management assigns resource IDs and derives ownership from the authenticated user.
Routes require `mlpipeline.manage` and use Dashboard audit, session and CSRF
policies. Workers receive owned handles; server filesystem paths stay private.
Uploads and worker outputs record SHA-256 digests and sizes. Published artifact
files must match the bytes registered for the producing attempt.

SQLite transactions use `BEGIN IMMEDIATE`. Submission identity, the run graph and
initial events commit together; output publication and task outcomes are also
atomic. Worker calls and file copying happen outside transactions. A crash during
upload can leave unreferenced bytes; automatic garbage collection is not provided.

## Connecting a worker

A coordinator integrates through these internal service methods:

1. `StartAttempt` checks dependency success and persists an attempt before returning
   its frozen `WorkerRequest`. Repeating it for an active task returns the same
   attempt ID; dispatch must be idempotent by that ID.
2. `Acknowledge` records the versioned worker acknowledgement. It cannot replace an
   existing worker handle with a different one.
3. `PutOutput` registers immutable bytes against an active, owned attempt. It checks
   the attempt again after copying, at the commit boundary.
4. `Complete` validates worker outputs and publishes resources, output indexes,
   task/attempt status and events together. Invalid results fail the attempt with
   diagnostics. Replaying an accepted final report preserves resource IDs.
5. `Recover` returns active requests, original attempt IDs, worker acknowledgements
   and cancellation intent. It does not create new identities, redispatch workers
   or infer job failure from management restart.

Worker inputs follow the outputs of direct dependency tasks: artifacts contribute
their published variants, and evaluations contribute only their explicit
`variant_id`. Inputs are deduplicated by variant ID and must be published in this
run and owned by the current user. Thus `train -> evaluate -> qualify` passes the
evaluated model to the third task. Evaluation results must reference a resolved
input. Restart recovery and retry use the same resolution.

Cancellation settles unstarted tasks immediately. Running tasks settle when the
coordinator reports their final outcomes. Retry retains successful work and attempt
history; it creates a new attempt only when unfinished work starts again.

## Validation

Run `make training-contract-check` from the repository root for shared Go/Python
contract checks and generated-artifact consistency.
From `dashboard/backend`:

```sh
go test -race ./training ./workflowstore
go test -race ./router -run '^TestTraining'
go vet ./training ./workflowstore ./handlers ./router
```

The tests use shared selector and neural fixtures, temporary stores and mock
worker results. They cover authenticated HTTP intake, concurrent submission,
restart recovery, cancellation/retry and owned artifact/evaluation publication.
They require no trainer, GPU or model downloads.
