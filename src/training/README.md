# Model training

This tree is the canonical source for producing, evaluating, and packaging the
models maintained with Semantic Router. Public artifacts map to their local
owners in [`model_artifacts.json`](model_artifacts.json). The Vela collection
includes the shared Base, Embedding, Reranker, and eight classifiers. Historical
MOM artifacts retain their original training owners.

## Top-level ownership

| Directory | Responsibility |
| --- | --- |
| `control_plane/` | shared training-contract validation and classifier provenance binding |
| `model_classifier/` | intent, policy, feedback, modality, PII, and safety classifiers |
| `model_embeddings/` | text, reranking, and multimodal embedding families |
| `model_eval/` | cross-family evaluation utilities |
| `model_experiment/` | experiments that are not release owners |
| `model_selection/` | learned model-selection research |
| `kv_mapper/` | cross-model KV mapper artifacts, ridge fit and distillation (#2976) |

Each release-owning family has a focused directory with a README,
machine-readable configuration, explicit data/output paths, train and export
entrypoints, and lightweight contract tests. Model weights, datasets, caches,
checkpoints, and run logs stay outside Git.

Start with the public [model training guide](../../website/docs/training/training-overview.md)
to choose a model family. Use the README beside a trainer for its environment,
data preparation, commands, evaluation, and artifact format.

## Training control-plane contract

`semantic-router.training/v2` defines shared resources and messages for selector
training and neural fine-tuning. The contract package defines shared types;
[Go management](../../dashboard/backend/training/README.md) implements durable
resources, run/attempt state, events and the first management HTTP operations.
New runs remain `pending` until a coordinator dispatches work. Worker scheduling,
capability integration, qualification publication and Console training flows
remain follow-up work.

v2 replaced v1 when the Router's embedded Candle and ONNX Runtime runtimes were
removed. Classifiers qualify on the model runtime, and v1 documents are refused.

The canonical [Go contract](../semantic-router/pkg/trainingcontract/) generates
[JSON Schema](../semantic-router/pkg/trainingcontract/training-v2.schema.json) and
[Console types](../../dashboard/frontend/src/generated/trainingContract.ts).
[Python validation](control_plane/contracts.py) consumes that schema directly.
The [OpenAPI contract](../semantic-router/pkg/trainingcontract/training-v2.openapi.yaml)
defines management operations, ownership, output discovery, submission idempotency,
cancellation and retry semantics.

`APIError.code` is an open string. New codes may be added within v2 without a
contract-version bump; existing codes retain their meanings. Clients must accept
unknown codes and handle them as generic errors using HTTP status and `message`.
The OpenAPI error response lists the well-known codes and their HTTP statuses.

Starting from a run ID, clients follow `RunGraph.outputs` to artifacts and
evaluations. Artifact variants map logical relative file names to owned handles,
so clients can reconstruct a model's file layout. The contract also defines
qualification evidence and binding proposals; their management routes remain
planned. The OpenAPI operation descriptions specify output discovery.

### Profiles and provenance

Profiles describe the dataset and output shape; `target_contract` selects the
selector, label-scores or spans profile. These shapes are distinct, and declaring
one does not imply that a worker supports it. The optional pinned `base_model`
belongs to the frozen run spec, allowing different base models to use one snapshot.
Classifier label indices must be non-null integers forming a unique, contiguous
sequence starting at zero; Go and Python both reject `null` indices.
Trainer-specific parameter validation belongs to the capability planner.

Go management owns resource identity, ownership, profile compatibility and state
transitions. Python validates message structure. The provenance envelope links
outputs to their run, task, attempt, snapshot and trainer; its optional
`manifest_bundle` preserves the existing
[Router Model provenance contract](model_eval/provenance/README.md).
Before publishing classifier qualification evidence,
[validate_classifier_provenance](control_plane/provenance.py) validates that bundle
and binds its label mapping and base model to the profile and frozen run spec.
It also requires the variant's complete file names, digests and sizes to match an
evaluated artifact manifest, then verifies the actual bytes through a worker-owned
handle resolver. The resolver supplies immutable files for the variant being
qualified; local paths remain private to the adapter.
Structural acceptance of a worker result alone does not qualify an artifact.

### Worker lifecycle

Workers receive frozen inputs, resolved input variants and run/task/attempt IDs.
They execute tasks; Go management checks dependencies and derives run status.
The [worker integration boundary](../../dashboard/backend/training/README.md#connecting-a-worker)
defines dispatch, dependency-output resolution and restart recovery.
The allowed run transitions are:

```text
pending -> running -> succeeded | failed
pending | running -> cancelling -> cancelled
failed | cancelled -> pending  (explicit retry)
```

`skipped` is a task-only dependency outcome. Worker submit, inspect and cancel use
`attempt_id`; submit is idempotent by that ID, including after a lost response.
Management must persist attempt identity before dispatch and reconcile it after
restart. The worker job handle is an opaque acknowledgement.

Workers report running, succeeded, failed or cancelled. Only successful results
publish outputs. Management validates them, assigns resource IDs and publishes the
resources, run output IDs and task outcome together. Evaluation and qualification
reuse published variant IDs. A protocol-invalid final result becomes a failed
attempt with diagnostics.

### Capability catalog and planning

The capability layer resolves a requested training outcome across independently extensible
trainer, architecture, hardware, artifact, and runtime capabilities without central switch
statements or hard-coded UI enums. Capability IDs follow `<domain>/<name>@<version>`
(e.g. `trainer/hf-peft@v1`, `architecture/hf-modernbert@v1`, `hardware/rocm@v1`, `runtime/model-runtime@v1`).

Training hardware requirements remain distinct from inference qualification hardware requirements:
a neural model trained on CUDA/ROCm GPUs may be planned and qualified across multiple runtime targets
(such as `runtime/model-runtime@v1`, the model runtime's OpenAPI 2.x contract, on a CPU or an AMD GPU).
The built-in catalog qualifies ModernBERT label-score and span classifiers from Safetensors
checkpoints on the model runtime and selectors on the native runtime. When a registered runtime
accepts only another format, the planner schedules the conversion a registered rule provides.

Clients query `GET /capabilities` to discover supported descriptors and `POST /capabilities/plan`
to validate proposed combinations. Unsupported combinations return stable machine-readable reason
codes (such as `INCOMPATIBLE_HARDWARE`, `INCOMPATIBLE_ARCHITECTURE`, `UNSUPPORTED_TARGET`,
`MISSING_FORMAT_CONVERSION`) and actionable remediation messages.

### Generate and verify

From the repository root, run `make training-contract-generate` after changing Go
types or the default capability descriptors, then `make training-contract-check`
to check generated drift, Go semantics, Python schema validation and provenance
compatibility. In `dashboard/frontend`,
run `npx vitest run src/utils/trainingContract.test.ts` for Console consumption.

The shared [selector and neural fixtures](../semantic-router/pkg/trainingcontract/testdata/)
exercise management/worker exchanges and output discovery across Go, Python and
TypeScript. Their IDs and digests are examples, not downloadable model artifacts.
These checks need no trainer, GPU or running management service; they do not
replace the [management HTTP tests](../../dashboard/backend/router/training_routes_test.go),
which exercise authenticated API requests, restart recovery and mock-worker
publication. See the management README for their commands.
