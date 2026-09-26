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

Each release-owning family has a focused directory with a README,
machine-readable configuration, explicit data/output paths, train and export
entrypoints, and lightweight contract tests. Model weights, datasets, caches,
checkpoints, and run logs stay outside Git.

Start with the public [model training guide](../../website/docs/training/training-overview.md)
to choose a model family. Use the README beside a trainer for its environment,
data preparation, commands, evaluation, and artifact format.

## Training control-plane contract

`semantic-router.training/v1` defines shared resources and messages for selector
training and neural fine-tuning. This is a contract for future management and
worker implementations; HTTP routes, persistence, scheduling and Console training
flows are not implemented by this package.

The canonical [Go contract](../semantic-router/pkg/trainingcontract/) generates
[JSON Schema](../semantic-router/pkg/trainingcontract/training-v1.schema.json) and
[Console types](../../dashboard/frontend/src/generated/trainingContract.ts).
[Python validation](control_plane/contracts.py) consumes that schema directly.
The [OpenAPI contract](../semantic-router/pkg/trainingcontract/training-v1.openapi.yaml)
defines management operations, ownership, output discovery, submission idempotency,
cancellation and retry semantics.

`APIError.code` is an open string. New codes may be added within v1 without a
contract-version bump; existing codes retain their meanings. Clients must accept
unknown codes and handle them as generic errors using HTTP status and `message`.
The OpenAPI error response lists the well-known codes and their HTTP statuses.

Starting from a run ID, clients follow `RunGraph.outputs` to artifacts, evaluations
and qualifications. Artifact variants map logical relative file names to owned
file handles, so clients can reconstruct a model's file layout. Qualification
evidence supplies the variant and qualification IDs for a binding proposal.
The OpenAPI operation descriptions specify this flow; clients need no fixture IDs
or worker messages to discover outputs.

### Profiles and provenance

Profiles describe the dataset and output shape; `target_contract` selects the
selector, label-scores or spans profile. These shapes are distinct, and declaring
one does not imply that a worker supports it. The optional pinned `base_model`
belongs to the frozen run spec, allowing different base models to use one snapshot.
Trainer-specific parameter validation belongs to the capability planner.

Go management owns resource identity, ownership, profile compatibility and state
transitions. Python validates message structure. The provenance envelope links
outputs to their run, task, attempt, snapshot and trainer; its optional
`manifest_bundle` preserves the existing
[Router Model provenance contract](model_eval/provenance/README.md).
Before publishing classifier qualification evidence,
[validate_classifier_provenance](control_plane/provenance.py) validates that bundle
and binds its label mapping and base model to the profile and frozen run spec.
Structural acceptance of a worker result alone does not qualify an artifact.

### Worker lifecycle

Workers receive frozen inputs, resolved input variants and run/task/attempt IDs.
They execute tasks; Go management schedules dependencies and derives run status.
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

### Generate and verify

From the repository root, run `make training-contract-generate` after changing Go
types, then `make training-contract-check` to check generated drift, Go semantics,
Python schema validation and provenance compatibility. In `dashboard/frontend`,
run `npx vitest run src/utils/trainingContract.test.ts` for Console consumption.

The shared [selector and neural fixtures](../semantic-router/pkg/trainingcontract/testdata/)
exercise management/worker exchanges and output discovery across Go, Python and
TypeScript. Their IDs and digests are examples, not downloadable model artifacts.
These checks need no trainer, GPU or running management service; they do not
replace HTTP integration tests when handlers are implemented.
