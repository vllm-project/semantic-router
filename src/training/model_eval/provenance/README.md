# Router Model provenance manifests

One machine-readable contract for the four objects a built-in Router Model
passes through.

| Kind | Answers |
| --- | --- |
| `dataset` | which rows, at which revision, under which license, preprocessed how |
| `run` | which base model, code revision, dependencies, seed, and hyperparameters produced the artifact |
| `artifact` | which bytes exist, under which immutable identity, with which class order and runtime requirements |
| `evaluation` | which artifact was measured on which split, with which harness settings, and what it scored |

Each manifest is a YAML mapping validated against a JSON Schema in `schemas/`.

## Validating

```
python -m provenance.cli validate <directory>   # whole bundle, including references
python -m provenance.cli check <file> --kind artifact
```

Both exit non-zero on the first problem and print every problem found.

## What fails validation

Schema level:

- an unknown field
- a `schema_version` other than `v1`
- a revision that is not a 40-character commit sha, so a branch or tag cannot be
  recorded as if it were immutable
- a label mapping that does not cover `0..n-1` exactly once
- a composite dataset that does not pin the upstreams it samples from
- YAML anchors, aliases, tags, or merge keys, which can hide or duplicate
  provenance
- a credential, an absolute host path, or an over-long string that reads as
  embedded sample data

Cross-reference level:

- a `dataset_ref`, `artifact_ref`, or `run_ref` that resolves to nothing
- an `artifact_ref` whose revision or digest disagrees with the artifact manifest
- an `identity.digest` that does not match the file list it claims to summarise
- an artifact with no training run
- an evaluation that scored more rows than the split holds, or fewer with no
  declared `sample_limit`
- `metrics.per_label` that omits or invents a label
- **a label mapping that differs between any two manifests**

The last one is the reason the contract exists. A checkpoint whose class order
differs from the order the harness assumed still produces plausible accuracy, so
the mismatch is invisible in metrics and only a cross-reference catches it.

## Emitting

`emit.py` builds manifests from a running workflow. It reads the resolved
revision, hashes the artifact on disk, and records the installed dependency
versions; it raises rather than writing a placeholder when a fact is
unavailable.

- `src/training/model_classifier/prompt_guard_fine_tuning_lora/` emits the
  dataset, run, and artifact manifests at the end of training.
- `src/training/model_eval/quality_baseline.py` emits the dataset, artifact, and
  evaluation manifests, and can reference an artifact manifest a training run
  already published rather than minting a second identity for the same bytes.
