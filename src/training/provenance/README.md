# Router Model Provenance Manifests

Machine-readable provenance contract for Router Model training. Four JSON
manifests describe one training lineage, and each manifest's identity is the
SHA-256 of its canonical JSON.

| Manifest | File | Records |
|---|---|---|
| dataset | `dataset.json` | sources with revision and license, split sizes, preprocessing, label or span mapping |
| run | `run.json` | task, `dataset_id`, base model and revision, code revision, dependencies, seed, hyperparameters |
| evaluation | `evaluation.json` | `run_id`, `dataset_id`, split, sample count, command, metrics |
| artifact | `artifact.json` | `run_id`, `evaluation_id`, format, per-file digests, tree digest, label mapping |

Cross-references use manifest identities, so an edited upstream manifest
invalidates every manifest that points at it.

## Validate

```bash
cd src/training
python -m provenance validate-bundle <dir> --artifact-dir <exported model dir>
python -m provenance validate <manifest.json>
python -m provenance id <manifest.json>
```

Validation fails when a required field is empty, a source lacks a license,
a cross-reference points at a different manifest, artifact file digests differ
from the export directory, or a key or value looks like a secret.

## When to validate

| Lifecycle step | Call | Gate |
|---|---|---|
| training finishes | `validate_manifest(dataset)`, `validate_manifest(run)` | do not start evaluation on an incomplete run |
| evaluation finishes | `validate_manifest(evaluation)` | do not export without recorded metrics |
| export finishes, before upload | `validate-bundle <dir> --artifact-dir <export dir>` | do not publish an artifact whose digests or references disagree |
| runtime or CI loads an artifact | `validate-bundle <dir> --artifact-dir <downloaded dir>` | do not load bytes that differ from the manifest |

Publishing and loading are the two hard gates; the earlier calls fail fast so a
problem is caught before compute is spent.

## Emit

```python
from provenance import RunManifest, dump_manifest, manifest_id

run = RunManifest(name=..., task=..., dataset_id=manifest_id(dataset), ...)
dump_manifest(run, output_dir / "run.json")
```

Use `provenance.digest.tree_files` on the exported model directory to fill
`ArtifactManifest.files`, and `tree_digest` for `ArtifactManifest.digest`.

## Test

```bash
cd src/training && python -m pytest provenance/test_provenance.py
```
