# PL-0043: mmBERT-32K Safety Classifier Reconstruction

## Goal

Add a canonical, reproducible training and release workflow for the binary and
nine-class safety classifiers, aligned with the repository's other
`mmbert-32k-yarn` classifier artifacts.

## Scope

- Reconstruct the published safety tasks from their model cards and artifact
  metadata where original source is unavailable.
- Define a deterministic AEGIS plus synthetic-data preparation contract.
- Preserve the legacy nine-class output order through an explicit, reviewed
  taxonomy crosswalk.
- Train LoRA adapters from `llm-semantic-router/mmbert-32k-yarn` and export
  matching merged checkpoints.
- Evaluate, validate, document, and publish non-overwriting artifacts with
  pinned input revisions and machine-readable run manifests.

## Non-Goals

- Bit-for-bit reproduction of checkpoints whose original source, split,
  random seed, and optimizer state were not published.
- Overwriting or silently rebasing the existing safety adapters.
- Redesigning the public safety taxonomy in the same experiment as the base
  model migration.
- Committing datasets, model weights, checkpoints, caches, or private machine
  details to the repository.

## Exit Criteria

- The repository contains narrow data, taxonomy, training, evaluation, export,
  and release entrypoints with lightweight CPU tests.
- Base-model and dataset revisions, seeds, splits, sampling, label mapping,
  hyperparameters, and artifact schemas are explicit.
- Both safety tasks complete accelerator training and evaluation from the
  checked-in workflow.
- Each task produces a loadable LoRA adapter and merged model whose logits
  agree within the documented tolerance.
- New Hugging Face artifacts are published without replacing the historical
  models, and their model cards identify the reconstructed decisions.
- Repository harness gates pass for the complete change set.

## Task List

- [x] `SAFETY-01` Capture the reconstruction contract and pinned dependencies.
- [x] `SAFETY-02` Implement deterministic data preparation and taxonomy tests.
- [x] `SAFETY-03` Implement training, evaluation, merge, and release commands.
- [ ] `SAFETY-04` Pass local unit, smoke, lint, and harness validation.
- [ ] `SAFETY-05` Train and evaluate the binary classifier on AMD accelerators.
- [ ] `SAFETY-06` Train and evaluate the legacy nine-class classifier on AMD accelerators.
- [ ] `SAFETY-07` Validate adapter/merged parity and publish all four artifacts.
- [ ] `SAFETY-08` Update public documentation and close the execution plan.

## Next Action

Build the pinned ROCm environment on the selected accelerator host, then run
eight-device Level 1 and Level 2 smoke training before starting full runs.

## Operating Rules

- Keep source-data taxonomy separate from model-output taxonomy in code and
  manifests.
- Change one controlled variable in the first experiment: the base checkpoint.
- Record reconstruction decisions as decisions, not as recovered historical
  facts.
- Use immutable model and dataset revisions for every release-quality run.
- Publish adapters and merged checkpoints as distinct artifact repositories.
- Keep raw accelerator receipts and private infrastructure details in ignored
  local evidence only.

## Related Docs

- [Training stack change skill](../../skills/training-stack-change/SKILL.md)
- [Safety classifier training guide](../../../../website/docs/training/mmbert-safety-classifier.md)
