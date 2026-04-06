# Training Model Classifier Notes

## Scope

- `src/training/model_classifier/**`

## Responsibilities

- Keep dataset verification, shared training utilities, and per-model-family training packages on separate seams.
- Treat the top-level verifier as a workflow entrypoint, not as the default home for dataset loading, judge parsing, correction export, and reporting logic.
- Keep individual classifier families isolated behind their own subdirectories and scripts.

## Change Rules

- `verify_text_classification_datasets.py` is a ratcheted hotspot. New CLI parsing, dataset loading, threaded judge execution, correction export, and reporting helpers should be split into adjacent modules instead of widening the one script.
- Do not mix family-specific fine-tuning logic back into shared top-level utilities when a subpackage already owns it.
- If a change touches both shared verification workflow and one classifier-family training package, separate the shared helper from the family-specific behavior first.
