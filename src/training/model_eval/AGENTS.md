# Training Model Eval Notes

## Scope

- `src/training/model_eval/**`

## Responsibilities

- Keep evaluation artifact parsing, contract defaults, and config-generation helpers on separate seams.
- Treat result-to-config translation as a narrow adapter onto the shared training-artifact contract, not as a second owner for runtime config defaults.
- Keep evaluation scripts separate from dashboard or router runtime ownership.

## Change Rules

- `result_to_config.py` is a ratcheted hotspot. New artifact parsing, default resolution, or config-shaping helpers should move into adjacent modules instead of widening the script entrypoint.
- Do not duplicate training-artifact path or default inventories that already live in shared contract owners.
- If a change touches both evaluation artifacts and runtime config contracts, keep the contract source of truth in the shared seam and make this subtree a thin translator.
