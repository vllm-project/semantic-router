# Candle Binding Core Notes

## Scope

- `candle-binding/src/core/**`

## Responsibilities

- Keep config loading, tokenization, similarity helpers, and shared error handling on separate seams.
- Treat `config_loader.rs` as the compatibility and config-adapter hotspot, not as the home for unrelated inference behavior.
- Keep binding-core support code separate from higher-level runtime policy owned outside the Rust binding.

## Change Rules

- `config_loader.rs` is a ratcheted hotspot. New YAML walking, default resolution, or compatibility fallback logic should move into adjacent helpers instead of widening the one file.
- Do not mix tokenization, similarity math, and config compatibility in the same Rust module when a sibling file already owns the behavior.
- If a change touches both binding config loading and router-side contract defaults, keep the canonical contract owner outside this crate and let the binding stay a thin adapter.
