# TD014: Candle Binding Crate-Wide Clippy Gate Blocks Diff-Scoped Validation

## Status

Closed

## Scope

candle-binding Rust lint gating versus diff-scoped PR validation

## Summary

The repository's canonical harness currently runs `cargo clippy --no-default-features --all-targets -- -D warnings ...` for the entire `candle-binding` crate whenever any Rust file under `candle-binding/src/**/*.rs` appears in the changed-file set. That means a narrowly scoped PR change such as `candle-binding/src/core/config_loader.rs` re-enters hundreds of pre-existing crate-wide Clippy failures that are unrelated to the touched code, so branch-level `make agent-lint` cannot distinguish new regressions from historical Rust debt.

## Evidence

- [candle-binding/src/core/config_loader.rs](../../../candle-binding/src/core/config_loader.rs)
- [candle-binding/src/core/unified_error.rs](../../../candle-binding/src/core/unified_error.rs)
- [candle-binding/src/ffi/mlp.rs](../../../candle-binding/src/ffi/mlp.rs)
- [candle-binding/Cargo.toml](../../../candle-binding/Cargo.toml)
- [tools/agent/scripts/agent_support.py](../../../tools/agent/scripts/agent_support.py)
- [tools/agent/repo-manifest.yaml](../../../tools/agent/repo-manifest.yaml)

## Why It Matters

- Narrow canonical-config fixes that legitimately touch Candle binding cannot get a truthful branch-level lint result because the harness collapses them into the crate's unrelated historical Clippy backlog.
- Reviewers and agents lose signal about whether a PR introduced a Rust regression or merely intersected an unretired crate-wide lint debt.
- The repo cannot honestly claim diff-scoped validation if one changed Rust file forces whole-crate warning cleanup with no baseline or ratchet.

## Desired End State

- Candle binding Rust validation either passes cleanly at the crate level or uses an explicit executable baseline/ratchet that isolates new violations from unrelated historical Clippy debt.
- PRs touching `candle-binding/src/core/config_loader.rs` can run canonical harness validation without being blocked by unrelated modules.

## Exit Criteria

- `make agent-lint CHANGED_FILES="...,candle-binding/src/core/config_loader.rs,..."` no longer fails because of unrelated pre-existing Clippy findings elsewhere in the crate.
- The executable rule layer documents how Candle binding Rust lint is scoped or ratcheted.
- If ratcheting is used, the repo records the baseline explicitly rather than treating whole-crate historical failures as PR-local regressions.

## Retirement Notes

- `tools/agent/scripts/agent_support.py` now runs Rust Clippy with JSON output and filters diagnostics to the changed Rust file set before deciding whether `agent-lint` fails.
- Branch-level `make agent-lint` continues to compile the crate, but pre-existing Clippy findings in untouched `candle-binding` modules no longer masquerade as PR-local regressions.
- `docs/agent/playbooks/rust-bindings.md` documents the new split: diff-scoped Clippy in `agent-lint`, crate-wide correctness in `cargo check` and the binding test targets.
