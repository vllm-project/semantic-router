# Rust Bindings Playbook

- Touch `candle-binding/`, `ml-binding/`, or `nlp-binding/` independently when possible
- Keep FFI boundaries thin and documented
- Run the agent fast gate first, then binding-specific feature tests
- `make agent-lint` now treats Rust Clippy as diff-scoped for changed `.rs` spans, while crate-wide correctness still comes from `cargo check`, `make test-binding-minimal`, and any affected feature tests
- Avoid leaking training-only or E2E-only concerns into runtime crates
