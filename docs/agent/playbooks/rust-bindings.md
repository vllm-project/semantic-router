# Rust Bindings Playbook

- Touch `candle-binding/`, `ml-binding/`, or `nlp-binding/` independently when possible
- Keep FFI boundaries thin and documented
- Run the agent fast gate first, then binding-specific feature tests
- Avoid leaking training-only or E2E-only concerns into runtime crates
