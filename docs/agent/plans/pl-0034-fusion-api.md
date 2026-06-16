# PL-0034 Fusion API

## Goal

Implement Support Fusion API for multi-model deliberation (#2193) end to end, then validate the final patch with vLLM Semantic Router AMD regression.

## Scope

- Add a Fusion-style multi-model deliberation API.
- Support direct invocation with a configured Fusion model slug.
- Support request-level `plugins[].id == "fusion"` overrides for judge and analysis models.
- Keep the implementation aligned with the router config, looper execution, Python CLI schema, docs, examples, and harness validation contracts.
- Preserve the existing auto model routing path and avoid treating Fusion slugs as normal explicit model aliases.

## Exit Criteria

- `model: "vllm-sr/fusion"` works by executing a configured Fusion decision.
- `model: "vllm-sr/fusion"` limits decision matching to Fusion-capable decisions; it does not fall back to normal non-Fusion routes.
- `model: "vllm-sr/auto"` is supported as a namespaced auto-routing alias alongside legacy `auto` and `MoM`.
- `model: "openrouter/fusion"` is supported only as an opt-in compatibility alias when configured.
- Request-level Fusion plugin overrides can set `model` and `analysis_models`.
- Panel model calls run concurrently and degrade on partial failure.
- All-panel failure returns a typed error.
- Judge parse failure falls back to raw panel synthesis.
- Fusion execution is guarded against recursive Fusion calls.
- Config catalog, reference config, docs, and CLI schema are synchronized.
- `global.integrations.looper.fusion` registers direct slugs only; route selection, judge, and analysis panel config remain under `routing.decisions[].algorithm.fusion`.
- `global.router.auto_model_names` registers auto-routing aliases only; it does not own route policy or algorithm defaults.
- Targeted unit tests and harness gates pass.
- AMD regression is run and the validation evidence is recorded in the final report.

## Task List

- [x] Inspect current looper, config, extproc, CLI, and docs contracts.
- [x] Add Fusion config and catalog surface.
- [x] Add request plugin parsing and direct slug routing.
- [x] Implement Fusion looper execution and response formatting.
- [x] Add targeted Go and Python tests.
- [x] Update reference config, fragments, and docs.
- [x] Run local harness validation and fix failures.
- [x] Run vLLM Semantic Router AMD regression.

## Next Action

Draft the companion website blog.

## Validation Notes

- Local harness gates passed for config, looper, extproc, Python CLI schema, dashboard type/unit checks, `make agent-validate`, `make vllm-sr-test`, `make test-semantic-router`, `make dashboard-check`, `make agent-lint`, and `make agent-ci-gate`.
- Local Docker integration was blocked by the local Docker daemon being unavailable; runtime coverage came from the AMD regression.
- Final AMD regression used router image tag `amd-regression-d5edebae-fusion`, installed the CLI from the same ref, validated the config, and started the router through the AMD local serve path.
- The AMD OpenRouter regression configured three external model providers as Fusion panel backends: `google/gemini-3-flash-preview`, `moonshotai/kimi-k2.6`, and `deepseek/deepseek-v4-pro`. The judge model was configured per Fusion decision, not under global runtime config.
- The AMD smoke covered direct `model: "vllm-sr/fusion"`, opt-in direct `model: "openrouter/fusion"` with a request-level `plugins[].id == "fusion"` override, and normal `model: "auto"` routing into a Fusion decision.
- The final AMD smoke completed 3/3 Fusion requests with assistant content, 3/3 structured analysis parses, 0 failed panel calls, and 15 upstream OpenAI-compatible model calls across panel, judge, and final synthesis paths. The matching single-model baseline over the same three prompts returned content in 4/9 calls, with the remaining calls producing reasoning-only payloads.

## Operating Rules

- Keep `processor_req_body.go` and `config.go` as orchestration/schema entrypoints only; place new helpers in adjacent focused files.
- Treat request-level Fusion plugins as OpenAI-compatible request extensions, not as decision plugins.
- Use the existing looper internal-request headers for plugin recursion safety and add Fusion-specific recursion protection where needed.
- Do not publish private AMD host details in public artifacts.

## Related Docs

- [Issue #2193](https://github.com/vllm-project/semantic-router/issues/2193)
- [docs/agent/module-boundaries.md](../module-boundaries.md)
- [docs/agent/change-surfaces.md](../change-surfaces.md)
- [docs/agent/amd-local.md](../amd-local.md)
