---
name: vllm-sr-agent-operations
description: Operate a vLLM Semantic Router directly through its CLI and HTTP contracts. Use when an agent needs to install or deploy vLLM-SR, author or change canonical YAML configuration and Recipes, validate and plan changes, test routing with preview and real end-to-end requests, run the Intelligence 1.0 benchmark suite, analyze replay evidence, or continuously optimize a physical or virtual model pool. Dashboard is never required.
---

# vLLM-SR agent operations

Treat the running Router as the authority for the version it supports. Use YAML
as the reviewable source, the management API or CLI as transport, and immutable
receipts as evidence. Do not invent a second DSL or scrape Dashboard state.

## Discover before acting

1. Check `GET /health` and `GET /ready` on the management listener.
2. Fetch `GET /api/v1?audience=agent&visibility=primary` for the compact
   capability directory.
3. Narrow discovery with `?capability=<name>` or fetch filtered OpenAPI from
   `/openapi.json?capability=<name>&audience=agent`.
4. Discover configuration progressively:
   - `vllm-sr config schema --endpoint <management-origin>`
   - add `--section <path>` or `--surface <kind:name>` for one contract;
   - use `--full` only when a complete schema is necessary.

The management origin normally serves `/api/v1/**`, `/openapi.json`, and
`/docs`. The routed inference origin separately serves OpenAI-compatible
requests such as `/v1/chat/completions`. Never infer one port from the other.

## Use the safe control loop

For configuration changes, follow this exact order:

1. Edit canonical YAML locally.
2. Run local validation, then authoritative Router validation.
3. Plan against current state without writing.
4. Apply with the ETag returned by the plan.
5. Confirm readiness and active configuration.
6. Preview representative routing cases without model calls.
7. Probe the Envoy-routed endpoint with real model calls and assertions.
8. Run the appropriate routing workload or full model benchmark.
9. Review replays, outcomes, latency, token use, cost, and failures.
10. Keep the change only when its declared gate passes; otherwise apply the
    previous version or use config rollback.

Read [configuration-loop.md](references/configuration-loop.md) when changing
configuration or Recipes. Read [evaluation-loop.md](references/evaluation-loop.md)
when testing or optimizing routing. Read
[deployment-loop.md](references/deployment-loop.md) for installation, serving,
or model-pool changes.

## Evidence boundaries

- `vllm-sr route preview` evaluates signals and a decision without calling a
  generation backend. Use it for fast route assertions.
- `vllm-sr route probe` sends one real request through the routed inference
  listener and emits a JSON receipt containing selected route headers, latency,
  response, and assertions.
- `vllm-sr benchmark` runs versioned routing workloads.
- `vllm-sr benchmark intelligence` plans or runs the six fixed Intelligence
  1.0 model benchmarks against any physical or virtual model ID. A virtual
  model is evaluated through its endpoint; never synthesize its score from
  member-model scores.

Route evidence and full task-quality evidence answer different questions. A
successful preview does not establish backend quality, and one successful
probe does not establish benchmark performance.

## Operational security and reproducibility

- Put credentials only in named environment variables. Pass the variable name,
  never its value, to `--token-env` or `--api-key-env`.
- Never put secrets in YAML, command arguments, logs, receipts, or prompts.
- Use exact runner and dataset revisions returned by
  `vllm-sr benchmark intelligence list`.
- Treat `--sample-limit` as smoke evidence only. It cannot enter the 1.0 index.
- HLE 1.0 is always the frozen 2,158-question text-only subset. Do not enable
  multimodal questions or substitute a rolling/verified subset.
- Preserve raw benchmark artifacts outside Git. Commit only intentional
  configuration, Recipe, documentation, or code changes.
