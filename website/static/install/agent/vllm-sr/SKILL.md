---
name: vllm-sr
description: Install, configure, validate, operate, evaluate, and improve vLLM Semantic Router through its CLI and Router API. Use when an agent should manage vLLM SR without depending on the Dashboard.
---

# vLLM Semantic Router

Use the CLI and Router API directly. The Dashboard is optional and must not be
a dependency of this workflow. The user's instructions and deployment
boundaries take precedence over this skill.

## Authoritative contracts

- Treat the installed CLI help, the running Router's discovery response, JSON
  Schema, and OpenAPI document as the current source of truth.
- Discover progressively. Start with `vllm-sr config schema`, then request only
  the relevant `--section` or `--surface`. Use `--full` only when the complete
  contract is required.
- Discover Router operations from `GET /api/v1`; fetch the full or filtered
  OpenAPI document from `/openapi.json` when request and response details are
  needed.
- Do not reuse remembered fields or endpoints when runtime discovery is
  available.

## Workflow

1. Clarify the requested outcome and inspect the host, existing installation,
   current config, model endpoints, container runtime, and accelerator with
   read-only commands.
2. If the CLI is missing, install the stable release without starting or
   changing a runtime yet:

   ```bash
   curl -fsSL https://vllm-sr.ai/install.sh | \
     bash -s -- --channel stable --mode cli --runtime skip --no-launch
   vllm-sr --version
   ```

3. Inspect `vllm-sr serve --help` and select the deployment path that matches
   the actual host. Do not assume a GPU platform. Do not expose a management
   listener publicly unless the user explicitly requests and secures it.
4. Discover the config surface before writing YAML:

   ```bash
   vllm-sr config schema
   vllm-sr config schema --section providers.models
   vllm-sr config schema --section routing.modelCards
   vllm-sr config schema --section routing.decisions.modelRefs
   ```

   Use `vllm-sr config schema --surface KIND:NAME` for a selected signal,
   projection, algorithm, or plugin. Query the narrowest path first; a broad
   section can still be large because it contains every valid nested choice.
5. Start from the running configuration when one exists by reading
   `vllm-sr config get`; otherwise create `config.yaml` with
   `vllm-sr config init`. Preserve fields outside the requested change.
   A physical model must be present in `providers.models`, represented by a
   matching `routing.modelCards` entry, and referenced from the applicable
   `routing.decisions[].modelRefs` before it can receive routed traffic. Keep
   credentials in environment variables and store only environment references
   in the config.
6. Validate locally, then ask the running Router to plan the exact mutation:

   ```bash
   vllm-sr config validate --config config.yaml
   vllm-sr config plan --config config.yaml
   ```

7. Review the plan. Apply only when it matches the user's requested scope:

   ```bash
   vllm-sr config apply --config config.yaml
   ```

8. Check routing separately from backend execution:

   ```bash
   vllm-sr route preview \
     --model vllm-sr/auto \
     --prompt 'Explain why this request should take this route.' \
     --trace --json

   vllm-sr route probe \
     --config config.yaml \
     --base-url http://localhost:8899/v1 \
     --model vllm-sr/auto \
     --prompt 'Return exactly: route-ok'
   ```

   Preview proves the decision path without invoking a model. Probe sends a
   real request through Envoy and records end-to-end evidence. `--base-url`
   accepts either the listener origin or its OpenAI `/v1` root. Use explicit
   `--expect-*` assertions when the intended recipe, decision, algorithm, or
   model is known.
9. For optimization, capture a baseline, make one coherent recipe change,
   validate and plan it, run representative previews and probes, and compare
   the requested quality, cost, latency, or safety objective. Keep a change
   only when the evidence improves the objective without violating hard
   constraints.
10. Run the reproducible benchmark workflow only when the user asks for model
    or Mixture-of-Models evaluation. Begin with
    `vllm-sr benchmark intelligence plan --help`; keep benchmark revisions,
    commands, results, and runtime identity together.

## Boundaries

- Never print, commit, or place secret values in command arguments or YAML.
- Ask before privileged actions, destructive changes, public exposure, or
  stopping unrelated services. Resolve exact container and file targets first.
- Preserve existing user configuration and unrelated workloads.
- Do not treat routing preview as model-quality evidence or an end-to-end probe
  as proof that every routing branch is correct; use both where appropriate.
- Leave the user with the config path, active revision, validation result,
  routing evidence, and any remaining limitation.

See the [Router API](https://vllm-sr.ai/docs/api/router) and
[agent evaluation loop](https://vllm-sr.ai/docs/benchmarking/agent-evaluation-loop)
when the task needs the full contract.
