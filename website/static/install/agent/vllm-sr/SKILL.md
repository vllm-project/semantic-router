---
name: vllm-sr
description: Install, configure, verify, and improve vLLM Semantic Router through its CLI and Router API. Use for deployment, recipe tuning, and single-model/MoM evaluation, with Dashboard verification when requested.
---

# vLLM Semantic Router operations

Work against the user's selected stack and objective. Inspect the installed CLI,
running configuration and available backends before choosing an approach.
Preserve unrelated workloads, credentials and the user's existing authorization.

## Discover the contract

Use `vllm-sr --help`, command-specific help and `vllm-sr config schema`.
For a running Router, `GET /api/v1` advertises its operations and schemas.
Management origin, inference listener and public model entrypoint are separate;
discover them instead of assuming default ports or a recipe name.

Read only the reference needed for the task:

| Task | Reference |
| --- | --- |
| Install, select hardware/runtime, isolate a stack, open Dashboard access | [Deployment](https://vllm-sr.ai/install/agent/vllm-sr/references/deployment-loop.md) |
| Change live config, activate a recipe, recover a revision | [Configuration](https://vllm-sr.ai/install/agent/vllm-sr/references/configuration-loop.md) |
| Verify routing, tools, context boundaries or delivery | [Route verification](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md) |
| Improve signal, decision or model-selection policy | [Recipe tuning](https://vllm-sr.ai/install/agent/vllm-sr/references/recipe-tuning.md) |
| Compare single models and MoM; run a measured optimization loop | [sr-bench](https://vllm-sr.ai/install/agent/vllm-sr/references/sr-bench.md) |

For installation or an authorized upgrade, default to the published dev package
unless the user selects another version:

```bash
curl -fsSL https://vllm-sr.ai/install.sh | \
  bash -s -- --channel dev --mode cli --runtime skip --no-launch
export PATH="$HOME/.local/bin:$PATH"
vllm-sr --version
```

## Work loop

- Establish the intended behavior and a small reproducible baseline.
- For a new stack, initialize and validate config before `serve`. For an existing
  stack, derive changes from fresh `config get`, then validate, plan and apply.
  Respect restart-required changes and verify the active revision afterward.
- Preview checks routing without generating an answer. Probe or live evaluation
  checks actual delivery. Verify the behavior affected by the change, including
  final output; readiness or HTTP 200 alone is insufficient.
- Compare the same workload before and after a coherent change. Use sr-bench
  when capability, cost or latency is the objective. Preserve unsuccessful
  attempts and distinguish small-sample evidence from a quality claim.

For Dashboard work, exercise the corresponding user flow against the same stack
and inspect the resulting artifacts. Leave the user with the active config or
recipe, access details, evidence and material limitations. Keep secret values and
private request content out of public artifacts.
