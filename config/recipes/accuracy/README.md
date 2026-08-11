# Accuracy routing recipe

This recipe uses orchestration as a bounded accuracy tool, not as the default
for every prompt.

## Policy

- `accuracy_workflow` uses a dynamic micro-agent workflow for explicit
  evidence gathering, tool use, decomposition, and verification tasks.
- `accuracy_deliberation` fuses independent frontier responses for adversarial
  review and competing-hypothesis judgments.
- `accuracy_long_context_direct` sends long context to one 1M-token worker,
  avoiding the latency and token multiplier of fan-out.
- `accuracy_direct` is the single-worker default.

Priority is intentional: explicit Workflow requests win over long context;
long context wins over Deliberation; Deliberation wins over the direct
fallback. The maintained probes include both priority collisions.

Per-request fan-out is capped at three workers. The planner is capped at 2,048
completion tokens, each Workflow request at four steps and 8,192 completion
tokens. Workflow and Fusion require two of three successful workers and use
`on_error: skip`, so one failed worker degrades the panel instead of failing
the whole request. Fusion keeps reasoning enabled
for panel workers but disables it for the coordinator's structured judge calls,
preventing reasoning-only completions from producing an empty synthesis.

The maintained OpenRouter worker IDs are
`anthropic/claude-opus-4.8`, `google/gemini-3.1-pro-preview`, and
`openai/gpt-5.5`. Contract tests pin those IDs and the orchestration bounds.

## Validate

```bash
vllm-sr validate --config config/recipes/accuracy/config.yaml
(cd src/semantic-router && go run ./cmd/dsl validate ../../config/recipes/accuracy/recipe.dsl)
python tools/agent/scripts/router_calibration_loop.py \
  eval \
  --router-url http://127.0.0.1:8080 \
  --probes config/recipes/accuracy/probes.yaml
```

The Eval suite proves routing without invoking OpenRouter or a local planner.
End-to-end generation still requires the backend credentials documented in
`config.yaml`.
