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

Per-request fan-out is capped at three workers. Workflow steps and completion
tokens are also bounded so accuracy improvements cannot grow work without a
limit.

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
