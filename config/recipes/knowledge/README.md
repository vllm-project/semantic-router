# Knowledge routing recipe

This use case routes knowledge questions using a versioned KB score rather than
hard-coding vendor or model-name policy. The included `mmlu_kb` data records
measured domain uplift between a local 7B lane and a frontier 72B lane.

## Policy

- `escalate_72b` directly matches the audited high-uplift labels: biology,
  business, computer science, economics, history, math, other, philosophy, and
  psychology.
- `keep_7b` handles lower-uplift domains through the `escalate_vs_keep` metric's
  `no_escalation` projection.
- This asymmetry is intentional: the selected `kb` label makes escalation
  explainable, while the metric provides a conservative local fallback.
- Small deterministic boundary guards cover terms such as matrix invertibility,
  blood cells, and legal consideration where adjacent MMLU labels can be
  semantically closer than the policy category. They correct known ambiguity;
  they do not replace the KB score.

MMLU is the seed dataset, not the use-case identity. Replace the built-in KB
rows with production benchmark or feedback evidence while retaining the same
policy shape.

## Assets and validation

- `config.yaml` is the runnable canonical configuration.
- `recipe.dsl` is the equivalent routing authoring surface.
- `probes.yaml` covers every escalation/local label family, including `other`,
  deterministic keyword overrides, and the per-label-versus-metric boundary.

```bash
vllm-sr validate --config config/recipes/knowledge/config.yaml

(cd src/semantic-router && \
  go run ./cmd/dsl validate ../../config/recipes/knowledge/recipe.dsl)

python tools/agent/scripts/router_calibration_loop.py \
  eval \
  --router-url http://127.0.0.1:8080 \
  --probes config/recipes/knowledge/probes.yaml
```

Eval does not call either backend model; it validates KB lookup, projection,
decision, and selected alias deterministically.
