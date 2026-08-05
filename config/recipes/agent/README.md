# Agent routing recipe

This recipe is the deployable agent use case previously named after its SAARS
Router Learning implementation. It combines privacy and jailbreak containment,
specialist domains, complexity projections, agentic structure, and replay data
collection across local and frontier model lanes.

## Policy

- Security and private-context decisions stay on the local AMD model.
- Simple math and general requests use the local fast path.
- Coding, STEM/research, legal/health, and business requests use dedicated
  specialist lanes.
- Complex non-domain work escalates to frontier models; the simple local lane
  is the single explicit fallback for otherwise unmatched traffic.
- Domain and general decisions include mutual-exclusion guards so priority is
  deterministic rather than relying on accidental overlap.

`router_replay` is enabled only on lanes intended to produce learning data.
The recipe remains runnable without training; future Router Learning updates
can recalibrate its projections and decision weights.

## Validate

```bash
vllm-sr validate --config config/recipes/agent/config.yaml
(cd src/semantic-router && go run ./cmd/dsl validate ../../config/recipes/agent/recipe.dsl)
python tools/agent/scripts/router_calibration_loop.py \
  eval \
  --router-url http://127.0.0.1:8080 \
  --probes config/recipes/agent/probes.yaml
```

Eval validates signal, projection, decision, plugin, and alias selection without
calling the five configured inference backends.
