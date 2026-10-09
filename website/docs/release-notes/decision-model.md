# Release note: choose the decision model

The Router selects its default judgment model through an exact deployment
reference. The default remains Vela 2.0 0.3B on CPU. Decision 1.0 and Decision
2.0 deployments can also answer routing tasks according to their native
question capabilities.

```yaml
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/Vela-2.0-9B
        device: rocm
    system:
      decision_model:
        deployment: primary
```

```bash
vllm-sr serve --platform rocm
```

## What changes

- The binding contains only `deployment`; model identity, device, profile and
  endpoint belong to `global.model_catalog.deployments`. Keys are exact and
  case-sensitive. Scalar model names are rejected.
- `serve MODEL` changes the configured default deployment's artifact, preserving
  its placement and profile unless explicit options override them. Select a
  different deployment key in the canonical binding. Status reports that key.
- Questions and selectors with no deployment use this default. Explicit task
  and question bindings override it without replacing the resource.
- Available tasks come from native model capabilities. Family names do not
  decide whether a model may perform routing judgments.
- The Dashboard separates the active resource, pending selection, task
  capability and deployment readiness. Model evaluation quality remains
  separate from runtime readiness.

## Measured

Through the Router on the router signal suite, against the Vela 1.0
specialists
([record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-decision-model-sizes.md)):

| Decision model | Held-out domain accuracy | Held-out prompt guard AUC | p50 on a GPU | p50 on 12 CPU cores |
| --- | ---: | ---: | ---: | ---: |
| `Vela-2.0-0.3B` | −0.037 | +0.026 | 6.6 ms | 79 ms |
| `Vela-2.0-0.8B` | +0.063 | +0.067 | 40.1 ms | about 3 s |
| `Vela-2.0-4B` | +0.122 | +0.098 | 55.2 ms | GPU only |
| `Vela-2.0-9B` | +0.138 | +0.097 | 76.5 ms | GPU only |

The 4B and 9B are ahead of Vela 1.0 on almost every signal. Every size is
behind on user feedback's fresh file and on PII in distribution.

## What you may need to do

- **Recipes that set rule thresholds** (`routing.signals.jailbreak[].threshold`
  and the like) keep them when you switch size; the built-in recipes set the
  0.3B's. The record maps each to every size.
- **`config/config.yaml`** now names `decision_model`. It no longer pins each
  signal to the 0.3B with a `system.<module>` line, a module `model_id` or
  `classifier.model_path`, or the 0.3B's module thresholds, any of which would
  keep that signal on the 0.3B whatever the decision model. A configuration
  that copied those lines keeps those signals on the 0.3B until it removes
  them.
- **The operator** no longer defaults `prompt_guard.model_id` and `threshold`,
  so the guard follows the decision model. A `SemanticRouter` created earlier
  keeps the values it stored.
- **The ROCm image** runs the 0.8B on a GPU only; the CPU image runs it on a
  CPU.
