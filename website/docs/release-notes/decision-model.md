# Release note: choose the decision model

The first release that includes
[#4719](https://github.com/vllm-project/semantic-router/issues/4719) lets you
choose the Router's decision model: the Vela model that answers every built-in
signal and every `decision` question that names no `deployment`, in one call
per request.

```bash
vllm-sr serve --decision-model Vela-2.0-9B --platform amd
```

```yaml
global:
  model_catalog:
    system:
      decision_model: Vela-2.0-9B
```

## What changes

- **One field:** `global.model_catalog.system.decision_model` takes
  `Vela-2.0-0.3B` (the default on every platform), `Vela-2.0-0.8B`,
  `Vela-2.0-4B`, `Vela-2.0-9B` or `Vela-1.0`, in any case. Every other name is
  an error, with its own message for a Decision 2.0 model.
- **`vllm-sr serve --decision-model`** (docker and kubernetes targets) writes
  the field into the active configuration as a new version, then serves; later
  starts keep it. `vllm-sr status` shows it and `vllm-sr config validate`
  checks it. `vllm-sr serve MODEL` (engine mode) refuses the flag.
- **Helm** has a `decisionModel` value and the **operator** a
  `spec.config.decision_model` field. The Dashboard's setup and its Router
  settings have a selector with each size's hardware.
- **Decision questions:** a `routing.signals.decision` question without a
  `deployment` asks the decision model, in the same call as the built-in
  signals. With `Vela-1.0` it is a load error that asks for a `deployment`.
- **Decision selectors:** an `algorithm.decision` without a `deployment` asks
  the same deployment which of the decision's `modelRefs` answers, so one
  model answers the signals and chooses the model. `Vela-1.0` again asks for a
  `deployment`.
- **Sizes:** the 0.8B, 4B and 9B are built-in models at the revisions the
  model runtime pins. The 4B and 9B need a GPU: `vllm-sr serve` refuses them on
  `--platform cpu` or on a host without the GPU, and the Router where the
  model runtime finds none.
- **Thresholds:** each size has module thresholds calibrated to keep the Vela
  1.0 specialists' operating points. A module that sets none takes those of the
  model it runs, so switching size switches them.

## Measured

Through the Router on the router signal suite, against the Vela 1.0
specialists
([record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-decision-model-sizes.md)):

| Decision model | Held-out domain accuracy | Held-out prompt guard AUC | p50 on a GPU | p50 on 12 CPU cores |
| --- | ---: | ---: | ---: | ---: |
| `Vela-2.0-0.3B` | −0.037 | +0.026 | 6.9 ms | 79 ms |
| `Vela-2.0-0.8B` | +0.063 | +0.067 | 40.7 ms | about 3 s |
| `Vela-2.0-4B` | +0.122 | +0.098 | 56.8 ms | GPU only |
| `Vela-2.0-9B` | +0.138 | +0.097 | 79.0 ms | GPU only |

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
