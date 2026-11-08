# Release note: one catalog for the models the Router asks

Every model the Router asks, the models behind its signals, embeddings and
reranking, is now declared once, by name, under
`global.model_catalog.deployments`. Everything that uses a model writes that
name. `provider` says which API the Router speaks to each deployment:
`model_runtime`, `systemone`, `classify`, `openai_chat`, `openai_embeddings` or
`rerank`.

```yaml
global:
  model_catalog:
    deployments:
      local-vela: {provider: model_runtime, artifact: vllm-sr/vela3-fast, device: cpu}
      d3-remote:
        provider: systemone
        endpoint: https://app.vllm-sr.ai/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: VSR_APP_KEY}
        timeout_ms: 1500
    system:
      decision_model: {deployment: d3-remote}
    bindings:
      pii: local-vela
```

Here PII runs on a local vela3-fast, and every other question-style signal asks
the remote System One model.

## What changes

- **One rule, three levels.** A rule that asks its own question (`decision`,
  `classifiers`, `safety` and its `hazard` part, `complexity`, and the decision
  selector) names its `deployment`. A signal type whose rules share one answer
  (`domain`, `jailbreak`, `pii`, `fact_check`, `user_feedback`, `modality`,
  `hallucination`, `preference`, `reask`) is bound with
  `global.model_catalog.bindings.<type>`, or `routing.model_bindings.<type>` in
  a recipe. Everything else asks the decision model,
  `global.model_catalog.system.decision_model`. Embeddings, reranking and the
  hazard head are named where they are used. See
  [Choose a model for a signal](../installation/choose-a-model-for-a-signal.md).
- **One connection block.** An attached model runtime and every remote service
  share `endpoint` (a full URL), `model`, `auth` (`api_key_env` or
  `api_key_file`; a key is never written in the configuration), `headers`,
  `timeout_ms`, `max_retries`, `tls` and `limits`. Request backends in
  `providers.models` use the same `endpoint`, `auth` and `headers`, and a native
  System One alias can name a `systemone` deployment. See the
  [model catalog reference](../installation/model-catalog.md).
- **Checks at load.** An undeclared name, a deployment that cannot answer the
  signal it is given, a binding key that is not a signal type, and a deployment
  written at the wrong level are refused when the configuration loads, with the
  reason. Nothing falls back silently.
- **Thresholds follow the model that answers.** A module threshold the
  configuration does not write comes from the answering deployment: the
  calibrated value of a built-in model; 0.5 for a `systemone` model, which
  publishes none; otherwise the Vela 1.0 defaults.

## Behavior changes

- A `systemone` decision model now answers the built-in signals too. Before,
  they ran on a local `vela3-fast`. Their thresholds are 0.5 unless the
  configuration writes them, and the load warns `uncalibrated_preset`.
- A safety rule's hazard part without its own model uses the pre-declared
  `hazard` deployment (Vela 1.0 Hazard) under every decision model. Before, a
  configuration with such a part failed to load under a `systemone` decision
  model.

Everything else resolves to the same models as before.

## Earlier configurations

The Router refuses a configuration written in the earlier forms, with a
one-line error for each earlier field that names what replaces it.
`vllm-sr config validate --config config.yaml` lists them, and
`vllm-sr config migrate --config config.yaml` rewrites them in the new form.
[Migrate model resources](../installation/model-resource-migration.md) shows
each earlier form next to the one that replaces it.
