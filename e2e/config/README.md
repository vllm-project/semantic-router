# E2E Config Manifests

This directory holds smoke, demo, and harness manifests that the repository uses for local validation and E2E flows.

- agent smoke configs
- authz and multi-provider demos
- remote embedding smoke config for manual OpenAI-compatible embedding-provider validation
- response API and hallucination test configs
- ONNX binding test configs

`config.remote-embedding-smoke.yaml` is a manual local smoke config for validating
`global.model_catalog.embeddings.semantic.embedding_config.backend: openai_compatible`
without changing the default `config/config.yaml`. It uses mock chat backends on
`host.docker.internal:18000` and `host.docker.internal:18001`, and reads the
embedding provider credential from `OPENAI_API_KEY`. For Azure OpenAI-compatible
providers, change `global.model_catalog.embeddings.semantic.endpoint.base_url` to
`https://<resource>.openai.azure.com/openai/v1` before running the smoke.

These files are intentionally outside `config/` so the public config directory stays limited to canonical user-facing config plus routing fragments.
