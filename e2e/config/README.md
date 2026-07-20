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
providers that accept bearer authentication, change
`global.model_catalog.embeddings.semantic.endpoint.base_url` to
`https://<resource>.openai.azure.com/openai/v1` before running the smoke. Endpoints
that require the legacy `api-key` header are not supported by the current adapter.

Run it against the current local image with:

```bash
make vllm-sr-dev
OPENAI_API_KEY="<provider-key>" vllm-sr serve \
  --config e2e/config/config.remote-embedding-smoke.yaml \
  --image-pull-policy never \
  --minimal
curl -fsS http://localhost:8080/startup-status | jq '.embedding_provider'
```

The automated `remote-embedding` profile uses an in-cluster deterministic mock
provider and does not require a real API key:

```bash
make e2e-test E2E_PROFILE=remote-embedding
```

These files are intentionally outside `config/` so the public config directory stays limited to canonical user-facing config plus routing fragments.
