# Remote Embedding Providers

## Overview

Semantic Router can send text embedding work to an external
OpenAI-compatible endpoint instead of loading a local embedding model. The
provider is configured once under `global.model_catalog.embeddings.semantic`;
embedding-backed signals and selectors keep the same route configuration.

Two fields have different roles:

- `embedding_config.model_type: remote` selects remote execution.
- `embedding_config.backend: openai_compatible` selects the API protocol.

## What Problem Does It Solve?

A deployment may already operate a managed embedding service or want embedding
capacity to scale independently from Router replicas. Remote execution avoids
loading the text embedding model in every Router process.

## Key Advantages

- Reuses a managed embedding service across Router consumers.
- Lets embedding capacity scale independently from Router replicas.
- Preserves route structure when changing the embedding execution backend.

## When to Use

Use a remote provider when an OpenAI-compatible embedding service is available
and the request text may cross that service boundary. Keep local execution for
offline deployments, data-locality requirements, or image embeddings.

## Configuration

Put the bearer token in the Router environment:

```bash
export EMBEDDING_API_KEY="<provider-key>"
```

Configure the shared embedding provider:

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          backend: openai_compatible
          model_type: remote
          preload_embeddings: false
          target_dimension: 1536
        endpoint:
          base_url: https://embedding.example.com/v1
          model: text-embedding-model
          api_key_env: EMBEDDING_API_KEY
          timeout_seconds: 10
          max_retries: 2
          dimensions: 1536
```

The Router appends `/embeddings` unless `base_url` already ends with that path.
When both dimensions are set, `endpoint.dimensions` and
`embedding_config.target_dimension` must match.

Embedding signals do not change:

```yaml
routing:
  signals:
    embeddings:
      - name: billing-support
        threshold: 0.72
        candidates:
          - billing invoice payment subscription refund
          - pricing renewal credit card receipt
```

## Consumers

The shared provider is used by text embedding consumers such as embedding,
complexity, reask, KB, contrastive jailbreak/preference, model selection, and
semantic tool filtering. Switching providers changes their embedding space even
when the output dimension stays the same.

## Limitations and Data Handling

- Remote image and audio embeddings are not supported.
- The adapter sends `Authorization: Bearer <token>` and does not support custom
  authentication headers or provider-specific request bodies.
- Request text sent for embedding is visible to the remote provider. Apply the
  provider's logging, residency, and retention policy to that data.
- Recalibrate signal thresholds after changing provider or model.
- Persistent response-cache, memory, RAG, and vector-store indexes must be
  rebuilt or migrated when their embedding model or dimension changes.

## Operations

Startup status exposes redacted provider metadata and health:

```bash
curl -fsS http://localhost:8080/startup-status | jq '.embedding_provider'
```

It reports the environment-variable name and whether it is set, never the key
value. The Dashboard exposes the same fields under **Global Config > Model
Catalog > Embedding Models**.

The maintained full example is
[`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml),
and the public endpoint fields are defined in
[`embedding_config.go`](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/config/embedding_config.go).
