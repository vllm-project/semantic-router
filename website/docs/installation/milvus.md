---
title: Milvus
sidebar_label: Milvus
description: Use Milvus as the durable vector backend for response cache and other Router storage features.
---

# Milvus

Milvus is a distributed vector database that Semantic Router can use for a
durable response cache and other vector-backed features. Choose it when the
dataset must outgrow a single Router process, survive restarts, or be shared by
several Router replicas.

For a small local deployment, an in-memory cache is simpler. Valkey or Redis
may be a better fit when your team already operates their vector-search
extensions. Qdrant is another dedicated vector-store option. See
[Data and Storage](storage-overview) before selecting a backend.

## What this page configures

The examples below use Milvus for `global.stores.response_cache`. A decision
still needs a `response_cache` plugin before requests use that store.

Milvus can also back agentic memory or the general vector store, but those
features have separate schemas and retention requirements. Use distinct
collections when their data lifecycle differs.

## Prerequisites

- a Kubernetes cluster and Helm, or an existing reachable Milvus deployment
- persistent storage appropriate for your durability target
- private network access from every Router replica to Milvus gRPC (19530 by
  default)
- an embedding dimension that matches the Router's selected embedding model

## Deploy Milvus with Helm

The Milvus project maintains the Helm chart. The following starts a standalone
deployment suitable for development and evaluation:

```bash
helm repo add milvus https://zilliztech.github.io/milvus-helm/
helm repo update

helm upgrade --install milvus milvus/milvus \
  --namespace milvus \
  --create-namespace \
  --set cluster.enabled=false
```

Wait for the workload and inspect its Service:

```bash
kubectl get pods,service -n milvus
kubectl wait --for=condition=Ready pod \
  -l app.kubernetes.io/instance=milvus \
  -n milvus --timeout=10m
```

For production, use the topology, object storage, metadata store, persistence,
backup, and upgrade procedure documented by your Milvus distribution. Pin a
chart version and review its values rather than copying development defaults.

## Configure response cache

Use the canonical `response_cache` key. `semantic_cache` is a deprecated input
alias retained only for migration compatibility.

```yaml
global:
  stores:
    response_cache:
      enabled: true
      backend_type: milvus
      similarity_threshold: 0.86
      max_entries: 50000
      ttl_seconds: 7200
      embedding_model: mmbert
      milvus:
        connection:
          host: milvus.milvus.svc.cluster.local
          port: 19530
          database: default
          timeout: 30
        collection:
          name: semantic_router_response_cache
          description: Semantic Router response-cache vectors
          vector_field:
            name: embedding
            dimension: 768
            metric_type: COSINE
          index:
            type: HNSW
            params:
              M: 16
              efConstruction: 200
        search:
          params:
            ef: 64
          topk: 10
          consistency_level: Bounded
        development:
          drop_collection_on_startup: false
          auto_create_collection: true
          verbose_errors: false
```

Set `dimension` to the output dimension of `embedding_model`. A mismatch causes
inserts or searches to fail.

Enable the route plugin on decisions that may read or populate the cache:

```yaml
routing:
  decisions:
    - name: general-chat
      description: General requests that may use response cache.
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: local/general
      plugins:
        - type: response_cache
          configuration:
            enabled: true
            semantic:
              similarity_threshold: 0.86
```

Run configuration validation before rollout:

```bash
vllm-sr validate --config config.yaml
```

## Network and transport security

The current response-cache connector opens an unauthenticated, plaintext gRPC
connection using `connection.host` and `connection.port`. It does not apply
Milvus username/password or TLS settings, so do not add those fields expecting
the response-cache client to enforce them.

Keep this connection on a private network. In Kubernetes, use NetworkPolicy to
allow only Router workloads to reach the Milvus Service and deny unrelated
namespaces. Do not expose the Service publicly. If your environment requires
authenticated or end-to-end TLS database connections, use a backend whose
current Router integration supports that requirement, or place a reviewed
in-cluster transport proxy in front of Milvus and test the complete path before
production rollout.

## Verify behavior

After deployment:

1. confirm the Router becomes ready and logs a successful Milvus connection;
2. send a request through a decision with the `response_cache` plugin;
3. repeat an equivalent request and inspect cache metrics or routing metadata;
4. confirm that the expected collection exists and its vector dimension is
   correct; and
5. exercise the same path from every Router replica.

Do not use a fixed latency expectation as a health check. Lookup time depends on
network distance, index size, index parameters, consistency level, storage, and
hardware. Measure it with your dataset and deployment.

## Migrate from another cache

Response-cache entries are derived data, so the safest migration is usually to
start a new empty Milvus collection and allow it to warm:

1. deploy and secure Milvus;
2. add the Milvus config without removing the old deployment's rollback path;
3. validate and roll out to a small traffic slice;
4. monitor connection errors, cache hit rate, memory, and request latency;
5. expand the rollout; and
6. retire the previous cache after its rollback window expires.

If the collection contains durable memory or uploaded documents rather than
reconstructible cache entries, follow a data migration and backup procedure
specific to that feature. Do not treat those collections as disposable.

## Backup and retention

Define retention from the data being stored, not from Milvus alone. Response
cache may contain request-derived embeddings, metadata, or responses. Limit
access, set TTLs, and document deletion behavior.

Use the Milvus project's supported backup tooling for durable collections and
test restoration into an isolated environment. Record the Milvus version,
collection schema, embedding model, and dimension with the backup.

## Troubleshooting

### `milvus configuration is required`

`backend_type: milvus` requires the nested
`global.stores.response_cache.milvus` block. Check indentation and validate the
complete config.

### Collection does not exist

For development, set `development.auto_create_collection: true`. In a
controlled production environment, pre-create the collection and leave
automatic creation disabled. Ensure the schema and vector dimension match the
Router config.

### Connection timeout

Check the Service and endpoints, DNS from the Router namespace, NetworkPolicy,
and the configured database:

```bash
kubectl get service,endpoints -n milvus
kubectl get networkpolicy -A
```

### Search quality is poor

Verify that training and inference use the same embedding model and dimension.
Then tune the response-cache similarity threshold and Milvus search/index
parameters against representative traffic. Do not copy thresholds from another
embedding model without evaluation.

## References

- [Milvus documentation](https://milvus.io/docs)
- [Response Cache plugin](../tutorials/plugin/response-cache)
- [Data and Storage](storage-overview)
