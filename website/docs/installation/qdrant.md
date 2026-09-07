---
sidebar_position: 7
---

# Qdrant

This guide covers deploying [Qdrant](https://qdrant.tech/) as a backend for the Semantic Router. Qdrant can serve as the semantic cache, agentic memory store, vector store, and router replay store.

## Prerequisites

- Docker or a Kubernetes cluster with `kubectl` configured
- For Kubernetes: Helm 3.x installed

## Deploy with Docker

### Quick Start

```bash
docker network inspect vllm-sr-network >/dev/null 2>&1 || \
  docker network create vllm-sr-network

docker run -d --name qdrant \
  --network vllm-sr-network \
  -p 127.0.0.1:6333:6333 \
  qdrant/qdrant:latest
```

Verify Qdrant is running:

```bash
curl http://localhost:6333/healthz
```

### With Persistence

```bash
docker run -d --name qdrant \
  --network vllm-sr-network \
  -p 127.0.0.1:6333:6333 \
  -v qdrant-data:/qdrant/storage \
  qdrant/qdrant:latest
```

### With API Key Authentication

```bash
export QDRANT_API_KEY="$(openssl rand -hex 32)"

docker run -d --name qdrant \
  --network vllm-sr-network \
  -p 127.0.0.1:6333:6333 \
  -v qdrant-data:/qdrant/storage \
  -e QDRANT__SERVICE__API_KEY="$QDRANT_API_KEY" \
  qdrant/qdrant:latest
```

When authentication is enabled, add the same environment reference to each
Qdrant block that the Router uses:

```yaml
api_key: ${QDRANT_API_KEY}
```

Keep the value in the process environment or a Kubernetes Secret; do not put a
literal API key in the Router config. Omit `api_key` when the Qdrant server does
not require authentication.

The host mapping exposes only the HTTP health/API port on loopback. The Router
uses Qdrant's gRPC port directly over the shared Docker network, so it does not
need to be published on the host. The hostname `qdrant` in the configuration below is Docker DNS on
`vllm-sr-network`. If you start the Router with a custom stack name, for example
`VLLM_SR_STACK_NAME=team-a vllm-sr serve`, attach Qdrant to
`team-a-vllm-sr-network` and use the matching reachable hostname.

The Docker examples use `latest` for short-lived evaluation. Pin a published
Qdrant version or image digest for a shared or production deployment.

## Deploy in Kubernetes

### Using Helm

```bash
helm repo add qdrant https://qdrant.github.io/qdrant-helm
helm repo update

helm install qdrant qdrant/qdrant \
  --namespace vllm-semantic-router-system --create-namespace \
  --set persistence.size=10Gi
```

### Using a StatefulSet

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: vllm-semantic-router-system
spec:
  serviceName: qdrant
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
        - name: qdrant
          image: qdrant/qdrant:latest
          ports:
            - containerPort: 6333
            - containerPort: 6334
          volumeMounts:
            - name: data
              mountPath: /qdrant/storage
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant
  namespace: vllm-semantic-router-system
spec:
  selector:
    app: qdrant
  ports:
    - name: rest
      port: 6333
      targetPort: 6333
    - name: grpc
      port: 6334
      targetPort: 6334
  clusterIP: None
```

The StatefulSet is an unauthenticated evaluation example. For a shared or
production cluster, pin a chart or image version, configure an API key through
a Kubernetes Secret, enable TLS in both Qdrant and the Router's `api_key` /
`use_tls` bindings, restrict access with NetworkPolicy, and define backup and
restore procedures for the persistent volume.

## Configure the Router

### Semantic Cache

```yaml
global:
  stores:
    response_cache:
      enabled: true
      backend_type: qdrant
      similarity_threshold: 0.90
      ttl_seconds: 7200
      embedding_model: bert
      qdrant:
        host: qdrant                   # Service name or hostname
        port: 6334
        use_tls: false
        collection_name: semantic_cache
        connect_timeout: 10
```

### Agentic Memory

```yaml
global:
  stores:
    memory:
      enabled: true
      backend: qdrant
      qdrant:
        host: qdrant
        port: 6334
        collection: agentic_memory
        dimension: 384               # Must match your embedding model
      embedding_model: bert
      default_retrieval_limit: 5
      default_similarity_threshold: 0.70
```

### Uploaded document vector store

```yaml
global:
  stores:
    vector_store:
      enabled: true
      backend_type: qdrant
      file_storage_dir: /var/lib/vsr/data
      embedding_model: multimodal
      embedding_dimension: 384
      qdrant:
        host: qdrant
        port: 6334
        use_tls: false
        connect_timeout: 10
        collection_prefix: "vsr_vs_"
      metadata_store: memory
```

Use durable shared metadata instead of `memory` when several Router replicas
must see the same uploaded-file registry. The embedding dimension must match
the configured embedding model.

### Router Replay Store

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: qdrant
      qdrant:
        host: qdrant
        port: 6334
        collection_name: router_replay
```

This enables the router-wide replay policy. Add a route-local `router_replay`
plugin only when a decision needs to override capture or retention behavior;
see the [Router Replay plugin](../tutorials/plugin/router-replay).

### Configuration reference

All four Qdrant bindings accept `host`, `port`, optional `api_key`, and
`use_tls`. Their collection fields are deliberately different:

| Capability | Collection field | Other Qdrant-specific fields |
| --- | --- | --- |
| Response cache | `collection_name` | `connect_timeout` |
| Agentic memory | `collection` | `dimension`, `connect_timeout` |
| Uploaded document vector store | `collection_prefix` | `connect_timeout` |
| Router Replay | `collection_name` | No connection-timeout field in the replay schema |

Use an environment reference such as `${QDRANT_API_KEY}` for `api_key` when
authentication is enabled. Validate the complete config rather than copying a
field from one capability into another.
