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
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant:latest
```

Verify Qdrant is running:

```bash
curl http://localhost:6333/healthz
```

### With Persistence

```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant-data:/qdrant/storage \
  qdrant/qdrant:latest
```

### With API Key Authentication

```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant-data:/qdrant/storage \
  -e QDRANT__SERVICE__API_KEY=your-secret-key \
  qdrant/qdrant:latest
```

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

## Configure the Router

### Semantic Cache

```yaml
global:
  stores:
    semantic_cache:
      enabled: true
      backend_type: qdrant
      similarity_threshold: 0.90
      ttl_seconds: 7200
      embedding_model: bert
      qdrant:
        host: qdrant                   # Service name or hostname
        port: 6334
        api_key: ""
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
        api_key: ""
        collection: agentic_memory
        dimension: 384               # Must match your embedding model
      embedding_model: bert
      default_retrieval_limit: 5
      default_similarity_threshold: 0.70
```

### Router Replay Store

```yaml
global:
  services:
    router_replay:
      store_backend: qdrant
      qdrant:
        host: qdrant
        port: 6334
        api_key: ""
        collection_name: router_replay
```

### Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | `localhost` | Qdrant server hostname |
| `port` | `6334` | Qdrant gRPC port |
| `api_key` | _(empty)_ | API key for authentication |
| `use_tls` | `false` | Enable TLS for the gRPC connection |
| `collection_name` | varies | Collection to use (auto-created if absent) |
| `connect_timeout` | `10` | Connection timeout in seconds |
