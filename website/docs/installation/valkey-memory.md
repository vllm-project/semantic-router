---
sidebar_position: 6
---

# Valkey Agentic Memory

This guide covers deploying Valkey as the agentic memory backend for the Semantic Router. Valkey provides a lightweight, Redis-compatible alternative to Milvus for vector similarity storage using the built-in Search module.

:::note
Valkey is optional. The default memory backend is Milvus. Use Valkey when you want a single-binary deployment without external dependencies like etcd or MinIO, or when you already run Valkey for caching.
:::

## When to Use Valkey vs Milvus

| Concern | Valkey | Milvus |
|---------|--------|--------|
| Deployment complexity | Single binary with Search module | Requires etcd, MinIO/S3, optional Pulsar |
| Horizontal scaling | Cluster mode (manual sharding) | Native distributed architecture |
| Memory model | In-memory with optional persistence | Disk-based with memory-mapped indexes |
| Best for | Small-to-medium workloads, dev/test, existing Redis/Valkey infra | Large-scale production, billions of vectors |
| Vector index | HNSW via FT.CREATE | HNSW, IVF_FLAT, IVF_SQ8, and more |

## Prerequisites

- Valkey 8.0+ **with the Search module** enabled
- The `valkey/valkey-bundle` Docker image includes Search out of the box
- For Kubernetes: Helm 3.x and `kubectl` configured

## Deploy with Docker

### Quick Start

```bash
docker run -d --name valkey-memory \
  -p 6379:6379 \
  valkey/valkey-bundle:latest
```

Verify the Search module is loaded:

```bash
docker exec valkey-memory valkey-cli MODULE LIST | grep search
```

### With Persistence

```bash
docker run -d --name valkey-memory \
  -p 6379:6379 \
  -v valkey-data:/data \
  valkey/valkey-bundle:latest \
  valkey-server --appendonly yes
```

## Deploy in Kubernetes

### Using a StatefulSet

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: valkey-memory
  namespace: vllm-semantic-router-system
spec:
  serviceName: valkey-memory
  replicas: 1
  selector:
    matchLabels:
      app: valkey-memory
  template:
    metadata:
      labels:
        app: valkey-memory
    spec:
      containers:
        - name: valkey
          image: valkey/valkey-bundle:latest
          ports:
            - containerPort: 6379
          args: ["valkey-server", "--appendonly", "yes"]
          # For production, add --requirepass or mount a Secret:
          # args: ["valkey-server", "--appendonly", "yes", "--requirepass", "$(VALKEY_PASSWORD)"]
          volumeMounts:
            - name: data
              mountPath: /data
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 5Gi
---
apiVersion: v1
kind: Service
metadata:
  name: valkey-memory
  namespace: vllm-semantic-router-system
spec:
  selector:
    app: valkey-memory
  ports:
    - port: 6379
      targetPort: 6379
  clusterIP: None
```

## Configure the Router

Add the Valkey memory backend to your `config.yaml`:

```yaml
global:
  stores:
    memory:
      enabled: true
      backend: valkey
      auto_store: true
      valkey:
        host: valkey-memory          # Service name or hostname
        port: 6379
        database: 0
        timeout: 10
        collection_prefix: "mem:"
        index_name: mem_idx
        dimension: 384               # Must match your embedding model
        metric_type: COSINE           # COSINE, L2, or IP
        index_m: 16
        index_ef_construction: 256
      embedding_model: bert
      default_retrieval_limit: 5
      default_similarity_threshold: 0.70
      hybrid_search: true
      hybrid_mode: rerank
      adaptive_threshold: true
```

### Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | `localhost` | Valkey server hostname |
| `port` | `6379` | Valkey server port |
| `database` | `0` | Database number (0-15) |
| `password` | _(empty)_ | Authentication password |
| `timeout` | `10` | Connection timeout in seconds |
| `collection_prefix` | `mem:` | Key prefix for HASH documents |
| `index_name` | `mem_idx` | FT.CREATE index name |
| `dimension` | `384` | Embedding vector dimension |
| `metric_type` | `COSINE` | Distance metric: `COSINE`, `L2`, or `IP` |
| `index_m` | `16` | HNSW M parameter (links per node) |
| `index_ef_construction` | `256` | HNSW build-time search width |

### Optional Redis Hot Cache

You can layer a Redis/Valkey hot cache in front of the Valkey memory store for frequently accessed memories:

```yaml
      redis_cache:
        enabled: true
        address: "valkey-memory:6379"
        ttl_seconds: 900
        db: 1                        # Use a different DB to avoid key collisions
        key_prefix: "memory_cache:"
```

## Per-Decision Memory Plugin

Routes can override global memory settings using the `memory` plugin:

```yaml
routing:
  decisions:
    - name: personalized_route
      plugins:
        - type: memory
          configuration:
            enabled: true
            retrieval_limit: 10
            similarity_threshold: 0.60
            auto_store: true
```

See the [Memory plugin tutorial](/docs/tutorials/plugin/memory) for details.

## Performance Tuning

### HNSW Index Parameters

- **`index_m`** (default 16): Higher values improve recall at the cost of memory. Use 32-64 for production workloads requiring high accuracy.
- **`index_ef_construction`** (default 256): Higher values improve index quality at the cost of slower builds. Use 512+ for production.

### Memory Sizing

Each memory entry uses approximately:

- HASH fields: ~500-2000 bytes (content, metadata, timestamps)
- Embedding vector: `dimension * 4` bytes (e.g., 384 * 4 = 1.5 KB for BERT)
- HNSW index overhead: ~`dimension * index_m * 4` bytes per entry

For 100K memories with 384-dimensional embeddings and M=16:

- Data: ~300 MB
- Index: ~240 MB
- **Total: ~540 MB** plus Valkey base overhead

### Persistence

Enable AOF (Append-Only File) for durability:

```bash
valkey-server --appendonly yes --appendfsync everysec
```

For RDB snapshots (point-in-time backups):

```bash
valkey-server --save 900 1 --save 300 10
```

## Troubleshooting

### Search Module Not Loaded

```
FT.CREATE failed: unknown command 'FT.CREATE'
```

Ensure you are using `valkey/valkey-bundle` (includes Search) rather than plain `valkey/valkey`:

```bash
valkey-cli MODULE LIST
# Should show: name search ver ...
```

### Connection Timeout

```
valkey: connection timeout
```

- Verify the hostname resolves: `nslookup valkey-memory`
- Check port connectivity: `nc -zv valkey-memory 6379`
- Increase `timeout` in the config if the network is slow

### Index Already Exists

The router checks for existing indexes on startup and skips creation if one exists. If you need to recreate the index (e.g., after changing `dimension` or `metric_type`):

```bash
valkey-cli FT.DROPINDEX mem_idx
```

The router will recreate it on the next request.

### Out of Memory

Valkey stores all data in memory. If you hit the memory limit:

1. Set `maxmemory` and `maxmemory-policy` in Valkey config
2. Use `quality_scoring.max_memories_per_user` to cap per-user storage
3. Enable memory consolidation to merge similar memories

## Migration from Milvus

To switch an existing deployment from Milvus to Valkey:

1. Update `config.yaml` to set `backend: valkey` and add the `valkey:` block
2. Remove or comment out the `milvus:` block
3. Restart the router — it will create the Valkey index automatically
4. Existing memories in Milvus are **not** automatically migrated

:::warning
Switching backends does not migrate data. If you need to preserve existing memories, export them from Milvus and re-import via the memory API before switching.
:::
