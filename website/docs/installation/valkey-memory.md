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
| Best for | Small-to-medium workloads, dev/test, existing Redis/Valkey infra | Larger or distributed vector workloads |
| Vector index | HNSW via FT.CREATE | HNSW, IVF_FLAT, IVF_SQ8, and more |

## Prerequisites

- A Valkey release with the Search module enabled. The
  `valkey/valkey-bundle` image includes the module.
- A Valkey and Search-module version pair supported by the release you deploy.
  Pin that release in production rather than following `latest` or an RC tag.
- If your Valkey distribution does not bundle Search, follow the upstream
  [Valkey Search quick start](https://github.com/valkey-io/valkey-search/blob/main/QUICK_START.md)
  and review its release notes before loading the module.
- For Kubernetes: Helm 3.x and `kubectl` configured

:::info Trouble with the Search module?
If you run into issues loading or using the Search module, please [open an issue](https://github.com/vllm-project/semantic-router/issues/new) so we can help.
:::

## Deploy with Docker

### Quick Start

```bash
docker network inspect vllm-sr-network >/dev/null 2>&1 || \
  docker network create vllm-sr-network

docker run -d --name valkey-memory \
  --network vllm-sr-network \
  valkey/valkey-bundle:latest
```

Verify the Search module is loaded:

```bash
docker exec valkey-memory valkey-cli MODULE LIST | grep search
```

### With Persistence

```bash
docker run -d --name valkey-memory \
  --network vllm-sr-network \
  -v valkey-data:/data \
  valkey/valkey-bundle:latest \
  valkey-server --appendonly yes
```

The hostname `valkey-memory` in the configuration below is Docker DNS on
`vllm-sr-network`. If you start the Router with a custom stack name, for example
`VLLM_SR_STACK_NAME=team-a vllm-sr serve`, attach Valkey to
`team-a-vllm-sr-network` and use the matching reachable hostname.

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

This manifest is an unauthenticated development example. For production, use
your Valkey operator or chart's Secret integration and network policy rather
than placing a password in the Pod command line. Set the same credential in the
Router's `global.stores.memory.valkey.password` through your secret-management
workflow.

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
| `dimension` | derived | Embedding vector dimension; when omitted, `mmbert` uses 256 and current other memory embedding models use 384 |
| `metric_type` | `COSINE` | Distance metric: `COSINE`, `L2`, or `IP` |
| `index_m` | `16` | HNSW M parameter (links per node) |
| `index_ef_construction` | `256` | HNSW build-time search width |
| `tls_enabled` | `false` | Connect to Valkey with TLS |
| `tls_ca_path` | _(empty)_ | PEM-encoded CA file mounted in the Router; an empty value uses the system trust store |
| `tls_insecure_skip_verify` | `false` | Skip certificate verification; keep `false` outside isolated development |

For a production TLS endpoint, mount the CA certificate into the Router and
keep the password in an environment-backed secret:

```yaml
global:
  stores:
    memory:
      enabled: true
      backend: valkey
      valkey:
        host: valkey.example.internal
        port: 6380
        password: ${VALKEY_PASSWORD}
        tls_enabled: true
        tls_ca_path: /etc/valkey/certs/ca.pem
        tls_insecure_skip_verify: false
```

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
document:
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

- **`index_m`** (default 16): Higher values can improve recall at the cost of
  more memory and index work.
- **`index_ef_construction`** (default 256): Higher values can improve index
  quality at the cost of slower construction.

Tune both parameters with representative data and measure recall, latency,
build time, and memory together. There is no production-safe value that applies
to every corpus or capacity target.

### Memory Sizing

The raw float32 embedding alone uses `dimension * 4` bytes per entry. Actual
memory is higher and depends on:

- serialized content, metadata, and timestamps;
- the Search module and Valkey versions;
- HNSW graph structure and index parameters; and
- allocator fragmentation and Valkey base overhead.

Do not size production capacity from a fixed per-entry multiplier. Load a
representative dataset, inspect `INFO memory` and `FT.INFO <index>`, and reserve
headroom for index construction, replication, and traffic growth.

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

Restart the Router (or otherwise reinitialize the memory store) after dropping
the index. Index creation runs when the Valkey store starts, not on the next
request. Test the rebuild against a copy of production data before using this
procedure on a live store.

### Out of Memory

Valkey stores all data in memory. If you hit the memory limit:

1. Inspect `INFO memory` and `FT.INFO <index>` to distinguish document, index,
   and allocator growth.
2. Remove expired or unwanted memories through the application workflow, or
   shorten the retention policy that creates them.
3. Add capacity or shard the dataset before changing `maxmemory-policy`;
   eviction can remove memories that the application expects to retain.

## Migration from Milvus

To switch an existing deployment from Milvus to Valkey:

1. Update `config.yaml` to set `backend: valkey` and add the `valkey:` block
2. Remove or comment out the `milvus:` block
3. Restart the router — it will create the Valkey index automatically
4. Existing memories in Milvus are **not** automatically migrated

:::warning
Switching backends does not migrate data. If you need to preserve existing memories, export them from Milvus and re-import via the memory API before switching.
:::
