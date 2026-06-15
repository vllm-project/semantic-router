# Hybrid Cache: HNSW + Milvus

The Hybrid Cache combines an in-memory HNSW index for fast search with a Milvus vector database for scalable, persistent storage.

## Overview

The hybrid architecture provides:

- **Fast search** via in-memory HNSW index
- **Scalable storage** via Milvus vector database
- **Persistence** with Milvus as the source of truth
- **Hot data caching** with local document cache

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  Hybrid Cache                     │
├──────────────────────────────────────────────────┤
│  ┌─────────────────┐      ┌──────────────────┐  │
│  │  In-Memory      │      │   Local Cache    │  │
│  │  HNSW Index     │◄─────┤   (Hot Data)     │  │
│  └────────┬────────┘      └──────────────────┘  │
│           │                                       │
│           │ ID Mapping                           │
│           ▼                                       │
│  ┌──────────────────────────────────────────┐   │
│  │         Milvus Vector Database           │   │
│  └──────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

## How It Works

### Write Path (AddEntry)

When adding a cache entry:

1. Generate embedding using the configured embedding model
2. Write entry to Milvus for persistence
3. Add entry to in-memory HNSW index (if space is available)
4. Add document to local cache

### Read Path (FindSimilar)

When searching for a similar query:

1. Generate query embedding
2. Search HNSW index for nearest neighbors
3. Check local cache for matching documents
   - If found in local cache: return immediately (hot path)
   - If not found: fetch from Milvus (cold path)
4. Cache fetched documents in local cache for future queries

### Memory Management

- **HNSW Index**: Limited to a configured maximum number of entries
- **Local Cache**: Limited to a configured number of documents
- **Eviction**: FIFO policy when limits are reached
- **Data Persistence**: All data remains in Milvus regardless of memory limits

## Configuration

### Basic Configuration

```yaml
semantic_cache:
  enabled: true
  backend_type: "hybrid"
  similarity_threshold: 0.85
  ttl_seconds: 3600
  
  # Hybrid-specific settings
  max_memory_entries: 100000  # Max entries in HNSW
  local_cache_size: 1000      # Local document cache size
  
  # HNSW parameters
  hnsw_m: 16
  hnsw_ef_construction: 200
  
  # Milvus configuration
  backend_config_path: "config/semantic-cache/milvus.yaml"
```

### Decision-Level Configuration (Plugin-Based)

You can also configure hybrid cache at the decision level using plugins:

```yaml
signals:
  domains:
    - name: "math"
      description: "Mathematical queries"
      mmlu_categories: ["math"]

decisions:
  - name: math_route
    description: "Route math queries with strict caching"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "math"
    modelRefs:
      - model: "openai/gpt-oss-120b"
        use_reasoning: true
    plugins:
      - type: "semantic-cache"
        configuration:
          enabled: true
          similarity_threshold: 0.95  # Very strict for math accuracy
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend_type` | string | - | Must be `"hybrid"` |
| `similarity_threshold` | float | 0.85 | Minimum similarity for cache hit |
| `max_memory_entries` | int | 100000 | Max entries in HNSW index |
| `local_cache_size` | int | 1000 | Hot document cache size |
| `hnsw_m` | int | 16 | HNSW bi-directional links |
| `hnsw_ef_construction` | int | 200 | HNSW construction quality |
| `backend_config_path` | string | - | Path to Milvus config file |

### Milvus Configuration

Create `config/semantic-cache/milvus.yaml`:

```yaml
milvus:
  address: "localhost:19530"
  collection_name: "semantic_cache"
  dimension: 384
  index_type: "HNSW"
  metric_type: "IP"
  params:
    M: 16
    efConstruction: 200
```

## Example Usage

### Go Code

```go
import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"

// Initialize hybrid cache
options := cache.HybridCacheOptions{
    Enabled:             true,
    SimilarityThreshold: 0.85,
    TTLSeconds:          3600,
    MaxMemoryEntries:    100000,
    HNSWM:               16,
    HNSWEfConstruction:  200,
    MilvusConfigPath:    "config/semantic-cache/milvus.yaml",
    LocalCacheSize:      1000,
}

hybridCache, err := cache.NewHybridCache(options)
if err != nil {
    log.Fatalf("Failed to create hybrid cache: %v", err)
}
defer hybridCache.Close()

// Add cache entry
err = hybridCache.AddEntry(
    "request-id-123",
    "gpt-4",
    "What is quantum computing?",
    []byte(`{"prompt": "What is quantum computing?"}`),
    []byte(`{"response": "Quantum computing is..."}`),
)

// Search for similar query
response, found, err := hybridCache.FindSimilar(
    "gpt-4",
    "Explain quantum computers",
)
if found {
    fmt.Printf("Cache hit! Response: %s\n", string(response))
}

// Get statistics
stats := hybridCache.GetStats()
fmt.Printf("Total entries in HNSW: %d\n", stats.TotalEntries)
fmt.Printf("Hit ratio: %.2f%%\n", stats.HitRatio * 100)
```

## Monitoring and Metrics

The hybrid cache exposes metrics for monitoring:

```go
stats := hybridCache.GetStats()

// Available metrics
stats.TotalEntries  // Entries in HNSW index
stats.HitCount      // Total cache hits
stats.MissCount     // Total cache misses
stats.HitRatio      // Hit ratio (0.0 - 1.0)
```

### Prometheus Metrics

```
# Cache entries in HNSW
semantic_cache_entries{backend="hybrid"}

# Cache operations
semantic_cache_operations_total{backend="hybrid",operation="find_similar",status="hit_local"}
semantic_cache_operations_total{backend="hybrid",operation="find_similar",status="hit_milvus"}
semantic_cache_operations_total{backend="hybrid",operation="find_similar",status="miss"}

# Cache hit ratio
semantic_cache_hit_ratio{backend="hybrid"}
```

## Multi-Instance Deployment

The hybrid cache supports multi-instance deployments where each instance maintains its own HNSW index and local cache, but shares Milvus for persistence and data consistency:

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  Instance 1 │   │  Instance 2 │   │  Instance 3 │
│  HNSW Cache │   │  HNSW Cache │   │  HNSW Cache │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
                  ┌──────▼──────┐
                  │   Milvus    │
                  │  (Shared)   │
                  └─────────────┘
```

## See Also

- [In-Memory Cache Documentation](./in-memory-cache.md)
- [Milvus Cache Documentation](./milvus-cache.md)
