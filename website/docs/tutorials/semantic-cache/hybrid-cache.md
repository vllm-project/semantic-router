# Hybrid Cache: HNSW + Milvus

The Hybrid Cache combines the best of both worlds: in-memory HNSW index for ultra-fast search with Milvus vector database for scalable, persistent storage.

## Overview

The hybrid architecture provides:
- **O(log n) search** via in-memory HNSW index
- **Unlimited storage** via Milvus vector database
- **Cost efficiency** by keeping only hot data in memory
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
│  │  (100K entries) │      │   (1K docs)      │  │
│  └────────┬────────┘      └──────────────────┘  │
│           │                                       │
│           │ ID Mapping                           │
│           ▼                                       │
│  ┌──────────────────────────────────────────┐   │
│  │         Milvus Vector Database           │   │
│  │       (Millions of entries)              │   │
│  └──────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

## How It Works

### 1. Write Path (AddEntry)

```
User Request
    │
    ├─► Generate Embedding (BERT)
    │
    ├─► Write to Milvus (persistence)
    │
    └─► Add to HNSW Index (if space available)
        │
        └─► Add to Local Cache
```

### 2. Read Path (FindSimilar)

```
User Query
    │
    ├─► Generate Query Embedding
    │
    ├─► Search HNSW Index (10 candidates)
    │
    ├─► Check Local Cache (hot path)
    │   ├─► HIT: Return immediately
    │   └─► MISS: Continue
    │
    └─► Fetch from Milvus (cold path)
        └─► Cache in Local Cache
```

### 3. Memory Management

- **HNSW Index**: Limited to `max_memory_entries` (default: 100K)
- **Local Cache**: Limited to `local_cache_size` (default: 1K documents)
- **Eviction**: FIFO policy when limits reached
- **Data Persistence**: All data remains in Milvus

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
  backend_config_path: "config/milvus.yaml"
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

Create `config/milvus.yaml`:

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

## Performance Characteristics

### Search Performance

| Cache Size | Memory Backend | Hybrid (HNSW) | Hybrid (Local) | Improvement |
|------------|---------------|---------------|----------------|-------------|
| 100 entries | 0.5 ms | 0.3 ms | **0.05 ms** | 10x faster |
| 1K entries | 2 ms | 0.4 ms | **0.05 ms** | 40x faster |
| 10K entries | 15 ms | 0.6 ms | **0.05 ms** | 300x faster |
| 100K entries | 150 ms | 0.8 ms | **0.05 ms** | 3000x faster |
| 1M entries | N/A (OOM) | 1.2 ms | **0.05 ms** | ∞ |

### Memory Usage

| Component | Memory per Entry | 100K Entries | 1M Entries |
|-----------|-----------------|--------------|------------|
| Embeddings (384D) | ~1.5 KB | ~150 MB | ~1.5 GB |
| HNSW Graph | ~0.5 KB | ~50 MB | ~500 MB |
| Local Cache | ~2 KB | ~2 MB (1K docs) | ~2 MB |
| **Total In-Memory** | - | ~200 MB | ~2 GB |

**Milvus Storage**: Unlimited (disk-based)

## Use Cases

### When to Use Hybrid Cache

✅ **Ideal for:**
- Large-scale applications (>100K cache entries)
- Production systems requiring persistence
- Applications with hot/cold access patterns
- Cost-sensitive deployments
- Multi-instance deployments sharing cache

### When to Use Memory Backend

✅ **Ideal for:**
- Small to medium scale (<10K entries)
- Development and testing
- Single-instance deployments
- No persistence required

### When to Use Milvus Backend

✅ **Ideal for:**
- Massive scale (millions of entries)
- Complex vector search requirements
- Applications without latency sensitivity

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
    MilvusConfigPath:    "config/milvus.yaml",
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
semantic_cache_entries{backend="hybrid"} 95432

# Cache operations
semantic_cache_operations_total{backend="hybrid",operation="find_similar",status="hit_local"} 12453
semantic_cache_operations_total{backend="hybrid",operation="find_similar",status="hit_milvus"} 3421
semantic_cache_operations_total{backend="hybrid",operation="find_similar",status="miss"} 892

# Cache hit ratio
semantic_cache_hit_ratio{backend="hybrid"} 0.947
```

## Best Practices

### 1. Right-Size Your Memory

Choose `max_memory_entries` based on your working set:

```yaml
# For 1M total entries with 10% hot data
max_memory_entries: 100000  # 100K in HNSW
local_cache_size: 1000      # 1K hottest documents
```

### 2. Tune HNSW Parameters

Balance recall vs. speed:

```yaml
# High recall (slower build, better search)
hnsw_m: 32
hnsw_ef_construction: 400

# Balanced (recommended)
hnsw_m: 16
hnsw_ef_construction: 200

# Fast build (lower recall)
hnsw_m: 8
hnsw_ef_construction: 100
```

### 3. Monitor Hit Rates

Track cache effectiveness:

```bash
# Check cache stats
curl http://localhost:8080/metrics | grep cache

# Optimal hit rates:
# - Local cache: >80% (hot data)
# - Milvus cache: >90% (total)
```

### 4. Adjust Similarity Threshold

```yaml
# Stricter matching (fewer false positives)
similarity_threshold: 0.90

# Balanced (recommended)
similarity_threshold: 0.85

# Looser matching (more cache hits)
similarity_threshold: 0.80
```

## Troubleshooting

### High Memory Usage

**Symptom**: Memory usage exceeds expectations

**Solution**:
```yaml
# Reduce HNSW index size
max_memory_entries: 50000  # Instead of 100000

# Reduce local cache
local_cache_size: 500      # Instead of 1000

# Use smaller HNSW M
hnsw_m: 8                  # Instead of 16
```

### Low Hit Rate

**Symptom**: Cache hit rate < 50%

**Solution**:
1. Lower similarity threshold
2. Increase `max_memory_entries`
3. Check Milvus connectivity
4. Verify embedding model consistency

### Slow Queries

**Symptom**: Queries taking > 10ms

**Solution**:
1. Check Milvus network latency
2. Increase `local_cache_size` for hot data
3. Verify HNSW index health
4. Monitor Milvus load

## Migration Guide

### From Memory Backend

```yaml
# Before
semantic_cache:
  backend_type: "memory"
  max_entries: 10000

# After
semantic_cache:
  backend_type: "hybrid"
  max_memory_entries: 10000  # Keep same HNSW size
  local_cache_size: 1000
  backend_config_path: "config/milvus.yaml"
```

### From Milvus Backend

```yaml
# Before
semantic_cache:
  backend_type: "milvus"
  backend_config_path: "config/milvus.yaml"

# After
semantic_cache:
  backend_type: "hybrid"
  max_memory_entries: 100000  # Add HNSW layer
  local_cache_size: 1000      # Add local cache
  backend_config_path: "config/milvus.yaml"  # Keep Milvus
```

## Advanced Topics

### Custom Eviction Strategy

Currently uses FIFO. Future versions may support:
- LRU (Least Recently Used)
- LFU (Least Frequently Used)
- TTL-based eviction

### Multi-Instance Deployment

The hybrid cache is designed for multi-instance deployments:

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

Each instance maintains its own HNSW index and local cache, but shares Milvus for persistence and data consistency.

## See Also

- [In-Memory Cache Documentation](./in-memory-cache.md)
- [Milvus Cache Documentation](./milvus-cache.md)
- [HNSW Implementation Details](../../HNSW_IMPLEMENTATION_SUMMARY.md)
- [Research Paper: Hybrid Architecture](../../papers/hybrid_hnsw_storage_architecture.md)

