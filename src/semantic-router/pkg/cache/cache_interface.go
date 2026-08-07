package cache

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// CacheEntry represents a complete cached request-response pair with associated metadata
type CacheEntry struct {
	RequestID    string
	RequestBody  []byte
	ResponseBody []byte
	Model        string
	Query        string
	Embedding    []float32
	Timestamp    time.Time // Creation time (when the entry was added or completed with a response)
	LastAccessAt time.Time // Last access time
	HitCount     int64     // Access count
	TTLSeconds   int       // Per-entry TTL in seconds (0 = not cached, -1 = use cache default, >0 = specific TTL)
	ExpiresAt    time.Time // Calculated expiration time based on TTL
}

// LookupResult carries the outcome of a semantic cache lookup as a single
// request-owned value.
//
// Semantics:
//   - Found=true: Body holds the cached response and Similarity is the matched
//     score for this lookup.
//   - Found=false: Body is nil and Similarity is zero.
//
// Callers must not read similarity from any global or backend-owned state:
// two concurrent lookups against the same backend instance would otherwise
// race and leak one caller's score into another (#2473).
type LookupResult struct {
	Body       []byte
	Found      bool
	Similarity float32
}

// ctxErr reports a context's cancellation or deadline error, treating a nil
// context as "no error". Cache backends call it to short-circuit a lookup or
// write before starting synchronous embedding/storage work (#2473).
func ctxErr(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	return ctx.Err()
}

func contextErrorOnFailure(ctx context.Context, operationErr error) error {
	if operationErr == nil {
		return nil
	}
	return ctxErr(ctx)
}

// releaseOnFailure runs one post-construction setup step and releases the
// partially built client when that step fails, so a failed constructor leaks no
// connection, pool, or background goroutine (#2473).
//
// Backends route their setup steps through it because the cleanup itself has no
// black-box signal: a constructor that leaks the client still returns
// (nil, err), and a network-level probe cannot tell either — the underlying
// drivers already drop a connection whose command failed. Centralizing the
// contract here keeps it testable for every backend, including the ones whose
// client cannot be built at all without a live server.
func releaseOnFailure(step func() error, release func()) error {
	if err := step(); err != nil {
		release()
		return err
	}
	return nil
}

// CacheBackend defines the interface for semantic cache implementations
type CacheBackend interface {
	// IsEnabled returns whether caching is currently active
	IsEnabled() bool

	// CheckConnection verifies the cache backend connection is healthy
	// Returns nil if the connection is healthy, error otherwise
	// For local caches (in-memory), this may be a no-op
	CheckConnection(ctx context.Context) error

	// AddPendingRequest stores a request awaiting its response. Model is an
	// exact cache partition key; entries from another model partition must
	// never be considered by lookup.
	AddPendingRequest(ctx context.Context, requestID string, model string, query string, requestBody []byte, ttlSeconds int) error

	// UpdateWithResponse completes a pending request with the received response
	UpdateWithResponse(ctx context.Context, requestID string, responseBody []byte, ttlSeconds int) error

	// AddEntry stores a complete request-response pair in the cache
	AddEntry(ctx context.Context, requestID string, model string, query string, requestBody, responseBody []byte, ttlSeconds int) error

	// FindSimilar searches the exact model partition using the backend's
	// configured similarity threshold. The returned LookupResult carries the
	// response body, hit flag, and per-request similarity score.
	FindSimilar(ctx context.Context, model string, query string) (LookupResult, error)

	// FindSimilarWithThreshold searches the exact model partition with a
	// caller-supplied threshold (used for category-specific thresholds). The
	// returned LookupResult carries the response body, hit flag, and per-request
	// similarity score.
	FindSimilarWithThreshold(ctx context.Context, model string, query string, threshold float32) (LookupResult, error)

	// Close releases all resources held by the cache backend
	Close() error

	// GetStats provides cache performance and usage metrics
	GetStats() CacheStats
}

// Compile-time assertions that every backend satisfies CacheBackend. They live
// at the interface definition site (no build tag) so a future contract
// migration — e.g. adding a parameter to a write method — fails to compile here
// rather than silently drifting until a downstream build breaks.
//
// Caveat worth knowing before relying on them for the windows||!cgo
// InMemoryCache/HybridCache stubs: nothing in this repository compiles that
// variant today. `CGO_ENABLED=0 go build ./pkg/cache/...` fails inside
// valkey-glide, which itself requires cgo, and CI has no windows job. The stub
// method sets are therefore still review-verified, not build-verified.
var (
	_ CacheBackend = (*InMemoryCache)(nil)
	_ CacheBackend = (*HybridCache)(nil)
	_ CacheBackend = (*MilvusCache)(nil)
	_ CacheBackend = (*QdrantCache)(nil)
	_ CacheBackend = (*RedisCache)(nil)
	_ CacheBackend = (*ValkeyCache)(nil)
)

// CacheStats holds performance metrics and usage statistics for cache operations
type CacheStats struct {
	TotalEntries    int        `json:"total_entries"`
	HitCount        int64      `json:"hit_count"`
	MissCount       int64      `json:"miss_count"`
	HitRatio        float64    `json:"hit_ratio"`
	LastCleanupTime *time.Time `json:"last_cleanup_time,omitempty"`
}

// CacheBackendType defines the available cache backend implementations
type CacheBackendType string

const (
	// InMemoryCacheType specifies the in-memory cache backend
	InMemoryCacheType CacheBackendType = "memory"

	// MilvusCacheType specifies the Milvus vector database backend
	MilvusCacheType CacheBackendType = "milvus"

	// RedisCacheType specifies the Redis vector database backend
	RedisCacheType CacheBackendType = "redis"

	// ValkeyCacheType specifies the Valkey vector database backend
	ValkeyCacheType CacheBackendType = "valkey"

	// HybridCacheType specifies the hybrid HNSW + Milvus backend
	HybridCacheType CacheBackendType = "hybrid"

	// QdrantCacheType specifies the Qdrant vector search engine backend
	QdrantCacheType CacheBackendType = "qdrant"
)

// EvictionPolicyType defines the available eviction policies
type EvictionPolicyType string

const (
	// FIFOEvictionPolicyType specifies the FIFO eviction policy
	FIFOEvictionPolicyType EvictionPolicyType = "fifo"

	// LRUEvictionPolicyType specifies the LRU eviction policy
	LRUEvictionPolicyType EvictionPolicyType = "lru"

	// LFUEvictionPolicyType specifies the LFU eviction policy
	LFUEvictionPolicyType EvictionPolicyType = "lfu"
)

// CacheConfig contains configuration settings shared across all cache backends
type CacheConfig struct {
	// BackendType specifies which cache implementation to use
	BackendType CacheBackendType `yaml:"backend_type"`

	// Enabled controls whether semantic caching is active
	Enabled bool `yaml:"enabled"`

	// SimilarityThreshold defines the minimum similarity score for cache hits (0.0-1.0)
	SimilarityThreshold float32 `yaml:"similarity_threshold"`

	// MaxEntries limits the number of cached entries (for in-memory backend)
	MaxEntries int `yaml:"max_entries,omitempty"`

	// TTLSeconds sets cache entry expiration time (0 disables expiration)
	TTLSeconds int `yaml:"ttl_seconds,omitempty"`

	// EvictionPolicy defines the eviction policy for in-memory cache ("fifo", "lru", "lfu")
	EvictionPolicy EvictionPolicyType `yaml:"eviction_policy,omitempty"`

	// Redis specific settings
	Redis *config.RedisConfig `yaml:"redis,omitempty"`

	// Valkey specific settings
	Valkey *config.ValkeyConfig `yaml:"valkey,omitempty"`

	// Milvus specific settings
	Milvus *config.MilvusConfig `yaml:"milvus,omitempty"`

	// Qdrant specific settings
	Qdrant *config.QdrantConfig `yaml:"qdrant,omitempty"`

	// UseHNSW enables HNSW index for faster search in memory backend
	UseHNSW bool `yaml:"use_hnsw,omitempty"`

	// HNSWM is the number of bi-directional links per node (default: 16)
	HNSWM int `yaml:"hnsw_m,omitempty"`

	// HNSWEfConstruction is the size of dynamic candidate list during construction (default: 200)
	HNSWEfConstruction int `yaml:"hnsw_ef_construction,omitempty"`

	// Hybrid cache specific settings
	MaxMemoryEntries int `yaml:"max_memory_entries,omitempty"` // Max entries in HNSW for hybrid cache

	// EmbeddingModel specifies which embedding model to use
	// Options: "bert" (default), "qwen3", "gemma", "mmbert", "multimodal"
	EmbeddingModel string `yaml:"embedding_model,omitempty"`
}
