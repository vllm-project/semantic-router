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

// LookupResult carries the request-owned outcome of one lookup. A hit includes
// the matched score; a below-threshold miss may include its rejected candidate's
// score. Errors carry no score.
type LookupResult struct {
	ResponseBody []byte
	Found        bool
	Similarity   float32
	StoredAt     time.Time
	ExpiresAt    time.Time
	Age          time.Duration
	AgeKnown     bool
}

// lookupResultFromTimestamps constructs a successful LookupResult and calculates Age / AgeKnown.
func lookupResultFromTimestamps(responseBody []byte, similarity float32, storedAt, expiresAt time.Time) LookupResult {
	var age time.Duration
	var ageKnown bool
	if !storedAt.IsZero() {
		age = time.Since(storedAt)
		ageKnown = true
	}
	return LookupResult{
		ResponseBody: responseBody,
		Found:        true,
		Similarity:   similarity,
		StoredAt:     storedAt,
		ExpiresAt:    expiresAt,
		Age:          age,
		AgeKnown:     ageKnown,
	}
}

// ExactCacheBackend is an optional exact-response fast path implemented by
// key-value-capable cache backends.
type ExactCacheBackend interface {
	FindExact(ctx context.Context, partition string, fingerprint string) (LookupResult, error)
	AddExact(ctx context.Context, partition string, fingerprint string, responseBody []byte, ttlSeconds int) error
}

// ctxErr treats a nil context as having no error.
//
// Embedding cannot be interrupted mid-flight; callers check before embedding
// and before publishing state.
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

// releaseOnFailure releases a partially constructed client when setup fails.
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

	// AddEntry stores a complete request-response pair in the cache
	AddEntry(ctx context.Context, requestID string, model string, query string, requestBody, responseBody []byte, ttlSeconds int) error

	// LookupSimilarWithThreshold returns response data and similarity from the
	// same lookup operation. Request paths should use this method instead of
	// backend-global similarity state. A canceled or expired context is
	// reported as an error rather than as a miss.
	LookupSimilarWithThreshold(ctx context.Context, model string, query string, threshold float32) (LookupResult, error)

	// Close releases all resources held by the cache backend
	Close() error

	// GetStats provides cache performance and usage metrics
	GetStats() CacheStats
}

// LegacyCacheBackend is the temporary backend-implementation seam. New request
// paths depend on TypedCacheStore; only backend tests and migration adapters
// should use these two-phase and convenience methods.
//
// These methods stay context-free on purpose: they are not on a request path,
// so there is no caller context to honor. Implementations forward
// context.Background() to the context-aware core.
type LegacyCacheBackend interface {
	CacheBackend
	AddPendingRequest(requestID string, model string, query string, requestBody []byte, ttlSeconds int) error
	UpdateWithResponse(requestID string, responseBody []byte, ttlSeconds int) error
	FindSimilar(model string, query string) ([]byte, bool, error)
	FindSimilarWithThreshold(model string, query string, threshold float32) ([]byte, bool, error)
}

// Compile-time assertions keep production implementations aligned with CacheBackend.
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
	TotalEntries        int        `json:"total_entries"`
	HitCount            int64      `json:"hit_count"`
	MissCount           int64      `json:"miss_count"`
	HitRatio            float64    `json:"hit_ratio"`
	LastCleanupTime     *time.Time `json:"last_cleanup_time,omitempty"`
	ExactHitCount       int64      `json:"exact_hit_count"`
	SemanticHitCount    int64      `json:"semantic_hit_count"`
	L1HitCount          int64      `json:"l1_hit_count"`
	L2HitCount          int64      `json:"l2_hit_count"`
	L1Entries           int        `json:"l1_entries"`
	SingleflightWaiters int64      `json:"singleflight_waiters"`
	InvalidationCount   int64      `json:"invalidation_count"`
	StaleMissCount      int64      `json:"stale_miss_count"`
	FailOpenCount       int64      `json:"fail_open_count"`
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
