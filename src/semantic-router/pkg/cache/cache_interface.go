package cache

import (
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

// CacheBackend defines the interface for semantic cache implementations
type CacheBackend interface {
	// IsEnabled returns whether caching is currently active
	IsEnabled() bool

	// CheckConnection verifies the cache backend connection is healthy
	// Returns nil if the connection is healthy, error otherwise
	// For local caches (in-memory), this may be a no-op
	CheckConnection() error

	// AddPendingRequest stores a request awaiting its response
	AddPendingRequest(requestID string, model string, query string, requestBody []byte, ttlSeconds int) error

	// UpdateWithResponse completes a pending request with the received response
	UpdateWithResponse(requestID string, responseBody []byte, ttlSeconds int) error

	// AddEntry stores a complete request-response pair in the cache
	AddEntry(requestID string, model string, query string, requestBody, responseBody []byte, ttlSeconds int) error

	// FindSimilar searches for semantically similar cached requests
	// Returns the cached response, match status, and any error
	FindSimilar(model string, query string) ([]byte, bool, error)

	// FindSimilarWithThreshold searches for semantically similar cached requests using a specific threshold
	// This allows category-specific similarity thresholds
	// Returns the cached response, match status, and any error
	FindSimilarWithThreshold(model string, query string, threshold float32) ([]byte, bool, error)

	// Close releases all resources held by the cache backend
	Close() error

	// GetStats provides cache performance and usage metrics
	GetStats() CacheStats
}

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

	// HybridCacheType specifies the hybrid HNSW + Milvus backend
	HybridCacheType CacheBackendType = "hybrid"
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

	// Milvus specific settings
	Milvus *config.MilvusConfig `yaml:"milvus,omitempty"`

	// BackendConfigPath points to backend-specific configuration files (Deprecated)
	BackendConfigPath string `yaml:"backend_config_path,omitempty"`

	// UseHNSW enables HNSW index for faster search in memory backend
	UseHNSW bool `yaml:"use_hnsw,omitempty"`

	// HNSWM is the number of bi-directional links per node (default: 16)
	HNSWM int `yaml:"hnsw_m,omitempty"`

	// HNSWEfConstruction is the size of dynamic candidate list during construction (default: 200)
	HNSWEfConstruction int `yaml:"hnsw_ef_construction,omitempty"`

	// Hybrid cache specific settings
	MaxMemoryEntries int `yaml:"max_memory_entries,omitempty"` // Max entries in HNSW for hybrid cache

	// EmbeddingModel specifies which embedding model to use
	// Options: "bert" (default), "qwen3", "gemma"
	EmbeddingModel string `yaml:"embedding_model,omitempty"`
}
