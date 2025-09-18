package cache

import "time"

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
}

// CacheBackend defines the interface for semantic cache implementations
type CacheBackend interface {
	// IsEnabled returns whether caching is currently active
	IsEnabled() bool

	// AddPendingRequest stores a request awaiting its response
	AddPendingRequest(requestID string, model string, query string, requestBody []byte) error

	// UpdateWithResponse completes a pending request with the received response
	UpdateWithResponse(requestID string, responseBody []byte) error

	// AddEntry stores a complete request-response pair in the cache
	AddEntry(requestID string, model string, query string, requestBody, responseBody []byte) error

	// FindSimilar searches for semantically similar cached requests
	// Returns the cached response, match status, and any error
	FindSimilar(model string, query string) ([]byte, bool, error)

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

	// BackendConfigPath points to backend-specific configuration files
	BackendConfigPath string `yaml:"backend_config_path,omitempty"`
}
