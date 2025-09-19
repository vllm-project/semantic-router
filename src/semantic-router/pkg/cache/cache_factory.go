package cache

import (
	"fmt"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

// NewCacheBackend creates a cache backend instance from the provided configuration
func NewCacheBackend(config CacheConfig) (CacheBackend, error) {
	if !config.Enabled {
		// Create a disabled cache backend
		observability.Debugf("Cache disabled - creating disabled in-memory cache backend")
		return NewInMemoryCache(InMemoryCacheOptions{
			Enabled: false,
		}), nil
	}

	switch config.BackendType {
	case InMemoryCacheType, "":
		// Use in-memory cache as the default backend
		observability.Debugf("Creating in-memory cache backend - MaxEntries: %d, TTL: %ds, Threshold: %.3f",
			config.MaxEntries, config.TTLSeconds, config.SimilarityThreshold)
		options := InMemoryCacheOptions{
			Enabled:             config.Enabled,
			SimilarityThreshold: config.SimilarityThreshold,
			MaxEntries:          config.MaxEntries,
			TTLSeconds:          config.TTLSeconds,
			EvictionPolicy:      config.EvictionPolicy,
		}
		return NewInMemoryCache(options), nil

	case MilvusCacheType:
		observability.Debugf("Creating Milvus cache backend - ConfigPath: %s, TTL: %ds, Threshold: %.3f",
			config.BackendConfigPath, config.TTLSeconds, config.SimilarityThreshold)
		if config.BackendConfigPath == "" {
			return nil, fmt.Errorf("backend_config_path is required for Milvus cache backend")
		}

		// Ensure the Milvus configuration file exists
		if _, err := os.Stat(config.BackendConfigPath); os.IsNotExist(err) {
			observability.Debugf("Milvus config file not found: %s", config.BackendConfigPath)
			return nil, fmt.Errorf("Milvus config file not found: %s", config.BackendConfigPath)
		}
		observability.Debugf("Milvus config file found: %s", config.BackendConfigPath)

		options := MilvusCacheOptions{
			Enabled:             config.Enabled,
			SimilarityThreshold: config.SimilarityThreshold,
			TTLSeconds:          config.TTLSeconds,
			ConfigPath:          config.BackendConfigPath,
		}
		return NewMilvusCache(options)

	default:
		observability.Debugf("Unsupported cache backend type: %s", config.BackendType)
		return nil, fmt.Errorf("unsupported cache backend type: %s", config.BackendType)
	}
}

// ValidateCacheConfig validates cache configuration parameters
func ValidateCacheConfig(config CacheConfig) error {
	if !config.Enabled {
		return nil // Skip validation for disabled cache
	}

	// Check similarity threshold range
	if config.SimilarityThreshold < 0.0 || config.SimilarityThreshold > 1.0 {
		return fmt.Errorf("similarity_threshold must be between 0.0 and 1.0, got: %f", config.SimilarityThreshold)
	}

	// Check TTL value
	if config.TTLSeconds < 0 {
		return fmt.Errorf("ttl_seconds cannot be negative, got: %d", config.TTLSeconds)
	}

	// Check max entries for in-memory cache
	if config.BackendType == InMemoryCacheType || config.BackendType == "" {
		if config.MaxEntries < 0 {
			return fmt.Errorf("max_entries cannot be negative for in-memory cache, got: %d", config.MaxEntries)
		}
	}

	// Check backend-specific requirements
	switch config.BackendType {
	case MilvusCacheType:
		if config.BackendConfigPath == "" {
			return fmt.Errorf("backend_config_path is required for Milvus cache backend")
		}
	}

	return nil
}

// GetDefaultCacheConfig provides sensible default cache configuration values
func GetDefaultCacheConfig() CacheConfig {
	return CacheConfig{
		BackendType:         InMemoryCacheType,
		Enabled:             true,
		SimilarityThreshold: 0.8,
		MaxEntries:          1000,
		TTLSeconds:          3600,
	}
}

// CacheBackendInfo describes the capabilities and features of a cache backend
type CacheBackendInfo struct {
	Type        CacheBackendType `json:"type"`
	Name        string           `json:"name"`
	Description string           `json:"description"`
	Features    []string         `json:"features"`
}

// GetAvailableCacheBackends returns metadata for all supported cache backends
func GetAvailableCacheBackends() []CacheBackendInfo {
	return []CacheBackendInfo{
		{
			Type:        InMemoryCacheType,
			Name:        "In-Memory Cache",
			Description: "High-performance in-memory semantic cache with BERT embeddings",
			Features: []string{
				"Fast access",
				"No external dependencies",
				"Automatic memory management",
				"TTL support",
				"Entry limit support",
			},
		},
		{
			Type:        MilvusCacheType,
			Name:        "Milvus Vector Database",
			Description: "Enterprise-grade semantic cache powered by Milvus vector database",
			Features: []string{
				"Highly scalable",
				"Persistent storage",
				"Distributed architecture",
				"Advanced indexing",
				"High availability",
				"TTL support",
			},
		},
	}
}
