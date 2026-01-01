package cache

import (
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// CacheManager manages multiple cache backends for different domains
// Each domain can have its own cache instance with different configurations
// (embedding models, similarity thresholds, TTLs, etc.)
type CacheManager struct {
	caches map[string]CacheBackend // domain -> cache instance
	config *CacheManagerConfig
	mu     sync.RWMutex
}

// CacheManagerConfig holds configuration for the cache manager
type CacheManagerConfig struct {
	// Per-domain cache configurations
	Domains map[string]CacheConfig `yaml:"domains"`

	// Fallback global cache config (used when domain not configured)
	GlobalCache *CacheConfig `yaml:"global_cache,omitempty"`
}

// NewCacheManager creates a new cache manager instance
func NewCacheManager(config *CacheManagerConfig) (*CacheManager, error) {
	if config == nil {
		return nil, fmt.Errorf("cache manager config cannot be nil")
	}

	logging.Debugf("CacheManager: initializing with %d domain configs", len(config.Domains))

	return &CacheManager{
		caches: make(map[string]CacheBackend),
		config: config,
	}, nil
}

// GetCache returns the cache for a specific domain
// Lazy initializes the cache on first access (thread-safe)
func (cm *CacheManager) GetCache(domain string) (CacheBackend, error) {
	// Fast path: check if cache already exists (read lock)
	cm.mu.RLock()
	if cache, exists := cm.caches[domain]; exists {
		cm.mu.RUnlock()
		logging.Debugf("CacheManager: returning existing cache for domain '%s'", domain)
		return cache, nil
	}
	cm.mu.RUnlock()

	// Slow path: create cache (write lock)
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// Double-check after acquiring write lock (another goroutine might have created it)
	if cache, exists := cm.caches[domain]; exists {
		logging.Debugf("CacheManager: cache for domain '%s' was created by another goroutine", domain)
		return cache, nil
	}

	// Get domain config
	domainConfig, hasDomainConfig := cm.config.Domains[domain]

	if !hasDomainConfig {
		// Try to use global cache as fallback
		if cm.config.GlobalCache != nil {
			logging.Infof("CacheManager: domain '%s' not configured, using global cache fallback", domain)
			domainConfig = *cm.config.GlobalCache
			// Override namespace to match domain for consistency
			if domainConfig.BackendType == RedisCacheType || domainConfig.BackendType == MilvusCacheType {
				// For Redis/Milvus, we can set namespace dynamically
				logging.Debugf("CacheManager: setting namespace to '%s' for global cache fallback", domain)
			}
		} else {
			logging.Warnf("CacheManager: no cache configuration for domain '%s' and no global fallback", domain)
			return nil, fmt.Errorf("no cache configuration for domain: %s", domain)
		}
	}

	// Create cache backend for domain
	logging.Infof("CacheManager: creating cache for domain '%s' (backend: %s, embedding: %s, threshold: %.2f)",
		domain, domainConfig.BackendType, domainConfig.EmbeddingModel, domainConfig.SimilarityThreshold)

	cache, err := NewCacheBackend(domainConfig)
	if err != nil {
		logging.Errorf("CacheManager: failed to create cache for domain '%s': %v", domain, err)
		return nil, fmt.Errorf("failed to create cache for domain %s: %w", domain, err)
	}

	// Store cache instance
	cm.caches[domain] = cache
	logging.Infof("CacheManager: successfully created and cached instance for domain '%s'", domain)

	return cache, nil
}

// GetCacheStats returns statistics for a specific domain's cache
func (cm *CacheManager) GetCacheStats(domain string) (CacheStats, error) {
	cm.mu.RLock()
	cache, exists := cm.caches[domain]
	cm.mu.RUnlock()

	if !exists {
		return CacheStats{}, fmt.Errorf("cache for domain %s not initialized", domain)
	}

	return cache.GetStats(), nil
}

// GetAllStats returns statistics for all initialized caches
func (cm *CacheManager) GetAllStats() map[string]CacheStats {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	stats := make(map[string]CacheStats, len(cm.caches))
	for domain, cache := range cm.caches {
		stats[domain] = cache.GetStats()
	}

	return stats
}

// CloseAll closes all cache backends
func (cm *CacheManager) CloseAll() error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	logging.Infof("CacheManager: closing all cache instances (%d total)", len(cm.caches))

	var firstError error
	for domain, cache := range cm.caches {
		if err := cache.Close(); err != nil {
			logging.Errorf("CacheManager: error closing cache for domain %s: %v", domain, err)
			if firstError == nil {
				firstError = err
			}
		} else {
			logging.Debugf("CacheManager: successfully closed cache for domain '%s'", domain)
		}
	}

	// Clear the cache map
	cm.caches = make(map[string]CacheBackend)

	if firstError != nil {
		return fmt.Errorf("errors occurred while closing caches: %w", firstError)
	}

	logging.Infof("CacheManager: all caches closed successfully")
	return nil
}

// GetInitializedDomains returns a list of domains that have initialized caches
func (cm *CacheManager) GetInitializedDomains() []string {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	domains := make([]string, 0, len(cm.caches))
	for domain := range cm.caches {
		domains = append(domains, domain)
	}

	return domains
}

// IsEnabled checks if caching is enabled for a specific domain
func (cm *CacheManager) IsEnabled(domain string) bool {
	// Check if domain has config
	if config, exists := cm.config.Domains[domain]; exists {
		return config.Enabled
	}

	// Check global config
	if cm.config.GlobalCache != nil {
		return cm.config.GlobalCache.Enabled
	}

	return false
}
