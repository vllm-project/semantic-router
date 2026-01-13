package cache

import (
	"sync"
	"testing"
)

// TestNewCacheManager verifies cache manager creation
func TestNewCacheManager(t *testing.T) {
	tests := []struct {
		name        string
		config      *CacheManagerConfig
		expectError bool
	}{
		{
			name: "valid config with domains",
			config: &CacheManagerConfig{
				Domains: map[string]CacheConfig{
					"medical": {
						BackendType:         InMemoryCacheType,
						Enabled:             true,
						SimilarityThreshold: 0.95,
						MaxEntries:          100,
						TTLSeconds:          3600,
						EmbeddingModel:      "bert",
					},
				},
			},
			expectError: false,
		},
		{
			name: "valid config with global cache",
			config: &CacheManagerConfig{
				GlobalCache: &CacheConfig{
					BackendType:         InMemoryCacheType,
					Enabled:             true,
					SimilarityThreshold: 0.80,
					MaxEntries:          100,
					TTLSeconds:          3600,
					EmbeddingModel:      "bert",
				},
			},
			expectError: false,
		},
		{
			name:        "nil config should error",
			config:      nil,
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cm, err := NewCacheManager(tt.config)
			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error but got none")
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
				}
				if cm == nil {
					t.Errorf("Expected cache manager but got nil")
				}
			}
		})
	}
}

// TestCacheManager_LazyInitialization verifies lazy initialization
func TestCacheManager_LazyInitialization(t *testing.T) {
	config := &CacheManagerConfig{
		Domains: map[string]CacheConfig{
			"medical": {
				BackendType:         InMemoryCacheType,
				Enabled:             true,
				SimilarityThreshold: 0.95,
				MaxEntries:          100,
				TTLSeconds:          3600,
				EmbeddingModel:      "bert",
			},
		},
	}

	cm, err := NewCacheManager(config)
	if err != nil {
		t.Fatalf("Failed to create cache manager: %v", err)
	}

	// Verify no caches are initialized yet
	initializedDomains := cm.GetInitializedDomains()
	if len(initializedDomains) != 0 {
		t.Errorf("Expected 0 initialized domains, got %d", len(initializedDomains))
	}

	// First call to GetCache should create the cache
	cache1, err := cm.GetCache("medical")
	if err != nil {
		t.Fatalf("Failed to get cache for medical domain: %v", err)
	}
	if cache1 == nil {
		t.Fatalf("Expected cache but got nil")
	}

	// Verify cache is now initialized
	initializedDomains = cm.GetInitializedDomains()
	if len(initializedDomains) != 1 {
		t.Errorf("Expected 1 initialized domain, got %d", len(initializedDomains))
	}

	// Second call should return same instance
	cache2, err := cm.GetCache("medical")
	if err != nil {
		t.Fatalf("Failed to get cache for medical domain: %v", err)
	}
	if cache1 != cache2 {
		t.Errorf("Expected same cache instance but got different instances")
	}
}

// TestCacheManager_FallbackToGlobal verifies fallback to global cache
func TestCacheManager_FallbackToGlobal(t *testing.T) {
	config := &CacheManagerConfig{
		Domains: map[string]CacheConfig{
			"medical": {
				BackendType:         InMemoryCacheType,
				Enabled:             true,
				SimilarityThreshold: 0.95,
				MaxEntries:          100,
				TTLSeconds:          3600,
				EmbeddingModel:      "bert",
			},
		},
		GlobalCache: &CacheConfig{
			BackendType:         InMemoryCacheType,
			Enabled:             true,
			SimilarityThreshold: 0.80,
			MaxEntries:          100,
			TTLSeconds:          3600,
			EmbeddingModel:      "bert",
		},
	}

	cm, err := NewCacheManager(config)
	if err != nil {
		t.Fatalf("Failed to create cache manager: %v", err)
	}

	// Request cache for unconfigured domain should use global cache
	cache, err := cm.GetCache("unconfigured_domain")
	if err != nil {
		t.Fatalf("Expected fallback to global cache, got error: %v", err)
	}
	if cache == nil {
		t.Fatalf("Expected cache but got nil")
	}
}

// TestCacheManager_DomainNotConfigured verifies error for unconfigured domain without global fallback
func TestCacheManager_DomainNotConfigured(t *testing.T) {
	config := &CacheManagerConfig{
		Domains: map[string]CacheConfig{
			"medical": {
				BackendType:         InMemoryCacheType,
				Enabled:             true,
				SimilarityThreshold: 0.95,
				MaxEntries:          100,
				TTLSeconds:          3600,
				EmbeddingModel:      "bert",
			},
		},
		// No global cache
	}

	cm, err := NewCacheManager(config)
	if err != nil {
		t.Fatalf("Failed to create cache manager: %v", err)
	}

	// Request cache for unconfigured domain should error
	cache, err := cm.GetCache("unconfigured_domain")
	if err == nil {
		t.Errorf("Expected error for unconfigured domain without global fallback")
	}
	if cache != nil {
		t.Errorf("Expected nil cache but got: %v", cache)
	}
}

// TestCacheManager_ConcurrentAccess verifies thread safety with concurrent access
func TestCacheManager_ConcurrentAccess(t *testing.T) {
	config := &CacheManagerConfig{
		Domains: map[string]CacheConfig{
			"medical": {
				BackendType:         InMemoryCacheType,
				Enabled:             true,
				SimilarityThreshold: 0.95,
				MaxEntries:          100,
				TTLSeconds:          3600,
				EmbeddingModel:      "bert",
			},
			"math": {
				BackendType:         InMemoryCacheType,
				Enabled:             true,
				SimilarityThreshold: 0.85,
				MaxEntries:          100,
				TTLSeconds:          3600,
				EmbeddingModel:      "qwen3",
			},
		},
	}

	cm, err := NewCacheManager(config)
	if err != nil {
		t.Fatalf("Failed to create cache manager: %v", err)
	}

	// Create multiple goroutines accessing the same domain cache
	const numGoroutines = 20
	var wg sync.WaitGroup
	caches := make([]CacheBackend, numGoroutines)
	errors := make([]error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			cache, err := cm.GetCache("medical")
			caches[index] = cache
			errors[index] = err
		}(i)
	}

	wg.Wait()

	// Verify all goroutines got the same cache instance
	for i := 0; i < numGoroutines; i++ {
		if errors[i] != nil {
			t.Errorf("Goroutine %d got error: %v", i, errors[i])
		}
		if caches[i] == nil {
			t.Errorf("Goroutine %d got nil cache", i)
		}
		if i > 0 && caches[i] != caches[0] {
			t.Errorf("Goroutine %d got different cache instance", i)
		}
	}

	// Verify only 1 domain was initialized (medical)
	initializedDomains := cm.GetInitializedDomains()
	if len(initializedDomains) != 1 {
		t.Errorf("Expected 1 initialized domain, got %d", len(initializedDomains))
	}
}

// TestCacheManager_GetCacheStats verifies cache statistics retrieval
func TestCacheManager_GetCacheStats(t *testing.T) {
	config := &CacheManagerConfig{
		Domains: map[string]CacheConfig{
			"medical": {
				BackendType:         InMemoryCacheType,
				Enabled:             true,
				SimilarityThreshold: 0.95,
				MaxEntries:          100,
				TTLSeconds:          3600,
				EmbeddingModel:      "bert",
			},
		},
	}

	cm, err := NewCacheManager(config)
	if err != nil {
		t.Fatalf("Failed to create cache manager: %v", err)
	}

	// Before initialization, GetCacheStats should error
	_, err = cm.GetCacheStats("medical")
	if err == nil {
		t.Errorf("Expected error for stats of uninitialized cache")
	}

	// Initialize cache
	_, err = cm.GetCache("medical")
	if err != nil {
		t.Fatalf("Failed to get cache: %v", err)
	}

	// After initialization, GetCacheStats should work
	stats, err := cm.GetCacheStats("medical")
	if err != nil {
		t.Errorf("Unexpected error getting stats: %v", err)
	}
	// Verify stats are returned (check for TotalEntries field)
	if stats.TotalEntries < 0 {
		t.Errorf("Expected valid TotalEntries in stats")
	}
}

// TestCacheManager_GetAllStats verifies all cache statistics retrieval
func TestCacheManager_GetAllStats(t *testing.T) {
	config := &CacheManagerConfig{
		Domains: map[string]CacheConfig{
			"medical": {
				BackendType:         InMemoryCacheType,
				Enabled:             true,
				SimilarityThreshold: 0.95,
				MaxEntries:          100,
				TTLSeconds:          3600,
				EmbeddingModel:      "bert",
			},
			"math": {
				BackendType:         InMemoryCacheType,
				Enabled:             true,
				SimilarityThreshold: 0.85,
				MaxEntries:          100,
				TTLSeconds:          3600,
				EmbeddingModel:      "qwen3",
			},
		},
	}

	cm, err := NewCacheManager(config)
	if err != nil {
		t.Fatalf("Failed to create cache manager: %v", err)
	}

	// Initialize both caches
	_, err = cm.GetCache("medical")
	if err != nil {
		t.Fatalf("Failed to get medical cache: %v", err)
	}
	_, err = cm.GetCache("math")
	if err != nil {
		t.Fatalf("Failed to get math cache: %v", err)
	}

	// Get all stats
	allStats := cm.GetAllStats()
	if len(allStats) != 2 {
		t.Errorf("Expected 2 cache stats, got %d", len(allStats))
	}
	if _, exists := allStats["medical"]; !exists {
		t.Errorf("Expected medical cache in stats")
	}
	if _, exists := allStats["math"]; !exists {
		t.Errorf("Expected math cache in stats")
	}
}

// TestCacheManager_IsEnabled verifies enabled check for domains
func TestCacheManager_IsEnabled(t *testing.T) {
	config := &CacheManagerConfig{
		Domains: map[string]CacheConfig{
			"medical": {
				BackendType:         InMemoryCacheType,
				Enabled:             true,
				SimilarityThreshold: 0.95,
				MaxEntries:          100,
				TTLSeconds:          3600,
				EmbeddingModel:      "bert",
			},
			"disabled_domain": {
				BackendType:         InMemoryCacheType,
				Enabled:             false,
				SimilarityThreshold: 0.95,
				MaxEntries:          100,
				TTLSeconds:          3600,
				EmbeddingModel:      "bert",
			},
		},
		GlobalCache: &CacheConfig{
			BackendType:         InMemoryCacheType,
			Enabled:             true,
			SimilarityThreshold: 0.80,
			MaxEntries:          100,
			TTLSeconds:          3600,
			EmbeddingModel:      "bert",
		},
	}

	cm, err := NewCacheManager(config)
	if err != nil {
		t.Fatalf("Failed to create cache manager: %v", err)
	}

	tests := []struct {
		domain   string
		expected bool
	}{
		{"medical", true},
		{"disabled_domain", false},
		{"unconfigured_domain", true}, // Should use global cache
	}

	for _, tt := range tests {
		t.Run(tt.domain, func(t *testing.T) {
			enabled := cm.IsEnabled(tt.domain)
			if enabled != tt.expected {
				t.Errorf("IsEnabled(%s) = %v, expected %v", tt.domain, enabled, tt.expected)
			}
		})
	}
}

// TestCacheManager_CloseAll verifies closing all caches
func TestCacheManager_CloseAll(t *testing.T) {
	config := &CacheManagerConfig{
		Domains: map[string]CacheConfig{
			"medical": {
				BackendType:         InMemoryCacheType,
				Enabled:             true,
				SimilarityThreshold: 0.95,
				MaxEntries:          100,
				TTLSeconds:          3600,
				EmbeddingModel:      "bert",
			},
			"math": {
				BackendType:         InMemoryCacheType,
				Enabled:             true,
				SimilarityThreshold: 0.85,
				MaxEntries:          100,
				TTLSeconds:          3600,
				EmbeddingModel:      "qwen3",
			},
		},
	}

	cm, err := NewCacheManager(config)
	if err != nil {
		t.Fatalf("Failed to create cache manager: %v", err)
	}

	// Initialize both caches
	_, err = cm.GetCache("medical")
	if err != nil {
		t.Fatalf("Failed to get medical cache: %v", err)
	}
	_, err = cm.GetCache("math")
	if err != nil {
		t.Fatalf("Failed to get math cache: %v", err)
	}

	// Verify caches are initialized
	initializedDomains := cm.GetInitializedDomains()
	if len(initializedDomains) != 2 {
		t.Errorf("Expected 2 initialized domains before close, got %d", len(initializedDomains))
	}

	// Close all caches
	err = cm.CloseAll()
	if err != nil {
		t.Errorf("Unexpected error closing caches: %v", err)
	}

	// Verify all caches are closed
	initializedDomains = cm.GetInitializedDomains()
	if len(initializedDomains) != 0 {
		t.Errorf("Expected 0 initialized domains after close, got %d", len(initializedDomains))
	}
}

// TestCacheManager_MultipleDomains verifies managing multiple domain caches
func TestCacheManager_MultipleDomains(t *testing.T) {
	domains := []string{"medical", "math", "psychology", "biology", "chemistry"}
	domainConfigs := make(map[string]CacheConfig)

	for _, domain := range domains {
		domainConfigs[domain] = CacheConfig{
			BackendType:         InMemoryCacheType,
			Enabled:             true,
			SimilarityThreshold: 0.85,
			MaxEntries:          100,
			TTLSeconds:          3600,
			EmbeddingModel:      "bert",
		}
	}

	config := &CacheManagerConfig{
		Domains: domainConfigs,
	}

	cm, err := NewCacheManager(config)
	if err != nil {
		t.Fatalf("Failed to create cache manager: %v", err)
	}

	// Initialize all domain caches
	for _, domain := range domains {
		cache, err := cm.GetCache(domain)
		if err != nil {
			t.Errorf("Failed to get cache for %s: %v", domain, err)
		}
		if cache == nil {
			t.Errorf("Got nil cache for %s", domain)
		}
	}

	// Verify all domains are initialized
	initializedDomains := cm.GetInitializedDomains()
	if len(initializedDomains) != len(domains) {
		t.Errorf("Expected %d initialized domains, got %d", len(domains), len(initializedDomains))
	}

	// Verify GetAllStats returns all domains
	allStats := cm.GetAllStats()
	if len(allStats) != len(domains) {
		t.Errorf("Expected %d domain stats, got %d", len(domains), len(allStats))
	}
}
