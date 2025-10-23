package cache

import (
	"fmt"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// TestInMemoryCacheIntegration tests the in-memory cache integration
func TestInMemoryCacheIntegration(t *testing.T) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		t.Skipf("Failed to initialize BERT model: %v", err)
	}

	cache := NewInMemoryCache(InMemoryCacheOptions{
		Enabled:             true,
		MaxEntries:          2,
		SimilarityThreshold: 0.9,
		EvictionPolicy:      "lfu",
		TTLSeconds:          0,
	})

	t.Run("InMemoryCacheIntegration", func(t *testing.T) {
		// Step 1: Add first entry
		err := cache.AddEntry("req1", "test-model", "Hello world",
			[]byte("request1"), []byte("response1"))
		if err != nil {
			t.Fatalf("Failed to add first entry: %v", err)
		}

		// Step 2: Add second entry (cache at capacity)
		err = cache.AddEntry("req2", "test-model", "Good morning",
			[]byte("request2"), []byte("response2"))
		if err != nil {
			t.Fatalf("Failed to add second entry: %v", err)
		}

		// Verify
		if len(cache.entries) != 2 {
			t.Errorf("Expected 2 entries, got %d", len(cache.entries))
		}
		if cache.entries[1].RequestID != "req2" {
			t.Errorf("Expected req2 to be the second entry, got %s", cache.entries[1].RequestID)
		}

		// Step 3: Access first entry multiple times to increase its frequency
		for range 2 {
			responseBody, found, findErr := cache.FindSimilar("test-model", "Hello world")
			if findErr != nil {
				t.Logf("FindSimilar failed (expected due to high threshold): %v", findErr)
			}
			if !found {
				t.Errorf("Expected to find similar entry for first query")
			}
			if string(responseBody) != "response1" {
				t.Errorf("Expected response1, got %s", string(responseBody))
			}
		}

		// Step 4: Access second entry once
		responseBody, found, err := cache.FindSimilar("test-model", "Good morning")
		if err != nil {
			t.Logf("FindSimilar failed (expected due to high threshold): %v", err)
		}
		if !found {
			t.Errorf("Expected to find similar entry for second query")
		}
		if string(responseBody) != "response2" {
			t.Errorf("Expected response2, got %s", string(responseBody))
		}

		// Step 5: Add third entry - should trigger LFU eviction
		err = cache.AddEntry("req3", "test-model", "Bye",
			[]byte("request3"), []byte("response3"))
		if err != nil {
			t.Fatalf("Failed to add third entry: %v", err)
		}

		// Verify
		if len(cache.entries) != 2 {
			t.Errorf("Expected 2 entries after eviction, got %d", len(cache.entries))
		}
		if cache.entries[0].RequestID != "req1" {
			t.Errorf("Expected req1 to be the first entry, got %s", cache.entries[0].RequestID)
		}
		if cache.entries[1].RequestID != "req3" {
			t.Errorf("Expected req3 to be the second entry, got %s", cache.entries[1].RequestID)
		}
		if cache.entries[0].HitCount != 2 {
			t.Errorf("Expected HitCount to be 2, got %d", cache.entries[0].HitCount)
		}
		if cache.entries[1].HitCount != 0 {
			t.Errorf("Expected HitCount to be 0, got %d", cache.entries[1].HitCount)
		}
	})
}

// TestInMemoryCachePendingRequestWorkflow tests the in-memory cache pending request workflow
func TestInMemoryCachePendingRequestWorkflow(t *testing.T) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		t.Skipf("Failed to initialize BERT model: %v", err)
	}

	cache := NewInMemoryCache(InMemoryCacheOptions{
		Enabled:        true,
		MaxEntries:     2,
		EvictionPolicy: "lru",
	})

	t.Run("PendingRequestFlow", func(t *testing.T) {
		// Step 1: Add pending request
		err := cache.AddPendingRequest("req1", "test-model", "test query", []byte("request"))
		if err != nil {
			t.Fatalf("Failed to add pending request: %v", err)
		}

		// Verify
		if len(cache.entries) != 1 {
			t.Errorf("Expected 1 entry after AddPendingRequest, got %d", len(cache.entries))
		}

		if string(cache.entries[0].ResponseBody) != "" {
			t.Error("Expected ResponseBody to be empty for pending request")
		}

		// Step 2: Update with response
		err = cache.UpdateWithResponse("req1", []byte("response1"))
		if err != nil {
			t.Fatalf("Failed to update with response: %v", err)
		}

		// Step 3: Try to find similar
		response, found, err := cache.FindSimilar("test-model", "test query")
		if err != nil {
			t.Logf("FindSimilar error (may be due to embedding): %v", err)
		}

		if !found {
			t.Errorf("Expected to find completed entry after UpdateWithResponse")
		}
		if string(response) != "response1" {
			t.Errorf("Expected response1, got %s", string(response))
		}
	})
}

// TestEvictionPolicySelection tests that the correct policy is selected
func TestEvictionPolicySelection(t *testing.T) {
	testCases := []struct {
		policy   string
		expected string
	}{
		{"lru", "*cache.LRUPolicy"},
		{"lfu", "*cache.LFUPolicy"},
		{"fifo", "*cache.FIFOPolicy"},
		{"", "*cache.FIFOPolicy"},        // Default
		{"invalid", "*cache.FIFOPolicy"}, // Default fallback
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("Policy_%s", tc.policy), func(t *testing.T) {
			cache := NewInMemoryCache(InMemoryCacheOptions{
				EvictionPolicy: EvictionPolicyType(tc.policy),
			})

			policyType := fmt.Sprintf("%T", cache.evictionPolicy)
			if policyType != tc.expected {
				t.Errorf("Expected policy type %s, got %s", tc.expected, policyType)
			}
		})
	}
}

// TestInMemoryCacheHNSW tests the HNSW index functionality
func TestInMemoryCacheHNSW(t *testing.T) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		t.Skipf("Failed to initialize BERT model: %v", err)
	}

	// Test with HNSW enabled
	cacheHNSW := NewInMemoryCache(InMemoryCacheOptions{
		Enabled:             true,
		MaxEntries:          100,
		SimilarityThreshold: 0.85,
		TTLSeconds:          0,
		UseHNSW:             true,
		HNSWM:               16,
		HNSWEfConstruction:  200,
	})

	// Test without HNSW (linear search)
	cacheLinear := NewInMemoryCache(InMemoryCacheOptions{
		Enabled:             true,
		MaxEntries:          100,
		SimilarityThreshold: 0.85,
		TTLSeconds:          0,
		UseHNSW:             false,
	})

	testQueries := []struct {
		query    string
		model    string
		response string
	}{
		{"What is machine learning?", "test-model", "ML is a subset of AI"},
		{"Explain neural networks", "test-model", "NNs are inspired by the brain"},
		{"How does backpropagation work?", "test-model", "Backprop calculates gradients"},
		{"What is deep learning?", "test-model", "DL uses multiple layers"},
		{"Define artificial intelligence", "test-model", "AI mimics human intelligence"},
	}

	t.Run("HNSW_Basic_Operations", func(t *testing.T) {
		// Add entries to both caches
		for i, q := range testQueries {
			reqID := fmt.Sprintf("req%d", i)
			err := cacheHNSW.AddEntry(reqID, q.model, q.query, []byte(q.query), []byte(q.response))
			if err != nil {
				t.Fatalf("Failed to add entry to HNSW cache: %v", err)
			}

			err = cacheLinear.AddEntry(reqID, q.model, q.query, []byte(q.query), []byte(q.response))
			if err != nil {
				t.Fatalf("Failed to add entry to linear cache: %v", err)
			}
		}

		// Verify HNSW index was built
		if cacheHNSW.hnswIndex == nil {
			t.Fatal("HNSW index is nil")
		}
		if len(cacheHNSW.hnswIndex.nodes) != len(testQueries) {
			t.Errorf("Expected %d HNSW nodes, got %d", len(testQueries), len(cacheHNSW.hnswIndex.nodes))
		}

		// Test exact match search
		response, found, err := cacheHNSW.FindSimilar("test-model", "What is machine learning?")
		if err != nil {
			t.Fatalf("HNSW FindSimilar error: %v", err)
		}
		if !found {
			t.Error("HNSW should find exact match")
		}
		if string(response) != "ML is a subset of AI" {
			t.Errorf("Expected 'ML is a subset of AI', got %s", string(response))
		}

		// Test similar query search
		response, found, err = cacheHNSW.FindSimilar("test-model", "What is ML?")
		if err != nil {
			t.Logf("HNSW FindSimilar error (may not find due to threshold): %v", err)
		}
		if found {
			t.Logf("HNSW found similar entry: %s", string(response))
		}

		// Compare stats
		statsHNSW := cacheHNSW.GetStats()
		statsLinear := cacheLinear.GetStats()

		t.Logf("HNSW Cache Stats: Entries=%d, Hits=%d, Misses=%d, HitRatio=%.2f",
			statsHNSW.TotalEntries, statsHNSW.HitCount, statsHNSW.MissCount, statsHNSW.HitRatio)
		t.Logf("Linear Cache Stats: Entries=%d, Hits=%d, Misses=%d, HitRatio=%.2f",
			statsLinear.TotalEntries, statsLinear.HitCount, statsLinear.MissCount, statsLinear.HitRatio)
	})

	t.Run("HNSW_Rebuild_After_Cleanup", func(t *testing.T) {
		// Create cache with short TTL
		cacheTTL := NewInMemoryCache(InMemoryCacheOptions{
			Enabled:             true,
			MaxEntries:          100,
			SimilarityThreshold: 0.85,
			TTLSeconds:          1,
			UseHNSW:             true,
			HNSWM:               16,
			HNSWEfConstruction:  200,
		})

		// Add an entry
		err := cacheTTL.AddEntry("req1", "test-model", "test query", []byte("request"), []byte("response"))
		if err != nil {
			t.Fatalf("Failed to add entry: %v", err)
		}

		initialNodes := len(cacheTTL.hnswIndex.nodes)
		if initialNodes != 1 {
			t.Errorf("Expected 1 HNSW node initially, got %d", initialNodes)
		}

		// Manually trigger cleanup (in real scenario, TTL would expire)
		cacheTTL.mu.Lock()
		cacheTTL.cleanupExpiredEntries()
		cacheTTL.mu.Unlock()

		t.Logf("After cleanup: %d entries, %d HNSW nodes",
			len(cacheTTL.entries), len(cacheTTL.hnswIndex.nodes))
	})
}

// ===== Benchmark Tests =====

// BenchmarkInMemoryCacheSearch benchmarks search performance with and without HNSW
func BenchmarkInMemoryCacheSearch(b *testing.B) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	// Test different cache sizes
	cacheSizes := []int{100, 500, 1000, 5000}

	for _, size := range cacheSizes {
		// Prepare test data
		entries := make([]struct {
			query    string
			response string
		}, size)

		for i := 0; i < size; i++ {
			entries[i].query = fmt.Sprintf("Test query number %d about machine learning and AI", i)
			entries[i].response = fmt.Sprintf("Response %d", i)
		}

		// Benchmark Linear Search
		b.Run(fmt.Sprintf("LinearSearch_%d_entries", size), func(b *testing.B) {
			cache := NewInMemoryCache(InMemoryCacheOptions{
				Enabled:             true,
				MaxEntries:          size * 2,
				SimilarityThreshold: 0.85,
				TTLSeconds:          0,
				UseHNSW:             false,
			})

			// Populate cache
			for i, entry := range entries {
				reqID := fmt.Sprintf("req%d", i)
				_ = cache.AddEntry(reqID, "test-model", entry.query, []byte(entry.query), []byte(entry.response))
			}

			// Benchmark search
			searchQuery := "What is machine learning and artificial intelligence?"
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _, _ = cache.FindSimilar("test-model", searchQuery)
			}
		})

		// Benchmark HNSW Search
		b.Run(fmt.Sprintf("HNSWSearch_%d_entries", size), func(b *testing.B) {
			cache := NewInMemoryCache(InMemoryCacheOptions{
				Enabled:             true,
				MaxEntries:          size * 2,
				SimilarityThreshold: 0.85,
				TTLSeconds:          0,
				UseHNSW:             true,
				HNSWM:               16,
				HNSWEfConstruction:  200,
			})

			// Populate cache
			for i, entry := range entries {
				reqID := fmt.Sprintf("req%d", i)
				_ = cache.AddEntry(reqID, "test-model", entry.query, []byte(entry.query), []byte(entry.response))
			}

			// Benchmark search
			searchQuery := "What is machine learning and artificial intelligence?"
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _, _ = cache.FindSimilar("test-model", searchQuery)
			}
		})
	}
}

// BenchmarkHNSWIndexConstruction benchmarks HNSW index construction time
func BenchmarkHNSWIndexConstruction(b *testing.B) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	entryCounts := []int{100, 500, 1000, 5000}

	for _, count := range entryCounts {
		b.Run(fmt.Sprintf("AddEntries_%d", count), func(b *testing.B) {
			// Generate test queries outside the benchmark loop
			testQueries := make([]string, count)
			for i := 0; i < count; i++ {
				testQueries[i] = fmt.Sprintf("Query %d: machine learning deep neural networks", i)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				cache := NewInMemoryCache(InMemoryCacheOptions{
					Enabled:             true,
					MaxEntries:          count * 2,
					SimilarityThreshold: 0.85,
					TTLSeconds:          0,
					UseHNSW:             true,
					HNSWM:               16,
					HNSWEfConstruction:  200,
				})
				b.StartTimer()

				// Add entries and build index
				for j := 0; j < count; j++ {
					reqID := fmt.Sprintf("req%d", j)
					_ = cache.AddEntry(reqID, "test-model", testQueries[j], []byte(testQueries[j]), []byte("response"))
				}
			}
		})
	}
}

// BenchmarkHNSWParameters benchmarks different HNSW parameter configurations
func BenchmarkHNSWParameters(b *testing.B) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	cacheSize := 1000
	testConfigs := []struct {
		name           string
		m              int
		efConstruction int
	}{
		{"M8_EF100", 8, 100},
		{"M16_EF200", 16, 200},
		{"M32_EF400", 32, 400},
	}

	// Prepare test data
	entries := make([]struct {
		query    string
		response string
	}, cacheSize)

	for i := 0; i < cacheSize; i++ {
		entries[i].query = fmt.Sprintf("Query %d about AI and machine learning", i)
		entries[i].response = fmt.Sprintf("Response %d", i)
	}

	for _, config := range testConfigs {
		b.Run(config.name, func(b *testing.B) {
			cache := NewInMemoryCache(InMemoryCacheOptions{
				Enabled:             true,
				MaxEntries:          cacheSize * 2,
				SimilarityThreshold: 0.85,
				TTLSeconds:          0,
				UseHNSW:             true,
				HNSWM:               config.m,
				HNSWEfConstruction:  config.efConstruction,
			})

			// Populate cache
			for i, entry := range entries {
				reqID := fmt.Sprintf("req%d", i)
				_ = cache.AddEntry(reqID, "test-model", entry.query, []byte(entry.query), []byte(entry.response))
			}

			// Benchmark search
			searchQuery := "What is artificial intelligence and machine learning?"
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _, _ = cache.FindSimilar("test-model", searchQuery)
			}
		})
	}
}

// BenchmarkCacheOperations benchmarks complete cache workflow
func BenchmarkCacheOperations(b *testing.B) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	b.Run("LinearSearch_AddAndFind", func(b *testing.B) {
		cache := NewInMemoryCache(InMemoryCacheOptions{
			Enabled:             true,
			MaxEntries:          10000,
			SimilarityThreshold: 0.85,
			TTLSeconds:          0,
			UseHNSW:             false,
		})

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			query := fmt.Sprintf("Test query %d", i%100)
			reqID := fmt.Sprintf("req%d", i)

			// Add entry
			_ = cache.AddEntry(reqID, "test-model", query, []byte(query), []byte("response"))

			// Find similar
			_, _, _ = cache.FindSimilar("test-model", query)
		}
	})

	b.Run("HNSWSearch_AddAndFind", func(b *testing.B) {
		cache := NewInMemoryCache(InMemoryCacheOptions{
			Enabled:             true,
			MaxEntries:          10000,
			SimilarityThreshold: 0.85,
			TTLSeconds:          0,
			UseHNSW:             true,
			HNSWM:               16,
			HNSWEfConstruction:  200,
		})

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			query := fmt.Sprintf("Test query %d", i%100)
			reqID := fmt.Sprintf("req%d", i)

			// Add entry
			_ = cache.AddEntry(reqID, "test-model", query, []byte(query), []byte("response"))

			// Find similar
			_, _, _ = cache.FindSimilar("test-model", query)
		}
	})
}

// BenchmarkHNSWRebuild benchmarks index rebuild performance
func BenchmarkHNSWRebuild(b *testing.B) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	sizes := []int{100, 500, 1000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Rebuild_%d_entries", size), func(b *testing.B) {
			// Create and populate cache
			cache := NewInMemoryCache(InMemoryCacheOptions{
				Enabled:             true,
				MaxEntries:          size * 2,
				SimilarityThreshold: 0.85,
				TTLSeconds:          0,
				UseHNSW:             true,
				HNSWM:               16,
				HNSWEfConstruction:  200,
			})

			// Populate with test data
			for i := 0; i < size; i++ {
				query := fmt.Sprintf("Query %d about machine learning", i)
				reqID := fmt.Sprintf("req%d", i)
				_ = cache.AddEntry(reqID, "test-model", query, []byte(query), []byte("response"))
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				cache.mu.Lock()
				cache.rebuildHNSWIndex()
				cache.mu.Unlock()
			}
		})
	}
}
