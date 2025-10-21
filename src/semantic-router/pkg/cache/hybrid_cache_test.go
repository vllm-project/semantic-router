//go:build !windows && cgo
// +build !windows,cgo

package cache

import (
	"fmt"
	"os"
	"testing"
	"time"
)

// TestHybridCacheDisabled tests that disabled hybrid cache returns immediately
func TestHybridCacheDisabled(t *testing.T) {
	cache, err := NewHybridCache(HybridCacheOptions{
		Enabled: false,
	})
	if err != nil {
		t.Fatalf("Failed to create disabled cache: %v", err)
	}
	defer cache.Close()

	if cache.IsEnabled() {
		t.Error("Cache should be disabled")
	}

	// All operations should be no-ops
	err = cache.AddEntry("req1", "model1", "test query", []byte("request"), []byte("response"))
	if err != nil {
		t.Errorf("AddEntry should not error on disabled cache: %v", err)
	}

	_, found, err := cache.FindSimilar("model1", "test query")
	if err != nil {
		t.Errorf("FindSimilar should not error on disabled cache: %v", err)
	}
	if found {
		t.Error("FindSimilar should not find anything on disabled cache")
	}
}

// TestHybridCacheBasicOperations tests basic cache operations
func TestHybridCacheBasicOperations(t *testing.T) {
	// Skip if Milvus is not configured
	if os.Getenv("MILVUS_URI") == "" {
		t.Skip("Skipping: MILVUS_URI not set")
	}

	// Create a test Milvus config
	milvusConfig := "/tmp/test_milvus_config.yaml"
	err := os.WriteFile(milvusConfig, []byte(`
milvus:
  address: "localhost:19530"
  collection_name: "test_hybrid_cache"
  dimension: 384
  index_type: "HNSW"
  metric_type: "IP"
  params:
    M: 16
    efConstruction: 200
`), 0o644)
	if err != nil {
		t.Fatalf("Failed to create test config: %v", err)
	}
	defer os.Remove(milvusConfig)

	cache, err := NewHybridCache(HybridCacheOptions{
		Enabled:             true,
		SimilarityThreshold: 0.8,
		TTLSeconds:          300,
		MaxMemoryEntries:    100,
		HNSWM:               16,
		HNSWEfConstruction:  200,
		MilvusConfigPath:    milvusConfig,
	})
	if err != nil {
		t.Fatalf("Failed to create hybrid cache: %v", err)
	}
	defer cache.Close()

	if !cache.IsEnabled() {
		t.Fatal("Cache should be enabled")
	}

	// Test AddEntry
	testQuery := "What is the meaning of life?"
	testResponse := []byte(`{"response": "42"}`)

	err = cache.AddEntry("req1", "gpt-4", testQuery, []byte("{}"), testResponse)
	if err != nil {
		t.Fatalf("Failed to add entry: %v", err)
	}

	// Verify stats
	stats := cache.GetStats()
	if stats.TotalEntries != 1 {
		t.Errorf("Expected 1 entry, got %d", stats.TotalEntries)
	}

	// Test FindSimilar with exact same query (should hit)
	time.Sleep(100 * time.Millisecond) // Allow indexing to complete

	response, found, err := cache.FindSimilar("gpt-4", testQuery)
	if err != nil {
		t.Fatalf("FindSimilar failed: %v", err)
	}
	if !found {
		t.Error("Expected to find cached entry")
	}
	if string(response) != string(testResponse) {
		t.Errorf("Response mismatch: got %s, want %s", string(response), string(testResponse))
	}

	// Test FindSimilar with similar query (should hit)
	_, found, err = cache.FindSimilar("gpt-4", "What's the meaning of life?")
	if err != nil {
		t.Fatalf("FindSimilar failed: %v", err)
	}
	if !found {
		t.Error("Expected to find similar cached entry")
	}

	// Test FindSimilar with dissimilar query (should miss)
	_, found, err = cache.FindSimilar("gpt-4", "How to cook pasta?")
	if err != nil {
		t.Fatalf("FindSimilar failed: %v", err)
	}
	if found {
		t.Error("Should not find dissimilar query")
	}

	// Verify updated stats
	stats = cache.GetStats()
	if stats.HitCount < 1 {
		t.Errorf("Expected at least 1 hit, got %d", stats.HitCount)
	}
	if stats.MissCount < 1 {
		t.Errorf("Expected at least 1 miss, got %d", stats.MissCount)
	}
}

// TestHybridCachePendingRequest tests pending request flow
func TestHybridCachePendingRequest(t *testing.T) {
	// Skip if Milvus is not configured
	if os.Getenv("MILVUS_URI") == "" {
		t.Skip("Skipping: MILVUS_URI not set")
	}

	milvusConfig := "/tmp/test_milvus_pending_config.yaml"
	err := os.WriteFile(milvusConfig, []byte(`
milvus:
  address: "localhost:19530"
  collection_name: "test_hybrid_pending"
  dimension: 384
  index_type: "HNSW"
  metric_type: "IP"
`),
		0o644)
	if err != nil {
		t.Fatalf("Failed to create test config: %v", err)
	}
	defer os.Remove(milvusConfig)

	cache, err := NewHybridCache(HybridCacheOptions{
		Enabled:             true,
		SimilarityThreshold: 0.8,
		TTLSeconds:          300,
		MaxMemoryEntries:    100,
		MilvusConfigPath:    milvusConfig,
	})
	if err != nil {
		t.Fatalf("Failed to create hybrid cache: %v", err)
	}
	defer cache.Close()

	// Add pending request
	testQuery := "Explain quantum computing"
	err = cache.AddPendingRequest("req1", "gpt-4", testQuery, []byte("{}"))
	if err != nil {
		t.Fatalf("Failed to add pending request: %v", err)
	}

	// Update with response
	testResponse := []byte(`{"answer": "Quantum computing uses qubits..."}`)
	err = cache.UpdateWithResponse("req1", testResponse)
	if err != nil {
		t.Fatalf("Failed to update with response: %v", err)
	}

	// Wait for indexing
	time.Sleep(100 * time.Millisecond)

	// Try to find it
	response, found, err := cache.FindSimilar("gpt-4", testQuery)
	if err != nil {
		t.Fatalf("FindSimilar failed: %v", err)
	}
	if !found {
		t.Error("Expected to find cached entry after update")
	}
	if string(response) != string(testResponse) {
		t.Errorf("Response mismatch: got %s, want %s", string(response), string(testResponse))
	}
}

// TestHybridCacheEviction tests memory eviction behavior
func TestHybridCacheEviction(t *testing.T) {
	// Skip if Milvus is not configured
	if os.Getenv("MILVUS_URI") == "" {
		t.Skip("Skipping: MILVUS_URI not set")
	}

	milvusConfig := "/tmp/test_milvus_eviction_config.yaml"
	err := os.WriteFile(milvusConfig, []byte(`
milvus:
  address: "localhost:19530"
  collection_name: "test_hybrid_eviction"
  dimension: 384
  index_type: "HNSW"
  metric_type: "IP"
`),
		0o644)
	if err != nil {
		t.Fatalf("Failed to create test config: %v", err)
	}
	defer os.Remove(milvusConfig)

	// Create cache with very small memory limit
	cache, err := NewHybridCache(HybridCacheOptions{
		Enabled:             true,
		SimilarityThreshold: 0.8,
		TTLSeconds:          300,
		MaxMemoryEntries:    5, // Only 5 entries in memory
		MilvusConfigPath:    milvusConfig,
	})
	if err != nil {
		t.Fatalf("Failed to create hybrid cache: %v", err)
	}
	defer cache.Close()

	// Add 10 entries (will trigger evictions)
	for i := 0; i < 10; i++ {
		query := fmt.Sprintf("Query number %d", i)
		response := []byte(fmt.Sprintf(`{"answer": "Response %d"}`, i))
		err = cache.AddEntry(fmt.Sprintf("req%d", i), "gpt-4", query, []byte("{}"), response)
		if err != nil {
			t.Fatalf("Failed to add entry %d: %v", i, err)
		}
	}

	// Check that we have at most MaxMemoryEntries in HNSW
	stats := cache.GetStats()
	if stats.TotalEntries > 5 {
		t.Errorf("Expected at most 5 entries in memory, got %d", stats.TotalEntries)
	}

	// All entries should still be in Milvus
	// Try to find a recent entry (should be in memory)
	time.Sleep(100 * time.Millisecond)
	_, found, err := cache.FindSimilar("gpt-4", "Query number 9")
	if err != nil {
		t.Fatalf("FindSimilar failed: %v", err)
	}
	if !found {
		t.Error("Expected to find recent entry")
	}

	// Try to find an old evicted entry (should be in Milvus)
	_, _, err = cache.FindSimilar("gpt-4", "Query number 0")
	if err != nil {
		t.Fatalf("FindSimilar failed: %v", err)
	}
	// May or may not find it depending on Milvus indexing speed
	// Just verify no error
}

// TestHybridCacheLocalCacheHit tests local cache hot path
func TestHybridCacheLocalCacheHit(t *testing.T) {
	// Skip if Milvus is not configured
	if os.Getenv("MILVUS_URI") == "" {
		t.Skip("Skipping: MILVUS_URI not set")
	}

	milvusConfig := "/tmp/test_milvus_local_config.yaml"
	err := os.WriteFile(milvusConfig, []byte(`
milvus:
  address: "localhost:19530"
  collection_name: "test_hybrid_local"
  dimension: 384
  index_type: "HNSW"
  metric_type: "IP"
`),
		0o644)
	if err != nil {
		t.Fatalf("Failed to create test config: %v", err)
	}
	defer os.Remove(milvusConfig)

	cache, err := NewHybridCache(HybridCacheOptions{
		Enabled:             true,
		SimilarityThreshold: 0.8,
		TTLSeconds:          300,
		MaxMemoryEntries:    100,
		MilvusConfigPath:    milvusConfig,
	})
	if err != nil {
		t.Fatalf("Failed to create hybrid cache: %v", err)
	}
	defer cache.Close()

	// Add an entry
	testQuery := "What is machine learning?"
	testResponse := []byte(`{"answer": "ML is..."}`)
	err = cache.AddEntry("req1", "gpt-4", testQuery, []byte("{}"), testResponse)
	if err != nil {
		t.Fatalf("Failed to add entry: %v", err)
	}

	time.Sleep(100 * time.Millisecond)

	// First search - should populate local cache
	_, found, err := cache.FindSimilar("gpt-4", testQuery)
	if err != nil {
		t.Fatalf("FindSimilar failed: %v", err)
	}
	if !found {
		t.Fatal("Expected to find entry")
	}

	// Second search - should hit local cache (much faster)
	startTime := time.Now()
	response, found, err := cache.FindSimilar("gpt-4", testQuery)
	localLatency := time.Since(startTime)
	if err != nil {
		t.Fatalf("FindSimilar failed: %v", err)
	}
	if !found {
		t.Fatal("Expected to find entry in local cache")
	}
	if string(response) != string(testResponse) {
		t.Errorf("Response mismatch: got %s, want %s", string(response), string(testResponse))
	}

	// Local cache should be very fast (< 10ms)
	if localLatency > 10*time.Millisecond {
		t.Logf("Local cache hit took %v (expected < 10ms, but may vary)", localLatency)
	}

	stats := cache.GetStats()
	if stats.HitCount < 2 {
		t.Errorf("Expected at least 2 hits, got %d", stats.HitCount)
	}
}

// BenchmarkHybridCacheAddEntry benchmarks adding entries to hybrid cache
func BenchmarkHybridCacheAddEntry(b *testing.B) {
	if os.Getenv("MILVUS_URI") == "" {
		b.Skip("Skipping: MILVUS_URI not set")
	}

	milvusConfig := "/tmp/bench_milvus_config.yaml"
	err := os.WriteFile(milvusConfig, []byte(`
milvus:
  address: "localhost:19530"
  collection_name: "bench_hybrid_cache"
  dimension: 384
  index_type: "HNSW"
  metric_type: "IP"
`),
		0o644)
	if err != nil {
		b.Fatalf("Failed to create test config: %v", err)
	}
	defer os.Remove(milvusConfig)

	cache, err := NewHybridCache(HybridCacheOptions{
		Enabled:             true,
		SimilarityThreshold: 0.8,
		TTLSeconds:          300,
		MaxMemoryEntries:    10000,
		MilvusConfigPath:    milvusConfig,
	})
	if err != nil {
		b.Fatalf("Failed to create hybrid cache: %v", err)
	}
	defer cache.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query := fmt.Sprintf("Benchmark query number %d", i)
		response := []byte(fmt.Sprintf(`{"answer": "Response %d"}`, i))
		err := cache.AddEntry(fmt.Sprintf("req%d", i), "gpt-4", query, []byte("{}"), response)
		if err != nil {
			b.Fatalf("AddEntry failed: %v", err)
		}
	}
}

// BenchmarkHybridCacheFindSimilar benchmarks searching in hybrid cache
func BenchmarkHybridCacheFindSimilar(b *testing.B) {
	if os.Getenv("MILVUS_URI") == "" {
		b.Skip("Skipping: MILVUS_URI not set")
	}

	milvusConfig := "/tmp/bench_milvus_search_config.yaml"
	err := os.WriteFile(milvusConfig, []byte(`
milvus:
  address: "localhost:19530"
  collection_name: "bench_hybrid_search"
  dimension: 384
  index_type: "HNSW"
  metric_type: "IP"
`),
		0o644)
	if err != nil {
		b.Fatalf("Failed to create test config: %v", err)
	}
	defer os.Remove(milvusConfig)

	cache, err := NewHybridCache(HybridCacheOptions{
		Enabled:             true,
		SimilarityThreshold: 0.8,
		TTLSeconds:          300,
		MaxMemoryEntries:    1000,
		MilvusConfigPath:    milvusConfig,
	})
	if err != nil {
		b.Fatalf("Failed to create hybrid cache: %v", err)
	}
	defer cache.Close()

	// Pre-populate cache
	for i := 0; i < 100; i++ {
		query := fmt.Sprintf("Benchmark query number %d", i)
		response := []byte(fmt.Sprintf(`{"answer": "Response %d"}`, i))
		err := cache.AddEntry(fmt.Sprintf("req%d", i), "gpt-4", query, []byte("{}"), response)
		if err != nil {
			b.Fatalf("AddEntry failed: %v", err)
		}
	}

	time.Sleep(500 * time.Millisecond) // Allow indexing

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query := fmt.Sprintf("Benchmark query number %d", i%100)
		_, _, err := cache.FindSimilar("gpt-4", query)
		if err != nil {
			b.Fatalf("FindSimilar failed: %v", err)
		}
	}
}
