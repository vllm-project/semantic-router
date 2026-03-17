package main

import (
	"fmt"
	"log"
	"os"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func main() {
	// Example: Setting up Valkey cache backend
	fmt.Println("Valkey Cache Backend Example")
	fmt.Println("============================")

	// Initialize the embedding model
	fmt.Println("\n0. Initializing embedding model...")
	err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true)
	if err != nil {
		log.Fatalf("Failed to initialize embedding model: %v", err)
	}
	fmt.Println("✓ Embedding model initialized")

	// Configuration for Valkey cache
	// Use environment variables if set, otherwise use defaults
	host := os.Getenv("VALKEY_HOST")
	if host == "" {
		host = "localhost"
	}
	port := 6379 // Default to 6379 for custom builds
	if portEnv := os.Getenv("VALKEY_PORT"); portEnv != "" {
		fmt.Sscanf(portEnv, "%d", &port)
	}

	valkeyConfig := &config.ValkeyConfig{}
	valkeyConfig.Connection.Host = host
	valkeyConfig.Connection.Port = port
	valkeyConfig.Connection.Timeout = 5000
	valkeyConfig.Connection.TLS.Enabled = false

	valkeyConfig.Index.Name = "semantic_cache_idx"
	valkeyConfig.Index.Prefix = "doc:"
	valkeyConfig.Index.VectorField.Name = "embedding"
	valkeyConfig.Index.VectorField.Dimension = 384
	valkeyConfig.Index.VectorField.MetricType = "COSINE"
	valkeyConfig.Index.IndexType = "HNSW"
	valkeyConfig.Index.Params.M = 16
	valkeyConfig.Index.Params.EfConstruction = 64

	valkeyConfig.Search.TopK = 5

	valkeyConfig.Development.DropIndexOnStartup = false
	valkeyConfig.Development.AutoCreateIndex = true

	cacheConfig := cache.CacheConfig{
		BackendType:         cache.ValkeyCacheType,
		Enabled:             true,
		SimilarityThreshold: 0.75, // Lower threshold for more lenient matching
		TTLSeconds:          3600, // Entries expire after 1 hour
		Valkey:              valkeyConfig,
	}

	// Create cache backend
	fmt.Println("\n1. Creating Valkey cache backend...")
	cacheBackend, err := cache.NewCacheBackend(cacheConfig)
	if err != nil {
		log.Fatalf("Failed to create cache backend: %v", err)
	}
	defer cacheBackend.Close()

	fmt.Println("✓ Valkey cache backend created successfully")

	// Optional: Clear any existing data for a clean demo
	// Note: In production, you wouldn't typically do this
	fmt.Println("\n1.5. Clearing any existing cache data for clean demo...")
	// We don't have a direct Clear() method, but we can note the starting stats
	initialStats := cacheBackend.GetStats()
	if initialStats.TotalEntries > 0 {
		fmt.Printf("⚠ Found %d existing entries in cache (from previous runs)\n", initialStats.TotalEntries)
		fmt.Println("   Consider flushing Valkey if you want a clean start: valkey-cli FLUSHDB")
	}

	// Example cache operations
	model := "gpt-4"
	query := "What is the capital of France?"
	requestID := "req-12345"
	requestBody := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"What is the capital of France?"}]}`)
	responseBody := []byte(`{"choices":[{"message":{"content":"The capital of France is Paris."}}]}`)

	// Add entry to cache
	fmt.Println("\n2. Adding entry to cache...")
	err = cacheBackend.AddEntry(requestID, model, query, requestBody, responseBody, 3600)
	if err != nil {
		log.Fatalf("Failed to add entry: %v", err)
	}
	fmt.Println("✓ Entry added to cache")

	// Wait for Valkey to index the entry
	// Note: Valkey-search needs time to index the vector
	fmt.Println("   Waiting for Valkey to index the entry...")
	time.Sleep(1 * time.Second)

	// Search for similar entry
	fmt.Println("\n3. Searching for similar query...")
	similarQuery := "What's the capital city of France?"
	cachedResponse, found, err := cacheBackend.FindSimilar(model, similarQuery)
	if err != nil {
		log.Fatalf("Failed to search cache: %v", err)
	}

	if found {
		fmt.Println("✓ Cache HIT! Found similar query")
		fmt.Printf("  Cached response: %s\n", string(cachedResponse))
	} else {
		fmt.Println("✗ Cache MISS - no similar query found")
		fmt.Println("  Note: This can happen due to indexing delays or similarity threshold")
	}

	// Get cache statistics
	fmt.Println("\n4. Cache Statistics:")
	stats := cacheBackend.GetStats()
	fmt.Printf("  Total Entries: %d\n", stats.TotalEntries)
	fmt.Printf("  Hits: %d\n", stats.HitCount)
	fmt.Printf("  Misses: %d\n", stats.MissCount)
	fmt.Printf("  Hit Ratio: %.2f%%\n", stats.HitRatio*100)

	// Example with custom threshold
	fmt.Println("\n5. Searching with custom threshold...")
	strictQuery := "Paris is the capital of which country?"
	cachedResponse, found, err = cacheBackend.FindSimilarWithThreshold(model, strictQuery, 0.75)
	if err != nil {
		log.Fatalf("Failed to search cache: %v", err)
	}

	if found {
		fmt.Println("✓ Cache HIT with threshold 0.75")
		fmt.Printf("  Cached response: %s\n", string(cachedResponse))
	} else {
		fmt.Println("✗ Cache MISS with threshold 0.75")
		fmt.Println("  Note: Lower threshold allows more lenient matching")
	}

	// Example: Pending request workflow
	fmt.Println("\n6. Pending Request Workflow:")
	newRequestID := "req-67890"
	newQuery := "What is machine learning?"
	newRequestBody := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"What is machine learning?"}]}`)

	fmt.Println("  Adding pending request...")
	err = cacheBackend.AddPendingRequest(newRequestID, model, newQuery, newRequestBody, 3600)
	if err != nil {
		log.Fatalf("Failed to add pending request: %v", err)
	}
	fmt.Println("  ✓ Pending request added")

	// Wait for Valkey to index the entry
	// valkey-search needs time to index TAG fields
	fmt.Println("   Waiting for Valkey to index the pending request...")
	time.Sleep(1 * time.Second)

	// Simulate getting response from LLM
	newResponseBody := []byte(`{"choices":[{"message":{"content":"Machine learning is a subset of AI..."}}]}`)

	fmt.Println("  Updating with response...")
	err = cacheBackend.UpdateWithResponse(newRequestID, newResponseBody, 3600)
	if err != nil {
		log.Fatalf("Failed to update response: %v", err)
	}
	fmt.Println("  ✓ Response updated")

	// Wait for vector index to update
	time.Sleep(500 * time.Millisecond)

	// Verify the entry is now cached
	cachedResponse, found, err = cacheBackend.FindSimilar(model, newQuery)
	if err != nil {
		log.Fatalf("Failed to search cache: %v", err)
	}

	if found {
		fmt.Println("  ✓ Entry is now in cache and searchable")
	} else {
		fmt.Println("  ⚠ Entry not yet searchable (may need more indexing time)")
	}

	// Final statistics
	fmt.Println("\n7. Final Statistics:")
	stats = cacheBackend.GetStats()
	fmt.Printf("  Total Entries: %d\n", stats.TotalEntries)
	fmt.Printf("  Hits: %d\n", stats.HitCount)
	fmt.Printf("  Misses: %d\n", stats.MissCount)
	fmt.Printf("  Hit Ratio: %.2f%%\n", stats.HitRatio*100)

	fmt.Println("\n✓ Example completed successfully!")
	fmt.Println("\nNote: Valkey is a high-performance Redis fork with full compatibility.")
	fmt.Println("This example demonstrates the same semantic caching capabilities as Redis.")
}
