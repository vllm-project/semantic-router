package main

import (
	"fmt"
	"log"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
)

func main() {
	// Example: Setting up Redis cache backend
	fmt.Println("Redis Cache Backend Example")
	fmt.Println("===========================")

	// Initialize the embedding model
	fmt.Println("\n0. Initializing embedding model...")
	err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true)
	if err != nil {
		log.Fatalf("Failed to initialize embedding model: %v", err)
	}
	fmt.Println("✓ Embedding model initialized")

	// Configuration for Redis cache
	config := cache.CacheConfig{
		BackendType:         cache.RedisCacheType,
		Enabled:             true,
		SimilarityThreshold: 0.85,
		TTLSeconds:          3600, // Entries expire after 1 hour
		BackendConfigPath:   "../../config/semantic-cache/redis.yaml",
	}

	// Create cache backend
	fmt.Println("\n1. Creating Redis cache backend...")
	cacheBackend, err := cache.NewCacheBackend(config)
	if err != nil {
		log.Fatalf("Failed to create cache backend: %v", err)
	}
	defer cacheBackend.Close()

	fmt.Println("✓ Redis cache backend created successfully")

	// Example cache operations
	model := "gpt-4"
	query := "What is the capital of France?"
	requestID := "req-12345"
	requestBody := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"What is the capital of France?"}]}`)
	responseBody := []byte(`{"choices":[{"message":{"content":"The capital of France is Paris."}}]}`)

	// Add entry to cache
	fmt.Println("\n2. Adding entry to cache...")
	err = cacheBackend.AddEntry(requestID, model, query, requestBody, responseBody)
	if err != nil {
		log.Fatalf("Failed to add entry: %v", err)
	}
	fmt.Println("✓ Entry added to cache")

	// Wait a moment for Redis to index the entry
	time.Sleep(100 * time.Millisecond)

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
	}

	// Example: Pending request workflow
	fmt.Println("\n6. Pending Request Workflow:")
	newRequestID := "req-67890"
	newQuery := "What is machine learning?"
	newRequestBody := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"What is machine learning?"}]}`)

	fmt.Println("  Adding pending request...")
	err = cacheBackend.AddPendingRequest(newRequestID, model, newQuery, newRequestBody)
	if err != nil {
		log.Fatalf("Failed to add pending request: %v", err)
	}
	fmt.Println("  ✓ Pending request added")

	// Wait a moment for Redis to index the entry
	time.Sleep(100 * time.Millisecond)

	// Simulate getting response from LLM
	newResponseBody := []byte(`{"choices":[{"message":{"content":"Machine learning is a subset of AI..."}}]}`)

	fmt.Println("  Updating with response...")
	err = cacheBackend.UpdateWithResponse(newRequestID, newResponseBody)
	if err != nil {
		log.Fatalf("Failed to update response: %v", err)
	}
	fmt.Println("  ✓ Response updated")

	// Verify the entry is now cached
	cachedResponse, found, err = cacheBackend.FindSimilar(model, newQuery)
	if err != nil {
		log.Fatalf("Failed to search cache: %v", err)
	}

	if found {
		fmt.Println("  ✓ Entry is now in cache and searchable")
	}

	// Final statistics
	fmt.Println("\n7. Final Statistics:")
	stats = cacheBackend.GetStats()
	fmt.Printf("  Total Entries: %d\n", stats.TotalEntries)
	fmt.Printf("  Hits: %d\n", stats.HitCount)
	fmt.Printf("  Misses: %d\n", stats.MissCount)
	fmt.Printf("  Hit Ratio: %.2f%%\n", stats.HitRatio*100)

	fmt.Println("\n✓ Example completed successfully!")
}
