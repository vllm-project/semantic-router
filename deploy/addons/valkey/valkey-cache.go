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

func newValkeyConfig() *config.ValkeyConfig {
	host := os.Getenv("VALKEY_HOST")
	if host == "" {
		host = "localhost"
	}
	port := 6379
	if portEnv := os.Getenv("VALKEY_PORT"); portEnv != "" {
		fmt.Sscanf(portEnv, "%d", &port)
	}

	cfg := &config.ValkeyConfig{}
	cfg.Connection.Host = host
	cfg.Connection.Port = port
	cfg.Connection.Timeout = 5 // seconds
	cfg.Connection.TLS.Enabled = false

	cfg.Index.Name = "semantic_cache_idx"
	cfg.Index.Prefix = "doc:"
	cfg.Index.VectorField.Name = "embedding"
	cfg.Index.VectorField.Dimension = 384
	cfg.Index.VectorField.MetricType = "COSINE"
	cfg.Index.IndexType = "HNSW"
	cfg.Index.Params.M = 16
	cfg.Index.Params.EfConstruction = 64

	cfg.Search.TopK = 5
	cfg.Development.DropIndexOnStartup = false
	cfg.Development.AutoCreateIndex = true

	return cfg
}

func demoCacheOperations(cacheBackend cache.CacheBackend) {
	model := "gpt-4"
	query := "What is the capital of France?"
	requestID := "req-12345"
	requestBody := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"What is the capital of France?"}]}`)
	responseBody := []byte(`{"choices":[{"message":{"content":"The capital of France is Paris."}}]}`)

	fmt.Println("\n2. Adding entry to cache...")
	err := cacheBackend.AddEntry(requestID, model, query, requestBody, responseBody, 3600)
	if err != nil {
		log.Fatalf("Failed to add entry: %v", err)
	}
	fmt.Println("✓ Entry added to cache")

	fmt.Println("   Waiting for Valkey to index the entry...")
	time.Sleep(1 * time.Second)

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

	fmt.Println("\n4. Cache Statistics:")
	stats := cacheBackend.GetStats()
	fmt.Printf("  Total Entries: %d\n", stats.TotalEntries)
	fmt.Printf("  Hits: %d\n", stats.HitCount)
	fmt.Printf("  Misses: %d\n", stats.MissCount)
	fmt.Printf("  Hit Ratio: %.2f%%\n", stats.HitRatio*100)

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
}

func demoPendingRequestWorkflow(cacheBackend cache.CacheBackend) {
	model := "gpt-4"
	fmt.Println("\n6. Pending Request Workflow:")
	newRequestID := "req-67890"
	newQuery := "What is machine learning?"
	newRequestBody := []byte(`{"model":"gpt-4","messages":[{"role":"user","content":"What is machine learning?"}]}`)

	fmt.Println("  Adding pending request...")
	err := cacheBackend.AddPendingRequest(newRequestID, model, newQuery, newRequestBody, 3600)
	if err != nil {
		log.Fatalf("Failed to add pending request: %v", err)
	}
	fmt.Println("  ✓ Pending request added")

	fmt.Println("   Waiting for Valkey to index the pending request...")
	time.Sleep(1 * time.Second)

	newResponseBody := []byte(`{"choices":[{"message":{"content":"Machine learning is a subset of AI..."}}]}`)
	fmt.Println("  Updating with response...")
	err = cacheBackend.UpdateWithResponse(newRequestID, newResponseBody, 3600)
	if err != nil {
		log.Fatalf("Failed to update response: %v", err)
	}
	fmt.Println("  ✓ Response updated")

	time.Sleep(500 * time.Millisecond)

	_, found, err := cacheBackend.FindSimilar(model, newQuery)
	if err != nil {
		log.Fatalf("Failed to search cache: %v", err)
	}
	if found {
		fmt.Println("  ✓ Entry is now in cache and searchable")
	} else {
		fmt.Println("  ⚠ Entry not yet searchable (may need more indexing time)")
	}
}

func printFinalStats(cacheBackend cache.CacheBackend) {
	fmt.Println("\n7. Final Statistics:")
	stats := cacheBackend.GetStats()
	fmt.Printf("  Total Entries: %d\n", stats.TotalEntries)
	fmt.Printf("  Hits: %d\n", stats.HitCount)
	fmt.Printf("  Misses: %d\n", stats.MissCount)
	fmt.Printf("  Hit Ratio: %.2f%%\n", stats.HitRatio*100)
	fmt.Println("\n✓ Example completed successfully!")
	fmt.Println("\nNote: Valkey is a high-performance Redis fork with full compatibility.")
	fmt.Println("This example demonstrates the same semantic caching capabilities as Redis.")
}

func main() {
	fmt.Println("Valkey Cache Backend Example")
	fmt.Println("============================")

	fmt.Println("\n0. Initializing embedding model...")
	err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true)
	if err != nil {
		log.Fatalf("Failed to initialize embedding model: %v", err)
	}
	fmt.Println("✓ Embedding model initialized")

	cacheConfig := cache.CacheConfig{
		BackendType:         cache.ValkeyCacheType,
		Enabled:             true,
		SimilarityThreshold: 0.75,
		TTLSeconds:          3600,
		Valkey:              newValkeyConfig(),
	}

	fmt.Println("\n1. Creating Valkey cache backend...")
	cacheBackend, err := cache.NewCacheBackend(cacheConfig)
	if err != nil {
		log.Fatalf("Failed to create cache backend: %v", err)
	}
	defer func() { _ = cacheBackend.Close() }()
	fmt.Println("✓ Valkey cache backend created successfully")

	initialStats := cacheBackend.GetStats()
	if initialStats.TotalEntries > 0 {
		fmt.Printf("⚠ Found %d existing entries in cache (from previous runs)\n", initialStats.TotalEntries)
	}

	demoCacheOperations(cacheBackend)
	demoPendingRequestWorkflow(cacheBackend)
	printFinalStats(cacheBackend)
}
