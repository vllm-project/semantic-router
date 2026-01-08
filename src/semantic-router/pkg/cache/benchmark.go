//go:build !windows && cgo

package cache

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// BenchmarkConfig defines the parameters for a benchmark run
type BenchmarkConfig struct {
	CacheSize         int     // Number of entries to pre-populate
	ConcurrencyLevels []int   // Different concurrency levels to test
	RequestsPerLevel  int     // Number of requests per concurrency level
	SimilarityThresh  float32 // Similarity threshold
	UseHNSW           bool    // Whether to use HNSW indexing
	EmbeddingModel    string  // "bert", "qwen3", or "gemma"
	HitRatio          float64 // Expected cache hit ratio (0.0-1.0)
}

// BenchmarkResult captures detailed timing and performance metrics
type BenchmarkResult struct {
	Config        BenchmarkConfig
	TotalRequests int
	Duration      time.Duration
	Throughput    float64 // requests per second

	// Latency breakdown (in milliseconds)
	OverallP50  float64
	OverallP90  float64
	OverallP95  float64
	OverallP99  float64
	OverallMax  float64
	OverallMean float64

	// Component-level latencies
	EmbeddingP50 float64
	EmbeddingP90 float64
	EmbeddingP95 float64
	EmbeddingP99 float64

	SearchP50 float64
	SearchP90 float64
	SearchP95 float64
	SearchP99 float64

	// Cache statistics
	CacheHitRate  float64
	CacheMissRate float64

	// Concurrency impact
	ConcurrencyLevel int
	ErrorCount       int64
}

// latencyMeasurement captures timing for different phases
type latencyMeasurement struct {
	EmbeddingTime time.Duration
	SearchTime    time.Duration
	TotalTime     time.Duration
	CacheHit      bool
	Error         error
}

// generateTestQueries creates a set of test queries with controlled diversity
func generateTestQueries(count int, diversity float64) []string {
	baseQueries := []string{
		"What is the capital of France?",
		"How do I reverse a string in Python?",
		"Explain quantum computing",
		"What are the benefits of meditation?",
		"How does photosynthesis work?",
		"What is machine learning?",
		"How to make chocolate chip cookies?",
		"Explain the theory of relativity",
		"What is the meaning of life?",
		"How to start a business?",
	}

	queries := make([]string, 0, count)
	// Note: rand is automatically seeded in Go 1.20+ (no manual seeding needed)

	for i := 0; i < count; i++ {
		if rand.Float64() < diversity {
			// Generate a diverse query by combining base queries
			idx1 := rand.Intn(len(baseQueries))
			idx2 := rand.Intn(len(baseQueries))
			queries = append(queries, fmt.Sprintf("%s Also, %s", baseQueries[idx1], baseQueries[idx2]))
		} else {
			// Reuse an existing query to simulate cache hits
			if len(queries) > 0 && rand.Float64() < 0.7 {
				// 70% chance to reuse an existing query
				queries = append(queries, queries[rand.Intn(len(queries))])
			} else {
				queries = append(queries, baseQueries[rand.Intn(len(baseQueries))])
			}
		}
	}

	return queries
}

// populateCache pre-fills the cache with entries using concurrent requests
// to leverage continuous batching for faster population
func populateCache(cache *InMemoryCache, size int) error {
	queries := generateTestQueries(size, 0.9) // High diversity for initial population

	// Use high concurrency for population to maximize continuous batching
	populateConcurrency := 64
	if size < 64 {
		populateConcurrency = size
	}

	var wg sync.WaitGroup
	semaphore := make(chan struct{}, populateConcurrency)
	errors := make(chan error, size)

	// Atomic counter for progress tracking
	var completed int64

	startPopulate := time.Now()

	for i := 0; i < size; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			// Acquire semaphore
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			requestID := fmt.Sprintf("req-%d", idx)
			query := queries[idx]
			responseBody := []byte(fmt.Sprintf("Response for: %s", query))

			err := cache.AddEntry(requestID,
				"test-model", query, []byte(query), responseBody, -1)
			if err != nil {
				errors <- fmt.Errorf("failed to add entry %d: %w", idx, err)
				return
			}

			// Increment and check progress (non-blocking)
			count := atomic.AddInt64(&completed, 1)
			if count%1000 == 0 {
				fmt.Printf("  Populated %d/%d entries\n", count, size)
			}
		}(i)
	}

	wg.Wait()
	close(errors)

	populateDuration := time.Since(startPopulate)

	// Check for any errors
	if len(errors) > 0 {
		return <-errors
	}

	throughput := float64(size) / populateDuration.Seconds()
	fmt.Printf("  ✓ Population complete: %d entries in %v (%.0f entries/sec)\n",
		size, populateDuration.Round(time.Millisecond), throughput)

	return nil
}

// cosineSimilarity computes cosine similarity using SIMD-optimized dot product
// Embeddings are normalized, so dot product = cosine similarity
func cosineSimilarity(a, b []float32) float32 {
	return dotProductSIMD(a, b)
}

// measureSearchLatency performs a search and measures component latencies
// IMPORTANT: Separates embedding generation time from pure search time
func measureSearchLatency(cache *InMemoryCache, model, query string) latencyMeasurement {
	measurement := latencyMeasurement{}

	startTotal := time.Now()

	// Measure embedding generation time ONCE
	startEmbed := time.Now()
	queryEmbedding, err := cache.generateEmbedding(query)
	measurement.EmbeddingTime = time.Since(startEmbed)

	if err != nil {
		measurement.Error = err
		measurement.TotalTime = time.Since(startTotal)
		return measurement
	}

	// Measure PURE search time (no embedding generation)
	// Use the pre-computed embedding to avoid double-counting
	startSearch := time.Now()

	cache.mu.RLock()
	var bestMatch *CacheEntry
	var bestSimilarity float32 = -1

	// Perform similarity search using HNSW or linear
	if cache.hnswIndex != nil {
		// HNSW search
		candidates := cache.hnswIndex.searchKNN(queryEmbedding, 1, 50, cache.entries)
		for _, idx := range candidates {
			if idx >= 0 && idx < len(cache.entries) {
				entry := &cache.entries[idx]
				similarity := cosineSimilarity(queryEmbedding, entry.Embedding)
				if similarity > bestSimilarity && similarity >= cache.similarityThreshold {
					bestSimilarity = similarity
					bestMatch = entry
				}
			}
		}
	} else {
		// Linear search
		for i := range cache.entries {
			entry := &cache.entries[i]
			similarity := cosineSimilarity(queryEmbedding, entry.Embedding)
			if similarity > bestSimilarity && similarity >= cache.similarityThreshold {
				bestSimilarity = similarity
				bestMatch = entry
			}
		}
	}
	cache.mu.RUnlock()

	measurement.SearchTime = time.Since(startSearch)
	measurement.CacheHit = bestMatch != nil

	measurement.TotalTime = time.Since(startTotal)
	return measurement
}

// runBenchmarkScenario executes a single benchmark scenario
func runBenchmarkScenario(config BenchmarkConfig, concurrency int) BenchmarkResult {
	result := BenchmarkResult{
		Config:           config,
		ConcurrencyLevel: concurrency,
	}

	// Use Qwen3 by default if no model specified (benefits from continuous batching)
	embeddingModel := config.EmbeddingModel
	if embeddingModel == "" {
		embeddingModel = "qwen3"
	}

	// Check if HNSW should be overridden by environment variable
	useHNSW := config.UseHNSW
	hnswEnv := os.Getenv("USE_HNSW")
	if hnswEnv != "" {
		switch hnswEnv {
		case "true", "1":
			useHNSW = true
		case "false", "0":
			useHNSW = false
		}
	}

	// Create cache with specified configuration
	cache := NewInMemoryCache(InMemoryCacheOptions{
		SimilarityThreshold: config.SimilarityThresh,
		MaxEntries:          config.CacheSize * 2, // Allow room for growth
		TTLSeconds:          0,                    // No TTL for benchmarking
		Enabled:             true,
		EvictionPolicy:      LRUEvictionPolicyType,
		UseHNSW:             useHNSW,
		HNSWM:               16,
		HNSWEfConstruction:  200,
		HNSWEfSearch:        50,
		EmbeddingModel:      embeddingModel,
	})
	defer cache.Close()

	fmt.Printf("\n=== Benchmark Scenario ===\n")
	fmt.Printf("Cache Size: %d, Concurrency: %d, HNSW: %t, Model: %s\n",
		config.CacheSize, concurrency, useHNSW, embeddingModel)
	fmt.Printf("Populating cache...\n")

	// Pre-populate cache
	if err := populateCache(cache, config.CacheSize); err != nil {
		fmt.Printf("Error populating cache: %v\n", err)
		return result
	}

	fmt.Printf("Cache populated with %d entries\n", config.CacheSize)

	// Generate test queries based on desired hit ratio
	queryCount := config.RequestsPerLevel
	diversity := 1.0 - config.HitRatio // Lower diversity = higher hit ratio
	testQueries := generateTestQueries(queryCount, diversity)

	// Storage for measurements
	measurements := make([]latencyMeasurement, queryCount)
	var errorCount int64

	// Run benchmark with specified concurrency
	fmt.Printf("Running %d requests with concurrency %d...\n", queryCount, concurrency)

	startTime := time.Now()

	var wg sync.WaitGroup
	semaphore := make(chan struct{}, concurrency)

	for i := 0; i < queryCount; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			// Acquire semaphore
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			query := testQueries[idx]
			measurement := measureSearchLatency(cache, "test-model", query)
			measurements[idx] = measurement

			if measurement.Error != nil {
				atomic.AddInt64(&errorCount, 1)
			}
		}(i)
	}

	wg.Wait()
	duration := time.Since(startTime)

	// Calculate statistics
	result.TotalRequests = queryCount
	result.Duration = duration
	result.Throughput = float64(queryCount) / duration.Seconds()
	result.ErrorCount = errorCount

	// Extract latency data
	totalLatencies := make([]float64, 0, queryCount)
	embeddingLatencies := make([]float64, 0, queryCount)
	searchLatencies := make([]float64, 0, queryCount)

	hitCount := 0
	for _, m := range measurements {
		if m.Error == nil {
			totalLatencies = append(totalLatencies, float64(m.TotalTime.Microseconds())/1000.0)
			embeddingLatencies = append(embeddingLatencies, float64(m.EmbeddingTime.Microseconds())/1000.0)
			searchLatencies = append(searchLatencies, float64(m.SearchTime.Microseconds())/1000.0)

			if m.CacheHit {
				hitCount++
			}
		}
	}

	// Calculate percentiles
	if len(totalLatencies) > 0 {
		result.OverallP50 = percentile(totalLatencies, 50)
		result.OverallP90 = percentile(totalLatencies, 90)
		result.OverallP95 = percentile(totalLatencies, 95)
		result.OverallP99 = percentile(totalLatencies, 99)
		result.OverallMax = percentile(totalLatencies, 100)
		result.OverallMean = mean(totalLatencies)

		result.EmbeddingP50 = percentile(embeddingLatencies, 50)
		result.EmbeddingP90 = percentile(embeddingLatencies, 90)
		result.EmbeddingP95 = percentile(embeddingLatencies, 95)
		result.EmbeddingP99 = percentile(embeddingLatencies, 99)

		result.SearchP50 = percentile(searchLatencies, 50)
		result.SearchP90 = percentile(searchLatencies, 90)
		result.SearchP95 = percentile(searchLatencies, 95)
		result.SearchP99 = percentile(searchLatencies, 99)

		result.CacheHitRate = float64(hitCount) / float64(len(totalLatencies))
		result.CacheMissRate = 1.0 - result.CacheHitRate
	}

	return result
}

// percentile calculates the Nth percentile of a sorted slice
func percentile(data []float64, p int) float64 {
	if len(data) == 0 {
		return 0
	}

	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)

	if p >= 100 {
		return sorted[len(sorted)-1]
	}

	index := int(math.Ceil(float64(len(sorted))*float64(p)/100.0)) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(sorted) {
		index = len(sorted) - 1
	}

	return sorted[index]
}

// mean calculates the average of a slice
func mean(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}

	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

// printResults displays benchmark results in a formatted table
func printResults(results []BenchmarkResult) {
	fmt.Printf("\n")
	fmt.Printf("╔═══════════════════════════════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║                          SEMANTIC CACHE BENCHMARK RESULTS                             ║\n")
	fmt.Printf("╚═══════════════════════════════════════════════════════════════════════════════════════╝\n")
	fmt.Printf("\n")

	for _, r := range results {
		fmt.Printf("┌─────────────────────────────────────────────────────────────────────────────────────┐\n")
		fmt.Printf("│ Configuration:                                                                      │\n")
		fmt.Printf("│   Cache Size: %-10d  Concurrency: %-5d  HNSW: %-5t  Model: %-10s     │\n",
			r.Config.CacheSize, r.ConcurrencyLevel, r.Config.UseHNSW, r.Config.EmbeddingModel)
		fmt.Printf("│   Total Requests: %-10d  Duration: %-10s  Throughput: %-10.2f req/s  │\n",
			r.TotalRequests, r.Duration.Round(time.Millisecond), r.Throughput)
		fmt.Printf("│   Cache Hit Rate: %-6.2f%%  Cache Miss Rate: %-6.2f%%                           │\n",
			r.CacheHitRate*100, r.CacheMissRate*100)
		fmt.Printf("├─────────────────────────────────────────────────────────────────────────────────────┤\n")
		fmt.Printf("│ Overall Latency (ms):                                                               │\n")
		fmt.Printf("│   P50:  %8.2f  │  P90:  %8.2f  │  P95:  %8.2f  │  P99:  %8.2f          │\n",
			r.OverallP50, r.OverallP90, r.OverallP95, r.OverallP99)
		fmt.Printf("│   Mean: %8.2f  │  Max:  %8.2f                                              │\n",
			r.OverallMean, r.OverallMax)
		fmt.Printf("├─────────────────────────────────────────────────────────────────────────────────────┤\n")
		fmt.Printf("│ Embedding Generation Latency (ms):                                                  │\n")
		fmt.Printf("│   P50:  %8.2f  │  P90:  %8.2f  │  P95:  %8.2f  │  P99:  %8.2f          │\n",
			r.EmbeddingP50, r.EmbeddingP90, r.EmbeddingP95, r.EmbeddingP99)
		fmt.Printf("├─────────────────────────────────────────────────────────────────────────────────────┤\n")
		fmt.Printf("│ Search Latency (ms):                                                                │\n")
		fmt.Printf("│   P50:  %8.2f  │  P90:  %8.2f  │  P95:  %8.2f  │  P99:  %8.2f          │\n",
			r.SearchP50, r.SearchP90, r.SearchP95, r.SearchP99)
		fmt.Printf("└─────────────────────────────────────────────────────────────────────────────────────┘\n")
		fmt.Printf("\n")
	}
}

var (
	// Ensure thread-safe single initialization using sync.Once
	modelsInitOnce sync.Once
	modelsInitErr  error
)

// InitEmbeddingModels initializes the embedding models needed for benchmarks
// This should be called once before running any benchmarks
// It initializes Qwen3 which has continuous batching support for better performance
//
// Thread-safe: Multiple concurrent calls will only initialize once
//
// Environment Variables:
//
//	QWEN3_MODEL_PATH - Path to Qwen3 embedding model (optional)
//	GEMMA_MODEL_PATH - Path to Gemma embedding model (optional)
//	USE_GPU - Set to "true" or "1" to use GPU instead of CPU (default: CPU)
//	USE_HNSW - Set to "true" or "1" to enable HNSW indexing, "false" or "0" to disable (default: read from config)
//
// Example:
//
//	export QWEN3_MODEL_PATH=/path/to/Qwen3-Embedding-0.6B
//	export USE_GPU=true
//	export USE_HNSW=true
func InitEmbeddingModels() error {
	// Thread-safe initialization - only executes once regardless of concurrent calls
	modelsInitOnce.Do(func() {
		modelsInitErr = initEmbeddingModelsOnce()
	})
	return modelsInitErr
}

// initEmbeddingModelsOnce performs the actual initialization (called by sync.Once)
func initEmbeddingModelsOnce() error {
	// Check if GPU should be used
	useGPU := false
	useGPUEnv := os.Getenv("USE_GPU")
	if useGPUEnv == "true" || useGPUEnv == "1" {
		useGPU = true
	}

	deviceType := "CPU"
	if useGPU {
		deviceType = "GPU"
	}

	fmt.Printf("Initializing Qwen3 embedding model with FIXED continuous batching on %s...\n", deviceType)

	// Check for environment variable first
	qwen3Path := os.Getenv("QWEN3_MODEL_PATH")

	// If environment variable not set, try common paths
	var qwen3Paths []string
	if qwen3Path != "" {
		fmt.Printf("Using Qwen3 model path from QWEN3_MODEL_PATH: %s\n", qwen3Path)
		qwen3Paths = []string{qwen3Path}
	} else {
		fmt.Println("QWEN3_MODEL_PATH not set, trying default paths...")
		qwen3Paths = []string{
			"./models/mom-embedding-pro",
			"./candle-binding/models/mom-embedding-pro",
			"../models/mom-embedding-pro",
			"models/mom-embedding-pro",
		}
	}

	// Continuous batching configuration
	maxBatchSize := 64      // Batch up to 64 requests together
	maxWaitMs := uint64(10) // Wait max 10ms for batch to fill

	var lastErr error
	useCPU := !useGPU
	for i, path := range qwen3Paths {
		fmt.Printf("  Attempt %d/%d: Trying %s (device: %s)\n", i+1, len(qwen3Paths), path, deviceType)

		// Use InitEmbeddingModelsBatched with FIXED scheduler (returns Vec instead of Tensor!)
		err := candle_binding.InitEmbeddingModelsBatched(path, maxBatchSize, maxWaitMs, useCPU)
		if err == nil {
			fmt.Printf("✓ Qwen3 embedding model initialized from: %s\n", path)
			fmt.Printf("  Device: %s\n", deviceType)
			fmt.Printf("  TRUE Continuous batching: ENABLED ✨ (FIXED - no CUDA context errors!)\n")
			fmt.Printf("    - Max batch size: %d requests\n", maxBatchSize)
			fmt.Printf("    - Max wait time: %dms\n", maxWaitMs)
			fmt.Printf("    - Expected throughput: 10-15x improvement with concurrency!\n")
			if useGPU {
				fmt.Printf("  GPU acceleration: ENABLED\n")
			}
			return nil
		}
		lastErr = err
	}

	return fmt.Errorf("failed to initialize Qwen3 model on %s with continuous batching (tried %d paths): %w", deviceType, len(qwen3Paths), lastErr)
}

// RunStandaloneBenchmark runs a standalone benchmark (not as a test)
// This function is exported so it can be called from the standalone benchmark tool
// Note: Models should be initialized once before calling this function
func RunStandaloneBenchmark(ctx context.Context, config BenchmarkConfig) []BenchmarkResult {
	var results []BenchmarkResult

	for _, concurrency := range config.ConcurrencyLevels {
		select {
		case <-ctx.Done():
			fmt.Printf("Benchmark cancelled\n")
			return results
		default:
			result := runBenchmarkScenario(config, concurrency)
			results = append(results, result)
		}
	}

	return results
}

// PrintBenchmarkResults displays benchmark results in a formatted table
// This is exported so the standalone tool can use the same formatting
func PrintBenchmarkResults(results []BenchmarkResult) {
	printResults(results)
}
