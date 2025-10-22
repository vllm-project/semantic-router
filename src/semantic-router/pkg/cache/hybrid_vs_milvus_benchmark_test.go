//go:build milvus && !windows && cgo
// +build milvus,!windows,cgo

package cache

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// BenchmarkResult stores detailed benchmark metrics
type BenchmarkResult struct {
	CacheType           string
	CacheSize           int
	Operation           string
	AvgLatencyNs        int64
	AvgLatencyMs        float64
	P50LatencyMs        float64
	P95LatencyMs        float64
	P99LatencyMs        float64
	QPS                 float64
	MemoryUsageMB       float64
	HitRate             float64
	DatabaseCalls       int64
	TotalRequests       int64
	DatabaseCallPercent float64
}

// LatencyDistribution tracks percentile latencies
type LatencyDistribution struct {
	latencies []time.Duration
	mu        sync.Mutex
}

func (ld *LatencyDistribution) Record(latency time.Duration) {
	ld.mu.Lock()
	defer ld.mu.Unlock()
	ld.latencies = append(ld.latencies, latency)
}

func (ld *LatencyDistribution) GetPercentile(p float64) float64 {
	ld.mu.Lock()
	defer ld.mu.Unlock()

	if len(ld.latencies) == 0 {
		return 0
	}

	// Sort latencies
	sorted := make([]time.Duration, len(ld.latencies))
	copy(sorted, ld.latencies)
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	idx := int(float64(len(sorted)) * p)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}

	return float64(sorted[idx].Nanoseconds()) / 1e6
}

// DatabaseCallCounter tracks Milvus database calls
type DatabaseCallCounter struct {
	calls int64
}

func (dcc *DatabaseCallCounter) Increment() {
	atomic.AddInt64(&dcc.calls, 1)
}

func (dcc *DatabaseCallCounter) Get() int64 {
	return atomic.LoadInt64(&dcc.calls)
}

func (dcc *DatabaseCallCounter) Reset() {
	atomic.StoreInt64(&dcc.calls, 0)
}

// getMilvusConfigPath returns the path to milvus.yaml config file
func getMilvusConfigPath() string {
	// Check for environment variable first
	if envPath := os.Getenv("MILVUS_CONFIG_PATH"); envPath != "" {
		if _, err := os.Stat(envPath); err == nil {
			return envPath
		}
	}

	// Try relative from project root (when run via make)
	configPath := "config/cache/milvus.yaml"
	if _, err := os.Stat(configPath); err == nil {
		return configPath
	}

	// Fallback to relative from test directory
	return "../../../../../config/cache/milvus.yaml"
}

// BenchmarkHybridVsMilvus is the comprehensive benchmark comparing hybrid cache vs pure Milvus
// This validates the claims from the hybrid HNSW storage architecture paper
func BenchmarkHybridVsMilvus(b *testing.B) {
	// Initialize BERT model
	useCPU := os.Getenv("USE_CPU") != "false"
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if err := candle_binding.InitModel(modelName, useCPU); err != nil {
		b.Fatalf("Failed to initialize BERT model: %v", err)
	}

	// Test configurations - realistic production scales
	cacheSizes := []int{
		10000,  // Medium: 10K entries
		50000,  // Large: 50K entries
		100000, // Extra Large: 100K entries
	}

	// CSV output file - save to project benchmark_results directory
	// Use PROJECT_ROOT environment variable, fallback to working directory
	projectRoot := os.Getenv("PROJECT_ROOT")
	if projectRoot == "" {
		// If not set, use current working directory
		var err error
		projectRoot, err = os.Getwd()
		if err != nil {
			b.Logf("Warning: Could not determine working directory: %v", err)
			projectRoot = "."
		}
	}
	resultsDir := filepath.Join(projectRoot, "benchmark_results", "hybrid_vs_milvus")
	os.MkdirAll(resultsDir, 0755)
	timestamp := time.Now().Format("20060102_150405")
	csvPath := filepath.Join(resultsDir, fmt.Sprintf("results_%s.csv", timestamp))
	csvFile, err := os.Create(csvPath)
	if err != nil {
		b.Logf("Warning: Could not create CSV file at %s: %v", csvPath, err)
	} else {
		defer csvFile.Close()
		b.Logf("Results will be saved to: %s", csvPath)
		// Write CSV header
		csvFile.WriteString("cache_type,cache_size,operation,avg_latency_ns,avg_latency_ms,p50_ms,p95_ms,p99_ms,qps,memory_mb,hit_rate,db_calls,total_requests,db_call_percent\n")
	}

	b.Logf("=== Hybrid Cache vs Pure Milvus Benchmark ===")
	b.Logf("")

	for _, cacheSize := range cacheSizes {
		b.Run(fmt.Sprintf("CacheSize_%d", cacheSize), func(b *testing.B) {
			// Generate test queries
			b.Logf("Generating %d test queries...", cacheSize)
			testQueries := make([]string, cacheSize)
			for i := 0; i < cacheSize; i++ {
				testQueries[i] = generateQuery(MediumContent, i)
			}

			// Test two realistic hit rate scenarios
			scenarios := []struct {
				name    string
				hitRate float64
			}{
				{"HitRate_5pct", 0.05},  // 5% hit rate - very realistic for semantic cache
				{"HitRate_20pct", 0.20}, // 20% hit rate - optimistic but realistic
			}

			// Generate search queries for each scenario
			allSearchQueries := make(map[string][]string)
			for _, scenario := range scenarios {
				queries := make([]string, 100)
				hitCount := int(scenario.hitRate * 100)

				// Hits: reuse cached queries
				for i := 0; i < hitCount; i++ {
					queries[i] = testQueries[i%cacheSize]
				}

				// Misses: generate new queries
				for i := hitCount; i < 100; i++ {
					queries[i] = generateQuery(MediumContent, cacheSize+i)
				}

				allSearchQueries[scenario.name] = queries
				b.Logf("Generated queries for %s: %d hits, %d misses",
					scenario.name, hitCount, 100-hitCount)
			}

			// ============================================================
			// 1. Benchmark Pure Milvus Cache (Optional via SKIP_MILVUS env var)
			// ============================================================
			b.Run("Milvus", func(b *testing.B) {
				if os.Getenv("SKIP_MILVUS") == "true" {
					b.Skip("Skipping Milvus benchmark (SKIP_MILVUS=true)")
					return
				}
				b.Logf("\n=== Testing Pure Milvus Cache ===")

				milvusCache, err := NewMilvusCache(MilvusCacheOptions{
					Enabled:             true,
					SimilarityThreshold: 0.80,
					TTLSeconds:          3600,
					ConfigPath:          getMilvusConfigPath(),
				})
				if err != nil {
					b.Fatalf("Failed to create Milvus cache: %v", err)
				}
				defer milvusCache.Close()

				// Wait for Milvus to be ready
				time.Sleep(2 * time.Second)

				// Populate cache using batch insert for speed
				b.Logf("Populating Milvus with %d entries (using batch insert)...", cacheSize)
				populateStart := time.Now()

				// Prepare all entries
				entries := make([]CacheEntry, cacheSize)
				for i := 0; i < cacheSize; i++ {
					entries[i] = CacheEntry{
						RequestID:    fmt.Sprintf("req-milvus-%d", i),
						Model:        "test-model",
						Query:        testQueries[i],
						RequestBody:  []byte(fmt.Sprintf("request-%d", i)),
						ResponseBody: []byte(fmt.Sprintf("response-%d-this-is-a-longer-response-body-to-simulate-realistic-llm-output", i)),
					}
				}

				// Insert in batches of 100
				batchSize := 100
				for i := 0; i < cacheSize; i += batchSize {
					end := i + batchSize
					if end > cacheSize {
						end = cacheSize
					}

					err := milvusCache.AddEntriesBatch(entries[i:end])
					if err != nil {
						b.Fatalf("Failed to add batch: %v", err)
					}

					if (i+batchSize)%1000 == 0 {
						b.Logf("  Populated %d/%d entries", i+batchSize, cacheSize)
					}
				}

				// Flush once after all batches
				b.Logf("Flushing Milvus...")
				if err := milvusCache.Flush(); err != nil {
					b.Logf("Warning: flush failed: %v", err)
				}

				populateTime := time.Since(populateStart)
				b.Logf("✓ Populated in %v (%.0f entries/sec)", populateTime, float64(cacheSize)/populateTime.Seconds())

				// Wait for Milvus to be ready
				time.Sleep(2 * time.Second)

				// Test each hit rate scenario
				for _, scenario := range scenarios {
					searchQueries := allSearchQueries[scenario.name]

					b.Run(scenario.name, func(b *testing.B) {
						// Benchmark search operations
						b.Logf("Running search benchmark for %s...", scenario.name)
						latencyDist := &LatencyDistribution{latencies: make([]time.Duration, 0, b.N)}
						dbCallCounter := &DatabaseCallCounter{}
						hits := 0
						misses := 0

						b.ResetTimer()
						start := time.Now()

						for i := 0; i < b.N; i++ {
							queryIdx := i % len(searchQueries)
							searchStart := time.Now()

							// Every Milvus FindSimilar is a database call
							dbCallCounter.Increment()

							_, found, err := milvusCache.FindSimilar("test-model", searchQueries[queryIdx])
							searchLatency := time.Since(searchStart)

							if err != nil {
								b.Logf("Warning: search error at iteration %d: %v", i, err)
							}

							latencyDist.Record(searchLatency)

							if found {
								hits++
							} else {
								misses++
							}
						}

						elapsed := time.Since(start)
						b.StopTimer()

						// Calculate metrics
						avgLatencyNs := elapsed.Nanoseconds() / int64(b.N)
						avgLatencyMs := float64(avgLatencyNs) / 1e6
						qps := float64(b.N) / elapsed.Seconds()
						hitRate := float64(hits) / float64(b.N) * 100
						dbCalls := dbCallCounter.Get()
						dbCallPercent := float64(dbCalls) / float64(b.N) * 100

						// Memory usage estimation
						memUsageMB := estimateMilvusMemory(cacheSize)

						result := BenchmarkResult{
							CacheType:           "milvus",
							CacheSize:           cacheSize,
							Operation:           "search",
							AvgLatencyNs:        avgLatencyNs,
							AvgLatencyMs:        avgLatencyMs,
							P50LatencyMs:        latencyDist.GetPercentile(0.50),
							P95LatencyMs:        latencyDist.GetPercentile(0.95),
							P99LatencyMs:        latencyDist.GetPercentile(0.99),
							QPS:                 qps,
							MemoryUsageMB:       memUsageMB,
							HitRate:             hitRate,
							DatabaseCalls:       dbCalls,
							TotalRequests:       int64(b.N),
							DatabaseCallPercent: dbCallPercent,
						}

						// Report results
						b.Logf("\n--- Milvus Results (%s) ---", scenario.name)
						b.Logf("Avg Latency: %.2f ms", avgLatencyMs)
						b.Logf("P50: %.2f ms, P95: %.2f ms, P99: %.2f ms", result.P50LatencyMs, result.P95LatencyMs, result.P99LatencyMs)
						b.Logf("QPS: %.0f", qps)
						b.Logf("Hit Rate: %.1f%% (expected: %.0f%%)", hitRate, scenario.hitRate*100)
						b.Logf("Hits: %d, Misses: %d out of %d total", hits, misses, b.N)
						b.Logf("Database Calls: %d/%d (%.0f%%)", dbCalls, b.N, dbCallPercent)
						b.Logf("Memory Usage: %.1f MB", memUsageMB)

						// Write to CSV
						if csvFile != nil {
							writeBenchmarkResultToCSV(csvFile, result)
						}

						b.ReportMetric(avgLatencyMs, "ms/op")
						b.ReportMetric(qps, "qps")
						b.ReportMetric(hitRate, "hit_rate_%")
					})
				}
			})

			// ============================================================
			// 2. Benchmark Hybrid Cache
			// ============================================================
			b.Run("Hybrid", func(b *testing.B) {
				b.Logf("\n=== Testing Hybrid Cache ===")

				hybridCache, err := NewHybridCache(HybridCacheOptions{
					Enabled:             true,
					SimilarityThreshold: 0.80,
					TTLSeconds:          3600,
					MaxMemoryEntries:    cacheSize,
					HNSWM:               16,
					HNSWEfConstruction:  200,
					MilvusConfigPath:    getMilvusConfigPath(),
				})
				if err != nil {
					b.Fatalf("Failed to create Hybrid cache: %v", err)
				}
				defer hybridCache.Close()

				// Wait for initialization
				time.Sleep(2 * time.Second)

				// Populate cache using batch insert for speed
				b.Logf("Populating Hybrid cache with %d entries (using batch insert)...", cacheSize)
				populateStart := time.Now()

				// Prepare all entries
				entries := make([]CacheEntry, cacheSize)
				for i := 0; i < cacheSize; i++ {
					entries[i] = CacheEntry{
						RequestID:    fmt.Sprintf("req-hybrid-%d", i),
						Model:        "test-model",
						Query:        testQueries[i],
						RequestBody:  []byte(fmt.Sprintf("request-%d", i)),
						ResponseBody: []byte(fmt.Sprintf("response-%d-this-is-a-longer-response-body-to-simulate-realistic-llm-output", i)),
					}
				}

				// Insert in batches of 100
				batchSize := 100
				for i := 0; i < cacheSize; i += batchSize {
					end := i + batchSize
					if end > cacheSize {
						end = cacheSize
					}

					err := hybridCache.AddEntriesBatch(entries[i:end])
					if err != nil {
						b.Fatalf("Failed to add batch: %v", err)
					}

					if (i+batchSize)%1000 == 0 {
						b.Logf("  Populated %d/%d entries", i+batchSize, cacheSize)
					}
				}

				// Flush once after all batches
				b.Logf("Flushing Milvus...")
				if err := hybridCache.Flush(); err != nil {
					b.Logf("Warning: flush failed: %v", err)
				}

				populateTime := time.Since(populateStart)
				b.Logf("✓ Populated in %v (%.0f entries/sec)", populateTime, float64(cacheSize)/populateTime.Seconds())

				// Wait for Milvus to be ready
				time.Sleep(2 * time.Second)

				// Test each hit rate scenario
				for _, scenario := range scenarios {
					searchQueries := allSearchQueries[scenario.name]

					b.Run(scenario.name, func(b *testing.B) {
						// Get initial memory stats
						var memBefore runtime.MemStats
						runtime.ReadMemStats(&memBefore)

						// Benchmark search operations
						b.Logf("Running search benchmark for %s...", scenario.name)
						latencyDist := &LatencyDistribution{latencies: make([]time.Duration, 0, b.N)}
						hits := 0
						misses := 0

						// Track database calls (Hybrid should make fewer calls due to threshold filtering)
						initialMilvusCallCount := hybridCache.milvusCache.hitCount + hybridCache.milvusCache.missCount

						b.ResetTimer()
						start := time.Now()

						for i := 0; i < b.N; i++ {
							queryIdx := i % len(searchQueries)
							searchStart := time.Now()

							_, found, err := hybridCache.FindSimilar("test-model", searchQueries[queryIdx])
							searchLatency := time.Since(searchStart)

							if err != nil {
								b.Logf("Warning: search error at iteration %d: %v", i, err)
							}

							latencyDist.Record(searchLatency)

							if found {
								hits++
							} else {
								misses++
							}
						}

						elapsed := time.Since(start)
						b.StopTimer()

						// Calculate database calls (both hits and misses involve Milvus calls)
						finalMilvusCallCount := hybridCache.milvusCache.hitCount + hybridCache.milvusCache.missCount
						dbCalls := finalMilvusCallCount - initialMilvusCallCount

						// Get final memory stats
						var memAfter runtime.MemStats
						runtime.ReadMemStats(&memAfter)

						// Fix: Prevent unsigned integer underflow if GC ran during benchmark
						var memUsageMB float64
						if memAfter.Alloc >= memBefore.Alloc {
							memUsageMB = float64(memAfter.Alloc-memBefore.Alloc) / 1024 / 1024
						} else {
							// GC ran, use estimation instead
							memUsageMB = estimateHybridMemory(cacheSize)
						}

						// Calculate metrics
						avgLatencyNs := elapsed.Nanoseconds() / int64(b.N)
						avgLatencyMs := float64(avgLatencyNs) / 1e6
						qps := float64(b.N) / elapsed.Seconds()
						hitRate := float64(hits) / float64(b.N) * 100
						dbCallPercent := float64(dbCalls) / float64(b.N) * 100

						result := BenchmarkResult{
							CacheType:           "hybrid",
							CacheSize:           cacheSize,
							Operation:           "search",
							AvgLatencyNs:        avgLatencyNs,
							AvgLatencyMs:        avgLatencyMs,
							P50LatencyMs:        latencyDist.GetPercentile(0.50),
							P95LatencyMs:        latencyDist.GetPercentile(0.95),
							P99LatencyMs:        latencyDist.GetPercentile(0.99),
							QPS:                 qps,
							MemoryUsageMB:       memUsageMB,
							HitRate:             hitRate,
							DatabaseCalls:       dbCalls,
							TotalRequests:       int64(b.N),
							DatabaseCallPercent: dbCallPercent,
						}

						// Report results
						b.Logf("\n--- Hybrid Cache Results (%s) ---", scenario.name)
						b.Logf("Avg Latency: %.2f ms", avgLatencyMs)
						b.Logf("P50: %.2f ms, P95: %.2f ms, P99: %.2f ms", result.P50LatencyMs, result.P95LatencyMs, result.P99LatencyMs)
						b.Logf("QPS: %.0f", qps)
						b.Logf("Hit Rate: %.1f%% (expected: %.0f%%)", hitRate, scenario.hitRate*100)
						b.Logf("Hits: %d, Misses: %d out of %d total", hits, misses, b.N)
						b.Logf("Database Calls: %d/%d (%.0f%%)", dbCalls, b.N, dbCallPercent)
						b.Logf("Memory Usage: %.1f MB", memUsageMB)

						// Write to CSV
						if csvFile != nil {
							writeBenchmarkResultToCSV(csvFile, result)
						}

						b.ReportMetric(avgLatencyMs, "ms/op")
						b.ReportMetric(qps, "qps")
						b.ReportMetric(hitRate, "hit_rate_%")
						b.ReportMetric(dbCallPercent, "db_call_%")
					})
				}
			})
		})
	}
}

// BenchmarkComponentLatency measures individual component latencies
func BenchmarkComponentLatency(b *testing.B) {
	// Initialize BERT model
	useCPU := os.Getenv("USE_CPU") != "false"
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if err := candle_binding.InitModel(modelName, useCPU); err != nil {
		b.Fatalf("Failed to initialize BERT model: %v", err)
	}

	cacheSize := 10000
	testQueries := make([]string, cacheSize)
	for i := 0; i < cacheSize; i++ {
		testQueries[i] = generateQuery(MediumContent, i)
	}

	b.Run("EmbeddingGeneration", func(b *testing.B) {
		query := testQueries[0]
		b.ResetTimer()
		start := time.Now()
		for i := 0; i < b.N; i++ {
			_, err := candle_binding.GetEmbedding(query, 0)
			if err != nil {
				b.Fatal(err)
			}
		}
		elapsed := time.Since(start)
		avgMs := float64(elapsed.Nanoseconds()) / float64(b.N) / 1e6
		b.Logf("Embedding generation: %.2f ms/op", avgMs)
		b.ReportMetric(avgMs, "ms/op")
	})

	b.Run("HNSWSearch", func(b *testing.B) {
		// Build HNSW index
		cache := NewInMemoryCache(InMemoryCacheOptions{
			Enabled:             true,
			SimilarityThreshold: 0.80,
			MaxEntries:          cacheSize,
			UseHNSW:             true,
			HNSWM:               16,
			HNSWEfConstruction:  200,
		})

		b.Logf("Building HNSW index with %d entries...", cacheSize)
		for i := 0; i < cacheSize; i++ {
			cache.AddEntry(fmt.Sprintf("req-%d", i), "model", testQueries[i], []byte("req"), []byte("resp"))
		}
		b.Logf("✓ HNSW index built")

		query := testQueries[0]

		b.ResetTimer()
		start := time.Now()
		for i := 0; i < b.N; i++ {
			// Note: HNSW search uses entries slice internally
			cache.FindSimilar("model", query)
		}
		elapsed := time.Since(start)
		avgMs := float64(elapsed.Nanoseconds()) / float64(b.N) / 1e6
		b.Logf("HNSW search: %.2f ms/op", avgMs)
		b.ReportMetric(avgMs, "ms/op")
	})

	b.Run("MilvusVectorSearch", func(b *testing.B) {
		milvusCache, err := NewMilvusCache(MilvusCacheOptions{
			Enabled:             true,
			SimilarityThreshold: 0.80,
			TTLSeconds:          3600,
			ConfigPath:          getMilvusConfigPath(),
		})
		if err != nil {
			b.Fatalf("Failed to create Milvus cache: %v", err)
		}
		defer milvusCache.Close()

		time.Sleep(2 * time.Second)

		b.Logf("Populating Milvus with %d entries...", cacheSize)
		for i := 0; i < cacheSize; i++ {
			milvusCache.AddEntry(fmt.Sprintf("req-%d", i), "model", testQueries[i], []byte("req"), []byte("resp"))
		}
		time.Sleep(2 * time.Second)
		b.Logf("✓ Milvus populated")

		query := testQueries[0]

		b.ResetTimer()
		start := time.Now()
		for i := 0; i < b.N; i++ {
			milvusCache.FindSimilar("model", query)
		}
		elapsed := time.Since(start)
		avgMs := float64(elapsed.Nanoseconds()) / float64(b.N) / 1e6
		b.Logf("Milvus vector search: %.2f ms/op", avgMs)
		b.ReportMetric(avgMs, "ms/op")
	})

	b.Run("MilvusGetByID", func(b *testing.B) {
		// This would test Milvus get by ID if we exposed that method
		b.Skip("Milvus GetByID not exposed in current implementation")
	})
}

// BenchmarkThroughputUnderLoad tests throughput with concurrent requests
func BenchmarkThroughputUnderLoad(b *testing.B) {
	// Initialize BERT model
	useCPU := os.Getenv("USE_CPU") != "false"
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if err := candle_binding.InitModel(modelName, useCPU); err != nil {
		b.Fatalf("Failed to initialize BERT model: %v", err)
	}

	cacheSize := 10000
	concurrencyLevels := []int{1, 10, 50, 100}

	testQueries := make([]string, cacheSize)
	for i := 0; i < cacheSize; i++ {
		testQueries[i] = generateQuery(MediumContent, i)
	}

	for _, concurrency := range concurrencyLevels {
		b.Run(fmt.Sprintf("Milvus_Concurrency_%d", concurrency), func(b *testing.B) {
			milvusCache, err := NewMilvusCache(MilvusCacheOptions{
				Enabled:             true,
				SimilarityThreshold: 0.80,
				TTLSeconds:          3600,
				ConfigPath:          getMilvusConfigPath(),
			})
			if err != nil {
				b.Fatalf("Failed to create Milvus cache: %v", err)
			}
			defer milvusCache.Close()

			time.Sleep(2 * time.Second)

			// Populate
			for i := 0; i < cacheSize; i++ {
				milvusCache.AddEntry(fmt.Sprintf("req-%d", i), "model", testQueries[i], []byte("req"), []byte("resp"))
			}
			time.Sleep(2 * time.Second)

			b.ResetTimer()
			b.SetParallelism(concurrency)
			start := time.Now()

			b.RunParallel(func(pb *testing.PB) {
				i := 0
				for pb.Next() {
					query := testQueries[i%len(testQueries)]
					milvusCache.FindSimilar("model", query)
					i++
				}
			})

			elapsed := time.Since(start)
			qps := float64(b.N) / elapsed.Seconds()
			b.Logf("QPS with %d concurrent workers: %.0f", concurrency, qps)
			b.ReportMetric(qps, "qps")
		})

		b.Run(fmt.Sprintf("Hybrid_Concurrency_%d", concurrency), func(b *testing.B) {
			hybridCache, err := NewHybridCache(HybridCacheOptions{
				Enabled:             true,
				SimilarityThreshold: 0.80,
				TTLSeconds:          3600,
				MaxMemoryEntries:    cacheSize,
				HNSWM:               16,
				HNSWEfConstruction:  200,
				MilvusConfigPath:    getMilvusConfigPath(),
			})
			if err != nil {
				b.Fatalf("Failed to create Hybrid cache: %v", err)
			}
			defer hybridCache.Close()

			time.Sleep(2 * time.Second)

			// Populate
			for i := 0; i < cacheSize; i++ {
				hybridCache.AddEntry(fmt.Sprintf("req-%d", i), "model", testQueries[i], []byte("req"), []byte("resp"))
			}
			time.Sleep(2 * time.Second)

			b.ResetTimer()
			b.SetParallelism(concurrency)
			start := time.Now()

			b.RunParallel(func(pb *testing.PB) {
				i := 0
				for pb.Next() {
					query := testQueries[i%len(testQueries)]
					hybridCache.FindSimilar("model", query)
					i++
				}
			})

			elapsed := time.Since(start)
			qps := float64(b.N) / elapsed.Seconds()
			b.Logf("QPS with %d concurrent workers: %.0f", concurrency, qps)
			b.ReportMetric(qps, "qps")
		})
	}
}

// Helper functions

func estimateMilvusMemory(cacheSize int) float64 {
	// Milvus memory estimation (rough)
	// - Embeddings: cacheSize × 384 × 4 bytes
	// - HNSW index: cacheSize × 16 × 2 × 4 bytes (M=16, bidirectional)
	// - Metadata: cacheSize × 0.5 KB
	embeddingMB := float64(cacheSize*384*4) / 1024 / 1024
	indexMB := float64(cacheSize*16*2*4) / 1024 / 1024
	metadataMB := float64(cacheSize) * 0.5 / 1024
	return embeddingMB + indexMB + metadataMB
}

func estimateHybridMemory(cacheSize int) float64 {
	// Hybrid memory estimation (in-memory HNSW only, documents in Milvus)
	// - Embeddings: cacheSize × 384 × 4 bytes
	// - HNSW index: cacheSize × 16 × 2 × 4 bytes (M=16, bidirectional)
	// - ID map: cacheSize × 50 bytes (average string length)
	embeddingMB := float64(cacheSize*384*4) / 1024 / 1024
	indexMB := float64(cacheSize*16*2*4) / 1024 / 1024
	idMapMB := float64(cacheSize*50) / 1024 / 1024
	return embeddingMB + indexMB + idMapMB
}

func writeBenchmarkResultToCSV(file *os.File, result BenchmarkResult) {
	line := fmt.Sprintf("%s,%d,%s,%d,%.3f,%.3f,%.3f,%.3f,%.0f,%.1f,%.1f,%d,%d,%.1f\n",
		result.CacheType,
		result.CacheSize,
		result.Operation,
		result.AvgLatencyNs,
		result.AvgLatencyMs,
		result.P50LatencyMs,
		result.P95LatencyMs,
		result.P99LatencyMs,
		result.QPS,
		result.MemoryUsageMB,
		result.HitRate,
		result.DatabaseCalls,
		result.TotalRequests,
		result.DatabaseCallPercent,
	)
	file.WriteString(line)
}

// TestHybridVsMilvusSmoke is a quick smoke test to verify both caches work
func TestHybridVsMilvusSmoke(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping smoke test in short mode")
	}

	// Initialize BERT model
	useCPU := os.Getenv("USE_CPU") != "false"
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if err := candle_binding.InitModel(modelName, useCPU); err != nil {
		t.Fatalf("Failed to initialize BERT model: %v", err)
	}

	// Test Milvus cache
	t.Run("Milvus", func(t *testing.T) {
		cache, err := NewMilvusCache(MilvusCacheOptions{
			Enabled:             true,
			SimilarityThreshold: 0.85,
			TTLSeconds:          3600,
			ConfigPath:          getMilvusConfigPath(),
		})
		if err != nil {
			t.Fatalf("Failed to create Milvus cache: %v", err)
		}
		defer cache.Close()

		time.Sleep(1 * time.Second)

		// Add entry
		err = cache.AddEntry("req-1", "model", "What is machine learning?", []byte("req"), []byte("ML is..."))
		if err != nil {
			t.Fatalf("Failed to add entry: %v", err)
		}

		time.Sleep(1 * time.Second)

		// Find similar
		resp, found, err := cache.FindSimilar("model", "What is machine learning?")
		if err != nil {
			t.Fatalf("FindSimilar failed: %v", err)
		}
		if !found {
			t.Fatalf("Expected to find entry, but got miss")
		}
		if string(resp) != "ML is..." {
			t.Fatalf("Expected 'ML is...', got '%s'", string(resp))
		}

		t.Logf("✓ Milvus cache smoke test passed")
	})

	// Test Hybrid cache
	t.Run("Hybrid", func(t *testing.T) {
		cache, err := NewHybridCache(HybridCacheOptions{
			Enabled:             true,
			SimilarityThreshold: 0.85,
			TTLSeconds:          3600,
			MaxMemoryEntries:    1000,
			HNSWM:               16,
			HNSWEfConstruction:  200,
			MilvusConfigPath:    getMilvusConfigPath(),
		})
		if err != nil {
			t.Fatalf("Failed to create Hybrid cache: %v", err)
		}
		defer cache.Close()

		time.Sleep(1 * time.Second)

		// Add entry
		err = cache.AddEntry("req-1", "model", "What is deep learning?", []byte("req"), []byte("DL is..."))
		if err != nil {
			t.Fatalf("Failed to add entry: %v", err)
		}

		time.Sleep(1 * time.Second)

		// Find similar
		resp, found, err := cache.FindSimilar("model", "What is deep learning?")
		if err != nil {
			t.Fatalf("FindSimilar failed: %v", err)
		}
		if !found {
			t.Fatalf("Expected to find entry, but got miss")
		}
		if string(resp) != "DL is..." {
			t.Fatalf("Expected 'DL is...', got '%s'", string(resp))
		}

		t.Logf("✓ Hybrid cache smoke test passed")
	})
}
