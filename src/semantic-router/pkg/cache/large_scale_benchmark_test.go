package cache

import (
	"fmt"
	"os"
	"testing"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// BenchmarkLargeScale tests HNSW vs Linear at scales where HNSW shows advantages (10K-100K entries)
func BenchmarkLargeScale(b *testing.B) {
	// Initialize BERT model (GPU by default)
	useCPU := os.Getenv("USE_CPU") == "true"
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if err := candle_binding.InitModel(modelName, useCPU); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	// Large scale cache sizes where HNSW shines
	cacheSizes := []int{10000, 50000, 100000}

	// Quick mode: only run 10K for fast demo
	if os.Getenv("BENCHMARK_QUICK") == "true" {
		cacheSizes = []int{10000}
	}

	// Use medium length queries for consistency
	contentLen := MediumContent

	// HNSW configurations
	// Only using default config since performance is similar across configs
	hnswConfigs := []struct {
		name string
		m    int
		ef   int
	}{
		{"HNSW_default", 16, 200},
	}

	// Open CSV file for results
	// Create benchmark_results directory if it doesn't exist
	resultsDir := "../../benchmark_results"
	if err := os.MkdirAll(resultsDir, 0o755); err != nil {
		b.Logf("Warning: Could not create results directory: %v", err)
	}

	csvFile, err := os.OpenFile(resultsDir+"/large_scale_benchmark.csv",
		os.O_APPEND|os.O_CREATE|os.O_WRONLY,
		0o644)
	if err != nil {
		b.Logf("Warning: Could not open CSV file: %v", err)
	} else {
		defer csvFile.Close()
		// Write header if file is new
		stat, _ := csvFile.Stat()
		if stat.Size() == 0 {
			header := "cache_size,search_method,hnsw_m,hnsw_ef,avg_latency_ns,iterations,speedup_vs_linear\n"
			if _, err := csvFile.WriteString(header); err != nil {
				b.Logf("Warning: failed to write CSV header: %v", err)
			}
		}
	}

	for _, cacheSize := range cacheSizes {
		b.Run(fmt.Sprintf("CacheSize_%d", cacheSize), func(b *testing.B) {
			// Generate test data
			b.Logf("Generating %d test queries...", cacheSize)
			testQueries := make([]string, cacheSize)
			for i := 0; i < cacheSize; i++ {
				testQueries[i] = generateQuery(contentLen, i)
			}

			// Generate query embeddings once
			useCPUStr := "CPU"
			if !useCPU {
				useCPUStr = "GPU"
			}
			b.Logf("Generating embeddings for %d queries using %s...", cacheSize, useCPUStr)
			testEmbeddings := make([][]float32, cacheSize)
			embStart := time.Now()
			embProgressInterval := cacheSize / 10
			if embProgressInterval < 1000 {
				embProgressInterval = 1000
			}

			for i := 0; i < cacheSize; i++ {
				emb, err := candle_binding.GetEmbedding(testQueries[i], 0)
				if err != nil {
					b.Fatalf("Failed to generate embedding: %v", err)
				}
				testEmbeddings[i] = emb

				// Progress indicator
				if (i+1)%embProgressInterval == 0 {
					elapsed := time.Since(embStart)
					embPerSec := float64(i+1) / elapsed.Seconds()
					remaining := time.Duration(float64(cacheSize-i-1) / embPerSec * float64(time.Second))
					b.Logf("  [Embeddings] %d/%d (%.0f%%, %.0f emb/sec, ~%v remaining)",
						i+1, cacheSize, float64(i+1)/float64(cacheSize)*100,
						embPerSec, remaining.Round(time.Second))
				}
			}
			b.Logf("✓ Generated %d embeddings in %v (%.0f emb/sec)",
				cacheSize, time.Since(embStart), float64(cacheSize)/time.Since(embStart).Seconds())

			// Test query (use a query similar to middle entries for realistic search)
			searchQuery := generateQuery(contentLen, cacheSize/2)

			var linearLatency float64

			// Benchmark Linear Search
			b.Run("Linear", func(b *testing.B) {
				b.Logf("=== Testing Linear Search with %d entries ===", cacheSize)
				cache := NewInMemoryCache(InMemoryCacheOptions{
					Enabled:             true,
					SimilarityThreshold: 0.8,
					MaxEntries:          cacheSize,
					UseHNSW:             false, // Linear search
				})

				// Populate cache
				b.Logf("Building cache with %d entries...", cacheSize)
				progressInterval := cacheSize / 10
				if progressInterval < 1000 {
					progressInterval = 1000
				}

				for i := 0; i < cacheSize; i++ {
					err := cache.AddEntry(
						fmt.Sprintf("req-%d", i),
						"test-model",
						testQueries[i],
						[]byte(fmt.Sprintf("request-%d", i)),
						[]byte(fmt.Sprintf("response-%d", i)),
					)
					if err != nil {
						b.Fatalf("Failed to add entry: %v", err)
					}

					if (i+1)%progressInterval == 0 {
						b.Logf("  [Linear] Added %d/%d entries (%.0f%%)",
							i+1, cacheSize, float64(i+1)/float64(cacheSize)*100)
					}
				}
				b.Logf("✓ Linear cache built. Starting search benchmark...")

				// Run search benchmark
				b.ResetTimer()
				start := time.Now()
				for i := 0; i < b.N; i++ {
					_, _, err := cache.FindSimilar("test-model", searchQuery)
					if err != nil {
						b.Fatalf("FindSimilar failed: %v", err)
					}
				}
				b.StopTimer()

				linearLatency = float64(time.Since(start).Nanoseconds()) / float64(b.N)
				b.Logf("✓ Linear search complete: %.2f ms per query (%d iterations)",
					linearLatency/1e6, b.N)

				// Write to CSV
				if csvFile != nil {
					line := fmt.Sprintf("%d,linear,0,0,%.0f,%d,1.0\n",
						cacheSize, linearLatency, b.N)
					if _, err := csvFile.WriteString(line); err != nil {
						b.Logf("Warning: failed to write to CSV: %v", err)
					}
				}

				b.ReportMetric(linearLatency/1e6, "ms/op")
			})

			// Benchmark HNSW configurations
			for _, config := range hnswConfigs {
				b.Run(config.name, func(b *testing.B) {
					b.Logf("=== Testing %s with %d entries (M=%d, ef=%d) ===",
						config.name, cacheSize, config.m, config.ef)
					cache := NewInMemoryCache(InMemoryCacheOptions{
						Enabled:             true,
						SimilarityThreshold: 0.8,
						MaxEntries:          cacheSize,
						UseHNSW:             true,
						HNSWM:               config.m,
						HNSWEfConstruction:  config.ef,
					})

					// Populate cache
					b.Logf("Building HNSW index with %d entries (M=%d, ef=%d)...",
						cacheSize, config.m, config.ef)
					buildStart := time.Now()
					progressInterval := cacheSize / 10
					if progressInterval < 1000 {
						progressInterval = 1000
					}

					for i := 0; i < cacheSize; i++ {
						err := cache.AddEntry(
							fmt.Sprintf("req-%d", i),
							"test-model",
							testQueries[i],
							[]byte(fmt.Sprintf("request-%d", i)),
							[]byte(fmt.Sprintf("response-%d", i)),
						)
						if err != nil {
							b.Fatalf("Failed to add entry: %v", err)
						}

						// Progress indicator
						if (i+1)%progressInterval == 0 {
							elapsed := time.Since(buildStart)
							entriesPerSec := float64(i+1) / elapsed.Seconds()
							remaining := time.Duration(float64(cacheSize-i-1) / entriesPerSec * float64(time.Second))
							b.Logf("  [%s] %d/%d entries (%.0f%%, %v elapsed, ~%v remaining, %.0f entries/sec)",
								config.name, i+1, cacheSize,
								float64(i+1)/float64(cacheSize)*100,
								elapsed.Round(time.Second),
								remaining.Round(time.Second),
								entriesPerSec)
						}
					}
					buildTime := time.Since(buildStart)
					b.Logf("✓ HNSW index built in %v (%.0f entries/sec)",
						buildTime, float64(cacheSize)/buildTime.Seconds())

					// Run search benchmark
					b.Logf("Starting search benchmark...")
					b.ResetTimer()
					start := time.Now()
					for i := 0; i < b.N; i++ {
						_, _, err := cache.FindSimilar("test-model", searchQuery)
						if err != nil {
							b.Fatalf("FindSimilar failed: %v", err)
						}
					}
					b.StopTimer()

					hnswLatency := float64(time.Since(start).Nanoseconds()) / float64(b.N)
					speedup := linearLatency / hnswLatency

					b.Logf("✓ HNSW search complete: %.2f ms per query (%d iterations)",
						hnswLatency/1e6, b.N)
					b.Logf("📊 SPEEDUP: %.1fx faster than linear search (%.2f ms vs %.2f ms)",
						speedup, hnswLatency/1e6, linearLatency/1e6)

					// Write to CSV
					if csvFile != nil {
						line := fmt.Sprintf("%d,%s,%d,%d,%.0f,%d,%.2f\n",
							cacheSize, config.name, config.m, config.ef,
							hnswLatency, b.N, speedup)
						if _, err := csvFile.WriteString(line); err != nil {
							b.Logf("Warning: failed to write to CSV: %v", err)
						}
					}

					b.ReportMetric(hnswLatency/1e6, "ms/op")
					b.ReportMetric(speedup, "speedup")
					b.ReportMetric(float64(buildTime.Milliseconds()), "build_ms")
				})
			}
		})
	}
}

// BenchmarkScalability tests how performance scales with cache size
func BenchmarkScalability(b *testing.B) {
	useCPU := os.Getenv("USE_CPU") == "true"
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if err := candle_binding.InitModel(modelName, useCPU); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	// Test cache sizes from small to very large
	cacheSizes := []int{1000, 5000, 10000, 25000, 50000, 100000}

	// CSV output
	resultsDir := "../../benchmark_results"
	if err := os.MkdirAll(resultsDir, 0o755); err != nil {
		b.Logf("Warning: Could not create results directory: %v", err)
	}

	csvFile, err := os.OpenFile(resultsDir+"/scalability_benchmark.csv",
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		b.Logf("Warning: Could not open CSV file: %v", err)
	} else {
		defer csvFile.Close()
		stat, _ := csvFile.Stat()
		if stat.Size() == 0 {
			header := "cache_size,method,avg_latency_ns,latency_ms,ops_per_sec\n"
			if _, err := csvFile.WriteString(header); err != nil {
				b.Logf("Warning: failed to write CSV header: %v", err)
			}
		}
	}

	for _, cacheSize := range cacheSizes {
		// Skip linear search for very large sizes (too slow)
		testLinear := cacheSize <= 25000

		b.Run(fmt.Sprintf("Size_%d", cacheSize), func(b *testing.B) {
			// Generate test data
			testQueries := make([]string, cacheSize)
			for i := 0; i < cacheSize; i++ {
				testQueries[i] = generateQuery(MediumContent, i)
			}
			searchQuery := generateQuery(MediumContent, cacheSize/2)

			if testLinear {
				b.Run("Linear", func(b *testing.B) {
					cache := NewInMemoryCache(InMemoryCacheOptions{
						Enabled:             true,
						SimilarityThreshold: 0.8,
						MaxEntries:          cacheSize,
						UseHNSW:             false,
					})

					for i := 0; i < cacheSize; i++ {
						if err := cache.AddEntry(fmt.Sprintf("req-%d", i), "model",
							testQueries[i], []byte("req"), []byte("resp")); err != nil {
							b.Fatalf("AddEntry failed: %v", err)
						}
					}

					b.ResetTimer()
					start := time.Now()
					for i := 0; i < b.N; i++ {
						if _, _, err := cache.FindSimilar("model", searchQuery); err != nil {
							b.Fatalf("FindSimilar failed: %v", err)
						}
					}
					elapsed := time.Since(start)

					avgLatency := float64(elapsed.Nanoseconds()) / float64(b.N)
					latencyMS := avgLatency / 1e6
					opsPerSec := float64(b.N) / elapsed.Seconds()

					if csvFile != nil {
						line := fmt.Sprintf("%d,linear,%.0f,%.3f,%.0f\n",
							cacheSize, avgLatency, latencyMS, opsPerSec)
						if _, err := csvFile.WriteString(line); err != nil {
							b.Logf("Warning: failed to write to CSV: %v", err)
						}
					}

					b.ReportMetric(latencyMS, "ms/op")
					b.ReportMetric(opsPerSec, "qps")
				})
			}

			b.Run("HNSW", func(b *testing.B) {
				cache := NewInMemoryCache(InMemoryCacheOptions{
					Enabled:             true,
					SimilarityThreshold: 0.8,
					MaxEntries:          cacheSize,
					UseHNSW:             true,
					HNSWM:               16,
					HNSWEfConstruction:  200,
				})

				buildStart := time.Now()
				for i := 0; i < cacheSize; i++ {
					if err := cache.AddEntry(fmt.Sprintf("req-%d", i), "model",
						testQueries[i], []byte("req"), []byte("resp")); err != nil {
						b.Fatalf("AddEntry failed: %v", err)
					}
					if (i+1)%10000 == 0 {
						b.Logf("  Built %d/%d entries", i+1, cacheSize)
					}
				}
				b.Logf("HNSW build time: %v", time.Since(buildStart))

				b.ResetTimer()
				start := time.Now()
				for i := 0; i < b.N; i++ {
					if _, _, err := cache.FindSimilar("model", searchQuery); err != nil {
						b.Fatalf("FindSimilar failed: %v", err)
					}
				}
				elapsed := time.Since(start)

				avgLatency := float64(elapsed.Nanoseconds()) / float64(b.N)
				latencyMS := avgLatency / 1e6
				opsPerSec := float64(b.N) / elapsed.Seconds()

				if csvFile != nil {
					line := fmt.Sprintf("%d,hnsw,%.0f,%.3f,%.0f\n",
						cacheSize, avgLatency, latencyMS, opsPerSec)
					if _, err := csvFile.WriteString(line); err != nil {
						b.Logf("Warning: failed to write to CSV: %v", err)
					}
				}

				b.ReportMetric(latencyMS, "ms/op")
				b.ReportMetric(opsPerSec, "qps")
			})
		})
	}
}

// BenchmarkHNSWParameterSweep tests different HNSW parameters at large scale
func BenchmarkHNSWParameterSweep(b *testing.B) {
	useCPU := os.Getenv("USE_CPU") == "true"
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if err := candle_binding.InitModel(modelName, useCPU); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	cacheSize := 50000 // 50K entries - good size to show differences

	// Parameter combinations to test
	// Test different M (connectivity) and efSearch (search quality) combinations
	// Fixed efConstruction=200 to focus on search-time performance
	configs := []struct {
		name     string
		m        int
		efSearch int
	}{
		// Low connectivity
		{"M8_efSearch10", 8, 10},
		{"M8_efSearch50", 8, 50},
		{"M8_efSearch100", 8, 100},
		{"M8_efSearch200", 8, 200},

		// Medium connectivity (recommended)
		{"M16_efSearch10", 16, 10},
		{"M16_efSearch50", 16, 50},
		{"M16_efSearch100", 16, 100},
		{"M16_efSearch200", 16, 200},
		{"M16_efSearch400", 16, 400},

		// High connectivity
		{"M32_efSearch50", 32, 50},
		{"M32_efSearch100", 32, 100},
		{"M32_efSearch200", 32, 200},
	}

	// Generate test data once
	b.Logf("Generating %d test queries...", cacheSize)
	testQueries := make([]string, cacheSize)
	for i := 0; i < cacheSize; i++ {
		testQueries[i] = generateQuery(MediumContent, i)
	}
	searchQuery := generateQuery(MediumContent, cacheSize/2)

	// CSV output
	resultsDir := "../../benchmark_results"
	if err := os.MkdirAll(resultsDir, 0o755); err != nil {
		b.Logf("Warning: Could not create results directory: %v", err)
	}

	csvFile, err := os.OpenFile(resultsDir+"/hnsw_parameter_sweep.csv",
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		b.Logf("Warning: Could not open CSV file: %v", err)
	} else {
		defer csvFile.Close()
		stat, _ := csvFile.Stat()
		if stat.Size() == 0 {
			header := "m,ef_search,build_time_ms,search_latency_ns,search_latency_ms,qps,memory_mb\n"
			if _, err := csvFile.WriteString(header); err != nil {
				b.Logf("Warning: failed to write CSV header: %v", err)
			}
		}
	}

	for _, config := range configs {
		b.Run(config.name, func(b *testing.B) {
			cache := NewInMemoryCache(InMemoryCacheOptions{
				Enabled:             true,
				SimilarityThreshold: 0.8,
				MaxEntries:          cacheSize,
				UseHNSW:             true,
				HNSWM:               config.m,
				HNSWEfConstruction:  200, // Fixed for consistent build quality
				HNSWEfSearch:        config.efSearch,
			})

			// Build index and measure time
			b.Logf("Building HNSW index: M=%d, efConstruction=200, efSearch=%d", config.m, config.efSearch)
			buildStart := time.Now()
			for i := 0; i < cacheSize; i++ {
				if err := cache.AddEntry(fmt.Sprintf("req-%d", i), "model",
					testQueries[i], []byte("req"), []byte("resp")); err != nil {
					b.Fatalf("AddEntry failed: %v", err)
				}
				if (i+1)%10000 == 0 {
					b.Logf("  Progress: %d/%d", i+1, cacheSize)
				}
			}
			buildTime := time.Since(buildStart)

			// Estimate memory usage (rough)
			// Embeddings: cacheSize × 384 × 4 bytes
			// HNSW graph: cacheSize × M × 2 × 4 bytes (bidirectional links)
			embeddingMemMB := float64(cacheSize*384*4) / 1024 / 1024
			graphMemMB := float64(cacheSize*config.m*2*4) / 1024 / 1024
			totalMemMB := embeddingMemMB + graphMemMB

			b.Logf("Build time: %v, Est. memory: %.1f MB", buildTime, totalMemMB)

			// Benchmark search
			b.ResetTimer()
			start := time.Now()
			for i := 0; i < b.N; i++ {
				if _, _, err := cache.FindSimilar("model", searchQuery); err != nil {
					b.Fatalf("FindSimilar failed: %v", err)
				}
			}
			elapsed := time.Since(start)

			avgLatency := float64(elapsed.Nanoseconds()) / float64(b.N)
			latencyMS := avgLatency / 1e6
			qps := float64(b.N) / elapsed.Seconds()

			// Write to CSV
			if csvFile != nil {
				line := fmt.Sprintf("%d,%d,%.0f,%.0f,%.3f,%.0f,%.1f\n",
					config.m, config.efSearch, float64(buildTime.Milliseconds()),
					avgLatency, latencyMS, qps, totalMemMB)
				if _, err := csvFile.WriteString(line); err != nil {
					b.Logf("Warning: failed to write to CSV: %v", err)
				}
			}

			b.ReportMetric(latencyMS, "ms/op")
			b.ReportMetric(qps, "qps")
			b.ReportMetric(float64(buildTime.Milliseconds()), "build_ms")
			b.ReportMetric(totalMemMB, "memory_mb")
		})
	}
}
