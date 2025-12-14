//go:build !windows && cgo

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"runtime/pprof"
	"strings"
	"syscall"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
)

const logo = `
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                      SEMANTIC CACHE PERFORMANCE BENCHMARK                             ║
║                                                                                       ║
║  Evaluates latency and throughput for semantic cache search operations               ║
║  Measures embedding generation, similarity search, and overall performance           ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
`

type Config struct {
	// Test parameters
	CacheSizes        []int
	ConcurrencyLevels []int
	RequestsPerTest   int

	// Cache configuration
	SimilarityThreshold float64
	UseHNSW             bool
	HNSWM               int
	HNSWEfConstruction  int
	HNSWEfSearch        int
	EmbeddingModel      string
	HitRatio            float64

	// Test modes
	RunQuick              bool
	RunFull               bool
	RunScalability        bool
	RunConcurrency        bool
	RunModelComparison    bool
	RunComponentBreakdown bool

	// Output options
	OutputJSON bool
	OutputFile string

	// Profiling
	CPUProfile string
	MemProfile string
}

func main() {
	fmt.Print(logo)

	config := parseFlags()

	// Setup signal handling for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		fmt.Println("\n\nReceived interrupt signal, shutting down gracefully...")
		cancel()
	}()

	// Start CPU profiling if requested
	if config.CPUProfile != "" {
		f, err := os.Create(config.CPUProfile)
		if err != nil {
			fmt.Printf("Error creating CPU profile: %v\n", err)
			cancel()
			return
		}
		defer f.Close()

		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Printf("Error starting CPU profile: %v\n", err)
			cancel()
			return
		}
		defer pprof.StopCPUProfile()

		fmt.Printf("CPU profiling enabled, writing to: %s\n", config.CPUProfile)
	}

	// Initialize embedding models once at the start
	fmt.Println("🔧 Initializing embedding models...")
	if err := cache.InitEmbeddingModels(); err != nil {
		fmt.Printf("Error initializing models: %v\n", err)
		cancel()
		return
	}
	fmt.Println()

	// Run benchmarks
	results := runBenchmarks(ctx, config)

	// Write memory profile if requested
	if config.MemProfile != "" {
		f, err := os.Create(config.MemProfile)
		if err != nil {
			fmt.Printf("Error creating memory profile: %v\n", err)
			cancel()
			return
		}
		defer f.Close()

		runtime.GC() // Get up-to-date statistics
		if err := pprof.WriteHeapProfile(f); err != nil {
			fmt.Printf("Error writing memory profile: %v\n", err)
			cancel()
			return
		}

		fmt.Printf("\nMemory profile written to: %s\n", config.MemProfile)
	}

	// Output results
	if config.OutputJSON {
		outputJSON(results, config.OutputFile)
	} else if len(results) > 0 {
		// Print results in human-readable format
		fmt.Println("\n" + strings.Repeat("=", 80))
		fmt.Println("BENCHMARK RESULTS")
		fmt.Println(strings.Repeat("=", 80))
		cache.PrintBenchmarkResults(results)
	}

	fmt.Println("\n✓ Benchmark completed successfully!")
}

func parseFlags() Config {
	config := Config{}

	// Cache sizes (comma-separated)
	cacheSizesStr := flag.String("cache-sizes", "100,1000,10000", "Comma-separated list of cache sizes to test")
	concurrencyStr := flag.String("concurrency", "1,10,50,100", "Comma-separated list of concurrency levels to test")

	flag.IntVar(&config.RequestsPerTest, "requests", 1000, "Number of requests per test scenario")
	flag.Float64Var(&config.SimilarityThreshold, "threshold", 0.85, "Similarity threshold for cache hits")
	flag.BoolVar(&config.UseHNSW, "hnsw", true, "Use HNSW indexing for faster search")
	flag.IntVar(&config.HNSWM, "hnsw-m", 16, "HNSW M parameter (connections per node)")
	flag.IntVar(&config.HNSWEfConstruction, "hnsw-ef-construction", 200, "HNSW ef_construction parameter")
	flag.IntVar(&config.HNSWEfSearch, "hnsw-ef-search", 50, "HNSW ef_search parameter")
	flag.StringVar(&config.EmbeddingModel, "model", "qwen3", "Embedding model (qwen3, bert, gemma) - qwen3 has continuous batching")
	flag.Float64Var(&config.HitRatio, "hit-ratio", 0.3, "Expected cache hit ratio (0.0-1.0)")

	// Test modes
	flag.BoolVar(&config.RunQuick, "quick", false, "Run quick benchmark (small cache, low concurrency)")
	flag.BoolVar(&config.RunFull, "full", false, "Run full benchmark suite")
	flag.BoolVar(&config.RunScalability, "scalability", false, "Run scalability analysis")
	flag.BoolVar(&config.RunConcurrency, "concurrency-test", false, "Run concurrency impact analysis")
	flag.BoolVar(&config.RunModelComparison, "model-comparison", false, "Compare different embedding models")
	flag.BoolVar(&config.RunComponentBreakdown, "component-breakdown", false, "Show component latency breakdown")

	// Output options
	flag.BoolVar(&config.OutputJSON, "json", false, "Output results in JSON format")
	flag.StringVar(&config.OutputFile, "output", "", "Output file path (default: stdout)")

	// Profiling
	flag.StringVar(&config.CPUProfile, "cpuprofile", "", "Write CPU profile to file")
	flag.StringVar(&config.MemProfile, "memprofile", "", "Write memory profile to file")

	flag.Parse()

	// Parse cache sizes - simple implementation for common values
	switch *cacheSizesStr {
	case "100,1000,10000":
		config.CacheSizes = []int{100, 1000, 10000}
	case "100,1000":
		config.CacheSizes = []int{100, 1000}
	case "1000,10000":
		config.CacheSizes = []int{1000, 10000}
	default:
		// Default or single value
		config.CacheSizes = []int{1000}
	}

	// Parse concurrency levels
	switch *concurrencyStr {
	case "1,10,50,100":
		config.ConcurrencyLevels = []int{1, 10, 50, 100}
	case "1,10,50":
		config.ConcurrencyLevels = []int{1, 10, 50}
	case "10,50,100":
		config.ConcurrencyLevels = []int{10, 50, 100}
	default:
		config.ConcurrencyLevels = []int{50} // default
	}

	// Quick mode overrides
	if config.RunQuick {
		config.CacheSizes = []int{100}
		config.ConcurrencyLevels = []int{10}
		config.RequestsPerTest = 500
	}

	// If no specific test mode is selected, run a default benchmark
	if !config.RunQuick && !config.RunFull && !config.RunScalability &&
		!config.RunConcurrency && !config.RunModelComparison && !config.RunComponentBreakdown {
		config.RunFull = true
	}

	return config
}

func runBenchmarks(ctx context.Context, config Config) []cache.BenchmarkResult {
	var allResults []cache.BenchmarkResult

	fmt.Printf("\n📊 Benchmark Configuration:\n")
	fmt.Printf("   Cache Sizes: %v\n", config.CacheSizes)
	fmt.Printf("   Concurrency Levels: %v\n", config.ConcurrencyLevels)
	fmt.Printf("   Requests per Test: %d\n", config.RequestsPerTest)
	fmt.Printf("   Similarity Threshold: %.2f\n", config.SimilarityThreshold)
	fmt.Printf("   Use HNSW: %t\n", config.UseHNSW)
	fmt.Printf("   Embedding Model: %s\n", config.EmbeddingModel)
	fmt.Printf("   Expected Hit Ratio: %.2f%%\n\n", config.HitRatio*100)

	if config.RunQuick {
		fmt.Println("🚀 Running QUICK benchmark...")
		results := runQuickBenchmark(ctx, config)
		allResults = append(allResults, results...)
	}

	if config.RunFull {
		fmt.Println("🔬 Running FULL benchmark suite...")
		results := runFullBenchmark(ctx, config)
		allResults = append(allResults, results...)
	}

	if config.RunScalability {
		fmt.Println("📈 Running SCALABILITY analysis...")
		results := runScalabilityAnalysis(ctx, config)
		allResults = append(allResults, results...)
	}

	if config.RunConcurrency {
		fmt.Println("⚡ Running CONCURRENCY impact analysis...")
		results := runConcurrencyAnalysis(ctx, config)
		allResults = append(allResults, results...)
	}

	if config.RunComponentBreakdown {
		fmt.Println("🔍 Running COMPONENT latency breakdown...")
		results := runComponentBreakdown(ctx, config)
		allResults = append(allResults, results...)
	}

	if config.RunModelComparison {
		fmt.Println("🤖 Running MODEL comparison...")
		results := runModelComparison(ctx, config)
		allResults = append(allResults, results...)
	}

	return allResults
}

func runQuickBenchmark(ctx context.Context, config Config) []cache.BenchmarkResult {
	benchConfig := cache.BenchmarkConfig{
		CacheSize:         config.CacheSizes[0],
		ConcurrencyLevels: config.ConcurrencyLevels,
		RequestsPerLevel:  config.RequestsPerTest,
		SimilarityThresh:  float32(config.SimilarityThreshold),
		UseHNSW:           config.UseHNSW,
		EmbeddingModel:    config.EmbeddingModel,
		HitRatio:          config.HitRatio,
	}

	return cache.RunStandaloneBenchmark(ctx, benchConfig)
}

func runFullBenchmark(ctx context.Context, config Config) []cache.BenchmarkResult {
	var results []cache.BenchmarkResult

	for _, cacheSize := range config.CacheSizes {
		for _, useHNSW := range []bool{false, true} {
			benchConfig := cache.BenchmarkConfig{
				CacheSize:         cacheSize,
				ConcurrencyLevels: config.ConcurrencyLevels,
				RequestsPerLevel:  config.RequestsPerTest,
				SimilarityThresh:  float32(config.SimilarityThreshold),
				UseHNSW:           useHNSW,
				EmbeddingModel:    config.EmbeddingModel,
				HitRatio:          config.HitRatio,
			}

			benchResults := cache.RunStandaloneBenchmark(ctx, benchConfig)
			results = append(results, benchResults...)

			// Check for cancellation
			select {
			case <-ctx.Done():
				return results
			default:
			}
		}
	}

	return results
}

func runScalabilityAnalysis(ctx context.Context, config Config) []cache.BenchmarkResult {
	var results []cache.BenchmarkResult

	// Reduced cache sizes for faster testing (still shows scalability trends)
	cacheSizes := []int{100, 1000, 10000}
	concurrency := 50

	fmt.Println("\n=== Cache Scalability Analysis ===")
	fmt.Printf("Testing how performance scales with cache size (Concurrency: %d)\n\n", concurrency)

	// Check if USE_HNSW environment variable is set to force a specific mode
	hnswModes := []bool{false, true}
	hnswEnv := os.Getenv("USE_HNSW")
	switch hnswEnv {
	case "true", "1":
		hnswModes = []bool{true} // Only test HNSW
	case "false", "0":
		hnswModes = []bool{false} // Only test Linear
	}

	for _, useHNSW := range hnswModes {
		indexType := "Linear"
		if useHNSW {
			indexType = "HNSW"
		}

		fmt.Printf("\nIndex Type: %s\n", indexType)
		fmt.Printf("%-12s %-15s %-12s %-12s %-12s\n",
			"Cache Size", "Throughput", "P50 (ms)", "P95 (ms)", "P99 (ms)")
		fmt.Println("────────────────────────────────────────────────────────────────")

		for _, size := range cacheSizes {
			benchConfig := cache.BenchmarkConfig{
				CacheSize:         size,
				ConcurrencyLevels: []int{concurrency},
				RequestsPerLevel:  config.RequestsPerTest,
				SimilarityThresh:  float32(config.SimilarityThreshold),
				UseHNSW:           useHNSW,
				EmbeddingModel:    config.EmbeddingModel,
				HitRatio:          config.HitRatio,
			}

			benchResults := cache.RunStandaloneBenchmark(ctx, benchConfig)
			if len(benchResults) > 0 {
				r := benchResults[0]
				fmt.Printf("%-12d %-15.2f %-12.2f %-12.2f %-12.2f\n",
					size, r.Throughput, r.OverallP50, r.OverallP95, r.OverallP99)
				results = append(results, r)
			}

			// Check for cancellation
			select {
			case <-ctx.Done():
				return results
			default:
			}
		}
	}

	return results
}

func runConcurrencyAnalysis(ctx context.Context, config Config) []cache.BenchmarkResult {
	var results []cache.BenchmarkResult

	cacheSize := 1000
	concurrencyLevels := []int{1, 5, 10, 25, 50, 100, 200, 500}

	fmt.Println("\n=== Concurrency Impact Analysis ===")
	fmt.Printf("Testing how concurrency affects performance (Cache Size: %d)\n\n", cacheSize)

	// Check if USE_HNSW environment variable is set to force a specific mode
	hnswModes := []bool{false, true}
	hnswEnv := os.Getenv("USE_HNSW")
	switch hnswEnv {
	case "true", "1":
		hnswModes = []bool{true} // Only test HNSW
	case "false", "0":
		hnswModes = []bool{false} // Only test Linear
	}

	for _, useHNSW := range hnswModes {
		indexType := "Linear"
		if useHNSW {
			indexType = "HNSW"
		}

		fmt.Printf("\nIndex Type: %s\n", indexType)
		fmt.Printf("%-15s %-15s %-12s %-12s %-12s\n",
			"Concurrency", "Throughput", "P50 (ms)", "P95 (ms)", "P99 (ms)")
		fmt.Println("────────────────────────────────────────────────────────────────────")

		for _, concurrency := range concurrencyLevels {
			benchConfig := cache.BenchmarkConfig{
				CacheSize:         cacheSize,
				ConcurrencyLevels: []int{concurrency},
				RequestsPerLevel:  config.RequestsPerTest,
				SimilarityThresh:  float32(config.SimilarityThreshold),
				UseHNSW:           useHNSW,
				EmbeddingModel:    config.EmbeddingModel,
				HitRatio:          config.HitRatio,
			}

			benchResults := cache.RunStandaloneBenchmark(ctx, benchConfig)
			if len(benchResults) > 0 {
				r := benchResults[0]
				fmt.Printf("%-15d %-15.2f %-12.2f %-12.2f %-12.2f\n",
					concurrency, r.Throughput, r.OverallP50, r.OverallP95, r.OverallP99)
				results = append(results, r)
			}

			// Check for cancellation
			select {
			case <-ctx.Done():
				return results
			default:
			}
		}
	}

	return results
}

func runComponentBreakdown(ctx context.Context, config Config) []cache.BenchmarkResult {
	benchConfig := cache.BenchmarkConfig{
		CacheSize:         1000,
		ConcurrencyLevels: []int{50},
		RequestsPerLevel:  config.RequestsPerTest,
		SimilarityThresh:  float32(config.SimilarityThreshold),
		UseHNSW:           config.UseHNSW,
		EmbeddingModel:    config.EmbeddingModel,
		HitRatio:          config.HitRatio,
	}

	results := cache.RunStandaloneBenchmark(ctx, benchConfig)

	if len(results) > 0 {
		r := results[0]

		fmt.Print("\n=== Component Latency Breakdown ===\n")

		embeddingPct := (r.EmbeddingP50 / r.OverallP50) * 100
		searchPct := (r.SearchP50 / r.OverallP50) * 100

		fmt.Printf("P50 Latency Breakdown:\n")
		fmt.Printf("  Total:              %.2f ms\n", r.OverallP50)
		fmt.Printf("  ├─ Embedding:       %.2f ms (%.1f%%)\n", r.EmbeddingP50, embeddingPct)
		fmt.Printf("  └─ Search:          %.2f ms (%.1f%%)\n\n", r.SearchP50, searchPct)

		embeddingPct95 := (r.EmbeddingP95 / r.OverallP95) * 100
		searchPct95 := (r.SearchP95 / r.OverallP95) * 100

		fmt.Printf("P95 Latency Breakdown:\n")
		fmt.Printf("  Total:              %.2f ms\n", r.OverallP95)
		fmt.Printf("  ├─ Embedding:       %.2f ms (%.1f%%)\n", r.EmbeddingP95, embeddingPct95)
		fmt.Printf("  └─ Search:          %.2f ms (%.1f%%)\n\n", r.SearchP95, searchPct95)
	}

	return results
}

func runModelComparison(ctx context.Context, config Config) []cache.BenchmarkResult {
	var results []cache.BenchmarkResult

	models := []string{"bert"} // Add "qwen3", "gemma" if available
	cacheSize := 1000
	concurrency := 50

	fmt.Println("\n=== Embedding Model Comparison ===")
	fmt.Printf("Cache Size: %d, Concurrency: %d\n\n", cacheSize, concurrency)

	fmt.Printf("%-12s %-15s %-12s %-12s %-12s\n",
		"Model", "Throughput", "P50 (ms)", "P95 (ms)", "P99 (ms)")
	fmt.Println("────────────────────────────────────────────────────────────────")

	for _, model := range models {
		benchConfig := cache.BenchmarkConfig{
			CacheSize:         cacheSize,
			ConcurrencyLevels: []int{concurrency},
			RequestsPerLevel:  config.RequestsPerTest,
			SimilarityThresh:  float32(config.SimilarityThreshold),
			UseHNSW:           config.UseHNSW,
			EmbeddingModel:    model,
			HitRatio:          config.HitRatio,
		}

		benchResults := cache.RunStandaloneBenchmark(ctx, benchConfig)
		if len(benchResults) > 0 {
			r := benchResults[0]
			fmt.Printf("%-12s %-15.2f %-12.2f %-12.2f %-12.2f\n",
				model, r.Throughput, r.OverallP50, r.OverallP95, r.OverallP99)
			results = append(results, r)
		}

		// Check for cancellation
		select {
		case <-ctx.Done():
			return results
		default:
		}
	}

	return results
}

func outputJSON(results []cache.BenchmarkResult, outputFile string) {
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		fmt.Printf("Error marshaling results to JSON: %v\n", err)
		return
	}

	if outputFile == "" {
		fmt.Println("\n=== JSON Output ===")
		fmt.Println(string(data))
	} else {
		if err := os.WriteFile(outputFile, data, 0o644); err != nil {
			fmt.Printf("Error writing results to file: %v\n", err)
			return
		}
		fmt.Printf("\n✓ Results written to: %s\n", outputFile)
	}
}
