package cache

import (
	"fmt"
	"os"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// ContentLength defines different query content sizes
type ContentLength int

const (
	ShortContent  ContentLength = 20  // ~20 words
	MediumContent ContentLength = 50  // ~50 words
	LongContent   ContentLength = 100 // ~100 words
)

func (c ContentLength) String() string {
	switch c {
	case ShortContent:
		return "short"
	case MediumContent:
		return "medium"
	case LongContent:
		return "long"
	default:
		return "unknown"
	}
}

// GenerateQuery generates a query of specified length
func generateQuery(length ContentLength, index int) string {
	words := []string{
		"machine", "learning", "artificial", "intelligence", "neural", "network",
		"deep", "training", "model", "algorithm", "data", "science", "prediction",
		"classification", "regression", "supervised", "unsupervised", "reinforcement",
		"optimization", "gradient", "descent", "backpropagation", "activation",
		"function", "layer", "convolutional", "recurrent", "transformer", "attention",
		"embedding", "vector", "semantic", "similarity", "clustering", "feature",
	}
	
	query := fmt.Sprintf("Query %d: ", index)
	for i := 0; i < int(length); i++ {
		query += words[i%len(words)] + " "
	}
	return query
}

// BenchmarkComprehensive runs comprehensive benchmarks across multiple dimensions
func BenchmarkComprehensive(b *testing.B) {
	// Initialize BERT model
	useCPU := os.Getenv("USE_CPU") != "false" // Default to CPU
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if err := candle_binding.InitModel(modelName, useCPU); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	// Determine hardware type
	hardware := "cpu"
	if !useCPU {
		hardware = "gpu"
	}

	// Test configurations
	cacheSizes := []int{100, 500, 1000, 5000}
	contentLengths := []ContentLength{ShortContent, MediumContent, LongContent}
	hnswConfigs := []struct {
		name string
		m    int
		ef   int
	}{
		{"default", 16, 200},
		{"fast", 8, 100},
		{"accurate", 32, 400},
	}

	// Open CSV file for results
	csvFile, err := os.OpenFile("../../benchmark_results/benchmark_data.csv", 
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		b.Logf("Warning: Could not open CSV file: %v", err)
	} else {
		defer csvFile.Close()
	}

	// Run benchmarks
	for _, cacheSize := range cacheSizes {
		for _, contentLen := range contentLengths {
			// Generate test data
			testQueries := make([]string, cacheSize)
			for i := 0; i < cacheSize; i++ {
				testQueries[i] = generateQuery(contentLen, i)
			}

			// Benchmark Linear Search
			b.Run(fmt.Sprintf("%s/Linear/%s/%dEntries", hardware, contentLen.String(), cacheSize), func(b *testing.B) {
				cache := NewInMemoryCache(InMemoryCacheOptions{
					Enabled:             true,
					MaxEntries:          cacheSize * 2,
					SimilarityThreshold: 0.85,
					TTLSeconds:          0,
					UseHNSW:             false,
				})

				// Populate cache
				for i, query := range testQueries {
					reqID := fmt.Sprintf("req%d", i)
					_ = cache.AddEntry(reqID, "test-model", query, []byte(query), []byte("response"))
				}

				searchQuery := generateQuery(contentLen, cacheSize/2)
				b.ResetTimer()
				
				for i := 0; i < b.N; i++ {
					_, _, _ = cache.FindSimilar("test-model", searchQuery)
				}
				
				b.StopTimer()
				
				// Write to CSV
				if csvFile != nil {
					nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
					
					line := fmt.Sprintf("%s,%s,%d,linear,0,0,%.0f,0,0,%d,1.0\n",
						hardware, contentLen.String(), cacheSize, nsPerOp, b.N)
					csvFile.WriteString(line)
				}
			})

			// Benchmark HNSW with different configurations
			for _, hnswCfg := range hnswConfigs {
				b.Run(fmt.Sprintf("%s/HNSW_%s/%s/%dEntries", hardware, hnswCfg.name, contentLen.String(), cacheSize), func(b *testing.B) {
					cache := NewInMemoryCache(InMemoryCacheOptions{
						Enabled:             true,
						MaxEntries:          cacheSize * 2,
						SimilarityThreshold: 0.85,
						TTLSeconds:          0,
						UseHNSW:             true,
						HNSWM:               hnswCfg.m,
						HNSWEfConstruction:  hnswCfg.ef,
					})

					// Populate cache
					for i, query := range testQueries {
						reqID := fmt.Sprintf("req%d", i)
						_ = cache.AddEntry(reqID, "test-model", query, []byte(query), []byte("response"))
					}

					searchQuery := generateQuery(contentLen, cacheSize/2)
					b.ResetTimer()
					
					for i := 0; i < b.N; i++ {
						_, _, _ = cache.FindSimilar("test-model", searchQuery)
					}
					
					b.StopTimer()
					
					// Write to CSV
					if csvFile != nil {
						nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
						
						line := fmt.Sprintf("%s,%s,%d,hnsw_%s,%d,%d,%.0f,0,0,%d,0.0\n",
							hardware, contentLen.String(), cacheSize, hnswCfg.name, 
							hnswCfg.m, hnswCfg.ef, nsPerOp, b.N)
						csvFile.WriteString(line)
					}
				})
			}
		}
	}
}

// BenchmarkIndexConstruction benchmarks HNSW index build time
func BenchmarkIndexConstruction(b *testing.B) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	cacheSizes := []int{100, 500, 1000, 5000}
	contentLengths := []ContentLength{ShortContent, MediumContent, LongContent}

	for _, cacheSize := range cacheSizes {
		for _, contentLen := range contentLengths {
			testQueries := make([]string, cacheSize)
			for i := 0; i < cacheSize; i++ {
				testQueries[i] = generateQuery(contentLen, i)
			}

			b.Run(fmt.Sprintf("BuildIndex/%s/%dEntries", contentLen.String(), cacheSize), func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					b.StopTimer()
					cache := NewInMemoryCache(InMemoryCacheOptions{
						Enabled:             true,
						MaxEntries:          cacheSize * 2,
						SimilarityThreshold: 0.85,
						TTLSeconds:          0,
						UseHNSW:             true,
						HNSWM:               16,
						HNSWEfConstruction:  200,
					})
					b.StartTimer()

					// Build index by adding entries
					for j, query := range testQueries {
						reqID := fmt.Sprintf("req%d", j)
						_ = cache.AddEntry(reqID, "test-model", query, []byte(query), []byte("response"))
					}
				}
			})
		}
	}
}

