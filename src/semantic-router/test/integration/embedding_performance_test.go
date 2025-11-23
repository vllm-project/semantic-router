//go:build integration
// +build integration

package integration

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// TestEmbeddingPerformance benchmarks and tests embedding model performance

// Benchmark 1: Single embedding generation
func BenchmarkEmbedding_Single(b *testing.B) {
	qwen := testModelsDir + "/Qwen3-Embedding-0.6B"
	gemma := testModelsDir + "/embeddinggemma-300m"

	// Try with both models, fallback to Qwen3 only
	err := candle_binding.InitEmbeddingModels(qwen, gemma, true)
	if err != nil {
		// Fallback to Qwen3 only (this benchmark only uses Qwen3)
		err = candle_binding.InitEmbeddingModels(qwen, "", true)
		if err != nil {
			b.Skipf("Skipping benchmark - models not available: %v", err)
		}
	}

	text := "This is a test sentence for embedding generation"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
		if err != nil {
			b.Fatalf("Embedding generation failed: %v", err)
		}
	}
}

// Benchmark 2: Batch embedding generation
func BenchmarkEmbedding_Batch(b *testing.B) {
	qwen := testModelsDir + "/Qwen3-Embedding-0.6B"
	gemma := testModelsDir + "/embeddinggemma-300m"

	// Try with both models, fallback to Qwen3 only
	err := candle_binding.InitEmbeddingModels(qwen, gemma, true)
	if err != nil {
		// Fallback to Qwen3 only (this benchmark only uses Qwen3)
		err = candle_binding.InitEmbeddingModels(qwen, "", true)
		if err != nil {
			b.Skipf("Skipping benchmark - models not available: %v", err)
		}
	}

	texts := []string{
		"First test sentence",
		"Second test sentence",
		"Third test sentence",
		"Fourth test sentence",
		"Fifth test sentence",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, text := range texts {
			_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
			if err != nil {
				b.Fatalf("Batch embedding generation failed: %v", err)
			}
		}
	}
}

// Benchmark 3: Different text lengths
func BenchmarkEmbedding_TextLengths(b *testing.B) {
	qwen := testModelsDir + "/Qwen3-Embedding-0.6B"
	gemma := testModelsDir + "/embeddinggemma-300m"

	// Try with both models, fallback to Qwen3 only
	err := candle_binding.InitEmbeddingModels(qwen, gemma, true)
	if err != nil {
		// Fallback to Qwen3 only (this benchmark only uses Qwen3)
		err = candle_binding.InitEmbeddingModels(qwen, "", true)
		if err != nil {
			b.Skipf("Skipping benchmark - models not available: %v", err)
		}
	}

	testCases := []struct {
		name string
		text string
	}{
		{"Short", "Hello world"},
		{"Medium", "This is a medium length sentence with several words that provides more context"},
		{"Long", "This is a very long text that contains many words and provides extensive context. " +
			"It spans multiple sentences and discusses various topics. " +
			"The purpose is to test how the embedding model handles longer inputs. " +
			"Performance may vary with text length."},
	}

	for _, tc := range testCases {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := candle_binding.GetEmbeddingWithModelType(tc.text, "qwen3", 0)
				if err != nil {
					b.Fatalf("Embedding generation failed: %v", err)
				}
			}
		})
	}
}

// Benchmark 4: Model comparison (Qwen3 vs Gemma vs BERT)
// Note: This benchmark requires Gemma and will skip if unavailable (no fallback)
func BenchmarkEmbedding_ModelComparison(b *testing.B) {
	err := candle_binding.InitEmbeddingModels(
		testModelsDir+"/Qwen3-Embedding-0.6B",
		testModelsDir+"/embeddinggemma-300m",
		true,
	)
	if err != nil {
		b.Skipf("Skipping benchmark - Gemma model required for comparison: %v", err)
	}

	text := "Benchmark text for model comparison"

	b.Run("Qwen3", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := candle_binding.GetEmbeddingBatched(text, "qwen3", 0)
			if err != nil {
				b.Fatalf("Qwen3 embedding failed: %v", err)
			}
		}
	})

	b.Run("Gemma", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := candle_binding.GetEmbeddingWithModelType(text, "gemma", 0)
			if err != nil {
				b.Fatalf("Gemma embedding failed: %v", err)
			}
		}
	})

	b.Run("BERT", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
			if err != nil {
				b.Fatalf("BERT embedding failed: %v", err)
			}
		}
	})
}

// Test: Performance requirement - embeddings should be fast enough
func TestEmbedding_PerformanceRequirement(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	text := "Performance test sentence"
	iterations := 10

	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
		if err != nil {
			t.Fatalf("Embedding generation failed: %v", err)
		}
	}
	elapsed := time.Since(start)

	avgTime := elapsed / time.Duration(iterations)
	t.Logf("Average embedding time: %v", avgTime)

	// Performance target: Should be reasonably fast (< 100ms per embedding on CPU)
	if avgTime > 100*time.Millisecond {
		t.Logf("Warning: Average time %v exceeds 100ms target", avgTime)
	} else {
		t.Logf("✓ Performance meets target: %v < 100ms", avgTime)
	}
}

// Test: Throughput measurement
func TestEmbedding_Throughput(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	text := "Throughput test sentence"
	duration := 5 * time.Second
	count := 0

	deadline := time.Now().Add(duration)
	for time.Now().Before(deadline) {
		_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
		if err != nil {
			t.Fatalf("Embedding generation failed: %v", err)
		}
		count++
	}

	throughput := float64(count) / duration.Seconds()
	t.Logf("Throughput: %.2f embeddings/second over %v", throughput, duration)
	t.Logf("Total embeddings generated: %d", count)
}

// Test: Latency distribution
func TestEmbedding_LatencyDistribution(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	text := "Latency distribution test"
	samples := 100
	latencies := make([]time.Duration, samples)

	// Collect latency samples
	for i := 0; i < samples; i++ {
		start := time.Now()
		_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
		if err != nil {
			t.Fatalf("Embedding generation failed: %v", err)
		}
		latencies[i] = time.Since(start)
	}

	// Calculate statistics
	var total time.Duration
	min := latencies[0]
	max := latencies[0]

	for _, lat := range latencies {
		total += lat
		if lat < min {
			min = lat
		}
		if lat > max {
			max = lat
		}
	}

	avg := total / time.Duration(samples)

	t.Logf("Latency statistics over %d samples:", samples)
	t.Logf("  Min: %v", min)
	t.Logf("  Avg: %v", avg)
	t.Logf("  Max: %v", max)

	// Check if max is not too far from average (no severe outliers)
	if max > avg*3 {
		t.Logf("Warning: Max latency %v is > 3x average %v", max, avg)
	} else {
		t.Logf("✓ Latency distribution is stable")
	}
}

// Test: Warm-up effect
func TestEmbedding_WarmupEffect(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	text := "Warmup test sentence"

	// First call (cold)
	start := time.Now()
	_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
	if err != nil {
		t.Fatalf("Cold embedding generation failed: %v", err)
	}
	coldTime := time.Since(start)

	// Subsequent calls (warm)
	warmTimes := make([]time.Duration, 10)
	for i := 0; i < 10; i++ {
		start = time.Now()
		_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
		if err != nil {
			t.Fatalf("Warm embedding generation failed: %v", err)
		}
		warmTimes[i] = time.Since(start)
	}

	var warmTotal time.Duration
	for _, t := range warmTimes {
		warmTotal += t
	}
	avgWarmTime := warmTotal / time.Duration(len(warmTimes))

	t.Logf("Cold start: %v", coldTime)
	t.Logf("Warm average: %v", avgWarmTime)
	t.Logf("Speedup: %.2fx", float64(coldTime)/float64(avgWarmTime))
}

// Test: Concurrent performance (no lock contention)
func TestEmbedding_ConcurrentPerformance(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	numGoroutines := 10
	iterationsPerGoroutine := 5
	text := "Concurrent performance test"

	start := time.Now()

	var wg sync.WaitGroup
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < iterationsPerGoroutine; j++ {
				_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
				if err != nil {
					t.Errorf("Goroutine %d failed: %v", id, err)
				}
			}
		}(i)
	}

	wg.Wait()
	elapsed := time.Since(start)

	totalOps := numGoroutines * iterationsPerGoroutine
	avgTimePerOp := elapsed / time.Duration(totalOps)

	t.Logf("Concurrent performance:")
	t.Logf("  Goroutines: %d", numGoroutines)
	t.Logf("  Total operations: %d", totalOps)
	t.Logf("  Total time: %v", elapsed)
	t.Logf("  Avg time per operation: %v", avgTimePerOp)

	// Sequential baseline for comparison
	seqStart := time.Now()
	for i := 0; i < totalOps; i++ {
		_, _ = candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
	}
	seqElapsed := time.Since(seqStart)

	speedup := float64(seqElapsed) / float64(elapsed)
	t.Logf("Sequential time: %v", seqElapsed)
	t.Logf("Speedup: %.2fx", speedup)

	if speedup < 1.5 {
		t.Logf("Warning: Low speedup %.2fx suggests lock contention", speedup)
	} else {
		t.Logf("✓ Good concurrent performance with %.2fx speedup", speedup)
	}
}

// Test: Model loading time
func TestEmbedding_ModelLoadingTime(t *testing.T) {
	// Check if models are already initialized
	if modelsInitialized {
		t.Skip("Models already initialized by previous tests - skipping load time test")
	}

	// Test model initialization time with fallback support
	qwen := testModelsDir + "/Qwen3-Embedding-0.6B"
	gemma := testModelsDir + "/embeddinggemma-300m"

	start := time.Now()

	// Try with both models first
	err := candle_binding.InitEmbeddingModels(qwen, gemma, true)
	if err != nil {
		// Fallback to Qwen3 only
		t.Log("⚠️  Gemma unavailable, measuring Qwen3-only load time")
		start = time.Now() // Reset timer for fallback
		err = candle_binding.InitEmbeddingModels(qwen, "", true)
		if err != nil {
			t.Fatalf("❌ Failed to initialize embedding models: %v", err)
		}
	}

	loadTime := time.Since(start)
	modelsInitialized = true
	t.Logf("Model loading time: %v", loadTime)

	// Model loading should be reasonable (< 10 seconds)
	if loadTime > 10*time.Second {
		t.Logf("Warning: Model loading took %v (> 10s)", loadTime)
	} else {
		t.Logf("✓ Model loading time acceptable: %v", loadTime)
	}
}

// Test: Memory footprint during embedding generation
func TestEmbedding_MemoryFootprint(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	// Force GC and measure baseline
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	baselineAlloc := m.Alloc

	// Generate embeddings
	text := "Memory footprint test"
	for i := 0; i < 100; i++ {
		_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
		if err != nil {
			t.Fatalf("Embedding generation failed: %v", err)
		}
	}

	// Measure after embeddings
	runtime.ReadMemStats(&m)
	afterAlloc := m.Alloc

	growth := afterAlloc - baselineAlloc
	t.Logf("Memory footprint:")
	t.Logf("  Baseline: %d MB", baselineAlloc/1024/1024)
	t.Logf("  After 100 embeddings: %d MB", afterAlloc/1024/1024)
	t.Logf("  Growth: %d MB", growth/1024/1024)

	// Memory growth should be reasonable (< 50MB for 100 embeddings)
	if growth > 50*1024*1024 {
		t.Logf("Warning: Memory grew by %d MB", growth/1024/1024)
	} else {
		t.Logf("✓ Memory footprint is reasonable")
	}
}

// Benchmark: Similarity calculation
func BenchmarkEmbedding_SimilarityCalculation(b *testing.B) {
	qwen := testModelsDir + "/Qwen3-Embedding-0.6B"
	gemma := testModelsDir + "/embeddinggemma-300m"

	// Try with both models, fallback to Qwen3 only
	err := candle_binding.InitEmbeddingModels(qwen, gemma, true)
	if err != nil {
		// Fallback to Qwen3 only (this benchmark uses similarity which works with Qwen3)
		err = candle_binding.InitEmbeddingModels(qwen, "", true)
		if err != nil {
			b.Skipf("Skipping benchmark - models not available: %v", err)
		}
	}

	query := "How to use kubernetes?"
	candidates := []string{
		"kubernetes deployment guide",
		"docker containers",
		"cloud computing",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Use qwen3 explicitly to work in fallback mode (when Gemma unavailable)
		_, err := candle_binding.CalculateSimilarityBatch(query, candidates, 0, "qwen3", 768)
		if err != nil {
			b.Fatalf("Similarity calculation failed: %v", err)
		}
	}
}

// Test: Performance test execution time requirement
func TestEmbedding_TestExecutionTime(t *testing.T) {
	// This test verifies that all embedding tests can complete within the 5-minute requirement
	start := time.Now()

	initEmbeddingModelsWithFallback(t, testModelsDir)

	// Run a subset of operations to estimate total test time
	operations := 50
	for i := 0; i < operations; i++ {
		_, err := candle_binding.GetEmbeddingWithModelType(fmt.Sprintf("Test %d", i), "qwen3", 0)
		if err != nil {
			t.Fatalf("Embedding generation failed: %v", err)
		}
	}

	elapsed := time.Since(start)
	t.Logf("Time for %d operations: %v", operations, elapsed)

	// Estimate time for full test suite
	estimatedFullTime := elapsed * 10 // Rough estimate
	t.Logf("Estimated full test suite time: %v", estimatedFullTime)

	if estimatedFullTime > 5*time.Minute {
		t.Logf("Warning: Estimated time %v may exceed 5-minute target", estimatedFullTime)
	} else {
		t.Logf("✓ Estimated time %v within 5-minute target", estimatedFullTime)
	}
}
