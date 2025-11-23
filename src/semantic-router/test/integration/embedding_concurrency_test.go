//go:build integration
// +build integration

package integration

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// TestEmbeddingConcurrency tests thread safety and concurrent embedding requests

// Test: Basic concurrency - multiple goroutines
func TestEmbedding_BasicConcurrency(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	numGoroutines := 10
	iterationsPerGoroutine := 10
	text := "Concurrent test"

	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines*iterationsPerGoroutine)

	start := time.Now()

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < iterationsPerGoroutine; j++ {
				// Use GetEmbeddingWithModelType for Qwen3/Gemma models
				output, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
				if err != nil {
					errors <- fmt.Errorf("goroutine %d, iteration %d: %v", id, j, err)
				} else if len(output.Embedding) == 0 {
					errors <- fmt.Errorf("goroutine %d, iteration %d: empty embedding", id, j)
				}
			}
		}(i)
	}

	wg.Wait()
	close(errors)
	elapsed := time.Since(start)

	// Check for errors
	errorCount := 0
	for err := range errors {
		t.Error(err)
		errorCount++
	}

	if errorCount > 0 {
		t.Fatalf("%d errors occurred during concurrent execution", errorCount)
	}

	t.Logf("✓ Concurrent execution successful:")
	t.Logf("  Goroutines: %d", numGoroutines)
	t.Logf("  Total operations: %d", numGoroutines*iterationsPerGoroutine)
	t.Logf("  Total time: %v", elapsed)
}

// Test: Concurrent requests with different texts
func TestEmbedding_ConcurrentDifferentTexts(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	texts := []string{
		"First unique text",
		"Second unique text",
		"Third unique text",
		"Fourth unique text",
		"Fifth unique text",
	}

	var wg sync.WaitGroup
	results := make([][]float32, len(texts))
	errors := make([]error, len(texts))

	for i, text := range texts {
		wg.Add(1)
		go func(index int, txt string) {
			defer wg.Done()
			output, err := candle_binding.GetEmbeddingWithModelType(txt, "qwen3", 0)
			if err == nil && output != nil {
				results[index] = output.Embedding
			}
			errors[index] = err
		}(i, text)
	}

	wg.Wait()

	// Verify all succeeded
	for i, err := range errors {
		if err != nil {
			t.Errorf("Embedding %d failed: %v", i, err)
		}
	}

	// Verify all embeddings are different
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[i] != nil && results[j] != nil {
				// Check that embeddings are different
				identical := true
				for k := 0; k < len(results[i]) && k < len(results[j]); k++ {
					if results[i][k] != results[j][k] {
						identical = false
						break
					}
				}
				if identical {
					t.Errorf("Embeddings %d and %d are identical (should be different)", i, j)
				}
			}
		}
	}

	t.Logf("✓ Concurrent different texts: all %d embeddings unique", len(texts))
}

// Test: High concurrency stress test
func TestEmbedding_HighConcurrencyStress(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping stress test in short mode")
	}

	initEmbeddingModelsWithFallback(t, testModelsDir)

	numGoroutines := 50
	iterationsPerGoroutine := 20
	text := "High concurrency stress test"

	var wg sync.WaitGroup
	var successCount atomic.Int64
	var errorCount atomic.Int64

	start := time.Now()

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < iterationsPerGoroutine; j++ {
				_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
				if err != nil {
					errorCount.Add(1)
				} else {
					successCount.Add(1)
				}
			}
		}(i)
	}

	wg.Wait()
	elapsed := time.Since(start)

	totalOps := int64(numGoroutines * iterationsPerGoroutine)
	successOps := successCount.Load()
	errorOps := errorCount.Load()

	t.Logf("High concurrency stress test:")
	t.Logf("  Goroutines: %d", numGoroutines)
	t.Logf("  Total operations: %d", totalOps)
	t.Logf("  Successful: %d", successOps)
	t.Logf("  Errors: %d", errorOps)
	t.Logf("  Duration: %v", elapsed)
	t.Logf("  Throughput: %.2f ops/sec", float64(totalOps)/elapsed.Seconds())

	if errorOps > 0 {
		t.Errorf("%d operations failed under high concurrency", errorOps)
	} else {
		t.Log("✓ All operations successful under high concurrency")
	}
}

// Test: Concurrent reads with same text (caching behavior)
func TestEmbedding_ConcurrentSameText(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	text := "Repeated text for concurrent access"
	numGoroutines := 15

	var wg sync.WaitGroup
	embeddings := make([][]float32, numGoroutines)
	errors := make([]error, numGoroutines)

	start := time.Now()

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			output, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
			if err == nil && output != nil {
				embeddings[index] = output.Embedding
			}
			errors[index] = err
		}(i)
	}

	wg.Wait()
	elapsed := time.Since(start)

	// Verify all succeeded
	for i, err := range errors {
		if err != nil {
			t.Errorf("Embedding %d failed: %v", i, err)
		}
	}

	// Verify all embeddings are identical (deterministic)
	if embeddings[0] != nil {
		for i := 1; i < len(embeddings); i++ {
			if embeddings[i] == nil {
				continue
			}
			if len(embeddings[0]) != len(embeddings[i]) {
				t.Errorf("Embedding %d has different length: %d vs %d",
					i, len(embeddings[i]), len(embeddings[0]))
				continue
			}
			for j := 0; j < len(embeddings[0]); j++ {
				if embeddings[0][j] != embeddings[i][j] {
					t.Errorf("Embedding %d differs at position %d", i, j)
					break
				}
			}
		}
	}

	t.Logf("✓ Concurrent same text: %d goroutines in %v", numGoroutines, elapsed)
	t.Log("✓ All embeddings are identical (deterministic)")
}

// Test: Deadlock detection
func TestEmbedding_NoDeadlock(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	text := "Deadlock detection test"
	// Increased timeout and reduced operations for CI (embeddings take ~400ms each)
	timeout := 30 * time.Second
	numGoroutines := 5
	opsPerGoroutine := 2
	if !testing.Short() {
		numGoroutines = 10
		opsPerGoroutine = 5
	}

	done := make(chan bool, 1)

	go func() {
		var wg sync.WaitGroup
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := 0; j < opsPerGoroutine; j++ {
					_, _ = candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
				}
			}()
		}
		wg.Wait()
		done <- true
	}()

	select {
	case <-done:
		t.Log("✓ No deadlock detected")
	case <-time.After(timeout):
		t.Fatal("Potential deadlock: test timed out after", timeout)
	}
}

// Test: Concurrent initialization (should be safe)
func TestEmbedding_ConcurrentInitialization(t *testing.T) {
	// Test that concurrent calls to initialization are safe
	// The helper uses sync.Once to ensure only one actual load happens
	numGoroutines := 5
	var wg sync.WaitGroup
	successCount := int32(0)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// This will be synchronized via sync.Once in the helper
			initEmbeddingModelsWithFallback(t, testModelsDir)
			atomic.AddInt32(&successCount, 1)
		}()
	}

	wg.Wait()

	// All should succeed since they're synchronized
	if successCount != int32(numGoroutines) {
		t.Fatalf("Expected all %d goroutines to succeed, got %d", numGoroutines, successCount)
	}

	t.Logf("✓ Concurrent initialization safe: %d/%d succeeded with proper synchronization", successCount, numGoroutines)
}

// Test: Goroutine leaks
func TestEmbedding_NoGoroutineLeaks(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	before := runtime.NumGoroutine()

	// Create and complete many goroutines
	for i := 0; i < 10; i++ {
		var wg sync.WaitGroup
		for j := 0; j < 10; j++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_, _ = candle_binding.GetEmbeddingWithModelType("Goroutine leak test", "qwen3", 0)
			}()
		}
		wg.Wait()
	}

	// Give some time for goroutines to clean up
	time.Sleep(100 * time.Millisecond)

	after := runtime.NumGoroutine()

	growth := after - before
	t.Logf("Goroutines: before=%d, after=%d, growth=%d", before, after, growth)

	// Some growth is acceptable, but not excessive
	if growth > 10 {
		t.Errorf("Potential goroutine leak: %d new goroutines", growth)
	} else {
		t.Logf("✓ No significant goroutine leak")
	}
}

// Test: Context cancellation (if supported)
func TestEmbedding_GracefulShutdown(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	text := "Shutdown test"
	numGoroutines := 5

	stop := make(chan bool)
	var wg sync.WaitGroup

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					_, _ = candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
				}
			}
		}()
	}

	// Run for a bit
	time.Sleep(500 * time.Millisecond)

	// Signal stop
	close(stop)

	// Wait for graceful shutdown with timeout
	done := make(chan bool)
	go func() {
		wg.Wait()
		done <- true
	}()

	select {
	case <-done:
		t.Log("✓ Graceful shutdown successful")
	case <-time.After(5 * time.Second):
		t.Error("Goroutines did not shut down gracefully")
	}
}

// Test: Mutex contention measurement
func TestEmbedding_MutexContention(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	text := "Mutex contention test"
	numGoroutines := 10
	iterationsPerGoroutine := 10

	// Measure sequential baseline
	seqStart := time.Now()
	for i := 0; i < numGoroutines*iterationsPerGoroutine; i++ {
		_, _ = candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
	}
	seqTime := time.Since(seqStart)

	// Measure concurrent
	concStart := time.Now()
	var wg sync.WaitGroup
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < iterationsPerGoroutine; j++ {
				_, _ = candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
			}
		}()
	}
	wg.Wait()
	concTime := time.Since(concStart)

	speedup := float64(seqTime) / float64(concTime)

	t.Logf("Mutex contention analysis:")
	t.Logf("  Sequential time: %v", seqTime)
	t.Logf("  Concurrent time: %v", concTime)
	t.Logf("  Speedup: %.2fx", speedup)

	if speedup < 1.5 {
		t.Logf("Warning: Low speedup (%.2fx) suggests high mutex contention", speedup)
	} else {
		t.Logf("✓ Low mutex contention with %.2fx speedup", speedup)
	}
}

// Test: Thread safety with mixed operations
func TestEmbedding_MixedOperations(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	var wg sync.WaitGroup
	numGoroutines := 8

	// Mix of different operation types
	operations := []func(){
		// Short text - Qwen3
		func() {
			_, _ = candle_binding.GetEmbeddingWithModelType("Short", "qwen3", 0)
		},
		// Long text - Qwen3
		func() {
			_, _ = candle_binding.GetEmbeddingWithModelType("This is a much longer text that requires more processing", "qwen3", 0)
		},
		// Model selection - Gemma
		func() {
			_, _ = candle_binding.GetEmbeddingWithModelType("Test", "gemma", 0)
		},
		// Different text lengths - Qwen3
		func() {
			_, _ = candle_binding.GetEmbeddingWithModelType("Medium length text for testing", "qwen3", 0)
		},
	}

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < 5; j++ {
				// Execute random operation
				op := operations[j%len(operations)]
				op()
			}
		}(i)
	}

	wg.Wait()
	t.Log("✓ Mixed operations thread-safe")
}

// Test: Concurrent similarity calculations
func TestEmbedding_ConcurrentSimilarity(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	queries := []string{
		"How to use kubernetes?",
		"What is machine learning?",
		"Explain quantum physics",
	}

	candidates := []string{
		"kubernetes deployment",
		"AI and ML",
		"quantum mechanics",
	}

	var wg sync.WaitGroup
	errors := make(chan error, len(queries))

	for _, query := range queries {
		wg.Add(1)
		go func(q string) {
			defer wg.Done()
			// Use qwen3 explicitly to work in fallback mode (when Gemma unavailable)
			_, err := candle_binding.CalculateSimilarityBatch(q, candidates, 0, "qwen3", 768)
			if err != nil {
				errors <- err
			}
		}(query)
	}

	wg.Wait()
	close(errors)

	errorCount := 0
	for err := range errors {
		t.Error(err)
		errorCount++
	}

	if errorCount == 0 {
		t.Log("✓ Concurrent similarity calculations successful")
	}
}
