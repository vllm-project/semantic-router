//go:build integration
// +build integration

package integration

import (
	"runtime"
	"testing"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// TestEmbeddingMemory tests memory usage and leak detection

// Test: Memory leak detection - long-running operations
func TestEmbedding_NoMemoryLeak(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	// Force GC and get baseline
	runtime.GC()
	time.Sleep(100 * time.Millisecond)

	var baseline runtime.MemStats
	runtime.ReadMemStats(&baseline)
	baselineAlloc := baseline.Alloc

	t.Logf("Baseline memory: %d MB", baselineAlloc/1024/1024)

	// Generate many embeddings
	iterations := 1000
	text := "Memory leak detection test"

	for i := 0; i < iterations; i++ {
		_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
		if err != nil {
			t.Fatalf("Embedding generation failed at iteration %d: %v", i, err)
		}

		// Sample memory every 100 iterations
		if i%100 == 0 {
			var m runtime.MemStats
			runtime.ReadMemStats(&m)
			t.Logf("Iteration %d: Memory = %d MB", i, m.Alloc/1024/1024)
		}
	}

	// Force GC to clean up
	runtime.GC()
	time.Sleep(100 * time.Millisecond)

	var final runtime.MemStats
	runtime.ReadMemStats(&final)
	finalAlloc := final.Alloc

	growth := int64(finalAlloc) - int64(baselineAlloc)
	growthMB := growth / 1024 / 1024

	t.Logf("Final memory: %d MB", finalAlloc/1024/1024)
	t.Logf("Memory growth: %d MB after %d embeddings", growthMB, iterations)

	// Memory growth should be minimal after GC (< 10MB for 1000 embeddings)
	threshold := int64(10)
	if growthMB > threshold {
		t.Errorf("Potential memory leak: grew by %d MB (threshold: %d MB)", growthMB, threshold)
	} else {
		t.Logf("✓ No memory leak detected: growth %d MB < %d MB threshold", growthMB, threshold)
	}
}

// Test: Memory growth rate
func TestEmbedding_MemoryGrowthRate(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	runtime.GC()
	var m runtime.MemStats

	// Measure memory at different intervals
	// Reduced iterations for CI to avoid timeout
	measurements := []int{10, 25, 50}
	if !testing.Short() {
		// Full test on local development
		measurements = []int{100, 200, 500, 1000}
	}
	text := "Memory growth rate test"

	for _, target := range measurements {
		runtime.GC()
		runtime.ReadMemStats(&m)
		before := m.Alloc

		// Generate embeddings
		for i := 0; i < target; i++ {
			_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
			if err != nil {
				t.Fatalf("Embedding generation failed: %v", err)
			}
		}

		runtime.GC()
		runtime.ReadMemStats(&m)
		after := m.Alloc

		growth := int64(after) - int64(before)
		perEmbedding := growth / int64(target)

		t.Logf("%d embeddings: growth=%d MB (%.2f KB per embedding)",
			target, growth/1024/1024, float64(perEmbedding)/1024)
	}

	t.Log("✓ Memory growth rate analysis completed")
}

// Test: Model memory footprint
func TestEmbedding_ModelMemoryFootprint(t *testing.T) {
	// Measure memory before loading models
	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)
	beforeMB := before.Alloc / 1024 / 1024

	t.Logf("Memory before model loading: %d MB", beforeMB)

	// Load models
	initEmbeddingModelsWithFallback(t, testModelsDir)

	// Measure memory after loading models
	runtime.GC()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)
	afterMB := after.Alloc / 1024 / 1024

	modelMemory := int64(afterMB) - int64(beforeMB)

	t.Logf("Memory after model loading: %d MB", afterMB)
	t.Logf("Model memory footprint: %d MB", modelMemory)

	// Model memory should be reasonable
	// Qwen3 (600M params) + Gemma (300M params) + BERT should fit in < 2GB on CPU
	if modelMemory > 2048 {
		t.Logf("Warning: Model memory %d MB exceeds 2GB", modelMemory)
	} else {
		t.Logf("✓ Model memory footprint is reasonable: %d MB", modelMemory)
	}
}

// Test: Garbage collection effectiveness
func TestEmbedding_GarbageCollection(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	text := "GC effectiveness test"

	// Generate embeddings (reduced in short mode for CI)
	iterations := 500
	if testing.Short() {
		iterations = 50
	}
	for i := 0; i < iterations; i++ {
		_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
		if err != nil {
			t.Fatalf("Embedding generation failed: %v", err)
		}
	}

	// Measure before GC
	var beforeGC runtime.MemStats
	runtime.ReadMemStats(&beforeGC)
	beforeMB := beforeGC.Alloc / 1024 / 1024

	// Force GC
	runtime.GC()
	time.Sleep(100 * time.Millisecond)

	// Measure after GC
	var afterGC runtime.MemStats
	runtime.ReadMemStats(&afterGC)
	afterMB := afterGC.Alloc / 1024 / 1024

	cleaned := int64(beforeMB) - int64(afterMB)

	t.Logf("Memory before GC: %d MB", beforeMB)
	t.Logf("Memory after GC: %d MB", afterMB)
	t.Logf("Memory cleaned: %d MB", cleaned)

	if cleaned < 0 {
		t.Log("Note: Memory increased after GC (may be normal)")
	} else if cleaned > 0 {
		t.Logf("✓ GC cleaned %d MB", cleaned)
	}
}

// Test: Memory per embedding size
func TestEmbedding_MemoryPerEmbeddingSize(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	testCases := []struct {
		name string
		text string
	}{
		{"Short", "Hi"},
		{"Medium", "This is a medium length sentence"},
		{"Long", "This is a very long text that contains many words and provides extensive context"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			runtime.GC()
			var before runtime.MemStats
			runtime.ReadMemStats(&before)

			// Generate embedding
			_, err := candle_binding.GetEmbeddingWithModelType(tc.text, "qwen3", 0)
			if err != nil {
				t.Fatalf("Embedding generation failed: %v", err)
			}

			var after runtime.MemStats
			runtime.ReadMemStats(&after)

			growth := int64(after.Alloc) - int64(before.Alloc)
			t.Logf("Text length %d: memory growth = %.2f KB",
				len(tc.text), float64(growth)/1024)
		})
	}
}

// Test: Batch memory usage
func TestEmbedding_BatchMemoryUsage(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	batchSizes := []int{10, 50, 100}
	text := "Batch memory test"

	for _, size := range batchSizes {
		runtime.GC()
		var before runtime.MemStats
		runtime.ReadMemStats(&before)

		// Generate batch
		for i := 0; i < size; i++ {
			_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
			if err != nil {
				t.Fatalf("Batch embedding failed: %v", err)
			}
		}

		var after runtime.MemStats
		runtime.ReadMemStats(&after)

		growth := int64(after.Alloc) - int64(before.Alloc)
		perEmbedding := growth / int64(size)

		t.Logf("Batch size %d: total=%d MB, per embedding=%.2f KB",
			size, growth/1024/1024, float64(perEmbedding)/1024)
	}

	t.Log("✓ Batch memory usage analysis completed")
}

// Test: Peak memory usage
func TestEmbedding_PeakMemoryUsage(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	runtime.GC()
	var baseline runtime.MemStats
	runtime.ReadMemStats(&baseline)

	text := "Peak memory test"
	var peak uint64 = baseline.Alloc

	// Generate embeddings and track peak (reduced in short mode for CI)
	iterations := 200
	if testing.Short() {
		iterations = 25
	}
	for i := 0; i < iterations; i++ {
		_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
		if err != nil {
			t.Fatalf("Embedding generation failed: %v", err)
		}

		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		if m.Alloc > peak {
			peak = m.Alloc
		}
	}

	peakMB := peak / 1024 / 1024
	baselineMB := baseline.Alloc / 1024 / 1024
	diff := peakMB - baselineMB

	t.Logf("Baseline memory: %d MB", baselineMB)
	t.Logf("Peak memory: %d MB", peakMB)
	t.Logf("Peak increase: %d MB", diff)

	if diff > 100 {
		t.Logf("Warning: Peak memory increased by %d MB", diff)
	} else {
		t.Logf("✓ Peak memory increase is acceptable: %d MB", diff)
	}
}

// Test: Memory cleanup after operations
func TestEmbedding_MemoryCleanup(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	runtime.GC()
	var start runtime.MemStats
	runtime.ReadMemStats(&start)
	startMB := start.Alloc / 1024 / 1024

	text := "Memory cleanup test"

	// Reduced iterations in short mode for CI
	iterations := 100
	if testing.Short() {
		iterations = 15
	}

	// Cycle 1
	for i := 0; i < iterations; i++ {
		_, _ = candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
	}
	runtime.GC()
	time.Sleep(50 * time.Millisecond)

	var mid runtime.MemStats
	runtime.ReadMemStats(&mid)
	midMB := mid.Alloc / 1024 / 1024

	// Cycle 2
	for i := 0; i < iterations; i++ {
		_, _ = candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
	}
	runtime.GC()
	time.Sleep(50 * time.Millisecond)

	var end runtime.MemStats
	runtime.ReadMemStats(&end)
	endMB := end.Alloc / 1024 / 1024

	t.Logf("Start: %d MB", startMB)
	t.Logf("After cycle 1: %d MB (growth: %d MB)", midMB, int64(midMB)-int64(startMB))
	t.Logf("After cycle 2: %d MB (growth: %d MB)", endMB, int64(endMB)-int64(midMB))

	// Growth between cycles should be similar (no accumulation)
	cycle1Growth := int64(midMB) - int64(startMB)
	cycle2Growth := int64(endMB) - int64(midMB)

	if cycle2Growth > cycle1Growth*2 {
		t.Errorf("Memory accumulation detected: cycle2=%d MB > 2*cycle1=%d MB",
			cycle2Growth, cycle1Growth)
	} else {
		t.Logf("✓ No memory accumulation between cycles")
	}
}

// Test: Memory stability over time
func TestEmbedding_MemoryStability(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping long-running test in short mode")
	}

	initEmbeddingModelsWithFallback(t, testModelsDir)

	runtime.GC()
	text := "Memory stability test"
	duration := 30 * time.Second
	sampleInterval := 5 * time.Second

	samples := []uint64{}
	deadline := time.Now().Add(duration)
	nextSample := time.Now()

	for time.Now().Before(deadline) {
		// Generate embeddings
		_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
		if err != nil {
			t.Fatalf("Embedding generation failed: %v", err)
		}

		// Sample memory periodically
		if time.Now().After(nextSample) {
			runtime.GC()
			var m runtime.MemStats
			runtime.ReadMemStats(&m)
			samples = append(samples, m.Alloc)
			nextSample = time.Now().Add(sampleInterval)
		}
	}

	t.Logf("Memory samples over %v:", duration)
	for i, sample := range samples {
		t.Logf("  Sample %d: %d MB", i, sample/1024/1024)
	}

	// Check if memory is stable (not continuously growing)
	if len(samples) >= 3 {
		lastThird := samples[len(samples)-len(samples)/3:]
		var avgLast uint64
		for _, s := range lastThird {
			avgLast += s / uint64(len(lastThird))
		}

		firstSample := samples[0]
		growth := int64(avgLast) - int64(firstSample)
		growthMB := growth / 1024 / 1024

		t.Logf("Memory growth over time: %d MB", growthMB)

		if growthMB > 50 {
			t.Errorf("Memory instability: grew by %d MB over %v", growthMB, duration)
		} else {
			t.Logf("✓ Memory is stable over time")
		}
	}
}

// Test: Memory allocation patterns
func TestEmbedding_AllocationPatterns(t *testing.T) {
	initEmbeddingModelsWithFallback(t, testModelsDir)

	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	text := "Allocation pattern test"

	// Generate embeddings (reduced in short mode for CI)
	iterations := 50
	if testing.Short() {
		iterations = 10
	}
	for i := 0; i < iterations; i++ {
		_, err := candle_binding.GetEmbeddingWithModelType(text, "qwen3", 0)
		if err != nil {
			t.Fatalf("Embedding generation failed: %v", err)
		}
	}

	// Force GC to free objects before checking stats
	runtime.GC()
	time.Sleep(100 * time.Millisecond)

	var after runtime.MemStats
	runtime.ReadMemStats(&after)

	// Calculate deltas using signed arithmetic to avoid underflow
	allocsDelta := int64(after.Mallocs) - int64(before.Mallocs)
	freesDelta := int64(after.Frees) - int64(before.Frees)
	heapObjectsDelta := int64(after.HeapObjects) - int64(before.HeapObjects)
	gcDelta := int64(after.NumGC) - int64(before.NumGC)

	t.Logf("Allocation statistics:")
	t.Logf("  Total allocs: %d", allocsDelta)
	t.Logf("  Total frees: %d", freesDelta)
	t.Logf("  Heap objects delta: %d", heapObjectsDelta)
	t.Logf("  GC runs: %d", gcDelta)

	// Check if allocations are being freed (after GC)
	allocs := after.Mallocs - before.Mallocs
	frees := after.Frees - before.Frees

	if allocs > 0 && frees == 0 {
		t.Error("No objects freed even after GC - potential memory leak")
	} else if allocs > 0 && float64(frees)/float64(allocs) < 0.5 {
		t.Logf("Warning: Low free ratio: %.2f%%", float64(frees)/float64(allocs)*100)
	} else {
		t.Logf("✓ Good allocation/free pattern: %.2f%% freed", float64(frees)/float64(allocs)*100)
	}
}
