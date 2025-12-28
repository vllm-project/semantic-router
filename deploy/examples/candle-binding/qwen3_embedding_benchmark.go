// Benchmark simulating a typical embedding server workload
//
// This benchmark tests:
// 1. Concurrent request handling (multiple goroutines)
// 2. Sustained throughput under load
// 3. Latency distribution (p50, p95, p99)
// 4. Real-world API server scenarios
//
// Unlike typical Go benchmarks, this simulates an actual embedding server
// where multiple clients send requests concurrently.
//
// Usage:
//   cd ../../candle-binding
//   LD_LIBRARY_PATH=$(pwd)/target/release go run ../examples/candle-binding/qwen3_embedding_benchmark.go

package main

import (
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

/*
#cgo LDFLAGS: -L${SRCDIR}/../../candle-binding/target/release -lcandle_semantic_router -ldl -lm
#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    float* data;
    int length;
    bool error;
    int model_type;
    int sequence_length;
    float processing_time_ms;
} EmbeddingResult;

extern bool init_embedding_models_batched(const char* qwen3_model_path, int max_batch_size, unsigned long long max_wait_ms, bool use_cpu);
extern int get_embedding_batched(const char* text, const char* model_type, int target_dim, EmbeddingResult* result);
*/
import "C"

// Test texts representing typical embedding workloads
var testTexts = []string{
	"How can I reset my password for this application?",
	"The weather forecast shows sunny skies for the weekend",
	"Machine learning models require large amounts of training data",
	"Please schedule a meeting with the engineering team",
	"What are the best practices for database optimization?",
	"The stock market experienced significant volatility today",
	"Can you help me troubleshoot this network connectivity issue?",
	"Artificial intelligence is transforming healthcare delivery",
}

type BenchmarkResult struct {
	TotalRequests   int
	SuccessRequests int
	FailedRequests  int
	TotalTime       time.Duration
	Latencies       []time.Duration
	Throughput      float64
	AvgLatency      time.Duration
	P50Latency      time.Duration
	P95Latency      time.Duration
	P99Latency      time.Duration
	MaxLatency      time.Duration
	MinLatency      time.Duration
}

func getEmbedding(text string) (time.Duration, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	cModelType := C.CString("qwen3")
	defer C.free(unsafe.Pointer(cModelType))

	var result C.EmbeddingResult

	start := time.Now()
	status := C.get_embedding_batched(cText, cModelType, -1, &result)
	duration := time.Since(start)

	if status != 0 || result.error {
		return 0, fmt.Errorf("failed to get embedding (status: %d)", status)
	}

	// Free the C memory
	C.free(unsafe.Pointer(result.data))

	return duration, nil
}

func runConcurrentBenchmark(numClients int, requestsPerClient int, quiet bool) *BenchmarkResult {
	var wg sync.WaitGroup
	var successCount, failCount int64
	latencies := make(chan time.Duration, numClients*requestsPerClient)

	startTime := time.Now()

	// Launch client goroutines
	for i := 0; i < numClients; i++ {
		wg.Add(1)
		go func(clientID int) {
			defer wg.Done()

			// Each client sends multiple requests sequentially
			for j := 0; j < requestsPerClient; j++ {
				// Use round-robin to select test text
				textIdx := (clientID + j) % len(testTexts)
				text := testTexts[textIdx]

				latency, err := getEmbedding(text)
				if err != nil {
					atomic.AddInt64(&failCount, 1)
					if !quiet {
						log.Printf("Client %d request %d failed: %v", clientID, j, err)
					}
					continue
				}

				atomic.AddInt64(&successCount, 1)
				latencies <- latency
			}
		}(i)
	}

	// Wait for all clients to complete
	wg.Wait()
	close(latencies)

	totalTime := time.Since(startTime)

	// Collect latencies
	latencySlice := make([]time.Duration, 0, numClients*requestsPerClient)
	for lat := range latencies {
		latencySlice = append(latencySlice, lat)
	}

	// Sort latencies for percentile calculations
	sort.Slice(latencySlice, func(i, j int) bool {
		return latencySlice[i] < latencySlice[j]
	})

	result := &BenchmarkResult{
		TotalRequests:   numClients * requestsPerClient,
		SuccessRequests: int(successCount),
		FailedRequests:  int(failCount),
		TotalTime:       totalTime,
		Latencies:       latencySlice,
	}

	if len(latencySlice) > 0 {
		// Calculate throughput
		result.Throughput = float64(result.SuccessRequests) / totalTime.Seconds()

		// Calculate average latency
		var totalLatency time.Duration
		for _, lat := range latencySlice {
			totalLatency += lat
		}
		result.AvgLatency = totalLatency / time.Duration(len(latencySlice))

		// Calculate percentiles
		result.MinLatency = latencySlice[0]
		result.MaxLatency = latencySlice[len(latencySlice)-1]
		result.P50Latency = latencySlice[len(latencySlice)*50/100]
		result.P95Latency = latencySlice[len(latencySlice)*95/100]
		if len(latencySlice) > 100 {
			result.P99Latency = latencySlice[len(latencySlice)*99/100]
		} else {
			result.P99Latency = result.MaxLatency
		}
	}

	return result
}

func printHeader(title string) {
	fmt.Println()
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println("  " + title)
	fmt.Println(strings.Repeat("=", 80))
	fmt.Println()
}

func printResult(result *BenchmarkResult, title string) {
	printHeader(title)

	fmt.Printf("📊 Requests:     %d total (%d success, %d failed)\n",
		result.TotalRequests, result.SuccessRequests, result.FailedRequests)
	fmt.Printf("⏱️  Total Time:   %.2f seconds\n", result.TotalTime.Seconds())
	fmt.Printf("🚀 Throughput:   %.2f embeddings/second\n", result.Throughput)
	fmt.Println()
	fmt.Println("Latency Distribution:")
	fmt.Printf("  Average:  %8.2f ms\n", result.AvgLatency.Seconds()*1000)
	fmt.Printf("  Min:      %8.2f ms\n", result.MinLatency.Seconds()*1000)
	fmt.Printf("  P50:      %8.2f ms\n", result.P50Latency.Seconds()*1000)
	fmt.Printf("  P95:      %8.2f ms\n", result.P95Latency.Seconds()*1000)
	fmt.Printf("  P99:      %8.2f ms\n", result.P99Latency.Seconds()*1000)
	fmt.Printf("  Max:      %8.2f ms\n", result.MaxLatency.Seconds()*1000)
	fmt.Println()
}

func runSingleThreadedBaseline(numRequests int) *BenchmarkResult {
	latencies := make([]time.Duration, 0, numRequests)
	var successCount, failCount int

	startTime := time.Now()

	for i := 0; i < numRequests; i++ {
		textIdx := i % len(testTexts)
		text := testTexts[textIdx]

		latency, err := getEmbedding(text)
		if err != nil {
			failCount++
			continue
		}

		successCount++
		latencies = append(latencies, latency)
	}

	totalTime := time.Since(startTime)

	// Sort latencies for percentiles
	sort.Slice(latencies, func(i, j int) bool {
		return latencies[i] < latencies[j]
	})

	result := &BenchmarkResult{
		TotalRequests:   numRequests,
		SuccessRequests: successCount,
		FailedRequests:  failCount,
		TotalTime:       totalTime,
		Latencies:       latencies,
	}

	if len(latencies) > 0 {
		result.Throughput = float64(successCount) / totalTime.Seconds()

		var totalLatency time.Duration
		for _, lat := range latencies {
			totalLatency += lat
		}
		result.AvgLatency = totalLatency / time.Duration(len(latencies))

		result.MinLatency = latencies[0]
		result.MaxLatency = latencies[len(latencies)-1]
		result.P50Latency = latencies[len(latencies)*50/100]
		result.P95Latency = latencies[len(latencies)*95/100]
		if len(latencies) > 100 {
			result.P99Latency = latencies[len(latencies)*99/100]
		} else {
			result.P99Latency = result.MaxLatency
		}
	}

	return result
}

func main() {
	printHeader("Qwen3 Embedding Server Benchmark")

	// Initialize model
	fmt.Println("🔧 Initializing Qwen3 Embedding Model...")
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		modelPath = "../models/mom-embedding-pro"
	}

	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	// Initialize with continuous batching
	maxBatchSize := 64      // Batch up to 64 requests
	maxWaitMs := uint64(10) // Wait max 10ms for batch to fill

	success := C.init_embedding_models_batched(cModelPath, C.int(maxBatchSize), C.ulonglong(maxWaitMs), false)
	if !success {
		log.Fatalf("❌ Failed to initialize batched embedding model from: %s", modelPath)
	}

	fmt.Printf("✅ Batched model loaded successfully from: %s\n", modelPath)
	fmt.Printf("   Continuous batching: max_batch=%d, max_wait=%dms\n", maxBatchSize, maxWaitMs)

	// Warm-up
	fmt.Println("\n🔥 Warming up model...")
	for i := 0; i < 3; i++ {
		_, err := getEmbedding(testTexts[0])
		if err != nil {
			log.Fatalf("❌ Warm-up failed: %v", err)
		}
	}
	fmt.Println("✅ Warm-up complete")

	// Benchmark 1: Single-threaded baseline
	printHeader("Benchmark 1: Single-Threaded Baseline")
	fmt.Println("Simulating sequential request processing (1 client)")
	fmt.Println("Total requests: 40")
	fmt.Println()

	baseline := runSingleThreadedBaseline(40)
	printResult(baseline, "Single-Threaded Results")

	// Benchmark 2: Low concurrency (typical API server)
	printHeader("Benchmark 2: Low Concurrency (8 clients)")
	fmt.Println("Simulating typical API server load")
	fmt.Println("Clients: 8, Requests per client: 5, Total: 40")
	fmt.Println()

	lowConcurrency := runConcurrentBenchmark(8, 5, false)
	printResult(lowConcurrency, "Low Concurrency Results")

	// Benchmark 3: Medium concurrency
	printHeader("Benchmark 3: Medium Concurrency (16 clients)")
	fmt.Println("Simulating moderate API server load")
	fmt.Println("Clients: 16, Requests per client: 5, Total: 80")
	fmt.Println()

	mediumConcurrency := runConcurrentBenchmark(16, 5, true)
	printResult(mediumConcurrency, "Medium Concurrency Results")

	// Benchmark 4: High concurrency
	printHeader("Benchmark 4: High Concurrency (32 clients)")
	fmt.Println("Simulating high API server load")
	fmt.Println("Clients: 32, Requests per client: 5, Total: 160")
	fmt.Println()

	highConcurrency := runConcurrentBenchmark(32, 5, true)
	printResult(highConcurrency, "High Concurrency Results")

	// Benchmark 5: Sustained throughput
	printHeader("Benchmark 5: Sustained Throughput Test")
	fmt.Println("Simulating sustained high load")
	fmt.Println("Clients: 16, Requests per client: 25, Total: 400")
	fmt.Println()

	sustained := runConcurrentBenchmark(16, 25, true)
	printResult(sustained, "Sustained Throughput Results")

	// Summary
	printHeader("📊 Benchmark Summary")

	fmt.Println("Throughput Comparison:")
	fmt.Printf("  Single-threaded:   %8.2f emb/s (baseline)\n", baseline.Throughput)
	fmt.Printf("  8 clients:         %8.2f emb/s (%6.2fx)\n",
		lowConcurrency.Throughput, lowConcurrency.Throughput/baseline.Throughput)
	fmt.Printf("  16 clients:        %8.2f emb/s (%6.2fx)\n",
		mediumConcurrency.Throughput, mediumConcurrency.Throughput/baseline.Throughput)
	fmt.Printf("  32 clients:        %8.2f emb/s (%6.2fx)\n",
		highConcurrency.Throughput, highConcurrency.Throughput/baseline.Throughput)
	fmt.Printf("  Sustained (16x25): %8.2f emb/s (%6.2fx)\n",
		sustained.Throughput, sustained.Throughput/baseline.Throughput)
	fmt.Println()

	fmt.Println("Latency Comparison (P95):")
	fmt.Printf("  Single-threaded:   %8.2f ms\n", baseline.P95Latency.Seconds()*1000)
	fmt.Printf("  8 clients:         %8.2f ms (%+.1f%%)\n",
		lowConcurrency.P95Latency.Seconds()*1000,
		(lowConcurrency.P95Latency.Seconds()-baseline.P95Latency.Seconds())/baseline.P95Latency.Seconds()*100)
	fmt.Printf("  16 clients:        %8.2f ms (%+.1f%%)\n",
		mediumConcurrency.P95Latency.Seconds()*1000,
		(mediumConcurrency.P95Latency.Seconds()-baseline.P95Latency.Seconds())/baseline.P95Latency.Seconds()*100)
	fmt.Printf("  32 clients:        %8.2f ms (%+.1f%%)\n",
		highConcurrency.P95Latency.Seconds()*1000,
		(highConcurrency.P95Latency.Seconds()-baseline.P95Latency.Seconds())/baseline.P95Latency.Seconds()*100)
	fmt.Println()

	// Performance assessment
	if lowConcurrency.Throughput > baseline.Throughput*1.5 {
		fmt.Println("✅ EXCELLENT: Concurrent processing shows significant speedup!")
		fmt.Println("   The model is effectively utilizing parallelism.")
	} else if lowConcurrency.Throughput > baseline.Throughput {
		fmt.Println("✓  GOOD: Some concurrent speedup observed.")
		fmt.Println("   Consider enabling continuous batching for better performance.")
	} else {
		fmt.Println("⚠️  NOTE: Limited concurrent speedup.")
		fmt.Println("   This is expected without continuous batching.")
		fmt.Println("   GPU operations are being serialized.")
	}

	fmt.Println()
	fmt.Println("💡 Recommendation:")
	fmt.Println("   For production embedding servers with high concurrency,")
	fmt.Println("   enable continuous batching for 10-15x throughput improvement!")

	printHeader("✅ Benchmark Complete!")
}
