package main

import (
	"fmt"
	"os"
	"sort"
	"sync"
	"time"

	candle "github.com/vllm-project/semantic-router/candle-binding"
	openvino "github.com/vllm-project/semantic-router/openvino-binding"
)

// Test input sizes
var (
	SmallInput = "This is a short test message for benchmarking."

	MediumInput = "This is a medium-length text that contains multiple sentences. " +
		"It is designed to test the performance of the embedding and classification systems " +
		"with a reasonable amount of content. This represents typical use cases where users " +
		"submit paragraphs of text for processing and analysis."

	LargeInput = "This is a large text input designed to stress test the performance of both " +
		"the OpenVINO and Candle bindings with ModernBERT models. It contains multiple paragraphs " +
		"and sentences that simulate real-world usage scenarios where users might submit " +
		"substantial amounts of text for semantic analysis, classification, or embedding generation. " +
		"In practical applications, we often encounter text of varying lengths, from short queries " +
		"to long documents. This benchmark aims to capture the performance characteristics across " +
		"these different input sizes. The system must be able to handle not just small snippets " +
		"but also larger chunks of text efficiently. Performance metrics like latency, throughput, " +
		"and resource utilization are critical for production deployments. Understanding how the " +
		"system scales with input size and concurrency helps in capacity planning and optimization."
)

// BenchmarkConfig holds configuration for a benchmark run
type BenchmarkConfig struct {
	Name        string
	InputSize   string
	Input       string
	Concurrency int
	Iterations  int
}

// BenchmarkResult holds the results of a benchmark run
type BenchmarkResult struct {
	Config     BenchmarkConfig
	Binding    string
	Operation  string
	Latencies  []time.Duration
	Mean       time.Duration
	Median     time.Duration
	P95        time.Duration
	P99        time.Duration
	Min        time.Duration
	Max        time.Duration
	Throughput float64
	ErrorCount int
}

func main() {
	fmt.Println(repeat("=", 80))
	fmt.Println("ModernBERT Binding Performance Benchmark")
	fmt.Println("OpenVINO vs Candle - Classification Comparison")
	fmt.Println(repeat("=", 80))
	fmt.Println()

	// Print environment variable hints
	fmt.Println("Environment Variables:")
	fmt.Println("  OPENVINO_MODEL_PATH    - Path to OpenVINO model XML file")
	fmt.Println("                           Default: ../../test_models/category_classifier_modernbert/openvino_model.xml")
	fmt.Println("  CANDLE_MODEL_PATH      - Path to Candle model directory")
	fmt.Println("                           Default: ../../../models/category_classifier_modernbert-base_model")
	fmt.Println("  OPENVINO_TOKENIZERS_LIB - Path to libopenvino_tokenizers.so")
	fmt.Println()

	// Initialize models
	fmt.Println("Initializing models...")
	if err := initializeModels(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize models: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("✓ Models initialized\n")

	// Verify classification results match
	fmt.Println("Verifying classification correctness...")
	if err := verifyClassificationResults(); err != nil {
		fmt.Fprintf(os.Stderr, "⚠ Classification verification warning: %v\n\n", err)
	} else {
		fmt.Println("✓ Classification results verified (OpenVINO matches Candle)\n")
	}

	// Define benchmark configurations
	configs := []BenchmarkConfig{
		// Small input - various concurrency levels
		{Name: "Small-1x", InputSize: "Small (~10 words)", Input: SmallInput, Concurrency: 1, Iterations: 10},
		{Name: "Small-5x", InputSize: "Small (~10 words)", Input: SmallInput, Concurrency: 5, Iterations: 10},
		{Name: "Small-10x", InputSize: "Small (~10 words)", Input: SmallInput, Concurrency: 10, Iterations: 10},
		{Name: "Small-20x", InputSize: "Small (~10 words)", Input: SmallInput, Concurrency: 20, Iterations: 10},

		// Medium input
		{Name: "Medium-1x", InputSize: "Medium (~50 words)", Input: MediumInput, Concurrency: 1, Iterations: 10},
		{Name: "Medium-5x", InputSize: "Medium (~50 words)", Input: MediumInput, Concurrency: 5, Iterations: 10},
		{Name: "Medium-10x", InputSize: "Medium (~50 words)", Input: MediumInput, Concurrency: 10, Iterations: 10},

		// Large input
		{Name: "Large-1x", InputSize: "Large (~200 words)", Input: LargeInput, Concurrency: 1, Iterations: 10},
		{Name: "Large-5x", InputSize: "Large (~200 words)", Input: LargeInput, Concurrency: 5, Iterations: 10},
	}

	allResults := []BenchmarkResult{}

	// Run benchmarks
	for _, config := range configs {
		fmt.Printf("\n%s\n", repeat("=", 80))
		fmt.Printf("Running: %s | Concurrency=%d | Iterations=%d\n", config.Name, config.Concurrency, config.Iterations)
		fmt.Printf("%s\n\n", repeat("=", 80))

		// OpenVINO Classification
		result := benchmarkOpenVINOClassification(config)
		allResults = append(allResults, result)
		printResult(result)

		// Candle Classification
		result = benchmarkCandleClassification(config)
		allResults = append(allResults, result)
		printResult(result)
	}

	// Print summary
	printSummary(allResults)
}

func verifyClassificationResults() error {
	// Test texts with different characteristics
	testTexts := []string{
		"This is a short test message",
		"This is a longer test message with more content to classify and analyze for proper categorization",
		"Hello world",
		SmallInput,
		MediumInput,
	}

	fmt.Println("  Testing with multiple inputs...")
	differences := 0

	for i, text := range testTexts {
		// Classify with OpenVINO
		ovResult, err := openvino.ClassifyModernBert(text)
		if err != nil {
			return fmt.Errorf("OpenVINO classification failed: %v", err)
		}

		// Classify with Candle
		candleResult, err := candle.ClassifyModernBertText(text)
		if err != nil {
			return fmt.Errorf("Candle classification failed: %v", err)
		}

		// Compare results
		if ovResult.Class != candleResult.Class {
			differences++
			fmt.Printf("    ⚠ DIFFERENCE in test %d:\n", i+1)
			fmt.Printf("      Text: '%.60s...'\n", text)
			fmt.Printf("      OpenVINO: class=%d, confidence=%.4f\n", ovResult.Class, ovResult.Confidence)
			fmt.Printf("      Candle:   class=%d, confidence=%.4f\n", candleResult.Class, candleResult.Confidence)
			fmt.Printf("      Delta:    Δclass=%d, Δconfidence=%.4f\n",
				int(ovResult.Class)-int(candleResult.Class),
				ovResult.Confidence-candleResult.Confidence)
		} else {
			// Same class, check confidence difference
			confDiff := ovResult.Confidence - candleResult.Confidence
			if confDiff < 0 {
				confDiff = -confDiff
			}

			if confDiff > 0.05 { // More than 5% difference
				fmt.Printf("    ℹ Test %d: Same class (%d) but confidence differs by %.4f\n",
					i+1, ovResult.Class, confDiff)
			}
		}
	}

	if differences > 0 {
		return fmt.Errorf("found %d classification differences (see details above)", differences)
	}

	return nil
}

func initializeModels() error {
	// Initialize OpenVINO
	// Use OPENVINO_MODEL_PATH environment variable or default to test_models directory
	ovClassifierPath := os.Getenv("OPENVINO_MODEL_PATH")
	if ovClassifierPath == "" {
		// Default: assume running from repository root or use relative path
		ovClassifierPath = "../../test_models/category_classifier_modernbert/openvino_model.xml"
	}

	if err := openvino.InitModernBertClassifier(ovClassifierPath, 14, "CPU"); err != nil {
		return fmt.Errorf("OpenVINO classifier init failed: %v\nSet OPENVINO_MODEL_PATH environment variable to specify model location", err)
	}

	// Initialize Candle (useCPU = true to force CPU usage)
	// Use CANDLE_MODEL_PATH environment variable or default
	candleClassifierPath := os.Getenv("CANDLE_MODEL_PATH")
	if candleClassifierPath == "" {
		// Default: assume models are in ../../../models relative to cmd/benchmark
		candleClassifierPath = "../../../models/category_classifier_modernbert-base_model"
	}

	if err := candle.InitModernBertClassifier(candleClassifierPath, true); err != nil {
		return fmt.Errorf("Candle classifier init failed: %v\nSet CANDLE_MODEL_PATH environment variable to specify model location", err)
	}

	return nil
}

func benchmarkOpenVINOClassification(config BenchmarkConfig) BenchmarkResult {
	return runBenchmark(config, "OpenVINO", "Classification", func() error {
		_, err := openvino.ClassifyModernBert(config.Input)
		return err
	})
}

func benchmarkOpenVINOEmbedding(config BenchmarkConfig) BenchmarkResult {
	return runBenchmark(config, "OpenVINO", "Embedding", func() error {
		_, err := openvino.GetModernBertEmbedding(config.Input, 512)
		return err
	})
}

func benchmarkCandleClassification(config BenchmarkConfig) BenchmarkResult {
	return runBenchmark(config, "Candle", "Classification", func() error {
		_, err := candle.ClassifyModernBertText(config.Input)
		return err
	})
}

func runBenchmark(config BenchmarkConfig, binding, operation string, fn func() error) BenchmarkResult {
	result := BenchmarkResult{
		Config:    config,
		Binding:   binding,
		Operation: operation,
		Latencies: make([]time.Duration, 0, config.Iterations*config.Concurrency),
	}

	var wg sync.WaitGroup
	var mu sync.Mutex

	startTime := time.Now()

	for i := 0; i < config.Concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			for j := 0; j < config.Iterations; j++ {
				iterStart := time.Now()
				err := fn()
				duration := time.Since(iterStart)

				mu.Lock()
				if err != nil {
					result.ErrorCount++
				} else {
					result.Latencies = append(result.Latencies, duration)
				}
				mu.Unlock()
			}
		}()
	}

	wg.Wait()
	totalTime := time.Since(startTime)

	// Calculate statistics
	if len(result.Latencies) > 0 {
		sort.Slice(result.Latencies, func(i, j int) bool {
			return result.Latencies[i] < result.Latencies[j]
		})

		result.Min = result.Latencies[0]
		result.Max = result.Latencies[len(result.Latencies)-1]
		result.Median = result.Latencies[len(result.Latencies)/2]

		p95Idx := int(float64(len(result.Latencies)) * 0.95)
		if p95Idx >= len(result.Latencies) {
			p95Idx = len(result.Latencies) - 1
		}
		result.P95 = result.Latencies[p95Idx]

		p99Idx := int(float64(len(result.Latencies)) * 0.99)
		if p99Idx >= len(result.Latencies) {
			p99Idx = len(result.Latencies) - 1
		}
		result.P99 = result.Latencies[p99Idx]

		var sum time.Duration
		for _, lat := range result.Latencies {
			sum += lat
		}
		result.Mean = sum / time.Duration(len(result.Latencies))

		result.Throughput = float64(len(result.Latencies)) / totalTime.Seconds()
	}

	return result
}

func printResult(result BenchmarkResult) {
	fmt.Printf("  %s %s:\n", result.Binding, result.Operation)
	fmt.Printf("    Mean:       %8.2f ms\n", float64(result.Mean.Microseconds())/1000.0)
	fmt.Printf("    Median:     %8.2f ms\n", float64(result.Median.Microseconds())/1000.0)
	fmt.Printf("    P95:        %8.2f ms\n", float64(result.P95.Microseconds())/1000.0)
	fmt.Printf("    P99:        %8.2f ms\n", float64(result.P99.Microseconds())/1000.0)
	fmt.Printf("    Min:        %8.2f ms\n", float64(result.Min.Microseconds())/1000.0)
	fmt.Printf("    Max:        %8.2f ms\n", float64(result.Max.Microseconds())/1000.0)
	fmt.Printf("    Throughput: %8.2f req/s\n", result.Throughput)
	if result.ErrorCount > 0 {
		fmt.Printf("    Errors:     %d\n", result.ErrorCount)
	}
	fmt.Println()
}

func printSummary(results []BenchmarkResult) {
	fmt.Printf("\n%s\n", repeat("=", 80))
	fmt.Println("SUMMARY")
	fmt.Printf("%s\n\n", repeat("=", 80))

	// Group by input size and concurrency
	type Key struct {
		InputSize   string
		Concurrency int
		Operation   string
	}

	summary := make(map[Key]*BenchmarkResult)

	for i := range results {
		result := &results[i]
		key := Key{
			InputSize:   result.Config.InputSize,
			Concurrency: result.Config.Concurrency,
			Operation:   result.Operation,
		}
		summary[key] = result
	}

	// Print comparison table
	fmt.Printf("%-25s %-10s %-20s %12s %12s %12s %15s\n",
		"Input Size", "Concurrency", "Operation", "Mean (ms)", "P95 (ms)", "P99 (ms)", "Throughput")
	fmt.Println(repeat("-", 115))

	for _, inputSize := range []string{"Small (~10 words)", "Medium (~50 words)", "Large (~200 words)"} {
		for _, concurrency := range []int{1, 5, 10, 20} {
			for _, operation := range []string{"Classification"} {
				key := Key{InputSize: inputSize, Concurrency: concurrency, Operation: operation}
				result := summary[key]

				if result != nil {
					meanMs := float64(result.Mean.Microseconds()) / 1000.0
					p95Ms := float64(result.P95.Microseconds()) / 1000.0
					p99Ms := float64(result.P99.Microseconds()) / 1000.0

					fmt.Printf("%-25s %-10d %-20s %12.2f %12.2f %12.2f %12.2f req/s\n",
						inputSize, concurrency, operation, meanMs, p95Ms, p99Ms, result.Throughput)
				}
			}
		}
	}

	fmt.Println()
}

func repeat(s string, count int) string {
	result := ""
	for i := 0; i < count; i++ {
		result += s
	}
	return result
}
