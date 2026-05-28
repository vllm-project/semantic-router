package main

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
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
	fmt.Println("ModernBERT Classifier Benchmark")
	fmt.Println("OpenVINO vs Candle - Classification Comparison")
	fmt.Println(repeat("=", 80))
	fmt.Println()

	fmt.Println("Environment Variables:")
	fmt.Println("  OPENVINO_MODEL_PATH               - Path to OpenVINO model XML file")
	fmt.Println("  CANDLE_MODEL_PATH                 - Path to Candle model directory")
	fmt.Println("  OPENVINO_TOKENIZERS_LIB           - Path to libopenvino_tokenizers.so")
	fmt.Println("  CLASSIFIER_MIN_MATCH_RATE         - Min class match ratio for correctness (default: 1.0)")
	fmt.Println("  CLASSIFIER_MAX_CONFIDENCE_DELTA   - Max confidence delta on matched class (default: 0.15)")
	fmt.Println()

	fmt.Println("Initializing models...")
	if err := initializeModels(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize models: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Models initialized\n")

	fmt.Println("Verifying classification correctness (required before benchmark)...")
	if err := verifyClassificationResults(); err != nil {
		fmt.Fprintf(os.Stderr, "Classification correctness check failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Classification correctness check passed\n")

	configs := []BenchmarkConfig{
		{Name: "Small-1x", InputSize: "Small (~10 words)", Input: SmallInput, Concurrency: 1, Iterations: 10},
		{Name: "Small-5x", InputSize: "Small (~10 words)", Input: SmallInput, Concurrency: 5, Iterations: 10},
		{Name: "Small-10x", InputSize: "Small (~10 words)", Input: SmallInput, Concurrency: 10, Iterations: 10},
		{Name: "Small-20x", InputSize: "Small (~10 words)", Input: SmallInput, Concurrency: 20, Iterations: 10},
		{Name: "Medium-1x", InputSize: "Medium (~50 words)", Input: MediumInput, Concurrency: 1, Iterations: 10},
		{Name: "Medium-5x", InputSize: "Medium (~50 words)", Input: MediumInput, Concurrency: 5, Iterations: 10},
		{Name: "Medium-10x", InputSize: "Medium (~50 words)", Input: MediumInput, Concurrency: 10, Iterations: 10},
		{Name: "Large-1x", InputSize: "Large (~200 words)", Input: LargeInput, Concurrency: 1, Iterations: 10},
		{Name: "Large-5x", InputSize: "Large (~200 words)", Input: LargeInput, Concurrency: 5, Iterations: 10},
	}

	allResults := []BenchmarkResult{}

	for _, config := range configs {
		fmt.Printf("\n%s\n", repeat("=", 80))
		fmt.Printf("Running: %s | Concurrency=%d | Iterations=%d\n", config.Name, config.Concurrency, config.Iterations)
		fmt.Printf("%s\n\n", repeat("=", 80))

		openvinoResult := benchmarkOpenVINOClassification(config)
		allResults = append(allResults, openvinoResult)
		printResult(openvinoResult)

		candleResult := benchmarkCandleClassification(config)
		allResults = append(allResults, candleResult)
		printResult(candleResult)

		printComparison(openvinoResult, candleResult)
	}

	printSummary(allResults)
}

func verifyClassificationResults() error {
	testTexts := []string{
		"This is a short test message",
		"This is a longer test message with more content to classify and analyze for proper categorization",
		"Hello world",
		"This content may include policy discussion around safety and security.",
		SmallInput,
		MediumInput,
	}

	minMatchRate := getenvFloat("CLASSIFIER_MIN_MATCH_RATE", 1.0)
	maxConfDelta := getenvFloat("CLASSIFIER_MAX_CONFIDENCE_DELTA", 0.15)

	fmt.Printf("  correctness thresholds: min_match_rate=%.3f max_confidence_delta=%.3f\n", minMatchRate, maxConfDelta)
	fmt.Println("  testing with multiple inputs...")

	matches := 0
	maxObservedConfDelta := 0.0
	differenceLines := make([]string, 0)

	for i, text := range testTexts {
		ovResult, err := openvino.ClassifyModernBert(text)
		if err != nil {
			return fmt.Errorf("OpenVINO classification failed: %v", err)
		}

		candleResult, err := candle.ClassifyModernBertText(text)
		if err != nil {
			return fmt.Errorf("Candle classification failed: %v", err)
		}

		confDelta := math.Abs(float64(ovResult.Confidence - candleResult.Confidence))
		if confDelta > maxObservedConfDelta {
			maxObservedConfDelta = confDelta
		}

		if ovResult.Class == candleResult.Class {
			matches++
			continue
		}

		differenceLines = append(differenceLines,
			fmt.Sprintf("test=%d ov_class=%d candle_class=%d ov_conf=%.4f candle_conf=%.4f text='%.60s...'",
				i+1,
				ovResult.Class,
				candleResult.Class,
				ovResult.Confidence,
				candleResult.Confidence,
				text,
			),
		)
	}

	matchRate := float64(matches) / float64(len(testTexts))
	fmt.Printf("  correctness summary: matches=%d/%d match_rate=%.3f max_confidence_delta=%.4f\n", matches, len(testTexts), matchRate, maxObservedConfDelta)

	if matchRate < minMatchRate {
		for _, line := range differenceLines {
			fmt.Printf("    mismatch: %s\n", line)
		}
		return fmt.Errorf("classification match_rate %.3f is below threshold %.3f", matchRate, minMatchRate)
	}

	if maxObservedConfDelta > maxConfDelta {
		return fmt.Errorf("max confidence delta %.4f exceeds threshold %.4f", maxObservedConfDelta, maxConfDelta)
	}

	return nil
}

func initializeModels() error {
	ovClassifierPath := os.Getenv("OPENVINO_MODEL_PATH")
	if ovClassifierPath == "" {
		ovClassifierPath = "../../test_models/category_classifier_modernbert/openvino_model.xml"
	}

	if err := openvino.InitModernBertClassifier(ovClassifierPath, 14, "CPU"); err != nil {
		return fmt.Errorf("OpenVINO classifier init failed: %v\nSet OPENVINO_MODEL_PATH environment variable to specify model location", err)
	}

	candleClassifierPath := os.Getenv("CANDLE_MODEL_PATH")
	if candleClassifierPath == "" {
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
	bindingLower := strings.ToLower(result.Binding)
	fmt.Printf("backend=%s cfg=%s n=%d errs=%d mean=%v p50=%v p95=%v p99=%v min=%v max=%v tps=%.2f\n",
		bindingLower,
		result.Config.Name,
		len(result.Latencies),
		result.ErrorCount,
		result.Mean,
		result.Median,
		result.P95,
		result.P99,
		result.Min,
		result.Max,
		result.Throughput,
	)

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

func printComparison(openvinoResult BenchmarkResult, candleResult BenchmarkResult) {
	if openvinoResult.Mean <= 0 || candleResult.Mean <= 0 {
		fmt.Printf("speedup(openvino/candle) cfg=%s : n/a (insufficient successful samples)\n\n", openvinoResult.Config.Name)
		return
	}

	speedup := float64(candleResult.Mean) / float64(openvinoResult.Mean)
	fmt.Printf("speedup(openvino/candle) cfg=%s : %.2fx (higher is better)\n\n", openvinoResult.Config.Name, speedup)
}

func printSummary(results []BenchmarkResult) {
	fmt.Printf("\n%s\n", repeat("=", 80))
	fmt.Println("CLASSIFIER SUMMARY (OpenVINO vs Candle)")
	fmt.Printf("%s\n\n", repeat("=", 80))

	type ResultPair struct {
		OpenVINO *BenchmarkResult
		Candle   *BenchmarkResult
	}

	pairs := make(map[string]*ResultPair)
	for i := range results {
		result := &results[i]
		pair, ok := pairs[result.Config.Name]
		if !ok {
			pair = &ResultPair{}
			pairs[result.Config.Name] = pair
		}

		switch strings.ToLower(result.Binding) {
		case "openvino":
			pair.OpenVINO = result
		case "candle":
			pair.Candle = result
		}
	}

	ordered := []string{"Small-1x", "Small-5x", "Small-10x", "Small-20x", "Medium-1x", "Medium-5x", "Medium-10x", "Large-1x", "Large-5x"}
	fmt.Printf("%-12s %-20s %-6s %12s %14s %12s %10s %10s\n",
		"Config", "Input", "Conc", "OV Mean", "Candle Mean", "Speedup", "OV TPS", "CA TPS")
	fmt.Println(repeat("-", 112))

	for _, cfgName := range ordered {
		pair, ok := pairs[cfgName]
		if !ok || pair.OpenVINO == nil || pair.Candle == nil {
			continue
		}

		ovMeanMs := float64(pair.OpenVINO.Mean.Microseconds()) / 1000.0
		candleMeanMs := float64(pair.Candle.Mean.Microseconds()) / 1000.0
		speedup := candleMeanMs / ovMeanMs

		fmt.Printf("%-12s %-20s %-6d %12.2f %14.2f %11.2fx %10.2f %10.2f\n",
			cfgName,
			pair.OpenVINO.Config.InputSize,
			pair.OpenVINO.Config.Concurrency,
			ovMeanMs,
			candleMeanMs,
			speedup,
			pair.OpenVINO.Throughput,
			pair.Candle.Throughput,
		)
	}

	fmt.Println()
}

func getenvFloat(key string, def float64) float64 {
	raw := os.Getenv(key)
	if raw == "" {
		return def
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
		fmt.Printf("invalid %s=%q, using default %.4f\n", key, raw, def)
		return def
	}
	return v
}

func repeat(s string, count int) string {
	result := ""
	for i := 0; i < count; i++ {
		result += s
	}
	return result
}
