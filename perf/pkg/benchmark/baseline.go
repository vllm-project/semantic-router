package benchmark

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// Baseline represents performance baseline data
type Baseline struct {
	Version    string                     `json:"version"`
	GitCommit  string                     `json:"git_commit"`
	Timestamp  time.Time                  `json:"timestamp"`
	Benchmarks map[string]BenchmarkMetric `json:"benchmarks"`
}

// BenchmarkMetric holds metrics for a single benchmark
type BenchmarkMetric struct {
	NsPerOp       int64   `json:"ns_per_op"`
	P50LatencyMs  float64 `json:"p50_latency_ms,omitempty"`
	P95LatencyMs  float64 `json:"p95_latency_ms,omitempty"`
	P99LatencyMs  float64 `json:"p99_latency_ms,omitempty"`
	ThroughputQPS float64 `json:"throughput_qps,omitempty"`
	AllocsPerOp   int64   `json:"allocs_per_op,omitempty"`
	BytesPerOp    int64   `json:"bytes_per_op,omitempty"`
}

// ComparisonResult represents the result of comparing current vs baseline
type ComparisonResult struct {
	BenchmarkName      string
	Baseline           BenchmarkMetric
	Current            BenchmarkMetric
	NsPerOpChange      float64 // Percentage change
	P95LatencyChange   float64
	ThroughputChange   float64
	RegressionDetected bool
	Threshold          float64 // Max allowed regression percentage
}

// LoadBaseline loads baseline data from a JSON file
func LoadBaseline(path string) (*Baseline, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("baseline file not found: %s", path)
		}
		return nil, fmt.Errorf("failed to read baseline file: %w", err)
	}

	var baseline Baseline
	if err := json.Unmarshal(data, &baseline); err != nil {
		return nil, fmt.Errorf("failed to parse baseline JSON: %w", err)
	}

	return &baseline, nil
}

// SaveBaseline saves baseline data to a JSON file
func SaveBaseline(baseline *Baseline, path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("failed to create baseline directory: %w", err)
	}

	data, err := json.MarshalIndent(baseline, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal baseline: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write baseline file: %w", err)
	}

	return nil
}

// CompareWithBaseline compares current metrics against baseline
func CompareWithBaseline(current, baseline *Baseline, thresholds *ThresholdsConfig) ([]ComparisonResult, error) {
	var results []ComparisonResult

	for benchName, currentMetric := range current.Benchmarks {
		baselineMetric, exists := baseline.Benchmarks[benchName]
		if !exists {
			// New benchmark, no baseline to compare
			continue
		}

		result := ComparisonResult{
			BenchmarkName: benchName,
			Baseline:      baselineMetric,
			Current:       currentMetric,
		}

		// Calculate percentage changes
		if baselineMetric.NsPerOp > 0 {
			result.NsPerOpChange = calculatePercentChange(
				float64(baselineMetric.NsPerOp),
				float64(currentMetric.NsPerOp),
			)
		}

		if baselineMetric.P95LatencyMs > 0 {
			result.P95LatencyChange = calculatePercentChange(
				baselineMetric.P95LatencyMs,
				currentMetric.P95LatencyMs,
			)
		}

		if baselineMetric.ThroughputQPS > 0 {
			result.ThroughputChange = calculatePercentChange(
				baselineMetric.ThroughputQPS,
				currentMetric.ThroughputQPS,
			)
		}

		// Determine threshold for this benchmark
		threshold := getThresholdForBenchmark(benchName, thresholds)
		result.Threshold = threshold

		// Detect regressions
		// Latency increase or throughput decrease beyond threshold = regression
		if result.NsPerOpChange > threshold ||
			result.P95LatencyChange > threshold ||
			(result.ThroughputChange < -threshold && baselineMetric.ThroughputQPS > 0) {
			result.RegressionDetected = true
		}

		results = append(results, result)
	}

	return results, nil
}

// calculatePercentChange calculates percentage change from baseline to current
// Positive = increase, negative = decrease
func calculatePercentChange(baseline, current float64) float64 {
	if baseline == 0 {
		return 0
	}
	return ((current - baseline) / baseline) * 100
}

// getThresholdForBenchmark retrieves the appropriate threshold for a benchmark
func getThresholdForBenchmark(benchName string, thresholds *ThresholdsConfig) float64 {
	// Default threshold
	defaultThreshold := 10.0

	if thresholds == nil {
		return defaultThreshold
	}

	// Try to find specific threshold based on benchmark name
	// This is a simplified approach - could be made more sophisticated
	for _, threshold := range thresholds.ComponentBenchmarks.Classification {
		if threshold.MaxRegressionPercent > 0 {
			return threshold.MaxRegressionPercent
		}
	}

	for _, threshold := range thresholds.ComponentBenchmarks.DecisionEngine {
		if threshold.MaxRegressionPercent > 0 {
			return threshold.MaxRegressionPercent
		}
	}

	for _, threshold := range thresholds.ComponentBenchmarks.Cache {
		if threshold.MaxRegressionPercent > 0 {
			return threshold.MaxRegressionPercent
		}
	}

	return defaultThreshold
}

// HasRegressions checks if any regressions were detected
func HasRegressions(results []ComparisonResult) bool {
	for _, result := range results {
		if result.RegressionDetected {
			return true
		}
	}
	return false
}

// PrintComparisonResults prints comparison results in a formatted table
func PrintComparisonResults(results []ComparisonResult) {
	fmt.Println("\n" + "===================================================================================")
	fmt.Println("                        PERFORMANCE COMPARISON RESULTS")
	fmt.Println("===================================================================================")
	fmt.Printf("%-50s %-15s %-15s %-15s\n", "Benchmark", "Baseline", "Current", "Change")
	fmt.Println("-----------------------------------------------------------------------------------")

	for _, result := range results {
		icon := "✓"
		if result.RegressionDetected {
			icon = "⚠️"
		}

		// Display ns/op comparison
		fmt.Printf("%s %-48s %-15d %-15d %+.2f%%\n",
			icon,
			result.BenchmarkName,
			result.Baseline.NsPerOp,
			result.Current.NsPerOp,
			result.NsPerOpChange,
		)

		// Display P95 latency if available
		if result.Baseline.P95LatencyMs > 0 {
			fmt.Printf("  └─ P95 Latency: %-15.2fms %-15.2fms %+.2f%%\n",
				result.Baseline.P95LatencyMs,
				result.Current.P95LatencyMs,
				result.P95LatencyChange,
			)
		}

		// Display throughput if available
		if result.Baseline.ThroughputQPS > 0 {
			fmt.Printf("  └─ Throughput:  %-15.2f qps %-15.2f qps %+.2f%%\n",
				result.Baseline.ThroughputQPS,
				result.Current.ThroughputQPS,
				result.ThroughputChange,
			)
		}
	}

	fmt.Println("===================================================================================")

	// Print summary
	regressionCount := 0
	for _, result := range results {
		if result.RegressionDetected {
			regressionCount++
		}
	}

	if regressionCount > 0 {
		fmt.Printf("\n⚠️  WARNING: %d regression(s) detected!\n", regressionCount)
	} else {
		fmt.Printf("\n✓ No regressions detected\n")
	}
}
