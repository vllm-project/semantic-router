package benchmark

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

// Baseline represents performance baseline data
type Baseline struct {
	Version    string                     `json:"version"`
	GitCommit  string                     `json:"git_commit"`
	Timestamp  time.Time                  `json:"timestamp"`
	Benchmarks map[string]BenchmarkMetric `json:"benchmarks"`
}

// BenchmarkMetric holds metrics for a single benchmark.
//
// NsPerOp is a float64 (not int64) because the Go benchmark runner reports
// sub-microsecond timings with a decimal, e.g. "628.5 ns/op", and
// update-baseline.sh writes that value verbatim into the baseline JSON. An
// int64 field made json.Unmarshal reject every such baseline (#2455 rc#4).
type BenchmarkMetric struct {
	NsPerOp       float64 `json:"ns_per_op"`
	P50LatencyMs  float64 `json:"p50_latency_ms,omitempty"`
	P95LatencyMs  float64 `json:"p95_latency_ms,omitempty"`
	P99LatencyMs  float64 `json:"p99_latency_ms,omitempty"`
	ThroughputQPS float64 `json:"throughput_qps,omitempty"`
	AllocsPerOp   int64   `json:"allocs_per_op,omitempty"`
	BytesPerOp    int64   `json:"bytes_per_op,omitempty"`
}

// ComparisonResult represents the result of comparing current vs baseline.
//
// RegressionDetected is driven only by the hardware-independent metrics
// (allocs/op, B/op). NsPerOpChange is advisory: NsAdvisory flags when it
// exceeded its configured bound, but it never sets RegressionDetected because
// absolute time is machine-dependent.
type ComparisonResult struct {
	BenchmarkName      string
	Baseline           BenchmarkMetric
	Current            BenchmarkMetric
	NsPerOpChange      float64 // advisory (machine-dependent)
	AllocsPerOpChange  float64 // blocking
	BytesPerOpChange   float64 // blocking
	P95LatencyChange   float64
	ThroughputChange   float64
	RegressionDetected bool                // any blocking metric exceeded its threshold
	NsAdvisory         bool                // ns/op exceeded its advisory bound (reported, non-blocking)
	Thresholds         RegressionThreshold // thresholds applied to this benchmark
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

// LoadBaselineDir loads and merges every *.json baseline file in dir into one
// Baseline. update-baseline.sh writes a separate file per suite
// (classification.json, decision.json, cache.json, extproc.json, looper.json)
// and never the single baseline.json the comparison path used to read, so the
// consumer must union them (#2455 rc#1). Later files win on name collisions;
// suites are disjoint by construction, so this only matters if a benchmark is
// re-categorized.
func LoadBaselineDir(dir string) (*Baseline, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("failed to read baseline directory %s: %w", dir, err)
	}

	merged := &Baseline{Benchmarks: make(map[string]BenchmarkMetric)}
	loaded := 0
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		b, err := LoadBaseline(filepath.Join(dir, entry.Name()))
		if err != nil {
			return nil, fmt.Errorf("failed to load baseline %s: %w", entry.Name(), err)
		}
		for name, metric := range b.Benchmarks {
			merged.Benchmarks[name] = metric
		}
		// Carry the newest file's provenance so the report shows a real commit.
		if b.Timestamp.After(merged.Timestamp) {
			merged.Version = b.Version
			merged.GitCommit = b.GitCommit
			merged.Timestamp = b.Timestamp
		}
		loaded++
	}

	if loaded == 0 {
		return nil, fmt.Errorf("no baseline *.json files found in %s", dir)
	}
	return merged, nil
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

		th := getThresholdsForBenchmark(benchName, thresholds)
		result.Thresholds = th

		// ns/op is advisory: computed and flagged, but never blocks the gate,
		// because absolute time depends on the runner's hardware.
		if baselineMetric.NsPerOp > 0 {
			result.NsPerOpChange = calculatePercentChange(baselineMetric.NsPerOp, currentMetric.NsPerOp)
		}
		result.NsAdvisory = th.MaxNsRegressionPercent > 0 &&
			baselineMetric.NsPerOp > 0 &&
			result.NsPerOpChange > th.MaxNsRegressionPercent

		// allocs/op and B/op are hardware-independent, so they gate the result.
		var allocsRegressed, bytesRegressed bool
		result.AllocsPerOpChange, allocsRegressed = metricRegression(
			baselineMetric.AllocsPerOp, currentMetric.AllocsPerOp, th.MaxAllocsRegressionPercent)
		result.BytesPerOpChange, bytesRegressed = metricRegression(
			baselineMetric.BytesPerOp, currentMetric.BytesPerOp, th.MaxBytesRegressionPercent)
		result.RegressionDetected = allocsRegressed || bytesRegressed

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

// metricRegression returns the percent change from baseline to current for a
// hardware-independent integer metric (allocs/op or B/op) and whether it counts
// as a regression. A zero baseline is special: any newly-introduced allocation
// is a regression even though the percentage is undefined.
func metricRegression(baseline, current int64, thresholdPercent float64) (float64, bool) {
	if baseline == 0 {
		return 0, current > 0
	}
	change := (float64(current) - float64(baseline)) / float64(baseline) * 100
	return change, change > thresholdPercent
}

// getThresholdsForBenchmark retrieves the regression thresholds for a benchmark
// by matching its name against the configured patterns. The first matching
// entry wins, so thresholds.yaml lists them most-specific first; benchmarks
// matching nothing use the configured default (#2455 rc#3).
//
// A metric a matching entry omits (leaves at 0) inherits the default's value —
// omitting a threshold means "use the default", never "zero tolerance".
func getThresholdsForBenchmark(benchName string, thresholds *ThresholdsConfig) RegressionThreshold {
	fallback := RegressionThreshold{MaxAllocsRegressionPercent: 10, MaxBytesRegressionPercent: 10, MaxNsRegressionPercent: 30}
	if thresholds == nil {
		return fallback
	}
	base := mergeThreshold(thresholds.ComponentBenchmarks.Default, fallback)
	for _, t := range thresholds.ComponentBenchmarks.Benchmarks {
		if t.Pattern == "" {
			continue
		}
		if matched, err := regexp.MatchString(t.Pattern, benchName); err == nil && matched {
			return mergeThreshold(t.RegressionThreshold, base)
		}
	}
	return base
}

// mergeThreshold returns primary with any unset (zero) metric filled from
// fallback, so an omitted YAML field inherits the default rather than becoming
// a 0% (zero-tolerance) bound.
func mergeThreshold(primary, fallback RegressionThreshold) RegressionThreshold {
	if primary.MaxAllocsRegressionPercent == 0 {
		primary.MaxAllocsRegressionPercent = fallback.MaxAllocsRegressionPercent
	}
	if primary.MaxBytesRegressionPercent == 0 {
		primary.MaxBytesRegressionPercent = fallback.MaxBytesRegressionPercent
	}
	if primary.MaxNsRegressionPercent == 0 {
		primary.MaxNsRegressionPercent = fallback.MaxNsRegressionPercent
	}
	return primary
}

// UngatedBenchmarks returns the names (sorted) of current benchmarks that have
// no baseline entry and were therefore not compared. Surfacing these keeps a
// gate pass honest: an empty or missing baseline suite is excluded, not checked.
func UngatedBenchmarks(current, baseline *Baseline) []string {
	return missingKeys(current.Benchmarks, baseline.Benchmarks)
}

// MissingBenchmarks returns the names (sorted) of baseline benchmarks that have
// no current result — benchmarks that were expected but did not run (a crashed
// suite, a rename, or a filtered run). Without surfacing these, a benchmark that
// disappears from the run is silently skipped and could pass the gate unnoticed.
func MissingBenchmarks(current, baseline *Baseline) []string {
	return missingKeys(baseline.Benchmarks, current.Benchmarks)
}

// missingKeys returns the keys of have that are absent from other, sorted.
func missingKeys(have, other map[string]BenchmarkMetric) []string {
	var names []string
	for name := range have {
		if _, ok := other[name]; !ok {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
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
	fmt.Println("        (allocs/op and B/op gate the result; ns/op is advisory only)")
	fmt.Println("===================================================================================")
	fmt.Printf("%-50s %-15s %-15s %-15s\n", "Benchmark / metric", "Baseline", "Current", "Change")
	fmt.Println("-----------------------------------------------------------------------------------")

	for _, result := range results {
		icon := "✓"
		if result.RegressionDetected {
			icon = "⚠️"
		}

		// Blocking metric: allocs/op (hardware-independent).
		fmt.Printf("%s %-48s %-15d %-15d %+.2f%%\n",
			icon,
			result.BenchmarkName,
			result.Baseline.AllocsPerOp,
			result.Current.AllocsPerOp,
			result.AllocsPerOpChange,
		)

		// Blocking metric: B/op (hardware-independent).
		fmt.Printf("  └─ B/op:        %-15d %-15d %+.2f%%\n",
			result.Baseline.BytesPerOp,
			result.Current.BytesPerOp,
			result.BytesPerOpChange,
		)

		// Advisory metric: ns/op (machine-dependent, never blocks).
		advisory := ""
		if result.NsAdvisory {
			advisory = "  ⚠️ advisory"
		}
		fmt.Printf("  └─ ns/op (adv): %-15.0f %-15.0f %+.2f%%%s\n",
			result.Baseline.NsPerOp,
			result.Current.NsPerOp,
			result.NsPerOpChange,
			advisory,
		)
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
		fmt.Printf("\nNo regressions detected\n")
	}
}
