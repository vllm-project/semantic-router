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

const ResultSchemaVersion = 2

// Baseline is the canonical result-set schema. The same shape is used for a
// live run and for a reviewed baseline so external CPU/GPU producers can join
// the comparison pipeline without a second report format.
type Baseline struct {
	SchemaVersion int                        `json:"schema_version"`
	Metadata      RunMetadata                `json:"metadata"`
	Benchmarks    map[string]BenchmarkMetric `json:"benchmarks"`

	// Legacy provenance fields remain readable while old baselines are being
	// retired. New result sets write Metadata instead.
	Version   string    `json:"version,omitempty"`
	GitCommit string    `json:"git_commit,omitempty"`
	Timestamp time.Time `json:"timestamp,omitempty"`
}

type RunMetadata struct {
	GeneratedAt     time.Time          `json:"generated_at"`
	GitCommit       string             `json:"git_commit"`
	GitBranch       string             `json:"git_branch"`
	Environment     string             `json:"environment"`
	EnvironmentKind string             `json:"environment_kind"`
	Accelerator     string             `json:"accelerator,omitempty"`
	Profile         string             `json:"profile"`
	GoVersion       string             `json:"go_version"`
	GOOS            string             `json:"goos"`
	GOARCH          string             `json:"goarch"`
	CPUModel        string             `json:"cpu_model,omitempty"`
	CPUCount        int                `json:"cpu_count"`
	Suites          []SuiteRunMetadata `json:"suites"`
	Trends          []TrendMetadata    `json:"trends,omitempty"`
}

type TrendMetadata struct {
	Name            string `json:"name"`
	Title           string `json:"title"`
	Description     string `json:"description,omitempty"`
	Suite           string `json:"suite"`
	Benchmark       string `json:"benchmark"`
	XDimension      string `json:"x_dimension"`
	SeriesDimension string `json:"series_dimension,omitempty"`
	Metric          string `json:"metric"`
	XScale          string `json:"x_scale,omitempty"`
}

type SuiteRunMetadata struct {
	Name            string              `json:"name"`
	Runner          string              `json:"runner"`
	DurationSeconds float64             `json:"duration_seconds"`
	BenchmarkCount  int                 `json:"benchmark_count"`
	Dimensions      map[string][]string `json:"dimensions,omitempty"`
}

type BenchmarkMetric struct {
	Suite              string             `json:"suite,omitempty"`
	Dimensions         map[string]string  `json:"dimensions,omitempty"`
	Iterations         int64              `json:"iterations,omitempty"`
	Samples            int                `json:"samples,omitempty"`
	NsPerOp            float64            `json:"ns_per_op"`
	NsStdDevPct        float64            `json:"ns_stddev_percent,omitempty"`
	AllocsPerOp        int64              `json:"allocs_per_op"`
	BytesPerOp         int64              `json:"bytes_per_op"`
	Custom             map[string]float64 `json:"custom,omitempty"`
	P50LatencyMs       float64            `json:"p50_latency_ms,omitempty"`
	P95LatencyMs       float64            `json:"p95_latency_ms,omitempty"`
	P99LatencyMs       float64            `json:"p99_latency_ms,omitempty"`
	ThroughputQPS      float64            `json:"throughput_qps,omitempty"`
	UpstreamCalls      float64            `json:"upstream_calls,omitempty"`
	InputTokens        float64            `json:"input_tokens,omitempty"`
	OutputTokens       float64            `json:"output_tokens,omitempty"`
	TokenAmplification float64            `json:"token_amplification,omitempty"`
}

type ComparisonResult struct {
	BenchmarkName            string              `json:"benchmark_name"`
	Suite                    string              `json:"suite"`
	Baseline                 BenchmarkMetric     `json:"baseline"`
	Current                  BenchmarkMetric     `json:"current"`
	NsPerOpChange            float64             `json:"ns_per_op_change"`
	AllocsPerOpChange        float64             `json:"allocs_per_op_change"`
	BytesPerOpChange         float64             `json:"bytes_per_op_change"`
	P95LatencyChange         float64             `json:"p95_latency_change,omitempty"`
	ThroughputChange         float64             `json:"throughput_change,omitempty"`
	UpstreamCallsChange      float64             `json:"upstream_calls_change,omitempty"`
	TokenAmplificationChange float64             `json:"token_amplification_change,omitempty"`
	RegressionDetected       bool                `json:"regression_detected"`
	NsAdvisory               bool                `json:"ns_advisory"`
	Thresholds               RegressionThreshold `json:"thresholds"`
}

type ComparisonDocument struct {
	SchemaVersion     int                        `json:"schema_version"`
	GeneratedAt       time.Time                  `json:"generated_at"`
	BaselineMetadata  RunMetadata                `json:"baseline_metadata"`
	CurrentMetadata   RunMetadata                `json:"current_metadata"`
	Results           []ComparisonResult         `json:"results"`
	CurrentBenchmarks map[string]BenchmarkMetric `json:"current_benchmarks,omitempty"`
	Ungated           []string                   `json:"ungated_benchmarks"`
	Missing           []string                   `json:"missing_benchmarks"`
	HasRegressions    bool                       `json:"has_regressions"`
	CoverageComplete  bool                       `json:"coverage_complete"`
}

func LoadBaseline(path string) (*Baseline, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read result set %s: %w", path, err)
	}
	var baseline Baseline
	if err := json.Unmarshal(data, &baseline); err != nil {
		return nil, fmt.Errorf("parse result set %s: %w", path, err)
	}
	if baseline.Benchmarks == nil {
		return nil, fmt.Errorf("result set %s has no benchmarks map", path)
	}
	if baseline.Metadata.GitCommit == "" {
		baseline.Metadata.GitCommit = baseline.GitCommit
	}
	if baseline.Metadata.GeneratedAt.IsZero() {
		baseline.Metadata.GeneratedAt = baseline.Timestamp
	}
	return &baseline, nil
}

func LoadBaselineDir(dir string) (*Baseline, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read baseline directory %s: %w", dir, err)
	}
	merged := &Baseline{SchemaVersion: ResultSchemaVersion, Benchmarks: make(map[string]BenchmarkMetric)}
	loaded := 0
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		baseline, err := LoadBaseline(filepath.Join(dir, entry.Name()))
		if err != nil {
			return nil, err
		}
		for name, metric := range baseline.Benchmarks {
			if _, duplicate := merged.Benchmarks[name]; duplicate {
				return nil, fmt.Errorf("duplicate benchmark %q across baseline files", name)
			}
			merged.Benchmarks[name] = metric
		}
		if baseline.Metadata.GeneratedAt.After(merged.Metadata.GeneratedAt) {
			merged.Metadata = baseline.Metadata
		}
		loaded++
	}
	if loaded == 0 {
		return nil, fmt.Errorf("no baseline *.json files found in %s", dir)
	}
	return merged, nil
}

func SaveBaseline(baseline *Baseline, path string) error {
	if baseline.SchemaVersion == 0 {
		baseline.SchemaVersion = ResultSchemaVersion
	}
	return saveJSON(path, baseline)
}

func SaveComparison(comparison *ComparisonDocument, path string) error {
	return saveJSON(path, comparison)
}

func saveJSON(path string, value any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create output directory: %w", err)
	}
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal %s: %w", path, err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}

func BuildComparison(current, baseline *Baseline, thresholds *ThresholdsConfig) (*ComparisonDocument, error) {
	if baseline.Metadata.Environment != "" && current.Metadata.Environment != "" &&
		baseline.Metadata.Environment != current.Metadata.Environment {
		return nil, fmt.Errorf("cannot compare environment %q with baseline environment %q", current.Metadata.Environment, baseline.Metadata.Environment)
	}
	results, err := CompareWithBaseline(current, baseline, thresholds)
	if err != nil {
		return nil, err
	}
	expectedBaseline := baselineForCurrentSuites(baseline, current)
	ungated := UngatedBenchmarks(current, expectedBaseline)
	missing := MissingBenchmarks(current, expectedBaseline)
	return &ComparisonDocument{
		SchemaVersion:     ResultSchemaVersion,
		GeneratedAt:       time.Now().UTC(),
		BaselineMetadata:  baseline.Metadata,
		CurrentMetadata:   current.Metadata,
		Results:           results,
		CurrentBenchmarks: current.Benchmarks,
		Ungated:           ungated,
		Missing:           missing,
		HasRegressions:    HasRegressions(results),
		CoverageComplete:  len(ungated) == 0 && len(missing) == 0,
	}, nil
}

func baselineForCurrentSuites(baseline, current *Baseline) *Baseline {
	selected := make(map[string]struct{}, len(current.Metadata.Suites))
	for _, suite := range current.Metadata.Suites {
		selected[suite.Name] = struct{}{}
	}
	if len(selected) == 0 {
		return baseline
	}
	filtered := &Baseline{Metadata: baseline.Metadata, Benchmarks: make(map[string]BenchmarkMetric)}
	for name, metric := range baseline.Benchmarks {
		if _, ok := selected[metric.Suite]; ok {
			filtered.Benchmarks[name] = metric
		}
	}
	return filtered
}

func CompareWithBaseline(current, baseline *Baseline, thresholds *ThresholdsConfig) ([]ComparisonResult, error) {
	expectedBaseline := baselineForCurrentSuites(baseline, current)
	results := make([]ComparisonResult, 0, len(current.Benchmarks))
	for benchmarkName, currentMetric := range current.Benchmarks {
		baselineMetric, exists := expectedBaseline.Benchmarks[benchmarkName]
		if !exists {
			continue
		}
		threshold := getThresholdsForBenchmark(benchmarkName, thresholds)
		results = append(results, compareBenchmark(benchmarkName, currentMetric, baselineMetric, threshold))
	}
	sort.Slice(results, func(i, j int) bool { return results[i].BenchmarkName < results[j].BenchmarkName })
	return results, nil
}

func compareBenchmark(
	benchmarkName string,
	current, baseline BenchmarkMetric,
	threshold RegressionThreshold,
) ComparisonResult {
	result := ComparisonResult{
		BenchmarkName: benchmarkName,
		Suite:         current.Suite,
		Baseline:      baseline,
		Current:       current,
		Thresholds:    threshold,
	}
	if baseline.NsPerOp > 0 {
		result.NsPerOpChange = calculatePercentChange(baseline.NsPerOp, current.NsPerOp)
	}
	result.NsAdvisory = threshold.MaxNsRegressionPercent > 0 &&
		baseline.NsPerOp > 0 && result.NsPerOpChange > threshold.MaxNsRegressionPercent

	var allocationsRegressed, bytesRegressed bool
	result.AllocsPerOpChange, allocationsRegressed = metricRegression(
		baseline.AllocsPerOp, current.AllocsPerOp, threshold.MaxAllocsRegressionPercent,
	)
	result.BytesPerOpChange, bytesRegressed = metricRegression(
		baseline.BytesPerOp, current.BytesPerOp, threshold.MaxBytesRegressionPercent,
	)
	externalRegressed := compareExternalMetrics(&result, current, baseline, threshold)
	result.RegressionDetected = allocationsRegressed || bytesRegressed || externalRegressed ||
		(threshold.GateNsPerOp && result.NsAdvisory)
	return result
}

func compareExternalMetrics(
	result *ComparisonResult,
	current, baseline BenchmarkMetric,
	threshold RegressionThreshold,
) bool {
	regressed := false
	if baseline.P95LatencyMs > 0 {
		if current.P95LatencyMs <= 0 {
			regressed = true
		} else {
			result.P95LatencyChange = calculatePercentChange(baseline.P95LatencyMs, current.P95LatencyMs)
			regressed = result.P95LatencyChange > threshold.MaxP95LatencyRegressionPercent
		}
	}
	if baseline.ThroughputQPS > 0 {
		if current.ThroughputQPS <= 0 {
			regressed = true
		} else {
			result.ThroughputChange = calculatePercentChange(baseline.ThroughputQPS, current.ThroughputQPS)
			regressed = regressed || result.ThroughputChange < -threshold.MaxThroughputRegressionPercent
		}
	}
	if baseline.UpstreamCalls > 0 {
		if current.UpstreamCalls <= 0 {
			regressed = true
		} else {
			result.UpstreamCallsChange = calculatePercentChange(baseline.UpstreamCalls, current.UpstreamCalls)
			regressed = regressed || result.UpstreamCallsChange > threshold.MaxUpstreamCallsRegressionPercent
		}
	}
	if baseline.TokenAmplification > 0 {
		if current.TokenAmplification <= 0 {
			regressed = true
		} else {
			result.TokenAmplificationChange = calculatePercentChange(baseline.TokenAmplification, current.TokenAmplification)
			regressed = regressed || result.TokenAmplificationChange > threshold.MaxTokenAmplificationRegressionPercent
		}
	}
	return regressed
}

func calculatePercentChange(baseline, current float64) float64 {
	if baseline == 0 {
		return 0
	}
	return ((current - baseline) / baseline) * 100
}

func metricRegression(baseline, current int64, thresholdPercent float64) (float64, bool) {
	if baseline == 0 {
		return 0, current > 0
	}
	change := (float64(current) - float64(baseline)) / float64(baseline) * 100
	return change, change > thresholdPercent
}

func getThresholdsForBenchmark(benchmarkName string, thresholds *ThresholdsConfig) RegressionThreshold {
	fallback := RegressionThreshold{
		MaxAllocsRegressionPercent:             10,
		MaxBytesRegressionPercent:              10,
		MaxNsRegressionPercent:                 30,
		MaxP95LatencyRegressionPercent:         20,
		MaxThroughputRegressionPercent:         10,
		MaxUpstreamCallsRegressionPercent:      10,
		MaxTokenAmplificationRegressionPercent: 10,
	}
	if thresholds == nil {
		return fallback
	}
	base := mergeThreshold(thresholds.ComponentBenchmarks.Default, fallback)
	for _, threshold := range thresholds.ComponentBenchmarks.Benchmarks {
		matched, err := regexp.MatchString(threshold.Pattern, benchmarkName)
		if err == nil && matched {
			return mergeThreshold(threshold.RegressionThreshold, base)
		}
	}
	return base
}

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
	if primary.MaxP95LatencyRegressionPercent == 0 {
		primary.MaxP95LatencyRegressionPercent = fallback.MaxP95LatencyRegressionPercent
	}
	if primary.MaxThroughputRegressionPercent == 0 {
		primary.MaxThroughputRegressionPercent = fallback.MaxThroughputRegressionPercent
	}
	if primary.MaxUpstreamCallsRegressionPercent == 0 {
		primary.MaxUpstreamCallsRegressionPercent = fallback.MaxUpstreamCallsRegressionPercent
	}
	if primary.MaxTokenAmplificationRegressionPercent == 0 {
		primary.MaxTokenAmplificationRegressionPercent = fallback.MaxTokenAmplificationRegressionPercent
	}
	return primary
}

func UngatedBenchmarks(current, baseline *Baseline) []string {
	return missingKeys(current.Benchmarks, baseline.Benchmarks)
}

func MissingBenchmarks(current, baseline *Baseline) []string {
	return missingKeys(baseline.Benchmarks, current.Benchmarks)
}

func missingKeys(have, other map[string]BenchmarkMetric) []string {
	names := make([]string, 0)
	for name := range have {
		if _, ok := other[name]; !ok {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}

func HasRegressions(results []ComparisonResult) bool {
	for _, result := range results {
		if result.RegressionDetected {
			return true
		}
	}
	return false
}

func PrintComparisonResults(comparison *ComparisonDocument) {
	fmt.Printf("\n%-52s %12s %12s %10s %10s\n", "benchmark", "base ns/op", "head ns/op", "allocs", "B/op")
	fmt.Println(strings.Repeat("-", 104))
	for _, result := range comparison.Results {
		status := "ok"
		if result.RegressionDetected {
			status = "REGRESSION"
		} else if result.NsAdvisory {
			status = "timing?"
		}
		fmt.Printf("%-52s %12.1f %12.1f %+9.1f%% %+9.1f%%  %s\n",
			result.BenchmarkName, result.Baseline.NsPerOp, result.Current.NsPerOp,
			result.AllocsPerOpChange, result.BytesPerOpChange, status)
	}
	if !comparison.CoverageComplete {
		fmt.Printf("coverage incomplete: %d unbaselined, %d missing\n", len(comparison.Ungated), len(comparison.Missing))
	}
	if comparison.HasRegressions {
		fmt.Println("performance regression detected")
	} else {
		fmt.Println("no blocking performance regression detected")
	}
}
