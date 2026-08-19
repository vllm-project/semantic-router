package benchmark

import (
	"os"
	"path/filepath"
	"testing"
)

// TestLoadBaseline_ParsesDecimalNsPerOp guards #2455 root cause #4: the Go
// benchmark runner emits sub-microsecond timings as decimals (e.g. "628.5
// ns/op"). The metric field must round-trip a decimal ns/op without erroring.
func TestLoadBaseline_ParsesDecimalNsPerOp(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	content := `{
  "version": "v1.0.0",
  "benchmarks": {
    "BenchmarkDecimal": {"iterations": 100, "ns_per_op": 628.5, "bytes_per_op": 112, "allocs_per_op": 5}
  }
}`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	b, err := LoadBaseline(path)
	if err != nil {
		t.Fatalf("LoadBaseline rejected a decimal ns_per_op: %v", err)
	}

	got := float64(b.Benchmarks["BenchmarkDecimal"].NsPerOp)
	if got != 628.5 {
		t.Fatalf("round-trip NsPerOp = %v, want 628.5", got)
	}
}

func TestCompareWithBaselineGatesExternalLatencyAndEfficiency(t *testing.T) {
	baseline := &Baseline{Benchmarks: map[string]BenchmarkMetric{
		"routed/batch=8/context=16384": {
			Suite: "gpu-serving", P95LatencyMs: 100, ThroughputQPS: 80,
			UpstreamCalls: 1, TokenAmplification: 1,
		},
	}}
	current := &Baseline{Benchmarks: map[string]BenchmarkMetric{
		"routed/batch=8/context=16384": {
			Suite: "gpu-serving", P50LatencyMs: 80, P95LatencyMs: 125, P99LatencyMs: 150,
			ThroughputQPS: 68, UpstreamCalls: 1.2, TokenAmplification: 1.2,
		},
	}}

	results, err := CompareWithBaseline(current, baseline, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || !results[0].RegressionDetected {
		t.Fatalf("external regression was not gated: %+v", results)
	}
	if results[0].P95LatencyChange != 25 || results[0].ThroughputChange != -15 {
		t.Fatalf("external changes = p95 %.1f, throughput %.1f", results[0].P95LatencyChange, results[0].ThroughputChange)
	}
}

func TestCompareWithBaselineFailsClosedOnMissingExternalMetrics(t *testing.T) {
	tests := []struct {
		name     string
		baseline BenchmarkMetric
	}{
		{name: "p95 latency", baseline: BenchmarkMetric{P95LatencyMs: 100}},
		{name: "throughput", baseline: BenchmarkMetric{ThroughputQPS: 80}},
		{name: "upstream calls", baseline: BenchmarkMetric{UpstreamCalls: 1}},
		{name: "token amplification", baseline: BenchmarkMetric{TokenAmplification: 1}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			baseline := &Baseline{Benchmarks: map[string]BenchmarkMetric{"external": test.baseline}}
			current := &Baseline{Benchmarks: map[string]BenchmarkMetric{"external": {}}}
			results, err := CompareWithBaseline(current, baseline, nil)
			if err != nil {
				t.Fatal(err)
			}
			if len(results) != 1 || !results[0].RegressionDetected {
				t.Fatalf("missing %s was not gated: %+v", test.name, results)
			}
		})
	}
}
