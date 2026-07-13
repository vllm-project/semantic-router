package benchmark

import (
	"path/filepath"
	"testing"
)

func testThresholds() *ThresholdsConfig {
	return &ThresholdsConfig{
		ComponentBenchmarks: ComponentBenchmarksThresholds{
			Default: RegressionThreshold{MaxRegressionPercent: 10},
			Benchmarks: []BenchmarkRegressionThreshold{
				{Name: "classification", Pattern: "^BenchmarkClassify", MaxRegressionPercent: 15},
				{Name: "decision_engine", Pattern: "^Benchmark(EvaluateDecisions|PrioritySelection)", MaxRegressionPercent: 5},
				{Name: "looper", Pattern: "^Benchmark(ReMoM|Fusion|Flow|Base)", MaxRegressionPercent: 25},
			},
		},
	}
}

// TestGetThresholdForBenchmark guards #2455 root cause #3: the threshold must be
// selected by the benchmark's own name, not the first non-zero entry in the
// config. In particular a decision-engine benchmark must resolve to its own 5%,
// never the classification 15% that the old code always returned.
func TestGetThresholdForBenchmark(t *testing.T) {
	cfg := testThresholds()
	cases := []struct {
		name  string
		bench string
		want  float64
	}{
		{"classification", "BenchmarkClassifyText", 15},
		{"decision resolves to its own bound, not the first-listed one", "BenchmarkEvaluateDecisions_SingleDomain", 5},
		{"priority shares the decision bound", "BenchmarkPrioritySelection", 5},
		{"looper subtest name matches", "BenchmarkReMoM_Execute/1x4", 25},
		{"unmatched benchmark falls back to default", "BenchmarkBrandNewThing", 10},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := getThresholdForBenchmark(c.bench, cfg); got != c.want {
				t.Errorf("getThresholdForBenchmark(%q) = %v, want %v", c.bench, got, c.want)
			}
		})
	}
}

func TestGetThresholdForBenchmark_NilConfig(t *testing.T) {
	if got := getThresholdForBenchmark("BenchmarkAnything", nil); got != 10 {
		t.Errorf("nil-config threshold = %v, want 10", got)
	}
}

// TestLoadThresholds_ShippedConfigMapsEverySuite loads the real shipped
// thresholds.yaml and asserts each suite — including looper, which the previous
// struct silently dropped — resolves to its configured bound. This locks the
// YAML and the Go struct together.
func TestLoadThresholds_ShippedConfigMapsEverySuite(t *testing.T) {
	cfg, err := LoadThresholds(filepath.Join("..", "..", "config", "thresholds.yaml"))
	if err != nil {
		t.Fatalf("LoadThresholds: %v", err)
	}
	cases := map[string]float64{
		"BenchmarkClassifyText":                   15,
		"BenchmarkEvaluateDecisions_SingleDomain": 5,
		"BenchmarkCacheSearch":                    10,
		"BenchmarkReMoM_DistributeRoundRobin":     25,
	}
	for bench, want := range cases {
		if got := getThresholdForBenchmark(bench, cfg); got != want {
			t.Errorf("shipped config: %s -> %v, want %v", bench, got, want)
		}
	}
}
