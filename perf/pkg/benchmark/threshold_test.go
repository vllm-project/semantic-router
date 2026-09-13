package benchmark

import (
	"path/filepath"
	"testing"
)

func testThresholds() *ThresholdsConfig {
	return &ThresholdsConfig{
		ComponentBenchmarks: ComponentBenchmarksThresholds{
			Default: RegressionThreshold{MaxAllocsRegressionPercent: 10, MaxBytesRegressionPercent: 10},
			Benchmarks: []BenchmarkRegressionThreshold{
				{Name: "classification", Pattern: "^BenchmarkClassify",
					RegressionThreshold: RegressionThreshold{MaxAllocsRegressionPercent: 10, MaxBytesRegressionPercent: 10, MaxNsRegressionPercent: 30}},
				{Name: "decision_engine", Pattern: "^Benchmark(EvaluateDecisions|PrioritySelection)",
					RegressionThreshold: RegressionThreshold{MaxAllocsRegressionPercent: 5, MaxBytesRegressionPercent: 5, MaxNsRegressionPercent: 20}},
				{Name: "looper", Pattern: "^Benchmark(ReMoM|Fusion|Flow|Base)",
					RegressionThreshold: RegressionThreshold{MaxAllocsRegressionPercent: 10, MaxBytesRegressionPercent: 15, MaxNsRegressionPercent: 40}},
			},
		},
	}
}

// TestGetThresholdsForBenchmark guards #2455 root cause #3: thresholds are
// selected by the benchmark's own name, not the first entry in the config. The
// decision engine keeps its own tight 5% allocs bound rather than inheriting
// classification's 10%.
func TestGetThresholdsForBenchmark(t *testing.T) {
	cfg := testThresholds()
	cases := []struct {
		name       string
		bench      string
		wantAllocs float64
	}{
		{"classification", "BenchmarkClassifyText", 10},
		{"decision keeps its own tight bound", "BenchmarkEvaluateDecisions_SingleDomain", 5},
		{"looper subtest name matches", "BenchmarkReMoM_Execute/1x4", 10},
		{"unmatched benchmark falls back to default", "BenchmarkBrandNewThing", 10},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := getThresholdsForBenchmark(c.bench, cfg).MaxAllocsRegressionPercent; got != c.wantAllocs {
				t.Errorf("getThresholdsForBenchmark(%q).MaxAllocsRegressionPercent = %v, want %v", c.bench, got, c.wantAllocs)
			}
		})
	}
}

// TestGetThresholdsForBenchmark_OmittedFieldInheritsDefault guards the config
// footgun: an entry that sets only some metrics must inherit the rest from the
// default, not silently get a zero-tolerance (0%) bound.
func TestGetThresholdsForBenchmark_OmittedFieldInheritsDefault(t *testing.T) {
	cfg := &ThresholdsConfig{
		ComponentBenchmarks: ComponentBenchmarksThresholds{
			Default: RegressionThreshold{MaxAllocsRegressionPercent: 10, MaxBytesRegressionPercent: 12, MaxNsRegressionPercent: 30},
			Benchmarks: []BenchmarkRegressionThreshold{
				// Only allocs set; bytes and ns omitted must inherit the default.
				{Name: "x", Pattern: "^BenchmarkX", RegressionThreshold: RegressionThreshold{MaxAllocsRegressionPercent: 3}},
			},
		},
	}
	th := getThresholdsForBenchmark("BenchmarkX", cfg)
	if th.MaxAllocsRegressionPercent != 3 {
		t.Errorf("allocs = %v, want 3 (explicit)", th.MaxAllocsRegressionPercent)
	}
	if th.MaxBytesRegressionPercent != 12 {
		t.Errorf("bytes = %v, want 12 (inherited from default, not 0)", th.MaxBytesRegressionPercent)
	}
	if th.MaxNsRegressionPercent != 30 {
		t.Errorf("ns = %v, want 30 (inherited from default, not 0)", th.MaxNsRegressionPercent)
	}
}

func TestGetThresholdsForBenchmark_NilConfig(t *testing.T) {
	th := getThresholdsForBenchmark("BenchmarkAnything", nil)
	if th.MaxAllocsRegressionPercent != 10 || th.MaxBytesRegressionPercent != 10 {
		t.Errorf("nil-config thresholds = %+v, want allocs=bytes=10", th)
	}
}

// TestLoadThresholds_ShippedConfigMapsEverySuite loads the real shipped
// thresholds.yaml and asserts the decision-engine tight bound and the looper
// block (previously silently dropped) both map, locking the YAML to the struct.
func TestLoadThresholds_ShippedConfigMapsEverySuite(t *testing.T) {
	cfg, err := LoadThresholds(filepath.Join("..", "..", "config", "thresholds.yaml"))
	if err != nil {
		t.Fatalf("LoadThresholds: %v", err)
	}
	if got := getThresholdsForBenchmark("BenchmarkEvaluateDecisions_SingleDomain", cfg).MaxAllocsRegressionPercent; got != 5 {
		t.Errorf("shipped config: decision allocs bound = %v, want 5", got)
	}
	looper := getThresholdsForBenchmark("BenchmarkReMoM_DistributeRoundRobin", cfg)
	if looper.MaxAllocsRegressionPercent == 0 {
		t.Error("shipped config: looper allocs bound is 0 — the looper block is not mapped")
	}
	if looper.MaxNsRegressionPercent == 0 {
		t.Error("shipped config: looper ns/op advisory bound is 0 — advisory not carried")
	}
}
