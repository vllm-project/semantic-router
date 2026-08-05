package benchmark

import "testing"

// TestCompareWithBaseline_UsesPerBenchmarkThreshold proves the gate flags a
// regression against each benchmark's OWN allocs bound: +8% allocations breaches
// the decision engine's tight 5% bound but stays within the looper's 10% bound.
func TestCompareWithBaseline_UsesPerBenchmarkThreshold(t *testing.T) {
	th := testThresholds() // decision allocs 5%, looper allocs 10%

	baseline := &Baseline{Benchmarks: map[string]BenchmarkMetric{
		"BenchmarkEvaluateDecisions_SingleDomain": {AllocsPerOp: 100, BytesPerOp: 100},
		"BenchmarkReMoM_Execute/1x4":              {AllocsPerOp: 100, BytesPerOp: 100},
	}}
	current := &Baseline{Benchmarks: map[string]BenchmarkMetric{
		"BenchmarkEvaluateDecisions_SingleDomain": {AllocsPerOp: 108, BytesPerOp: 100}, // +8% vs 5% bound
		"BenchmarkReMoM_Execute/1x4":              {AllocsPerOp: 108, BytesPerOp: 100}, // +8% vs 10% bound
	}}

	results, err := CompareWithBaseline(current, baseline, th)
	if err != nil {
		t.Fatalf("CompareWithBaseline: %v", err)
	}
	regressed := make(map[string]bool, len(results))
	for _, r := range results {
		regressed[r.BenchmarkName] = r.RegressionDetected
	}
	if !regressed["BenchmarkEvaluateDecisions_SingleDomain"] {
		t.Error("decision-engine allocs +8% should regress against its 5% bound")
	}
	if regressed["BenchmarkReMoM_Execute/1x4"] {
		t.Error("looper allocs +8% should be within its 10% bound")
	}
	if !HasRegressions(results) {
		t.Error("HasRegressions should report true when any benchmark regressed")
	}
}
