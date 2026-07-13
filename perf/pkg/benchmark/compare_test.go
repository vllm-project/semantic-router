package benchmark

import "testing"

// TestCompareWithBaseline_UsesPerBenchmarkThreshold proves the comparison flags
// a regression against each benchmark's OWN threshold: a +10% decision-engine
// slowdown breaches its 5% bound, while the same +10% on a looper micro-bench
// stays within its looser 25% bound. This is the behavior the CI gate rides on.
func TestCompareWithBaseline_UsesPerBenchmarkThreshold(t *testing.T) {
	thresholds := testThresholds() // default 10, decision 5, looper 25

	baseline := &Baseline{Benchmarks: map[string]BenchmarkMetric{
		"BenchmarkEvaluateDecisions_SingleDomain": {NsPerOp: 100},
		"BenchmarkReMoM_Execute/1x4":              {NsPerOp: 100},
	}}
	current := &Baseline{Benchmarks: map[string]BenchmarkMetric{
		"BenchmarkEvaluateDecisions_SingleDomain": {NsPerOp: 110}, // +10% vs 5% bound
		"BenchmarkReMoM_Execute/1x4":              {NsPerOp: 110}, // +10% vs 25% bound
	}}

	results, err := CompareWithBaseline(current, baseline, thresholds)
	if err != nil {
		t.Fatalf("CompareWithBaseline: %v", err)
	}

	regressed := make(map[string]bool, len(results))
	for _, r := range results {
		regressed[r.BenchmarkName] = r.RegressionDetected
	}
	if !regressed["BenchmarkEvaluateDecisions_SingleDomain"] {
		t.Error("decision-engine +10% should regress against its 5% threshold")
	}
	if regressed["BenchmarkReMoM_Execute/1x4"] {
		t.Error("looper +10% should be within its 25% threshold")
	}
	if !HasRegressions(results) {
		t.Error("HasRegressions should report true when any benchmark regressed")
	}
}
