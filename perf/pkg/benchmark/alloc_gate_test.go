package benchmark

import "testing"

// TestCompareWithBaseline_GatesOnAllocsNotTime is the core of the option-A
// design: the gate blocks on hardware-independent metrics (allocs/op, B/op) and
// treats ns/op as advisory. Absolute time is machine-dependent, so a slower
// wall-clock must not fail the gate; a change in allocations must.
func TestCompareWithBaseline_GatesOnAllocsNotTime(t *testing.T) {
	cfg := testThresholds() // decision: allocs/bytes 5%, ns 20% (advisory)
	base := BenchmarkMetric{NsPerOp: 100, AllocsPerOp: 10, BytesPerOp: 100}

	cases := []struct {
		name      string
		current   BenchmarkMetric
		wantBlock bool
	}{
		{"time doubles but allocs+bytes flat -> not blocking",
			BenchmarkMetric{NsPerOp: 200, AllocsPerOp: 10, BytesPerOp: 100}, false},
		{"allocs +20% over its 5% bound -> blocks",
			BenchmarkMetric{NsPerOp: 100, AllocsPerOp: 12, BytesPerOp: 100}, true},
		{"bytes +20% over its 5% bound -> blocks",
			BenchmarkMetric{NsPerOp: 100, AllocsPerOp: 10, BytesPerOp: 120}, true},
		{"everything flat -> clean",
			BenchmarkMetric{NsPerOp: 100, AllocsPerOp: 10, BytesPerOp: 100}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cur := &Baseline{Benchmarks: map[string]BenchmarkMetric{"BenchmarkEvaluateDecisions_X": c.current}}
			bl := &Baseline{Benchmarks: map[string]BenchmarkMetric{"BenchmarkEvaluateDecisions_X": base}}
			results, err := CompareWithBaseline(cur, bl, cfg)
			if err != nil {
				t.Fatalf("CompareWithBaseline: %v", err)
			}
			if len(results) != 1 {
				t.Fatalf("want 1 result, got %d", len(results))
			}
			if got := results[0].RegressionDetected; got != c.wantBlock {
				t.Errorf("RegressionDetected = %v, want %v", got, c.wantBlock)
			}
		})
	}
}

// TestCompareWithBaseline_NewAllocationsAreRegression: going from zero
// allocations to some is a real regression even though the percentage is
// undefined, so a zero baseline is handled explicitly.
func TestCompareWithBaseline_NewAllocationsAreRegression(t *testing.T) {
	cfg := testThresholds()
	bl := &Baseline{Benchmarks: map[string]BenchmarkMetric{"BenchmarkEvaluateDecisions_Zero": {NsPerOp: 50, AllocsPerOp: 0}}}
	cur := &Baseline{Benchmarks: map[string]BenchmarkMetric{"BenchmarkEvaluateDecisions_Zero": {NsPerOp: 50, AllocsPerOp: 3}}}
	results, err := CompareWithBaseline(cur, bl, cfg)
	if err != nil {
		t.Fatalf("CompareWithBaseline: %v", err)
	}
	if !results[0].RegressionDetected {
		t.Error("introducing allocations (0 -> 3) should be a regression")
	}
}

// TestCompareWithBaseline_NsAdvisoryFlaggedNotBlocking: a ns/op regression past
// its advisory bound is surfaced (NsAdvisory) but never blocks the gate.
func TestCompareWithBaseline_NsAdvisoryFlaggedNotBlocking(t *testing.T) {
	cfg := testThresholds() // decision ns advisory 20%
	bl := &Baseline{Benchmarks: map[string]BenchmarkMetric{"BenchmarkEvaluateDecisions_Slow": {NsPerOp: 100, AllocsPerOp: 10, BytesPerOp: 100}}}
	cur := &Baseline{Benchmarks: map[string]BenchmarkMetric{"BenchmarkEvaluateDecisions_Slow": {NsPerOp: 200, AllocsPerOp: 10, BytesPerOp: 100}}}
	results, err := CompareWithBaseline(cur, bl, cfg)
	if err != nil {
		t.Fatalf("CompareWithBaseline: %v", err)
	}
	if results[0].RegressionDetected {
		t.Error("a ns/op-only regression must not block the gate")
	}
	if !results[0].NsAdvisory {
		t.Error("ns/op +100% over its 20% advisory bound should set NsAdvisory")
	}
}
