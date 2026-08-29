package benchmark

import (
	"reflect"
	"testing"
)

// TestUngatedBenchmarks reports current benchmarks that have no baseline entry
// and were therefore not compared, so the gate can surface reduced coverage
// instead of silently passing suites it never checked.
func TestUngatedBenchmarks(t *testing.T) {
	baseline := &Baseline{Benchmarks: map[string]BenchmarkMetric{"A": {}}}
	current := &Baseline{Benchmarks: map[string]BenchmarkMetric{"A": {}, "C": {}, "B": {}}}

	got := UngatedBenchmarks(current, baseline)
	want := []string{"B", "C"} // sorted; A has a baseline so it is gated
	if !reflect.DeepEqual(got, want) {
		t.Errorf("UngatedBenchmarks = %v, want %v", got, want)
	}
}

// TestMissingBenchmarks reports baseline benchmarks with no current result —
// benchmarks that were expected but did not run (a crashed suite, a rename, or
// a filtered run). Without this they would be silently skipped and a
// disappeared benchmark could pass the gate.
func TestMissingBenchmarks(t *testing.T) {
	baseline := &Baseline{Benchmarks: map[string]BenchmarkMetric{"A": {}, "Gone": {}, "AlsoGone": {}}}
	current := &Baseline{Benchmarks: map[string]BenchmarkMetric{"A": {}, "New": {}}}

	got := MissingBenchmarks(current, baseline)
	want := []string{"AlsoGone", "Gone"} // sorted; A ran, New has no baseline (that's UngatedBenchmarks)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("MissingBenchmarks = %v, want %v", got, want)
	}
}

func TestBuildComparisonFailsClosedForSuiteInventory(t *testing.T) {
	baseline := &Baseline{
		Metadata: RunMetadata{Environment: "cpu"},
		Benchmarks: map[string]BenchmarkMetric{
			"A":          {Suite: "core"},
			"Missing":    {Suite: "core"},
			"OtherSuite": {Suite: "other"},
		},
	}
	current := &Baseline{
		Metadata: RunMetadata{
			Environment: "cpu",
			Suites:      []SuiteRunMetadata{{Name: "core"}},
		},
		Benchmarks: map[string]BenchmarkMetric{
			"A":   {Suite: "core"},
			"New": {Suite: "core"},
		},
	}
	comparison, err := BuildComparison(current, baseline, nil)
	if err != nil {
		t.Fatal(err)
	}
	if comparison.CoverageComplete {
		t.Fatal("comparison with new and missing measurements must be incomplete")
	}
	if !reflect.DeepEqual(comparison.Ungated, []string{"New"}) {
		t.Fatalf("ungated = %v", comparison.Ungated)
	}
	if !reflect.DeepEqual(comparison.Missing, []string{"Missing"}) {
		t.Fatalf("missing = %v", comparison.Missing)
	}
}

func TestBuildComparisonRejectsEnvironmentMismatch(t *testing.T) {
	current := &Baseline{Metadata: RunMetadata{Environment: "cpu"}, Benchmarks: map[string]BenchmarkMetric{}}
	baseline := &Baseline{Metadata: RunMetadata{Environment: "amd-gpu"}, Benchmarks: map[string]BenchmarkMetric{}}
	if _, err := BuildComparison(current, baseline, nil); err == nil {
		t.Fatal("expected environment mismatch to fail")
	}
}
