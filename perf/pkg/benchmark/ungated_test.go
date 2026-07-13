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
