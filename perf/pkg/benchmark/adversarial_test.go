package benchmark

import (
	"strings"
	"testing"
)

// --- metricRegression: boundaries and direction ---------------------------

func TestMetricRegression_Boundaries(t *testing.T) {
	cases := []struct {
		name         string
		baseline     int64
		current      int64
		thresholdPct float64
		wantChange   float64
		wantRegress  bool
	}{
		{"exactly at threshold is not a regression (strict >)", 100, 105, 5, 5, false},
		{"just over threshold regresses", 100, 106, 5, 6, true},
		{"a decrease is never a regression", 100, 50, 5, -50, false},
		{"no change", 100, 100, 5, 0, false},
		{"zero baseline, zero current is not new", 0, 0, 5, 0, false},
		{"zero baseline, positive current is new (change undefined -> 0)", 0, 3, 5, 0, true},
		{"new allocation regresses even at 0% threshold", 0, 1, 0, 0, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			change, regressed := metricRegression(c.baseline, c.current, c.thresholdPct)
			if change != c.wantChange || regressed != c.wantRegress {
				t.Errorf("metricRegression(%d,%d,%v) = (%v,%v), want (%v,%v)",
					c.baseline, c.current, c.thresholdPct, change, regressed, c.wantChange, c.wantRegress)
			}
		})
	}
}

// --- CompareWithBaseline: mixed directions ---------------------------------

// A benchmark that allocates fewer times but far more bytes must still be
// blocked (bytes regressed), and the allocs change must read as the improvement
// it is.
func TestCompareWithBaseline_BytesRegressesWhileAllocsImprove(t *testing.T) {
	cfg := testThresholds() // decision allocs/bytes 5%
	bl := &Baseline{Benchmarks: map[string]BenchmarkMetric{
		"BenchmarkEvaluateDecisions_Mixed": {AllocsPerOp: 100, BytesPerOp: 100},
	}}
	cur := &Baseline{Benchmarks: map[string]BenchmarkMetric{
		"BenchmarkEvaluateDecisions_Mixed": {AllocsPerOp: 50, BytesPerOp: 200}, // allocs -50%, bytes +100%
	}}
	results, err := CompareWithBaseline(cur, bl, cfg)
	if err != nil {
		t.Fatal(err)
	}
	r := results[0]
	if !r.RegressionDetected {
		t.Error("bytes +100% must block even though allocs improved")
	}
	if r.AllocsPerOpChange >= 0 {
		t.Errorf("allocs change should be negative (improvement), got %v", r.AllocsPerOpChange)
	}
}

// --- ParseBenchOutput: hostile / unusual input -----------------------------

func TestParseBenchOutput_HostileInput(t *testing.T) {
	input := `goos: linux
goarch: amd64
--- FAIL: BenchmarkCrashed-8
panic: boom
BenchmarkNoBenchmem-8   	 1000000	      1234 ns/op
BenchmarkCustomMetric-8   	  100	     5.20 MB/s	   628 ns/op	   112 B/op	     5 allocs/op
   	garbage line with numbers 123 456
PASS
ok  	pkg	1.2s
`
	b, err := ParseBenchOutput(strings.NewReader(input))
	if err != nil {
		t.Fatalf("ParseBenchOutput: %v", err)
	}

	// A crashed benchmark prints no result line, so it must not appear.
	if _, ok := b.Benchmarks["BenchmarkCrashed"]; ok {
		t.Error("a crashed benchmark (--- FAIL, no result line) must not be parsed as a result")
	}
	// A line without -benchmem still yields ns/op; alloc columns default to 0.
	if m := b.Benchmarks["BenchmarkNoBenchmem"]; m.NsPerOp != 1234 || m.AllocsPerOp != 0 || m.BytesPerOp != 0 {
		t.Errorf("BenchmarkNoBenchmem = %+v, want NsPerOp=1234, allocs/bytes=0", m)
	}
	// A leading custom metric (MB/s) must not shift the ns/op, B/op, allocs/op mapping.
	if m := b.Benchmarks["BenchmarkCustomMetric"]; m.NsPerOp != 628 || m.BytesPerOp != 112 || m.AllocsPerOp != 5 {
		t.Errorf("BenchmarkCustomMetric = %+v, want ns=628, B=112, allocs=5", m)
	}
	if len(b.Benchmarks) != 2 {
		t.Errorf("parsed %d benchmarks, want 2 (%v)", len(b.Benchmarks), b.Benchmarks)
	}
}
