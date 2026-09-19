package benchmark

import (
	"strings"
	"testing"
)

// sampleBenchOutput mirrors real `go test -bench -benchmem` output, including
// the tab-aligned columns, a subtest name with a slash, and a trailing
// GOMAXPROCS "-8" suffix that must be stripped so the parsed name matches the
// keys update-baseline.sh writes.
const sampleBenchOutput = `goos: linux
goarch: amd64
pkg: github.com/vllm-project/semantic-router/perf/benchmarks
cpu: AMD EPYC
BenchmarkEvaluateDecisions_SingleDomain-8   	60389019	       628.5 ns/op	     112 B/op	       5 allocs/op
BenchmarkReMoM_DistributeRoundRobin-8   	 4413338	       241.9 ns/op	     384 B/op	       1 allocs/op
BenchmarkBase_Execute/models_1-8   	    3306	    356542 ns/op	   39404 B/op	     478 allocs/op
BenchmarkNoMem-4   	 1000000	      1234 ns/op
PASS
ok  	github.com/vllm-project/semantic-router/perf/benchmarks	12.345s
`

// TestParseBenchOutput guards #2455 root cause #2: nothing produced the
// current.json the comparison needs. ParseBenchOutput turns raw `go test`
// output into a Baseline so a current set can be generated on demand.
func TestParseBenchOutput(t *testing.T) {
	b, err := ParseBenchOutput(strings.NewReader(sampleBenchOutput))
	if err != nil {
		t.Fatalf("ParseBenchOutput: %v", err)
	}

	if len(b.Benchmarks) != 4 {
		t.Fatalf("parsed %d benchmarks, want 4 (%v)", len(b.Benchmarks), b.Benchmarks)
	}

	m := b.Benchmarks["BenchmarkEvaluateDecisions_SingleDomain"]
	if m.NsPerOp != 628.5 {
		t.Errorf("NsPerOp = %v, want 628.5 (decimal must survive)", m.NsPerOp)
	}
	if m.BytesPerOp != 112 || m.AllocsPerOp != 5 {
		t.Errorf("BytesPerOp=%d AllocsPerOp=%d, want 112/5", m.BytesPerOp, m.AllocsPerOp)
	}

	// Subtest names keep their slash; only the -8 GOMAXPROCS suffix is stripped.
	if got := b.Benchmarks["BenchmarkBase_Execute/models_1"].NsPerOp; got != 356542 {
		t.Errorf("BenchmarkBase_Execute/models_1 NsPerOp = %v, want 356542", got)
	}

	// A line without -benchmem still parses ns/op; alloc columns default to 0.
	noMem := b.Benchmarks["BenchmarkNoMem"]
	if noMem.NsPerOp != 1234 || noMem.BytesPerOp != 0 {
		t.Errorf("BenchmarkNoMem = %+v, want NsPerOp=1234 BytesPerOp=0", noMem)
	}
}

func TestParseBenchOutputWithProgressBetweenNameAndResult(t *testing.T) {
	// Captured shape from the cache benchmarks: Go prints the name before its
	// calibration iterations, whose setup emits multiple progress messages.
	input := `BenchmarkCacheSearch_1000Entries-4

=== Benchmark Scenario ===
Cache Size: 1000, Concurrency: 1, HNSW: true, Model: mmbert
  Populated 1000/1000 entries
Running 6759 requests with concurrency 1...
    6759 457599 ns/op 90.13 hit_rate_% 0.08100 p95_ms 16288 qps 22665 B/op 80 allocs/op
BenchmarkOrdinary-4 1 12 ns/op 0 B/op 0 allocs/op
BenchmarkCacheSearch_Linear
=== Benchmark Scenario ===
Running 2886 requests with concurrency 1...
    2886 1059609 ns/op 0.001000 embedding_p95_ms 0.1820 search_p95_ms 2868 B/op 14 allocs/op
BenchmarkHeaderProgress-4 Populating cache...
Running 7 requests with concurrency 1...
    7 105 ns/op 128 B/op 3 allocs/op
PASS
ok package 1.2s
`
	parsed, err := ParseBenchOutput(strings.NewReader(input))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]BenchmarkMetric{
		"BenchmarkCacheSearch_1000Entries": {NsPerOp: 457599, BytesPerOp: 22665, AllocsPerOp: 80},
		"BenchmarkCacheSearch_Linear":      {NsPerOp: 1059609, BytesPerOp: 2868, AllocsPerOp: 14},
		"BenchmarkOrdinary":                {NsPerOp: 12},
		"BenchmarkHeaderProgress":          {NsPerOp: 105, BytesPerOp: 128, AllocsPerOp: 3},
	}
	if len(parsed.Benchmarks) != len(want) {
		t.Fatalf("got %d measurements, want %d", len(parsed.Benchmarks), len(want))
	}
	for name, metric := range want {
		if got := parsed.Benchmarks[name]; got != metric {
			t.Errorf("%s: got %+v, want %+v", name, got, metric)
		}
	}
}

func TestParseBenchOutputRejectsIncompleteOrAmbiguousSplitResults(t *testing.T) {
	for name, input := range map[string]string{
		"missing result":       "BenchmarkA-4\nprogress\n",
		"next benchmark":       "BenchmarkA-4\nBenchmarkB-4 1 12 ns/op\n",
		"package boundary":     "BenchmarkA-4\nPASS\n1 12 ns/op\n",
		"failure boundary":     "BenchmarkA-4\n--- FAIL: BenchmarkA\n1 12 ns/op\n",
		"result without name":  "1 12 ns/op 0 B/op 0 allocs/op\n",
		"duplicate result":     "BenchmarkA-4\n1 12 ns/op\n1 13 ns/op\n",
		"duplicate benchmark":  "BenchmarkA-4\n1 12 ns/op\nBenchmarkA-4\n1 13 ns/op\n",
		"zero iteration count": "BenchmarkA-4\n0 12 ns/op\n1 13 ns/op\n",
		"inline zero count":    "BenchmarkA-4 0 12 ns/op\n",
		"invalid inline count": "BenchmarkA-4 nope 12 ns/op\n",
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := ParseBenchOutput(strings.NewReader(input)); err == nil {
				t.Fatal("accepted incomplete or ambiguous benchmark output")
			}
		})
	}
}
