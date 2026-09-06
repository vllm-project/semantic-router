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
