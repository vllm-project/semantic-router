package benchmark

import (
	"os"
	"path/filepath"
	"testing"
)

// TestLoadBaseline_ParsesDecimalNsPerOp guards #2455 root cause #4: the Go
// benchmark runner emits sub-microsecond timings as decimals (e.g. "628.5
// ns/op"), and update-baseline.sh writes them into the JSON verbatim. The
// metric field must therefore round-trip a decimal ns/op without erroring.
func TestLoadBaseline_ParsesDecimalNsPerOp(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	content := `{
  "version": "v1.0.0",
  "benchmarks": {
    "BenchmarkDecimal": {"iterations": 100, "ns_per_op": 628.5, "bytes_per_op": 112, "allocs_per_op": 5}
  }
}`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	b, err := LoadBaseline(path)
	if err != nil {
		t.Fatalf("LoadBaseline rejected a decimal ns_per_op: %v", err)
	}

	got := float64(b.Benchmarks["BenchmarkDecimal"].NsPerOp)
	if got != 628.5 {
		t.Fatalf("round-trip NsPerOp = %v, want 628.5", got)
	}
}
