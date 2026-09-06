package benchmark

import (
	"os"
	"path/filepath"
	"testing"
)

func writeBaselineFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// TestLoadBaselineDir_MergesSuiteFiles guards #2455 root cause #1: the producer
// (update-baseline.sh) writes one file per suite — decision.json, looper.json,
// classification.json, … — and never a combined baseline.json. The comparison
// side must therefore union every *.json in the directory.
func TestLoadBaselineDir_MergesSuiteFiles(t *testing.T) {
	dir := t.TempDir()
	writeBaselineFile(t, filepath.Join(dir, "decision.json"),
		`{"benchmarks":{"BenchmarkEvaluateDecisions":{"ns_per_op":628.5}}}`)
	writeBaselineFile(t, filepath.Join(dir, "looper.json"),
		`{"benchmarks":{"BenchmarkReMoM_DistributeRoundRobin":{"ns_per_op":241.9}}}`)
	// Non-JSON files in the directory must be ignored, not parsed.
	writeBaselineFile(t, filepath.Join(dir, "README.txt"), "not a baseline")

	b, err := LoadBaselineDir(dir)
	if err != nil {
		t.Fatalf("LoadBaselineDir: %v", err)
	}
	if len(b.Benchmarks) != 2 {
		t.Fatalf("merged benchmark count = %d, want 2 (%v)", len(b.Benchmarks), b.Benchmarks)
	}
	if got := b.Benchmarks["BenchmarkEvaluateDecisions"].NsPerOp; got != 628.5 {
		t.Errorf("BenchmarkEvaluateDecisions NsPerOp = %v, want 628.5", got)
	}
	if got := b.Benchmarks["BenchmarkReMoM_DistributeRoundRobin"].NsPerOp; got != 241.9 {
		t.Errorf("BenchmarkReMoM_DistributeRoundRobin NsPerOp = %v, want 241.9", got)
	}
}

// TestLoadBaselineDir_NoFilesErrors ensures a directory with no baseline files
// is a clear error rather than an empty, silently-passing comparison.
func TestLoadBaselineDir_NoFilesErrors(t *testing.T) {
	if _, err := LoadBaselineDir(t.TempDir()); err == nil {
		t.Fatal("expected an error for a directory with no baseline *.json files")
	}
}
