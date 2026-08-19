package benchmark

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReportWritesJSONMarkdownAndHTML(t *testing.T) {
	comparison := &ComparisonDocument{
		CurrentMetadata: RunMetadata{Environment: "cpu", EnvironmentKind: "cpu", Profile: "ci", GitCommit: "abc"},
		Results: []ComparisonResult{{
			BenchmarkName: "BenchmarkCore", Suite: "core",
			Baseline:      BenchmarkMetric{NsPerOp: 100, AllocsPerOp: 1, BytesPerOp: 8, Custom: map[string]float64{"input_bytes": 1024}, P95LatencyMs: 20, ThroughputQPS: 100},
			Current:       BenchmarkMetric{NsPerOp: 110, AllocsPerOp: 1, BytesPerOp: 8, Custom: map[string]float64{"input_bytes": 1024}, Samples: 3, P95LatencyMs: 22, ThroughputQPS: 95},
			NsPerOpChange: 10,
		}},
		CoverageComplete: true,
	}
	report := GenerateReport(comparison)
	dir := t.TempDir()
	if err := report.SaveAll(dir); err != nil {
		t.Fatalf("SaveAll: %v", err)
	}
	for _, name := range []string{"report.json", "report.md", "report.html"} {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		if !strings.Contains(string(data), "BenchmarkCore") {
			t.Errorf("%s does not contain the measurement", name)
		}
		if !strings.Contains(string(data), "22") {
			t.Errorf("%s does not contain external latency metrics", name)
		}
		if !strings.Contains(string(data), "input_bytes") {
			t.Errorf("%s does not contain custom metrics", name)
		}
	}
}
