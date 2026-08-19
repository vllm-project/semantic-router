package benchmark

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReportWritesJSONMarkdownAndHTML(t *testing.T) {
	comparison := &ComparisonDocument{
		CurrentMetadata: RunMetadata{
			Environment: "cpu", EnvironmentKind: "cpu", Profile: "ci", GitCommit: "abc",
			Trends: []TrendMetadata{{
				Name: "core-scaling", Title: "Core scaling", Suite: "core",
				Benchmark: "^BenchmarkCore/", XDimension: "items", Metric: "latency_us_per_op",
			}},
		},
		CurrentBenchmarks: map[string]BenchmarkMetric{
			"BenchmarkCore/items=1": {Suite: "core", Dimensions: map[string]string{"items": "1"}, NsPerOp: 100},
			"BenchmarkCore/items=2": {Suite: "core", Dimensions: map[string]string{"items": "2"}, NsPerOp: 200},
		},
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
	for _, name := range []string{"report.json", "report.md", "report.html", "trends.json"} {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		if name != "trends.json" && !strings.Contains(string(data), "BenchmarkCore") {
			t.Errorf("%s does not contain the measurement", name)
		}
		if name != "trends.json" && !strings.Contains(string(data), "22") {
			t.Errorf("%s does not contain external latency metrics", name)
		}
		if name != "trends.json" && !strings.Contains(string(data), "input_bytes") {
			t.Errorf("%s does not contain custom metrics", name)
		}
	}
	assertMarkdownHasNoArtifactRelativeCharts(t, filepath.Join(dir, "report.md"))
	chart, err := os.ReadFile(filepath.Join(dir, "charts", "core-scaling.svg"))
	if err != nil {
		t.Fatalf("read trend chart: %v", err)
	}
	if !strings.Contains(string(chart), "items") {
		t.Fatal("trend chart does not label its x dimension")
	}
}

func assertMarkdownHasNoArtifactRelativeCharts(t *testing.T, path string) {
	t.Helper()
	markdown, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read Markdown report: %v", err)
	}
	if strings.Contains(string(markdown), "](charts/") {
		t.Fatal("Markdown report contains an artifact-relative chart link")
	}
}
