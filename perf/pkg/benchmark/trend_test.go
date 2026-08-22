package benchmark

import (
	"strconv"
	"strings"
	"testing"
)

func TestBuildTrendReportsGroupsAndSortsSeries(t *testing.T) {
	definitions := []TrendMetadata{{
		Name: "signals", Title: "Signal scaling", Suite: "signals",
		Benchmark: "^BenchmarkSignals/", XDimension: "context_tokens",
		SeriesDimension: "learned_signals", Metric: "latency_us_per_op", XScale: "log2",
	}}
	benchmarks := map[string]BenchmarkMetric{
		"BenchmarkSignals/context_tokens=2048/learned_signals=4": trendFixture(2048, 4, 800_000),
		"BenchmarkSignals/context_tokens=128/learned_signals=4":  trendFixture(128, 4, 200_000),
		"BenchmarkSignals/context_tokens=2048/learned_signals=1": trendFixture(2048, 1, 400_000),
		"BenchmarkSignals/context_tokens=128/learned_signals=1":  trendFixture(128, 1, 100_000),
	}

	reports := buildTrendReports(definitions, benchmarks)
	if len(reports) != 1 || len(reports[0].Series) != 2 {
		t.Fatalf("trend reports = %+v", reports)
	}
	if reports[0].Series[0].Name != "learned_signals=1" || reports[0].Series[0].Points[0].X != 128 {
		t.Fatalf("series were not stably sorted: %+v", reports[0].Series)
	}
	if !strings.Contains(reports[0].SVG, "log2 scale") || !strings.Contains(reports[0].SVG, "learned_signals=4") {
		t.Fatal("SVG is missing scale or series labels")
	}
}

func trendFixture(contextTokens, learnedSignals int, nsPerOp float64) BenchmarkMetric {
	return BenchmarkMetric{
		Suite: "signals", NsPerOp: nsPerOp,
		Dimensions: map[string]string{
			"context_tokens":  strconv.Itoa(contextTokens),
			"learned_signals": strconv.Itoa(learnedSignals),
		},
	}
}
