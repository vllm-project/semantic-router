package catalog

import (
	"math"
	"strings"
	"testing"
)

func TestValidateBenchmarkAcceptsTagsAndMetricNormalization(t *testing.T) {
	minimum, maximum := 500.0, 2500.0
	benchmark := BenchmarkDefinition{
		ID:             "example/bench@1.0.0",
		DisplayName:    "Example",
		Domain:         "reasoning",
		Tags:           []string{"core"},
		DefaultProfile: "standard",
		Profiles: []BenchmarkProfile{{
			ID: "standard", DisplayName: "Standard", Description: "Published standard profile.",
		}},
		Metrics: []BenchmarkMetric{{
			ID: "elo", Unit: "elo", Direction: "higher_is_better", Range: [2]float64{0, 3000},
			Normalization: &Normalization{Type: "linear_clamp", Min: &minimum, Max: &maximum},
		}},
	}
	if err := validateBenchmark(benchmark, "benchmarks[0]"); err != nil {
		t.Fatalf("valid benchmark rejected: %v", err)
	}

	benchmark.Tags = []string{"not a slug"}
	if err := validateBenchmark(benchmark, "benchmarks[0]"); err == nil || !strings.Contains(err.Error(), "must be a slug") {
		t.Fatalf("invalid tag accepted: %v", err)
	}
	benchmark.Tags = []string{"core", "core"}
	if err := validateBenchmark(benchmark, "benchmarks[0]"); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("duplicate tag accepted: %v", err)
	}
	benchmark.Tags = []string{}
	if err := validateBenchmark(benchmark, "benchmarks[0]"); err == nil || !strings.Contains(err.Error(), "cannot be empty") {
		t.Fatalf("empty declared tags accepted: %v", err)
	}

	benchmark.Tags = []string{"core"}
	invalid := math.Inf(1)
	benchmark.Metrics[0].Normalization.Min = &invalid
	if err := validateBenchmark(benchmark, "benchmarks[0]"); err == nil || !strings.Contains(err.Error(), "min < max") {
		t.Fatalf("non-finite normalization accepted: %v", err)
	}
}
