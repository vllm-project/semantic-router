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

func TestIndexComponentProfilesRequiresOneUniqueProfileForm(t *testing.T) {
	component := IndexComponent{BenchmarkProfiles: []string{"independent", "published"}}
	profiles, err := indexComponentProfiles(component, "indices[0].components[0]")
	if err != nil {
		t.Fatal(err)
	}
	if len(profiles) != 2 || profiles[0] != "independent" || profiles[1] != "published" {
		t.Fatalf("profiles = %v", profiles)
	}

	component.BenchmarkProfile = "published"
	if _, err := indexComponentProfiles(component, "indices[0].components[0]"); err == nil ||
		!strings.Contains(err.Error(), "exactly one") {
		t.Fatalf("mixed profile forms accepted: %v", err)
	}

	component.BenchmarkProfile = ""
	component.BenchmarkProfiles = []string{"published", "published"}
	if _, err := indexComponentProfiles(component, "indices[0].components[0]"); err == nil ||
		!strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("duplicate profiles accepted: %v", err)
	}
}

func TestNestedIndexComponentRejectsBenchmarkFields(t *testing.T) {
	component := IndexComponent{
		Index:            "example/base@1.0.0",
		BenchmarkProfile: "published",
	}
	err := validateIndexComponentReference(
		component,
		"indices[0].components[0]",
		map[string]IndexDefinition{"example/base@1.0.0": {}},
		nil,
		&indexValidationSummary{directDomainWeights: map[string]float64{}},
	)
	if err == nil || !strings.Contains(err.Error(), "nested index") {
		t.Fatalf("nested benchmark fields accepted: %v", err)
	}
}
