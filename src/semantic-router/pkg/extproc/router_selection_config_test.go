package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildMultiFactorSelectionConfigCopiesQualityEvidencePolicy(t *testing.T) {
	minimumScore := 40.0
	got := buildMultiFactorSelectionConfig(&config.MultiFactorSelectionConfig{
		LatencyMetric: "ttft",
		Objective: &config.MultiFactorObjectiveConfig{
			Strategy: config.MultiFactorObjectiveLexicographic,
			Priorities: []config.MultiFactorPriorityConfig{
				{Factor: config.MultiFactorFactorQuality, Tolerance: 0.02},
				{Factor: config.MultiFactorFactorCost},
			},
		},
		Quality: &config.QualityEvidenceConfig{
			Index:       "vllm-sr/agentic@1.0.0",
			OnMissing:   "exclude",
			MinCoverage: 0.8,
			MinScore:    &minimumScore,
		},
	})
	assertString(t, got.QualityIndex, "vllm-sr/agentic@1.0.0", "multi_factor.quality.index")
	assertString(t, got.LatencyMetric, "ttft", "multi_factor.latency_metric")
	assertString(t, got.QualityOnMissing, "exclude", "multi_factor.quality.on_missing")
	assertFloat(t, got.QualityMinCoverage, 0.8, "multi_factor.quality.min_coverage")
	if got.QualityMinScore == nil {
		t.Fatal("multi_factor.quality.min_score was not copied")
	}
	assertFloat(t, *got.QualityMinScore, minimumScore, "multi_factor.quality.min_score")
	assertString(t, got.Objective.Strategy, config.MultiFactorObjectiveLexicographic, "multi_factor.objective.strategy")
	if len(got.Objective.Priorities) != 2 {
		t.Fatalf("multi_factor.objective.priorities = %#v, want two entries", got.Objective.Priorities)
	}
	assertString(t, got.Objective.Priorities[0].Factor, config.MultiFactorFactorQuality, "multi_factor.objective.priorities[0].factor")
	assertFloat(t, got.Objective.Priorities[0].Tolerance, 0.02, "multi_factor.objective.priorities[0].tolerance")

	strict := buildMultiFactorSelectionConfig(&config.MultiFactorSelectionConfig{
		Quality: &config.QualityEvidenceConfig{Index: "vllm-sr/intelligence@1.0.0"},
	})
	assertString(t, strict.QualityOnMissing, "exclude", "multi_factor.quality.on_missing default")
}

func assertFloat(t *testing.T, got, want float64, field string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %v, want %v", field, got, want)
	}
}

func assertString(t *testing.T, got, want, field string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %q, want %q", field, got, want)
	}
}
