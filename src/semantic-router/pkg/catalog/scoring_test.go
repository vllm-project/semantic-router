package catalog

import (
	"strings"
	"testing"
)

func TestSelectedEvaluationRecordsRejectsEqualDuplicateMetric(t *testing.T) {
	selected := selectedEvaluationRecords{
		values:   map[string]map[string]map[string]float64{},
		evidence: map[string]map[string]map[string]string{},
	}
	metrics := map[string]metricDefinition{
		"acme/benchmark@1.0.0#score": {
			metric:   BenchmarkMetric{Range: [2]float64{0, 1}},
			profiles: map[string]struct{}{"published": {}},
		},
	}
	record := EvaluationRecord{
		ID: "acme/run-1", Model: "acme/model", Benchmark: "acme/benchmark@1.0.0",
		BenchmarkProfile: "published", ReasoningEffort: "default",
	}
	if err := selected.addMetric(record, "score", 0.8, "evaluations.records[0]", metrics); err != nil {
		t.Fatal(err)
	}
	record.ID = "acme/run-2"
	err := selected.addMetric(record, "score", 0.8, "evaluations.records[1]", metrics)
	if err == nil || !strings.Contains(err.Error(), "duplicates another available value") {
		t.Fatalf("expected equal duplicate metric to be rejected, got %v", err)
	}
}

func TestPreferredEvaluationEffortFallsBackToUnspecifiedMeasurement(t *testing.T) {
	values := map[string]map[string]IndexResult{
		"unspecified": {"acme/index@1.0.0": {Status: "available"}},
	}
	got := preferredEvaluationEffort(EffectiveModelCard{}, values, nil)
	if got != "unspecified" {
		t.Fatalf("preferred effort = %q, want unspecified", got)
	}
}

func TestIndexEvaluatorPrefersFirstAvailableCompatibleProfile(t *testing.T) {
	component := IndexComponent{
		Benchmark: "acme/benchmark@1.0.0", Metric: "score",
		BenchmarkProfiles: []string{"independent", "published"},
		Weight:            1, Normalization: Normalization{Type: "identity"},
	}
	evaluator := indexEvaluator{
		metrics: map[string]metricDefinition{
			"acme/benchmark@1.0.0#score": {domain: "reasoning"},
		},
		values: map[string]float64{
			evaluationMetricKey(component.Benchmark, "published", component.Metric):   0.9,
			evaluationMetricKey(component.Benchmark, "independent", component.Metric): 0.7,
		},
		evidence: map[string]string{
			evaluationMetricKey(component.Benchmark, "published", component.Metric):   "acme/published",
			evaluationMetricKey(component.Benchmark, "independent", component.Metric): "acme/independent",
		},
	}

	evaluated, err := evaluator.evaluateComponent(component)
	if err != nil {
		t.Fatal(err)
	}
	if !evaluated.present || evaluated.raw != 0.7 || evaluated.result.BenchmarkProfile != "independent" {
		t.Fatalf("unexpected selected component: %+v", evaluated)
	}
	if len(evaluated.provenance) != 1 || evaluated.provenance[0] != "acme/independent" {
		t.Fatalf("unexpected provenance: %+v", evaluated.provenance)
	}
}

func TestRequireAllIndexPreservesPartialCoverageWithoutScore(t *testing.T) {
	definition := IndexDefinition{
		ID: "acme/index@1.0.0", Scale: [2]float64{0, 100},
		Missing: MissingPolicy{Policy: "require_all"},
		Components: []IndexComponent{
			{Benchmark: "acme/one@1.0.0", Metric: "score", BenchmarkProfile: "standard", Weight: 0.5, Normalization: Normalization{Type: "identity"}},
			{Benchmark: "acme/two@1.0.0", Metric: "score", BenchmarkProfile: "standard", Weight: 0.5, Normalization: Normalization{Type: "identity"}},
		},
	}
	evaluator := indexEvaluator{
		model: "acme/model", effort: "default",
		indices: map[string]IndexDefinition{definition.ID: definition},
		metrics: map[string]metricDefinition{
			"acme/one@1.0.0#score": {domain: "reasoning"},
			"acme/two@1.0.0#score": {domain: "coding"},
		},
		values: map[string]float64{
			evaluationMetricKey("acme/one@1.0.0", "standard", "score"): 0.8,
		},
		evidence: map[string]string{}, memo: map[string]IndexResult{}, visiting: map[string]bool{},
	}

	result, err := evaluator.compute(definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if result.Status != "partial" || result.Score != nil || result.Coverage != 0.5 {
		t.Fatalf("unexpected partial result: %+v", result)
	}
}
