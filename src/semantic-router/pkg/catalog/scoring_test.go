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
