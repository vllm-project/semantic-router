package evaluationplane

import (
	"strings"
	"testing"
)

func validMetricAnalysisProvenance(exclusions int) MetricAnalysisProvenance {
	return validMetricAnalysisProvenanceFor("routing.accuracy", exclusions)
}

func validMetricAnalysisProvenanceFor(metricID string, exclusions int) MetricAnalysisProvenance {
	specification, err := registeredMetricAnalysisSpec(metricID)
	if err != nil {
		panic(err)
	}
	return MetricAnalysisProvenance{
		ContractVersion:  MetricAnalysisContractVersion,
		EstimatorID:      specification.estimatorID,
		EstimatorVersion: specification.estimatorVersion,
		AnalysisUnit:     specification.analysisUnit,
		ClusterUnit:      specification.clusterUnit,
		Weighting:        specification.weighting,
		Missingness:      specification.missingness,
		ExclusionPolicy:  specification.exclusionPolicy,
		ObservedExclusions: func() *int {
			value := exclusions
			return &value
		}(),
	}
}

func TestValidateReportMetricsRequiresVersionedAnalysisProvenance(t *testing.T) {
	value := 0.8
	metric := Metric{
		ID: "routing.accuracy", Name: "Routing accuracy", TrackID: "routing",
		Value: &value, Unit: "fraction", Direction: "higher_is_better", SampleCount: 4,
		AnalysisProvenance: validMetricAnalysisProvenance(0),
	}
	if err := validateReportMetrics([]Metric{metric}, []TrackID{"routing"}); err != nil {
		t.Fatalf("valid metric provenance rejected: %v", err)
	}
	cases := []struct {
		name   string
		mutate func(*Metric)
		match  string
	}{
		{"missing", func(metric *Metric) { metric.AnalysisProvenance = MetricAnalysisProvenance{} }, "contract_version"},
		{"illegal version", func(metric *Metric) { metric.AnalysisProvenance.ContractVersion = "metric-analysis.v0" }, "contract_version"},
		{"blank estimator", func(metric *Metric) { metric.AnalysisProvenance.EstimatorID = " " }, "registered estimator"},
		{"swapped legal weighting", func(metric *Metric) { metric.AnalysisProvenance.Weighting = "uniform_repetition" }, "registered estimator"},
		{"swapped legal analysis unit", func(metric *Metric) { metric.AnalysisProvenance.AnalysisUnit = "load_level" }, "registered estimator"},
		{"missing observed exclusions", func(metric *Metric) { metric.AnalysisProvenance.ObservedExclusions = nil }, "observed_exclusions"},
		{"negative observed exclusions", func(metric *Metric) { value := -1; metric.AnalysisProvenance.ObservedExclusions = &value }, "observed_exclusions"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			candidate := metric
			test.mutate(&candidate)
			err := validateReportMetrics([]Metric{candidate}, []TrackID{"routing"})
			if err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("validation error=%v, want %q", err, test.match)
			}
		})
	}
}

func TestValidateReportMetricsRejectsUnknownMetricIDsWithoutFallback(t *testing.T) {
	value := 1.0
	metric := Metric{
		ID: "system.total_cost", Name: "Unknown worker metric", TrackID: "routing",
		Value: &value, Unit: "usd", Direction: "lower_is_better", SampleCount: 1,
		AnalysisProvenance: validMetricAnalysisProvenance(0),
	}
	err := validateReportMetrics([]Metric{metric}, []TrackID{"routing"})
	if err == nil || !strings.Contains(err.Error(), "metric id is not registered") {
		t.Fatalf("unknown metric validation error=%v, want fail-closed registration error", err)
	}
}
