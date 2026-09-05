package config

import (
	"math"
	"strings"
	"testing"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

func TestValidateUserEvaluationAcceptsSmallGenericSurface(t *testing.T) {
	err := validateUserEvaluation(modelcatalog.UserEvaluation{
		Benchmark:  "acme/support-bench@1",
		Metrics:    map[string]float64{"resolution_rate": 0.82},
		MeasuredAt: "2026-09-01",
		Metadata:   map[string]any{"runtime": "vllm", "tensor_parallel": 2},
	}, "routing.modelCards[private].evaluations[0]")
	if err != nil {
		t.Fatalf("validateUserEvaluation() error = %v", err)
	}
}

func TestValidateUserEvaluationRejectsAmbiguousOrNonFiniteData(t *testing.T) {
	tests := []struct {
		name       string
		evaluation modelcatalog.UserEvaluation
		want       string
	}{
		{
			name:       "unversioned benchmark",
			evaluation: modelcatalog.UserEvaluation{Benchmark: "support", Metrics: map[string]float64{"score": 1}},
			want:       "namespaced, versioned identity",
		},
		{
			name:       "non-finite metric",
			evaluation: modelcatalog.UserEvaluation{Benchmark: "acme/support@1", Metrics: map[string]float64{"score": math.NaN()}},
			want:       "finite numeric metric",
		},
		{
			name:       "nested metadata",
			evaluation: modelcatalog.UserEvaluation{Benchmark: "acme/support@1", Metrics: map[string]float64{"score": 1}, Metadata: map[string]any{"runtime": map[string]any{"name": "vllm"}}},
			want:       "scalar key/value pairs",
		},
		{
			name:       "invalid date",
			evaluation: modelcatalog.UserEvaluation{Benchmark: "acme/support@1", Metrics: map[string]float64{"score": 1}, MeasuredAt: "September 1"},
			want:       "YYYY-MM-DD",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateUserEvaluation(test.evaluation, "evaluation")
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("validateUserEvaluation() error = %v, want %q", err, test.want)
			}
		})
	}
}
