package config

import (
	"math"
	"reflect"
	"strings"
	"testing"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

func TestUnknownOperatorBenchmarkSurvivesMaterializationAndCanonicalExport(t *testing.T) {
	cfg, err := ParseYAMLBytesWithoutEnvExpansion([]byte(`
version: v0.3
providers:
  models:
    - name: private-reasoner
      api_format: openai
      backend_refs:
        - name: primary
          provider: vllm
          base_url: http://127.0.0.1:8000/v1
routing:
  modelCards:
    - name: private-reasoner
      evaluations:
        - benchmark: acme/support-bench@1
          benchmark_profile: production
          reasoning_effort: high
          metrics:
            resolution_rate: 0.82
          source: https://evals.example/runs/42
          measured_at: 2026-09-01
          metadata:
            runtime: vllm
`))
	if err != nil {
		t.Fatal(err)
	}

	want := modelcatalog.UserEvaluation{
		Benchmark:        "acme/support-bench@1",
		BenchmarkProfile: "production",
		ReasoningEffort:  "high",
		Metrics:          map[string]float64{"resolution_rate": 0.82},
		Source:           "https://evals.example/runs/42",
		MeasuredAt:       "2026-09-01",
		Metadata:         map[string]any{"runtime": "vllm"},
	}
	assertUnknownOperatorEvaluationPreserved(t, cfg, want)
	assertUnknownOperatorEvaluationNotIndexed(t, cfg, want.Benchmark)
}

func assertUnknownOperatorEvaluationPreserved(
	t *testing.T,
	cfg *RouterConfig,
	want modelcatalog.UserEvaluation,
) {
	t.Helper()
	effective, ok := cfg.EffectiveModelRegistry.Model("private-reasoner")
	if !ok || len(effective.Card.Evaluations) != 1 || !reflect.DeepEqual(effective.Card.Evaluations[0], want) {
		t.Fatalf("effective custom evaluation = %#v, want %#v", effective.Card.Evaluations, want)
	}
	if got := cfg.ModelConfig["private-reasoner"].Evaluations; len(got) != 1 || !reflect.DeepEqual(got[0], want) {
		t.Fatalf("runtime custom evaluation = %#v, want %#v", got, want)
	}

	exported := CanonicalConfigFromRouterConfig(cfg)
	if len(exported.Routing.ModelCards) != 1 || len(exported.Routing.ModelCards[0].Evaluations) != 1 ||
		!reflect.DeepEqual(exported.Routing.ModelCards[0].Evaluations[0], want) {
		t.Fatalf("canonical custom evaluation = %#v, want %#v", exported.Routing.ModelCards, want)
	}
}

func assertUnknownOperatorEvaluationNotIndexed(t *testing.T, cfg *RouterConfig, benchmark string) {
	t.Helper()
	effective, ok := cfg.EffectiveModelRegistry.Model("private-reasoner")
	if !ok {
		t.Fatal("effective custom model is missing")
	}
	if results := effective.IndicesByEffort["high"]; len(results) > 0 {
		for _, result := range results {
			for _, component := range result.Components {
				if component.Benchmark == benchmark {
					t.Fatalf("unknown operator benchmark entered repository index: %#v", result)
				}
			}
		}
	}
}

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
