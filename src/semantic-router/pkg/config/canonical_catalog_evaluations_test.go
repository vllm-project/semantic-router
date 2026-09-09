package config

import (
	"math"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"

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

func TestCustomEvaluationCatalogComputesAndRoundTripsIndex(t *testing.T) {
	document := []byte(`
version: v0.3
evaluation_catalog:
  benchmarks:
    - id: acme/support-bench@1.0.0
      display_name: ACME Support Bench
      domain: support
      source: https://evals.example/support-bench
      default_profile: production
      profiles:
        - id: production
          display_name: Production
          description: Frozen production support set.
      metrics:
        - id: resolution_rate
          unit: percent
          direction: higher_is_better
          range: [0, 100]
  indices:
    - id: acme/support-quality@1.0.0
      display_name: ACME Support Quality
      aggregation: weighted_mean
      scale: [0, 100]
      missing: {policy: require_all}
      domains: {support: 1}
      components:
        - benchmark: acme/support-bench@1.0.0
          benchmark_profile: production
          metric: resolution_rate
          weight: 1
          normalization: {type: linear_clamp, min: 0, max: 100}
providers:
  models:
    - name: private-reasoner
      api_format: openai
      backend_refs:
        - provider: vllm
          base_url: http://127.0.0.1:8000/v1
routing:
  modelCards:
    - name: private-reasoner
      evaluations:
        - benchmark: acme/support-bench@1.0.0
          benchmark_profile: production
          reasoning_effort: high
          metrics: {resolution_rate: 82}
          source: https://evals.example/runs/42
          measured_at: 2026-09-01
`)

	cfg, err := ParseYAMLBytesWithoutEnvExpansion(document)
	if err != nil {
		t.Fatal(err)
	}
	assertCustomIndexScore(t, cfg, 82)

	exported := CanonicalConfigFromRouterConfig(cfg)
	if exported.EvaluationCatalog == nil || len(exported.EvaluationCatalog.Benchmarks) != 1 || len(exported.EvaluationCatalog.Indices) != 1 {
		t.Fatalf("exported evaluation catalog = %#v", exported.EvaluationCatalog)
	}
	encoded, err := yaml.Marshal(exported)
	if err != nil {
		t.Fatalf("marshal canonical config: %v", err)
	}
	replayed, err := ParseYAMLBytesWithoutEnvExpansion(encoded)
	if err != nil {
		t.Fatalf("reparse canonical config: %v\n%s", err, encoded)
	}
	assertCustomIndexScore(t, replayed, 82)
}

func TestCustomEvaluationCatalogCanAdmitExplicitPartialCoverage(t *testing.T) {
	document := []byte(`
version: v0.3
evaluation_catalog:
  benchmarks:
    - id: acme/support-bench@1.0.0
      display_name: ACME Support Bench
      domain: support
      default_profile: production
      profiles:
        - id: production
          display_name: Production
          description: Frozen production support set.
      metrics:
        - id: resolution_rate
          unit: proportion
          direction: higher_is_better
          range: [0, 1]
        - id: grounded_rate
          unit: proportion
          direction: higher_is_better
          range: [0, 1]
  indices:
    - id: acme/support-quality@1.0.0
      display_name: ACME Support Quality
      aggregation: weighted_mean
      scale: [0, 100]
      missing: {policy: require_coverage, minimum: 0.5}
      domains: {support: 1}
      components:
        - benchmark: acme/support-bench@1.0.0
          benchmark_profile: production
          metric: resolution_rate
          weight: 0.5
          normalization: {type: identity}
        - benchmark: acme/support-bench@1.0.0
          benchmark_profile: production
          metric: grounded_rate
          weight: 0.5
          normalization: {type: identity}
providers:
  models:
    - name: private-reasoner
      api_format: openai
      backend_refs:
        - provider: vllm
          base_url: http://127.0.0.1:8000/v1
routing:
  modelCards:
    - name: private-reasoner
      evaluations:
        - benchmark: acme/support-bench@1.0.0
          benchmark_profile: production
          reasoning_effort: high
          metrics: {resolution_rate: 0.82}
`)

	cfg, err := ParseYAMLBytesWithoutEnvExpansion(document)
	if err != nil {
		t.Fatal(err)
	}
	result, ok := cfg.ModelConfig["private-reasoner"].EvidenceResultAt(
		"acme/support-quality@1.0.0", "high",
	)
	if !ok || result.Score == nil || math.Abs(*result.Score-82) > 1e-9 || math.Abs(result.Coverage-0.5) > 1e-9 {
		t.Fatalf("partial custom index result = %#v, want available score 82 at 0.5 coverage", result)
	}
}

func assertCustomIndexScore(t *testing.T, cfg *RouterConfig, want float64) {
	t.Helper()
	params := cfg.ModelConfig["private-reasoner"]
	result, ok := params.IndexResultsByEffort["high"]["acme/support-quality@1.0.0"]
	if !ok || result.Status != "available" || result.Score == nil || math.Abs(*result.Score-want) > 1e-9 {
		t.Fatalf("custom index result = %#v, want available score %.1f", result, want)
	}
}

func TestCustomEvaluationCatalogRejectsUnversionedOrBuiltInIdentities(t *testing.T) {
	tests := []struct {
		name      string
		catalog   CanonicalEvaluationCatalog
		wantError string
	}{
		{
			name: "unversioned benchmark",
			catalog: CanonicalEvaluationCatalog{Benchmarks: []modelcatalog.BenchmarkDefinition{{
				ID: "support-bench",
			}}},
			wantError: "namespaced, versioned identity",
		},
		{
			name: "built-in benchmark shadow",
			catalog: CanonicalEvaluationCatalog{Benchmarks: []modelcatalog.BenchmarkDefinition{{
				ID: "tiger-ai-lab/mmlu-pro@1.0.0",
			}}},
			wantError: "conflicts with an existing benchmark",
		},
		{
			name: "built-in index shadow",
			catalog: CanonicalEvaluationCatalog{Indices: []modelcatalog.IndexDefinition{{
				ID: "vllm-sr/intelligence@1.0.0",
			}}},
			wantError: "conflicts with an existing index",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := canonicalCatalogInput(&CanonicalConfig{EvaluationCatalog: &test.catalog})
			if err == nil || !strings.Contains(err.Error(), test.wantError) {
				t.Fatalf("canonicalCatalogInput() error = %v, want %q", err, test.wantError)
			}
		})
	}
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
