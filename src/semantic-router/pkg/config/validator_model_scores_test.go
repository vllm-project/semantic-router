package config

import (
	"math"
	"strings"
	"testing"
)

func testDomainCategoryWithScores(scores ...ModelScore) Category {
	category := testDomainCategory("math")
	category.ModelScores = scores
	return category
}

func testConfigWithDomainCategory(category Category) *RouterConfig {
	return &RouterConfig{
		IntelligentRouting: IntelligentRouting{
			Signals: Signals{
				Categories: []Category{category},
			},
		},
	}
}

func TestValidateDomainContractsRejectsNonFiniteModelScores(t *testing.T) {
	for name, score := range map[string]float64{
		"NaN":           math.NaN(),
		"positive +Inf": math.Inf(1),
		"negative -Inf": math.Inf(-1),
	} {
		t.Run(name, func(t *testing.T) {
			cfg := testConfigWithDomainCategory(testDomainCategoryWithScores(
				ModelScore{Model: "model-a", Score: score},
			))

			err := validateDomainContracts(cfg)
			if err == nil {
				t.Fatalf("expected non-finite model score error for %v", score)
			}
			if !strings.Contains(err.Error(), "model_scores") || !strings.Contains(err.Error(), "must be finite") {
				t.Fatalf("expected named finite model score error, got: %v", err)
			}
		})
	}
}

func TestValidateDomainContractsRejectsOutOfBandModelScores(t *testing.T) {
	for name, score := range map[string]float64{
		"negative":   -0.5,
		"above band": 1.5,
	} {
		t.Run(name, func(t *testing.T) {
			cfg := testConfigWithDomainCategory(testDomainCategoryWithScores(
				ModelScore{Model: "model-a", Score: score},
			))

			err := validateDomainContracts(cfg)
			if err == nil {
				t.Fatalf("expected out-of-band model score error for %v", score)
			}
			if !strings.Contains(err.Error(), "model_scores") || !strings.Contains(err.Error(), "between 0.0 and 1.0") {
				t.Fatalf("expected named range model score error, got: %v", err)
			}
		})
	}
}

func TestValidateDomainContractsAllowsBoundaryModelScores(t *testing.T) {
	cfg := testConfigWithDomainCategory(testDomainCategoryWithScores(
		ModelScore{Model: "model-a", Score: 0},
		ModelScore{Model: "model-b", Score: 0.9},
		ModelScore{Model: "model-c", Score: 1},
	))

	if err := validateDomainContracts(cfg); err != nil {
		t.Fatalf("unexpected validation error: %v", err)
	}
}

func TestParseYAMLBytesRejectsNaNModelScore(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`
version: v0.3
routing:
  signals:
    domains:
      - name: math
        model_scores:
          - model: model-a
            score: .nan
          - model: model-b
            score: 0.9
`))
	if err == nil {
		t.Fatal("expected NaN model score to be rejected at load")
	}
	if !strings.Contains(err.Error(), "model_scores") || !strings.Contains(err.Error(), "must be finite") {
		t.Fatalf("expected named finite model score error, got: %v", err)
	}
}

func TestParseYAMLBytesRejectsOutOfBandModelScore(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`
version: v0.3
routing:
  signals:
    domains:
      - name: math
        model_scores:
          - model: model-a
            score: 1.5
`))
	if err == nil {
		t.Fatal("expected out-of-band model score to be rejected at load")
	}
	if !strings.Contains(err.Error(), "model_scores") || !strings.Contains(err.Error(), "between 0.0 and 1.0") {
		t.Fatalf("expected named range model score error, got: %v", err)
	}
}
