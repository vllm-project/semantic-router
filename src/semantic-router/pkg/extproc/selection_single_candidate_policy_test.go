package extproc

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestSingleCandidateMultiFactorEnforcesEligibility(t *testing.T) {
	for _, test := range []struct {
		name   string
		policy *config.MultiFactorSelectionConfig
	}{
		{
			name: "cost ceiling",
			policy: &config.MultiFactorSelectionConfig{
				SLO: &config.MultiFactorSLOConfig{MaxCostPer1M: 1}, OnNoCandidates: "fail",
			},
		},
		{
			name: "missing quality evidence",
			policy: &config.MultiFactorSelectionConfig{
				Weights:        &config.MultiFactorWeightsConfig{Quality: 1},
				Quality:        &config.QualityEvidenceConfig{Index: "test/quality@1", OnMissing: config.QualityEvidenceOnMissingExclude},
				OnNoCandidates: "fail",
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{
				ModelConfig: map[string]config.ModelParams{"only": {Pricing: config.ModelPricing{PromptPer1M: 10, CompletionPer1M: 10}}},
			}}}
			selected, _, err := router.selectModelFromCandidates(
				&selection.SelectionContext{DecisionName: "strict", CandidateModels: []config.ModelRef{{Model: "only"}}},
				&config.AlgorithmConfig{Type: config.DecisionAlgorithmMultiFactor, MultiFactor: test.policy},
				&RequestContext{},
			)
			if !errors.Is(err, selection.ErrNoEligibleCandidates) || selected != nil {
				t.Fatalf("single candidate bypassed policy: selected=%+v, err=%v", selected, err)
			}
		})
	}
}

func TestSingleCandidateMultiFactorHonorsExplicitFallback(t *testing.T) {
	for _, fallback := range []string{"first", "cheapest"} {
		router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{"only": {Pricing: config.ModelPricing{PromptPer1M: 10, CompletionPer1M: 10}}},
		}}}
		selected, method, err := router.selectModelFromCandidates(
			&selection.SelectionContext{DecisionName: "explicit-fallback", CandidateModels: []config.ModelRef{{Model: "only"}}},
			&config.AlgorithmConfig{Type: config.DecisionAlgorithmMultiFactor, MultiFactor: &config.MultiFactorSelectionConfig{
				SLO: &config.MultiFactorSLOConfig{MaxCostPer1M: 1}, OnNoCandidates: fallback,
			}}, &RequestContext{},
		)
		if err != nil || selected == nil || selected.Model != "only" || method != string(selection.MethodMultiFactor) {
			t.Fatalf("fallback %s: selected=%+v, method=%s, err=%v", fallback, selected, method, err)
		}
	}
}
