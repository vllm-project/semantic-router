package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestSelectModelForEvalUsesLiveMultiFactorPolicy(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"lower-quality":  {QualityScore: 0.2},
				"higher-quality": {QualityScore: 0.95},
			},
		},
	}}
	decision := &config.Decision{
		Name: "quality-route",
		ModelRefs: []config.ModelRef{
			{Model: "lower-quality"},
			{Model: "higher-quality"},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: "multi_factor",
			MultiFactor: &config.MultiFactorSelectionConfig{
				Weights: &config.MultiFactorWeightsConfig{Quality: 1},
			},
		},
	}

	result := router.SelectModelForEval(services.EvalModelSelectionInput{
		Decision: decision,
		Query:    "Compare two designs.",
	})
	if result.Status != services.EvalSelectionSelected || result.SelectedModel != "higher-quality" {
		t.Fatalf("Eval selection = %+v", result)
	}
	if result.Method != "multi_factor" {
		t.Fatalf("Eval selection method = %q", result.Method)
	}
}

func TestSelectModelForEvalDoesNotPretendLooperCandidateIsFinal(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	decision := &config.Decision{
		Name:      "fusion-route",
		ModelRefs: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		Algorithm: &config.AlgorithmConfig{Type: "fusion"},
	}

	result := router.SelectModelForEval(services.EvalModelSelectionInput{Decision: decision})
	if result.Status != services.EvalSelectionExecutionRequired || result.SelectedModel != "" {
		t.Fatalf("looper Eval selection = %+v", result)
	}
}

func TestSelectModelForEvalReportsConfiguredLooperFinalModel(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	decision := &config.Decision{
		Name:      "fusion-route",
		ModelRefs: []config.ModelRef{{Model: "panel-a"}, {Model: "panel-b"}},
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmFusion,
			Fusion: &config.FusionAlgorithmConfig{
				Model: "judge-model",
			},
		},
	}

	result := router.SelectModelForEval(services.EvalModelSelectionInput{Decision: decision})
	if result.Status != services.EvalSelectionPlannedFinal || result.SelectedModel != "judge-model" {
		t.Fatalf("configured Looper final selection = %+v", result)
	}
}

func TestSelectModelForEvalDoesNotClaimBaseSelectorIsFinalWhenLearningCanChangeIt(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		RouterLearning: config.RouterLearningConfig{Enabled: true},
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"model-a": {QualityScore: 0.9},
				"model-b": {QualityScore: 0.1},
			},
		},
	}}
	decision := &config.Decision{
		Name:      "adaptive-route",
		ModelRefs: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmMultiFactor,
			MultiFactor: &config.MultiFactorSelectionConfig{
				Weights: &config.MultiFactorWeightsConfig{Quality: 1},
			},
		},
	}

	result := router.SelectModelForEval(services.EvalModelSelectionInput{Decision: decision})
	if result.Status != services.EvalSelectionExecutionRequired || result.SelectedModel != "" {
		t.Fatalf("learning-aware Eval selection = %+v, want no fabricated final model", result)
	}
}
