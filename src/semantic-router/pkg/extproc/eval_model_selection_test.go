package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestSelectModelForEvalUsesLiveMultiFactorPolicy(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"lower-quality":  modelParamsWithTestQuality(0.2),
				"higher-quality": modelParamsWithTestQuality(0.95),
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

func TestSelectModelForEvalPreservesFailClosedPolicy(t *testing.T) {
	decision := &config.Decision{
		Name: "strict-quality-route",
		ModelRefs: []config.ModelRef{
			{Model: "model-a"},
			{Model: "model-b"},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: config.DecisionAlgorithmMultiFactor,
			MultiFactor: &config.MultiFactorSelectionConfig{
				Weights: &config.MultiFactorWeightsConfig{Quality: 1},
				Quality: &config.QualityEvidenceConfig{
					Index:     "vllm-sr/intelligence@1.0.0",
					OnMissing: config.QualityEvidenceOnMissingExclude,
				},
				OnNoCandidates: "fail",
			},
		},
	}

	for _, test := range []struct {
		name     string
		learning config.RouterLearningConfig
	}{
		{name: "without learning"},
		{
			name: "with protection",
			learning: config.RouterLearningConfig{
				Enabled:    true,
				Adaptation: config.RouterLearningAdaptationConfig{Enabled: extprocBoolPtr(false)},
				Protection: config.RouterLearningProtectionConfig{Enabled: extprocBoolPtr(true)},
			},
		},
		{
			name: "with adaptation",
			learning: config.RouterLearningConfig{
				Enabled:    true,
				Adaptation: config.RouterLearningAdaptationConfig{Enabled: extprocBoolPtr(true)},
				Protection: config.RouterLearningProtectionConfig{Enabled: extprocBoolPtr(false)},
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{
				RouterLearning: test.learning,
				BackendModels: config.BackendModels{
					ModelConfig: map[string]config.ModelParams{
						"model-a": {},
						"model-b": {},
					},
				},
			}}
			result := router.SelectModelForEval(services.EvalModelSelectionInput{Decision: decision})
			if result.Status != services.EvalSelectionUnavailable || result.SelectedModel != "" {
				t.Fatalf("fail-closed Eval selection = %+v", result)
			}
			if !strings.Contains(result.Reason, selection.ErrNoEligibleCandidates.Error()) {
				t.Fatalf("fail-closed Eval reason = %q", result.Reason)
			}
		})
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

func TestSelectModelForEvalReturnsLearningSnapshotProvenance(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		RouterLearning: config.RouterLearningConfig{Enabled: true},
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"model-a": modelParamsWithTestQuality(0.9),
				"model-b": modelParamsWithTestQuality(0.1),
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
	if result.Status != services.EvalSelectionSelected || result.SelectedModel == "" || result.Provenance == nil || !result.Provenance.StateDependent || result.Provenance.Mode != "read_only_snapshot" || !result.Provenance.Sampled {
		t.Fatalf("learning-aware Eval selection = %+v", result)
	}
}
