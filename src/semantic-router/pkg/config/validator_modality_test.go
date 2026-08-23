package config

import "testing"

func TestValidateBothModalityDecisionDefersModelFreeRecipe(t *testing.T) {
	cfg := &RouterConfig{BackendModels: BackendModels{ModelConfig: map[string]ModelParams{}}}
	decision := Decision{Name: "multimodal"}

	if err := validateBothModalityDecision(cfg, decision); err != nil {
		t.Fatalf("model-free Recipe decision should be validated after assignment: %v", err)
	}
}

func TestValidateBothModalityDecisionRejectsIncompleteAssignment(t *testing.T) {
	cfg := &RouterConfig{BackendModels: BackendModels{ModelConfig: map[string]ModelParams{
		"text-model": {Modality: "ar"},
	}}}
	decision := Decision{
		Name:      "multimodal",
		ModelRefs: []ModelRef{{Model: "text-model"}},
	}

	if err := validateBothModalityDecision(cfg, decision); err == nil {
		t.Fatal("BOTH decision with only an AR assignment should be rejected")
	}
}
