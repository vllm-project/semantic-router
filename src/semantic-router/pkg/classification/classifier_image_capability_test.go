package classification

import (
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

func TestImageClassificationRequiresEveryRoutedImageDeployment(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.SystemOneRules = []config.SystemOneRule{{Name: "vision", Deployment: "vision", Input: config.SystemOneInput{Image: true}}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"vision": {Provider: config.ModelRuntimeProvider}}
	classifier := &Classifier{Config: cfg, decisionCards: map[string]modelservice.ModelCard{"vision": {ID: "image", Surfaces: []string{"decisions"}, QuestionTypes: []string{"noul"}, Modalities: []string{"text", "image"}}}}
	if err := classifier.ValidateImageClassification(); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("unused image rule should fail preflight: %v", err)
	}
	cfg.Decisions = []config.Decision{{Name: "route", Rules: config.RuleNode{Type: config.SignalTypeSystemOne, Name: "vision"}}}
	if err := classifier.ValidateImageClassification(); err != nil {
		t.Fatalf("prepared routed image rule rejected: %v", err)
	}
	classifier.decisionCards["vision"] = modelservice.ModelCard{Modalities: []string{"text"}}
	if err := classifier.ValidateImageClassification(); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("text-only model accepted: %v", err)
	}
	delete(classifier.decisionCards, "vision")
	if err := classifier.ValidateImageClassification(); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("unobserved model accepted: %v", err)
	}
	classifier.decisionCards["vision"] = modelservice.ModelCard{Modalities: []string{"image"}}
	cfg.SystemOneRules = append(cfg.SystemOneRules, config.SystemOneRule{Name: "vision2", Deployment: "other", Input: config.SystemOneInput{Image: true}})
	cfg.ModelDeployments["other"] = config.ModelDeployment{Provider: config.ModelRuntimeProvider}
	cfg.Decisions = append(cfg.Decisions, config.Decision{Name: "route2", Rules: config.RuleNode{Type: config.SignalTypeSystemOne, Name: "vision2"}})
	classifier.decisionCards["other"] = modelservice.ModelCard{Modalities: []string{"text"}}
	if err := classifier.ValidateImageClassification(); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("a second incompatible routed model was accepted: %v", err)
	}
	classifier.decisionCards["other"] = modelservice.ModelCard{Modalities: []string{"image"}}
	if err := classifier.ValidateImageClassification(); err != nil {
		t.Fatalf("two compatible routed models were rejected: %v", err)
	}
	cfg.ModelDeployments["vision"] = config.ModelDeployment{Provider: "systemone"}
	if err := classifier.ValidateImageClassification(); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("remote provider accepted: %v", err)
	}
}

func TestPreparedImageRuleForwardsImageToDecisionModel(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.SystemOneRules = []config.SystemOneRule{{Name: "vision", Deployment: "vision", Input: config.SystemOneInput{Image: true}, Question: config.SystemOneQuestion{Type: "noul", Instructions: "Is this an image?"}}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"vision": {Provider: config.ModelRuntimeProvider}}
	cfg.Decisions = []config.Decision{{Name: "image_route", Rules: config.RuleNode{Type: config.SignalTypeSystemOne, Name: "vision"}}}
	classifier, err := newClassifierWithOptions(cfg)
	if err != nil {
		t.Fatal(err)
	}
	classifier.decisionCards = map[string]modelservice.ModelCard{"vision": {ID: "image", Surfaces: []string{"decisions"}, QuestionTypes: []string{"noul"}, Modalities: []string{"text", "image"}}}
	decider := newFakeDecider(map[string]modelservice.Answer{"systemone:vision": {Type: "noul", Noul: 0.9, InputCoverage: "complete"}})
	classifier.SetDecisionDecider(decider)
	if err := classifier.ValidateImageClassification(); err != nil {
		t.Fatal(err)
	}
	image := "data:image/png;base64,YQ=="
	result := classifier.EvaluateAllSignalsWithRequestFacts("", "", "", nil, nil, false, false, "", nil, ConversationFacts{}, "", RequestFacts{Context: context.Background(), Images: []string{image}})
	if !slices.Equal(decider.requests["vision"].Images, []string{image}) {
		t.Fatalf("image not forwarded: %+v; errors: %v; calls: %v", decider.requests["vision"], result.SignalErrors, decider.calls)
	}
	if !slices.Contains(result.MatchedSystemOneRules, "vision") {
		t.Fatalf("image signal not evaluated: %+v", result.MatchedSystemOneRules)
	}
}
