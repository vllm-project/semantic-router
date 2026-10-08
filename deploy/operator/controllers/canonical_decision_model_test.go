package controllers

import (
	"context"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// routerConfigOf builds the ConfigMap document of a resource and loads it as the Router does.
func routerConfigOf(t *testing.T, spec vllmv1alpha1.ConfigSpec) (*routerconfig.CanonicalConfig, *routerconfig.RouterConfig) {
	t.Helper()
	r := &SemanticRouterReconciler{}
	canonical, err := r.buildCanonicalConfig(context.Background(), &vllmv1alpha1.SemanticRouter{Spec: vllmv1alpha1.SemanticRouterSpec{Config: spec}})
	if err != nil {
		t.Fatal(err)
	}
	data, err := yaml.Marshal(canonical)
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := routerconfig.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("the Router refuses the operator's document: %v", err)
	}
	return canonical, cfg
}

func TestTheDecisionModelReachesEveryModuleWithItsThresholds(t *testing.T) {
	canonical, cfg := routerConfigOf(t, vllmv1alpha1.ConfigSpec{DecisionModel: "vela-2.0-9b"})
	if system := canonical.Global.ModelCatalog.System; system != (routerconfig.CanonicalSystemModels{DecisionModel: "Vela-2.0-9B"}) {
		t.Fatalf("the ConfigMap must name the decision model and pin no module, got %+v", system)
	}
	if cfg.DecisionModel != "Vela-2.0-9B" || cfg.PromptGuard.ModelID != "models/Vela-2.0-9B" || cfg.CategoryModel.ModelID != "models/Vela-2.0-9B" {
		t.Fatalf("every module runs the 9B, got guard %q domain %q", cfg.PromptGuard.ModelID, cfg.CategoryModel.ModelID)
	}
	want := routerconfig.ModuleThresholdsOf("models/Vela-2.0-9B")
	if cfg.PromptGuard.Threshold != want.PromptGuard || cfg.CategoryModel.Threshold != want.Domain || cfg.PIIModel.Threshold != want.PII ||
		cfg.HallucinationMitigation.FactCheckModel.Threshold != want.FactCheck || cfg.FeedbackDetector.Threshold != want.Feedback {
		t.Fatalf("modules the resource leaves at defaults take the 9B's thresholds %+v", want)
	}
}

func TestAGuardTheResourceSetsKeepsItsModelAndThreshold(t *testing.T) {
	_, cfg := routerConfigOf(t, vllmv1alpha1.ConfigSpec{
		DecisionModel: "Vela-2.0-0.8B",
		PromptGuard:   &vllmv1alpha1.PromptGuardConfig{Enabled: true, ModelID: "models/Vela-1.0-Encoder-307M-Guard", Threshold: "0.6", UseCPU: true},
	})
	if cfg.PromptGuard.ModelID != "models/Vela-1.0-Encoder-307M-Guard" || cfg.PromptGuard.Threshold != 0.6 {
		t.Fatalf("an explicit guard stays, got %q at %v", cfg.PromptGuard.ModelID, cfg.PromptGuard.Threshold)
	}
	if cfg.CategoryModel.ModelID != "models/Vela-2.0-0.8B" {
		t.Fatalf("the other modules follow the decision model, got %q", cfg.CategoryModel.ModelID)
	}
	_, cfg = routerConfigOf(t, vllmv1alpha1.ConfigSpec{
		DecisionModel: "Vela-2.0-0.8B",
		PromptGuard:   &vllmv1alpha1.PromptGuardConfig{Enabled: true, UseCPU: true},
	})
	if cfg.PromptGuard.ModelID != "models/Vela-2.0-0.8B" || cfg.PromptGuard.Threshold != routerconfig.ModuleThresholdsOf("models/Vela-2.0-0.8B").PromptGuard {
		t.Fatalf("a guard without model or threshold follows the decision model, got %q at %v", cfg.PromptGuard.ModelID, cfg.PromptGuard.Threshold)
	}
}

func TestAnUnknownDecisionModelFailsReconciliation(t *testing.T) {
	r := &SemanticRouterReconciler{}
	_, err := r.buildCanonicalConfig(context.Background(), &vllmv1alpha1.SemanticRouter{Spec: vllmv1alpha1.SemanticRouterSpec{Config: vllmv1alpha1.ConfigSpec{DecisionModel: "Decision-2.0-Kai-0.6B"}}})
	if err == nil || !strings.Contains(err.Error(), "config.decision_model") || !strings.Contains(err.Error(), "is a Decision 2.0 model") {
		t.Fatalf("want the decision model error, got %v", err)
	}
}
