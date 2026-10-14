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
	canonical, cfg := routerConfigOf(t, decisionDeploymentSpec(t, "primary", "vllm-sr/Vela-2.0-9B"))
	if system := canonical.Global.ModelCatalog.System; system != (routerconfig.CanonicalSystemModels{DecisionModel: routerconfig.DecisionModelBinding{Deployment: "primary"}}) {
		t.Fatalf("the ConfigMap must name the decision model and pin no module, got %+v", system)
	}
	if cfg.DecisionModel != "primary" || cfg.PromptGuard.ModelID != "models/Vela-2.0-9B" || cfg.CategoryModel.ModelID != "models/Vela-2.0-9B" {
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
		DecisionModel:    &vllmv1alpha1.DecisionModelBinding{Deployment: "primary"},
		ModelDeployments: rawCanonicalRoutingJSON(t, `{"primary":{"provider":"model_runtime","artifact":"vllm-sr/Vela-2.0-0.8B"}}`),
		PromptGuard:      &vllmv1alpha1.PromptGuardConfig{Enabled: true, ModelID: "models/Vela-1.0-Encoder-307M-Guard", Threshold: "0.6", UseCPU: true},
	})
	if cfg.PromptGuard.ModelID != "models/Vela-1.0-Encoder-307M-Guard" || cfg.PromptGuard.Threshold != 0.6 {
		t.Fatalf("an explicit guard stays, got %q at %v", cfg.PromptGuard.ModelID, cfg.PromptGuard.Threshold)
	}
	if cfg.CategoryModel.ModelID != "models/Vela-2.0-0.8B" {
		t.Fatalf("the other modules follow the decision model, got %q", cfg.CategoryModel.ModelID)
	}
	_, cfg = routerConfigOf(t, vllmv1alpha1.ConfigSpec{
		DecisionModel:    &vllmv1alpha1.DecisionModelBinding{Deployment: "primary"},
		ModelDeployments: rawCanonicalRoutingJSON(t, `{"primary":{"provider":"model_runtime","artifact":"vllm-sr/Vela-2.0-0.8B"}}`),
		PromptGuard:      &vllmv1alpha1.PromptGuardConfig{Enabled: true, UseCPU: true},
	})
	if cfg.PromptGuard.ModelID != "models/Vela-2.0-0.8B" || cfg.PromptGuard.Threshold != routerconfig.ModuleThresholdsOf("models/Vela-2.0-0.8B").PromptGuard {
		t.Fatalf("a guard without model or threshold follows the decision model, got %q at %v", cfg.PromptGuard.ModelID, cfg.PromptGuard.Threshold)
	}
}

func decisionDeploymentSpec(t *testing.T, name, artifact string) vllmv1alpha1.ConfigSpec {
	t.Helper()
	return vllmv1alpha1.ConfigSpec{
		DecisionModel:    &vllmv1alpha1.DecisionModelBinding{Deployment: name},
		ModelDeployments: rawCanonicalRoutingJSON(t, `{"`+name+`":{"provider":"model_runtime","artifact":"`+artifact+`"}}`),
	}
}

func TestAnUnknownDecisionDeploymentFailsReconciliation(t *testing.T) {
	r := &SemanticRouterReconciler{}
	_, err := r.buildCanonicalConfig(context.Background(), &vllmv1alpha1.SemanticRouter{Spec: vllmv1alpha1.SemanticRouterSpec{Config: vllmv1alpha1.ConfigSpec{DecisionModel: &vllmv1alpha1.DecisionModelBinding{Deployment: "missing"}}}})
	if err == nil || !strings.Contains(err.Error(), "config.decision_model.deployment") {
		t.Fatalf("want exact deployment error, got %v", err)
	}
}

func TestDecisionModelFamilyIsNotAnOperatorAllowlist(t *testing.T) {
	_, cfg := routerConfigOf(t, decisionDeploymentSpec(t, "kai", "vllm-sr/Decision-2.0-Kai-0.6B"))
	if cfg.DecisionModel != "kai" {
		t.Fatalf("default reference changed: %q", cfg.DecisionModel)
	}
}
