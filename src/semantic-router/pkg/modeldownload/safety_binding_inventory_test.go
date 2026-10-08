package modeldownload

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestSafetyInventoryUsesPerRuleBindingAndRetainsOtherDefaultHead(t *testing.T) {
	cfg := &config.RouterConfig{MoMRegistry: map[string]string{"models/old-safety": "test/old-safety", "models/hazard": "test/hazard", "models/bound": "test/bound"}}
	cfg.SafetyModels = config.SafetyModelsConfig{Safety: config.SequenceHeadModelConfig{ModelID: "models/old-safety", UseCPU: true}, Hazard: config.SequenceHeadModelConfig{ModelID: "models/hazard", UseCPU: true}}
	cfg.SafetyRules = []config.SafetyRule{{Name: "risk", Hazard: &config.SafetyHazardRule{Labels: []string{"a", "b"}}}}
	cfg.Decisions = []config.Decision{{Name: "block", Rules: config.RuleNode{Type: config.SignalTypeSafety, Name: "risk"}}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"local": {Provider: config.ModelRuntimeProvider, Artifact: "models/bound"}, "remote": {Provider: "http", ExternalModel: "endpoint"}}
	cfg.ExternalModels = []config.ExternalModelConfig{{Name: "endpoint", ModelRole: config.ModelRoleClassification, ModelName: "binary", ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "localhost", Port: 8000}}}
	cfg.ModelBindings = map[string]config.ModelBinding{"safety.risk": {Deployment: "local", Adapter: "modernbert", Contract: config.RemoteClassifierContractLabelDistribution}}
	for _, remote := range []bool{false, true} {
		if remote {
			cfg.ModelBindings["safety.risk"] = config.ModelBinding{Deployment: "remote", Adapter: config.RemoteClassifierProtocolHTTPClassify, Contract: config.RemoteClassifierContractLabelDistribution}
		}
		specs, err := BuildModelSpecs(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if _, exists := findSpecByPath(specs, "models/old-safety"); exists {
			t.Fatal("downloaded overridden module")
		}
		if _, exists := findSpecByPath(specs, "models/hazard"); !exists {
			t.Fatalf("lost the implicit Hazard package: %+v", specs)
		}
		if _, exists := findSpecByPath(specs, "models/bound"); exists {
			t.Fatalf("the router downloaded a runtime-served deployment: %+v", specs)
		}
	}
	if cfg.SafetyModels.Safety.ModelID != "models/old-safety" {
		t.Fatal("mutated canonical defaults")
	}
}
