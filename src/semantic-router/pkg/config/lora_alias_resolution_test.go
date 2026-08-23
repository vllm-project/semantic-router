package config

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func TestNativeLoRAAssignmentUsesModelBackendAndReasoning(t *testing.T) {
	document := strings.Replace(entrypointRulesYAML,
		"      reasoning: {type: reasoning_effort, efforts: [high]}\n",
		"      reasoning: {type: reasoning_effort, efforts: [high]}\n      loras: [general-expert]\n", 1)
	document = strings.Replace(document,
		"{provider: private-test, endpoint: http://model-c.example, model: model-c}",
		"{provider: private-test, endpoint: http://model-c.example, model: model-c, credential: base}", 1)
	document = strings.Replace(document,
		"  services:\n    backend_egress:",
		"  services:\n    backend_credentials:\n      base:\n        credential_adapter_id: bearer\n        secret_env: TEST_BASE_MODEL_KEY\n    backend_egress:", 1)
	document = strings.Replace(document,
		"              - model: model-c\n",
		"              - model: model-c\n                lora: general-expert\n", 1)

	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	var premium *RoutingRecipe
	for index := range cfg.Entrypoints[0].Rules {
		if cfg.Entrypoints[0].Rules[index].Name == "premium" {
			premium = cfg.Entrypoints[0].Rules[index].derivedRecipe
		}
	}
	if premium == nil || premium.Profile.Decisions[0].ModelRefs[0].LoRAName != "general-expert" {
		t.Fatalf("LoRA assignment was not compiled: %+v", premium)
	}
	if family := cfg.GetModelReasoningFamily("model-c"); family == nil || family.Parameter != "reasoning_effort" {
		t.Fatalf("Model reasoning family was not compiled: %#v", family)
	}
	var model routingsnapshot.Model
	for _, candidate := range cfg.RoutingSnapshot.Models {
		if candidate.Name == "model-c" {
			model = candidate
		}
	}
	if len(model.LoRAs) != 1 || model.LoRAs[0] != "general-expert" || model.Backends[0].ProviderCredentialID != "base" {
		t.Fatalf("Model backend contract was not preserved: %+v", model)
	}
}
