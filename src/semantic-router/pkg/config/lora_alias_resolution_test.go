package config

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func TestNativeLoRAAssignmentUsesModelBackendAndReasoning(t *testing.T) {
	document := strings.Replace(strictV03AuthoringYAML,
		"    - name: model-c\n      reasoning: {type: reasoning_effort, efforts: [high]}\n",
		"    - name: model-c\n      reasoning: {type: reasoning_effort, efforts: [high]}\n      loras: [{name: general-expert}]\n", 1)
	document = strings.Replace(document,
		"        - {provider: private-test, endpoint: http://model-c.example}\n",
		"        - {provider: private-test, endpoint: http://model-c.example, api_key_env: TEST_BASE_MODEL_KEY}\n", 1)
	document = strings.Replace(document,
		"          - model: model-c\n",
		"          - model: model-c\n            lora: general-expert\n", 1)

	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	recipe, found := cfg.RecipeForRequestModel("vllm-sr/edge")
	if !found || recipe.Profile.Decisions[0].ModelRefs[0].LoRAName != "general-expert" {
		t.Fatalf("LoRA assignment was not compiled: %+v", recipe)
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
	if len(model.LoRAs) != 1 || model.LoRAs[0] != "general-expert" ||
		len(model.Backends) != 1 || model.Backends[0].ProviderCredentialID == "" {
		t.Fatalf("Model backend contract was not preserved: %+v", model)
	}
	credential := cfg.BackendCredentials.File[model.Backends[0].ProviderCredentialID]
	if credential.SecretEnv != "TEST_BASE_MODEL_KEY" || credential.SecretValue != "" {
		t.Fatalf("Model backend credential was not compiled: %+v", credential)
	}
}
