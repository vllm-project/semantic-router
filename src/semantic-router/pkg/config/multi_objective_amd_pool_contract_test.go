package config

import (
	"slices"
	"testing"

	yamlv3 "gopkg.in/yaml.v3"
)

func TestMultiObjectiveRecipePreservesAMDModelPool(t *testing.T) {
	const asset = "config/recipes/multi-objective/config.yaml"
	expectedEndpoints := map[string][]string{
		"local/omni":                       {"vllm-omni:8000"},
		"local/qwen3.5-122b-frontier":      {"vllm:8000"},
		"local/qwen3.5-9b-economy":         {"vllm-qwen35-economy:8000"},
		"local/qwen3.5-9b-economy-replica": {"vllm-qwen35-economy-replica:8000"},
		"local/qwen3.5-9b-private":         {"vllm-qwen35-economy:8000"},
		"local/qwen3.6-27b-coder":          {"vllm-qwen36-coder:8000"},
		"local/qwen3.6-35b-flash":          {"vllm-qwen36-flash:8000"},
		"local/gemma4-26b-balanced":        {"vllm-gemma4-balanced:8000"},
		"local/deepseek-v4-flash-analyst":  {"vllm-deepseek-v4:8000"},
	}

	var recipe CanonicalConfig
	if err := yamlv3.Unmarshal(mustReadRepoFile(t, asset), &recipe); err != nil {
		t.Fatalf("failed to decode %s: %v", asset, err)
	}

	if recipe.Providers.Defaults.DefaultModel != "local/qwen3.5-9b-economy" {
		t.Fatalf("unexpected multi-objective default model %q", recipe.Providers.Defaults.DefaultModel)
	}
	if len(recipe.Providers.Models) != len(expectedEndpoints) {
		t.Fatalf("expected %d logical model lanes, got %d", len(expectedEndpoints), len(recipe.Providers.Models))
	}

	physicalEndpoints := make([]string, 0, len(recipe.Providers.Models))
	for _, model := range recipe.Providers.Models {
		expectedEndpoint, ok := expectedEndpoints[model.Name]
		if !ok {
			t.Fatalf("unexpected multi-objective model lane %q", model.Name)
		}
		if len(model.BackendRefs) != len(expectedEndpoint) {
			t.Fatalf("model lane %q backend count = %d, want %d", model.Name, len(model.BackendRefs), len(expectedEndpoint))
		}
		actualEndpoints := make([]string, 0, len(model.BackendRefs))
		for _, backend := range model.BackendRefs {
			actualEndpoints = append(actualEndpoints, backend.Endpoint)
			physicalEndpoints = append(physicalEndpoints, backend.Endpoint)
		}
		slices.Sort(actualEndpoints)
		slices.Sort(expectedEndpoint)
		if !slices.Equal(actualEndpoints, expectedEndpoint) {
			t.Fatalf("model lane %q endpoints = %v, want %v", model.Name, actualEndpoints, expectedEndpoint)
		}
	}

	slices.Sort(physicalEndpoints)
	physicalEndpoints = slices.Compact(physicalEndpoints)
	if len(physicalEndpoints) != 8 {
		t.Fatalf("expected eight serving endpoints, got %v", physicalEndpoints)
	}
	if got := recipe.Global.Integrations.Looper.Endpoint; got != "http://vllm-sr-envoy-container:8899/v1/chat/completions" {
		t.Fatalf("Looper endpoint must re-enter Envoy for cross-backend orchestration, got %q", got)
	}
}
