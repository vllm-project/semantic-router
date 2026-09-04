package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestCanonicalExportKeepsRouterReplayDisabled(t *testing.T) {
	cfg := &RouterConfig{
		RouterReplay: RouterReplayConfig{
			Enabled:      false,
			StoreBackend: "memory",
			TTLSeconds:   600,
		},
	}

	encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatalf("marshal canonical config: %v", err)
	}

	var document map[interface{}]interface{}
	if err := yaml.Unmarshal(encoded, &document); err != nil {
		t.Fatalf("unmarshal canonical config: %v", err)
	}
	global := requireYAMLMap(t, document["global"], "global")
	services := requireYAMLMap(t, global["services"], "global.services")
	replay := requireYAMLMap(t, services["router_replay"], "global.services.router_replay")
	if got, ok := replay["enabled"]; !ok || got != false {
		t.Fatalf("router_replay.enabled = %#v, want explicit false; YAML:\n%s", got, encoded)
	}
}

func TestCanonicalExportPreservesAuthoredModelDeclaration(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  defaults:
    model: private-model
  models:
    - name: private-model
      api_format: openai
      reasoning:
        type: reasoning_effort
        parameter: reasoning_effort
        levels: [low, medium, high]
        default: medium
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
          api_key_env: PRIVATE_MODEL_API_KEY
routing: {}
`))
	if err != nil {
		t.Fatalf("parse inline reasoning config: %v", err)
	}

	exported := CanonicalConfigFromRouterConfig(cfg)
	assertExportedAuthoredModel(t, &exported)
	assertReplayedAuthoredModel(t, &exported)
	assertBuiltInReasoningStaysInternal(t)
}

func assertExportedAuthoredModel(t *testing.T, exported *CanonicalConfig) {
	t.Helper()
	if len(exported.Providers.Models) != 1 {
		t.Fatalf("exported providers.models = %#v", exported.Providers.Models)
	}
	reasoning := exported.Providers.Models[0].Reasoning
	if reasoning == nil || reasoning.Type != "reasoning_effort" || reasoning.Default != "medium" {
		t.Fatalf("exported authored reasoning = %#v", reasoning)
	}
	if reasoning.Family != "" || len(reasoning.Levels) != 3 {
		t.Fatalf("exported inline reasoning changed shape: %#v", reasoning)
	}
	if got := exported.Providers.Models[0].BackendRefs[0].APIKeyEnv; got != "PRIVATE_MODEL_API_KEY" {
		t.Fatalf("exported api_key_env = %q", got)
	}
}

func assertReplayedAuthoredModel(t *testing.T, exported *CanonicalConfig) {
	t.Helper()
	encoded, err := yaml.Marshal(exported)
	if err != nil {
		t.Fatalf("marshal canonical config: %v", err)
	}
	replayed, err := ParseYAMLBytes(encoded)
	if err != nil {
		t.Fatalf("reparse exported inline reasoning: %v\n%s", err, encoded)
	}
	if got := replayed.ModelConfig["private-model"].AuthoredModel; got == nil || got.Reasoning == nil || got.Reasoning.Default != "medium" {
		t.Fatalf("replayed authored model = %#v", got)
	}
}

func assertBuiltInReasoningStaysInternal(t *testing.T) {
	t.Helper()
	// Effective built-in metadata remains internal unless the operator authored
	// an explicit reasoning binding on providers.models.
	builtInDerived := canonicalProviderModelFromRuntime(
		"built-in",
		ModelParams{Catalog: "vendor/model", ReasoningFamily: "vendor-family"},
		nil,
		nil,
		nil,
	)
	if builtInDerived.Reasoning != nil {
		t.Fatalf("derived built-in reasoning leaked into user config: %#v", builtInDerived.Reasoning)
	}
}

func TestMergeProviderHeadersUsesCatalogDefaultsAndOperatorOverrides(t *testing.T) {
	merged := mergeProviderHeaders(
		map[string]string{"anthropic-version": "2023-06-01", "x-owner": "catalog"},
		map[string]string{"x-owner": "operator"},
	)
	if merged["anthropic-version"] != "2023-06-01" || merged["x-owner"] != "operator" {
		t.Fatalf("merged headers = %#v", merged)
	}
}

func TestCatalogBackedModelRejectsOperatorReasoningDefinition(t *testing.T) {
	_, err := canonicalCatalogInput(&CanonicalConfig{
		Providers: CanonicalProviders{Models: []CanonicalProviderModel{{
			Name:      "built-in",
			Catalog:   "vllm-sr/mom-v1-lite",
			Reasoning: &CanonicalReasoning{Family: "qwen3"},
		}}},
	})
	if err == nil || !strings.Contains(err.Error(), "only valid for a custom model without catalog") {
		t.Fatalf("canonicalCatalogInput() error = %v", err)
	}
}

func TestCatalogInputRejectsAliasNamedBuiltInOverride(t *testing.T) {
	_, err := canonicalCatalogInput(&CanonicalConfig{
		Providers: CanonicalProviders{Models: []CanonicalProviderModel{{
			Name:    "production",
			Catalog: "vllm-sr/mom-v1-lite",
		}}},
		Routing: CanonicalRouting{ModelCards: []RoutingModel{{Name: "production"}}},
	})
	if err == nil || !strings.Contains(err.Error(), "does not match a providers.models catalog identity") {
		t.Fatalf("canonicalCatalogInput() error = %v", err)
	}
}

func TestCatalogInputAllowsDeclaredLoRAAliasCard(t *testing.T) {
	_, err := canonicalCatalogInput(&CanonicalConfig{
		Providers: CanonicalProviders{Models: []CanonicalProviderModel{{
			Name: "base-model",
		}}},
		Routing: CanonicalRouting{ModelCards: []RoutingModel{
			{Name: "base-model", LoRAs: []LoRAAdapter{{Name: "general-expert"}}},
			{Name: "general-expert"},
		}},
	})
	if err != nil {
		t.Fatalf("canonicalCatalogInput() error = %v", err)
	}
}

func requireYAMLMap(t *testing.T, value interface{}, path string) map[interface{}]interface{} {
	t.Helper()
	result, ok := value.(map[interface{}]interface{})
	if !ok {
		t.Fatalf("%s = %#v, want YAML mapping", path, value)
	}
	return result
}
