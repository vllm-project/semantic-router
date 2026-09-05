package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v2"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
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
	if cfg.DefaultReasoningEffort != "" {
		t.Fatalf("omitted reasoning effort became %q", cfg.DefaultReasoningEffort)
	}

	exported := CanonicalConfigFromRouterConfig(cfg)
	if exported.Providers.Defaults.DefaultReasoningEffort != "" {
		t.Fatalf("exported unconfigured reasoning effort %q", exported.Providers.Defaults.DefaultReasoningEffort)
	}
	assertExportedAuthoredModel(t, &exported)
	assertReplayedAuthoredModel(t, &exported)
	assertBuiltInReasoningStaysInternal(t)
}

func TestCanonicalOmittedDefaultsRetainCatalogReasoningDefault(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  models:
    - name: qwen
      catalog: qwen/qwen3.8-max
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: dashscope
routing: {}
`))
	if err != nil {
		t.Fatalf("parse catalog-backed config without defaults: %v", err)
	}
	if cfg.DefaultReasoningEffort != "" {
		t.Fatalf("omitted reasoning effort became %q", cfg.DefaultReasoningEffort)
	}
	if family := cfg.ReasoningFamilies["qwen3.8"]; family.Default != "xhigh" {
		t.Fatalf("catalog reasoning default = %q, want xhigh", family.Default)
	}

	encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatalf("marshal canonical config: %v", err)
	}
	var document map[interface{}]interface{}
	if err := yaml.Unmarshal(encoded, &document); err != nil {
		t.Fatalf("unmarshal canonical config: %v", err)
	}
	providers := requireYAMLMap(t, document["providers"], "providers")
	if _, present := providers["defaults"]; present {
		t.Fatalf("unconfigured provider defaults were written back:\n%s", encoded)
	}
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

func TestCatalogBackedModelMaterializesAPIFormat(t *testing.T) {
	for _, test := range []struct {
		name       string
		catalog    string
		apiFormat  string
		wantFormat string
		wantWire   string
	}{
		{
			name:       "provider default for chat binding",
			catalog:    "openai/gpt-5.4",
			wantFormat: APIFormatOpenAI,
			wantWire:   "openai/chat-completions@1",
		},
		{
			name:       "explicit responses override",
			catalog:    "openai/gpt-5.4",
			apiFormat:  APIFormatResponses,
			wantFormat: APIFormatResponses,
			wantWire:   "openai/responses@1",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := CanonicalProviderModel{
				Name: "production", Catalog: test.catalog, APIFormat: test.apiFormat,
				BackendRefs: []CanonicalBackendRef{{Name: "primary", Provider: "openai"}},
			}
			input, err := canonicalCatalogInput(&CanonicalConfig{
				Version:   "v0.3",
				Providers: CanonicalProviders{Models: []CanonicalProviderModel{model}},
			})
			if err != nil {
				t.Fatal(err)
			}
			registry, err := modelcatalog.BuiltIn()
			if err != nil {
				t.Fatal(err)
			}
			effective, err := registry.Compile(input)
			if err != nil {
				t.Fatal(err)
			}
			cfg := &RouterConfig{}
			if err := applyEffectiveModelRegistry(cfg, effective, []CanonicalProviderModel{model}); err != nil {
				t.Fatal(err)
			}
			if got := cfg.GetModelAPIFormat("production"); got != test.wantFormat {
				t.Fatalf("api format = %q, want %q", got, test.wantFormat)
			}
			profile, ok := cfg.ProviderProfiles["production_primary"]
			if !ok || profile.Protocol != test.wantWire {
				t.Fatalf("provider profile = %+v, want protocol %q", profile, test.wantWire)
			}
		})
	}
}

func TestParseYAMLBytesHonorsExplicitResponsesForCatalogModel(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  models:
    - name: production
      catalog: openai/gpt-5.4
      api_format: responses
      backend_refs:
        - name: primary
          provider: openai
routing: {}
`))
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.GetModelAPIFormat("production"); got != APIFormatResponses {
		t.Fatalf("api format = %q, want %q", got, APIFormatResponses)
	}
	profile, ok := cfg.ProviderProfiles["production_primary"]
	if !ok || profile.Protocol != "openai/responses@1" {
		t.Fatalf("provider profile = %+v, want Responses protocol", profile)
	}
}

func TestCatalogInputRejectsUnsupportedExplicitAPIFormat(t *testing.T) {
	_, err := canonicalCatalogInput(&CanonicalConfig{
		Providers: CanonicalProviders{Models: []CanonicalProviderModel{{
			Name: "production", Catalog: "openai/gpt-5.4", APIFormat: "openai-ish",
		}}},
	})
	if err == nil || !strings.Contains(err.Error(), "api_format \"openai-ish\" is unsupported") {
		t.Fatalf("expected unsupported api_format error, got %v", err)
	}
}

func TestCanonicalBackendExplicitEmptyAuthPrefixOverridesCatalogDefault(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  models:
    - name: private
      provider_model_id: private
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: https://private.example/v1
          auth_header: x-api-key
          auth_prefix: ""
          api_key: static-secret
routing: {}
`))
	if err != nil {
		t.Fatal(err)
	}
	profile, ok := cfg.ProviderProfiles["private_primary"]
	if !ok {
		t.Fatal("materialized provider profile is missing")
	}
	header, prefix, err := profile.ResolveAuthHeader()
	if err != nil {
		t.Fatal(err)
	}
	if header != "x-api-key" || prefix != "" || !profile.AuthPrefixSet {
		t.Fatalf("resolved auth = (%q, %q, set=%v), want raw x-api-key", header, prefix, profile.AuthPrefixSet)
	}
	if got := cfg.GetModelAccessKeyForProvider("private", "vllm"); got != "static-secret" {
		t.Fatalf("runtime credential = %q, want static-secret", got)
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

func TestCatalogInputKeepsImplicitCustomCardSeparateFromBuiltInIdentity(t *testing.T) {
	input, err := canonicalCatalogInput(&CanonicalConfig{
		Providers: CanonicalProviders{Models: []CanonicalProviderModel{{
			Name: "openai/gpt-5.4",
			BackendRefs: []CanonicalBackendRef{{
				Name: "local", Endpoint: "127.0.0.1:8000", Provider: "vllm",
			}},
		}}},
	})
	if err != nil {
		t.Fatalf("canonicalCatalogInput() error = %v", err)
	}
	registry, err := modelcatalog.BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	effective, err := registry.Compile(input)
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	model, ok := effective.Model("openai/gpt-5.4")
	if !ok {
		t.Fatal("custom model is missing")
	}
	if model.Card.Provenance["id"] != modelcatalog.SourceOperator {
		t.Fatalf("custom card inherited built-in identity: %+v", model.Card.Provenance)
	}
	result := model.Indices["vllm-sr/intelligence@1.0.0"]
	if result.Score != nil || result.Status != "missing" || len(result.Provenance) != 0 {
		t.Fatalf("custom card inherited built-in evidence: %+v", result)
	}
}

func TestCanonicalConfigRejectsMixedOwnershipForOneCardIdentity(t *testing.T) {
	err := validateCanonicalContract(&CanonicalConfig{
		Version: "v0.3",
		Providers: CanonicalProviders{Models: []CanonicalProviderModel{
			{
				Name: "openai/gpt-5.4",
				BackendRefs: []CanonicalBackendRef{{
					Name: "local", Endpoint: "127.0.0.1:8000", Provider: "vllm",
				}},
			},
			{
				Name: "production-gpt", Catalog: "openai/gpt-5.4",
				BackendRefs: []CanonicalBackendRef{{
					Name: "cloud", Provider: "openai",
				}},
			},
		}},
	})
	if err == nil || !strings.Contains(err.Error(), "cannot represent both a catalog-backed and custom model") {
		t.Fatalf("validateCanonicalContract() error = %v", err)
	}
}

func TestCanonicalExportPreservesOperatorModelCardReleaseMetadata(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  models:
    - name: private-model
      backend_refs:
        - provider: vllm
          endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: private-model
      revision: private-model-v2
      released_at: 2026-08-31
      knowledge_cutoff: 2026-06
      lifecycle: experimental
`))
	if err != nil {
		t.Fatal(err)
	}

	exported := CanonicalConfigFromRouterConfig(cfg)
	if len(exported.Routing.ModelCards) != 1 {
		t.Fatalf("exported model cards = %#v, want one card", exported.Routing.ModelCards)
	}
	card := exported.Routing.ModelCards[0]
	for _, field := range []struct{ name, got, want string }{
		{"revision", card.Revision, "private-model-v2"},
		{"released_at", card.ReleasedAt, "2026-08-31"},
		{"knowledge_cutoff", card.KnowledgeCutoff, "2026-06"},
		{"lifecycle", card.Lifecycle, "experimental"},
	} {
		if field.got != field.want {
			t.Fatalf("exported %s = %q, want %q", field.name, field.got, field.want)
		}
	}

	encoded, err := yaml.Marshal(exported)
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := ParseYAMLBytes(encoded)
	if err != nil {
		t.Fatalf("reparse exported config: %v\n%s", err, encoded)
	}
	replayedModel, ok := replayed.EffectiveModelRegistry.Model("private-model")
	if !ok {
		t.Fatal("replayed custom model is missing")
	}
	replayedCard := replayedModel.Card.Card
	for _, field := range []struct{ name, got, want string }{
		{"revision", replayedCard.Revision, "private-model-v2"},
		{"released_at", replayedCard.ReleasedAt, "2026-08-31"},
		{"knowledge_cutoff", replayedCard.KnowledgeCutoff, "2026-06"},
		{"lifecycle", replayedCard.Lifecycle, "experimental"},
	} {
		if field.got != field.want {
			t.Fatalf("replayed %s = %q, want %q", field.name, field.got, field.want)
		}
	}
}

func TestCanonicalExportDeduplicatesCatalogOverrideSharedByAliases(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.3
providers:
  models:
    - name: primary-gpt
      catalog: openai/gpt-5.4
      backend_refs:
        - name: primary
          provider: openai
    - name: fallback-gpt
      catalog: openai/gpt-5.4
      backend_refs:
        - name: fallback
          provider: openai
routing:
  modelCards:
    - name: openai/gpt-5.4
      description: Restricted production profile
`))
	if err != nil {
		t.Fatal(err)
	}

	exported := CanonicalConfigFromRouterConfig(cfg)
	if len(exported.Routing.ModelCards) != 1 {
		t.Fatalf("exported model cards = %#v, want one shared catalog override", exported.Routing.ModelCards)
	}
	if card := exported.Routing.ModelCards[0]; card.Name != "openai/gpt-5.4" || card.Description != "Restricted production profile" {
		t.Fatalf("exported shared catalog override = %#v", card)
	}

	encoded, err := yaml.Marshal(exported)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ParseYAMLBytes(encoded); err != nil {
		t.Fatalf("reparse exported shared override: %v\n%s", err, encoded)
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
