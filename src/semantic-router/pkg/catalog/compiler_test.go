package catalog

import (
	"strings"
	"testing"
)

func TestBuiltInRegistryOwnsProviderProtocolAndPresentation(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	provider, ok := registry.Provider("openai")
	if !ok {
		t.Fatal("openai provider is missing")
	}
	if provider.DefaultProtocol != "openai/chat-completions@1" {
		t.Fatalf("default protocol = %q", provider.DefaultProtocol)
	}
	if provider.Auth.Header != "Authorization" || provider.Auth.InjectedHeader != "x-user-openai-key" {
		t.Fatalf("unexpected auth contract: %+v", provider.Auth)
	}
	if provider.Presentation.Logo == "" {
		t.Fatal("provider logo metadata is missing")
	}
	protocol, ok := registry.Protocol(provider.DefaultProtocol)
	if !ok || len(protocol.Operations) == 0 || protocol.Operations[0].Path != "/v1/chat/completions" {
		t.Fatalf("unexpected protocol contract: %+v", protocol)
	}
}

func TestProviderLookupReturnsDefensiveDefaultHeaders(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	provider, ok := registry.Provider("anthropic")
	if !ok || provider.DefaultHeaders["anthropic-version"] == "" {
		t.Fatalf("unexpected provider headers: %+v", provider.DefaultHeaders)
	}
	provider.DefaultHeaders["anthropic-version"] = "mutated"
	provider.SupportedOperations[0] = "mutated"
	reloaded, _ := registry.Provider("anthropic")
	if reloaded.DefaultHeaders["anthropic-version"] != "2023-06-01" {
		t.Fatalf("registry headers were mutated: %+v", reloaded.DefaultHeaders)
	}
	if reloaded.SupportedOperations[0] == "mutated" {
		t.Fatalf("registry operations were mutated: %+v", reloaded.SupportedOperations)
	}
}

func TestCompileCustomRuntimeCardAndBuiltInReasoning(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	displayName := "Private Qwen"
	description := "Operator-owned AWQ model"
	context := 131072
	maxOutput := 32768
	capabilities := []string{"chat", "tools", "reasoning"}
	modalities := Modalities{Input: []string{"text"}, Output: []string{"text"}}
	reasoning := "qwen3"

	effective, err := registry.Compile(CompileInput{
		Defaults: Defaults{Model: "private", QualityIndex: "vllm-sr/intelligence@1.0.0"},
		Providers: []ProviderInstance{{
			Name: "lab", Catalog: "vllm", BaseURL: "http://model-gateway.example/v1",
		}},
		Models: []ModelAlias{{
			Name: "private", Catalog: "acme/qwen-custom",
			Providers: []ModelProviderBinding{{Name: "lab", ModelID: "qwen-custom-awq"}},
		}},
		ModelCards: []ModelCardOverlay{{
			Name: "acme/qwen-custom", DisplayName: &displayName, Description: &description,
			ContextWindowSize: &context, MaxOutputTokens: &maxOutput,
			Capabilities: &capabilities, Modalities: &modalities, ReasoningFamily: &reasoning,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	model, ok := effective.Model("private")
	if !ok {
		t.Fatal("compiled alias is missing")
	}
	if model.Catalog != "acme/qwen-custom" || model.Card.Card.ReasoningFamily != "qwen3" {
		t.Fatalf("unexpected effective model: %+v", model)
	}
	if model.Card.Provenance["capabilities"] != SourceOperator {
		t.Fatalf("capability provenance = %q", model.Card.Provenance["capabilities"])
	}
	if result := model.Indices["vllm-sr/intelligence@1.0.0"]; result.Status != "missing" || result.Score != nil {
		t.Fatalf("missing evidence became a score: %+v", result)
	}
}

func TestCompileRejectsImplicitNameJoin(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	_, err = registry.Compile(CompileInput{
		Providers: []ProviderInstance{{Name: "primary", Catalog: "openai"}},
		Models:    []ModelAlias{{Name: "frontier", Catalog: "frontier", Providers: []ModelProviderBinding{{Name: "primary", ModelID: "frontier"}}}},
	})
	if err == nil || !strings.Contains(err.Error(), "has no built-in or handwritten model card") {
		t.Fatalf("expected explicit catalog/card error, got %v", err)
	}
}

func TestCompileRejectsUnverifiedCapabilityWidening(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	capabilities := []string{"chat", "reasoning", "tools", "multimodal", "unverified-new-capability"}
	_, err = registry.Compile(CompileInput{ModelCards: []ModelCardOverlay{{Name: "vllm-sr/mom-v1-blend", Capabilities: &capabilities}}})
	if err == nil || !strings.Contains(err.Error(), "without verification") {
		t.Fatalf("expected widening error, got %v", err)
	}
}

func TestIndexComputationPreservesMissingAndLineage(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	displayName := "Measured"
	description := "Measured custom model"
	capabilities := []string{"chat"}
	modalities := Modalities{Input: []string{"text"}, Output: []string{"text"}}
	effective, err := registry.Compile(CompileInput{
		Providers:  []ProviderInstance{{Name: "lab", Catalog: "vllm", BaseURL: "http://127.0.0.1:8000/v1"}},
		Models:     []ModelAlias{{Name: "measured", Catalog: "acme/measured", Providers: []ModelProviderBinding{{Name: "lab", ModelID: "measured"}}}},
		ModelCards: []ModelCardOverlay{{Name: "acme/measured", DisplayName: &displayName, Description: &description, Capabilities: &capabilities, Modalities: &modalities}},
		Evaluations: EvaluationConfig{
			Benchmarks: []BenchmarkDefinition{{ID: "acme/support@1.0.0", DisplayName: "Support", Domain: "support", Metrics: []BenchmarkMetric{{ID: "resolution", Unit: "proportion", Direction: "higher_is_better", Range: [2]float64{0, 1}}}}},
			Records:    []EvaluationRecord{{ID: "acme/run-1", Model: "acme/measured", Status: "available", Metrics: map[string]float64{"acme/support@1.0.0#resolution": 0.82}, Evidence: EvaluationEvidence{Provenance: "operator", Verification: "reproduced", Redistributable: true}}},
			Indices:    []IndexDefinition{{ID: "acme/readiness@1.0.0", DisplayName: "Readiness", Aggregation: "weighted_mean", Scale: [2]float64{0, 100}, Missing: MissingPolicy{Policy: "require_all"}, Components: []IndexComponent{{Metric: "acme/support@1.0.0#resolution", Weight: 1, Normalization: Normalization{Type: "identity"}}}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	model, _ := effective.Model("measured")
	custom := model.Indices["acme/readiness@1.0.0"]
	if custom.Score == nil || *custom.Score != 82 || custom.Coverage != 1 || len(custom.Provenance) != 1 || custom.Provenance[0] != "acme/run-1" {
		t.Fatalf("unexpected custom score: %+v", custom)
	}
	builtin := model.Indices["vllm-sr/intelligence@1.0.0"]
	if builtin.Score != nil || builtin.Status != "missing" {
		t.Fatalf("partial unrelated evidence became a headline score: %+v", builtin)
	}
}

func TestVirtualModelIndicesAreNotApplicable(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	effective, err := registry.Compile(CompileInput{
		Models: []ModelAlias{{Name: "auto", Catalog: "vllm-sr/mom-v1-blend"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	model, ok := effective.Model("auto")
	if !ok {
		t.Fatal("compiled virtual alias is missing")
	}
	result := model.Indices["vllm-sr/intelligence@1.0.0"]
	if result.Status != "not_applicable" || result.Score != nil {
		t.Fatalf("virtual model received an intelligence score: %+v", result)
	}
}

func TestEffectiveModelLookupIsDefensive(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	effective, err := registry.Compile(CompileInput{
		Providers: []ProviderInstance{{
			Name: "private-anthropic", Catalog: "anthropic", BaseURL: "https://gateway.example/v1",
			Headers: map[string]string{"x-tenant": "original"},
		}},
		Models: []ModelAlias{{
			Name: "private", Catalog: "acme/private",
			Providers: []ModelProviderBinding{{
				Name: "private-anthropic", ModelID: "private-v1",
				ExternalModelIDs: map[string]string{"anthropic": "private-v1"},
			}},
			BindingDefaults: ModelProviderBinding{
				ExternalModelIDs: map[string]string{"default": "private-v1"},
			},
		}},
		ModelCards: []ModelCardOverlay{{Name: "acme/private"}},
	})
	if err != nil {
		t.Fatal(err)
	}

	first, ok := effective.Model("private")
	if !ok {
		t.Fatal("compiled model is missing")
	}
	first.Providers[0].Binding.ExternalModelIDs["anthropic"] = "mutated"
	first.Providers[0].Provider.Instance.Headers["x-tenant"] = "mutated"
	first.Providers[0].Provider.Definition.DefaultHeaders["anthropic-version"] = "mutated"
	first.BindingDefaults.ExternalModelIDs["default"] = "mutated"
	indexID := "vllm-sr/intelligence@1.0.0"
	first.Indices[indexID].Components[0].Status = "mutated"

	second, ok := effective.Model("private")
	if !ok {
		t.Fatal("compiled model disappeared")
	}
	provider := second.Providers[0]
	if provider.Binding.ExternalModelIDs["anthropic"] != "private-v1" ||
		provider.Provider.Instance.Headers["x-tenant"] != "original" ||
		provider.Provider.Definition.DefaultHeaders["anthropic-version"] != "2023-06-01" ||
		second.BindingDefaults.ExternalModelIDs["default"] != "private-v1" ||
		second.Indices[indexID].Components[0].Status == "mutated" {
		t.Fatalf("effective registry was mutated through lookup: %+v", second)
	}
}
