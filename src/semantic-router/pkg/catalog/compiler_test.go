package catalog

import (
	"strings"
	"testing"
)

func TestValidateEndpointURLRejectsQueryAndInvalidPort(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{name: "query", raw: "https://api.example.test/v1?region=west", want: "query parameters"},
		{name: "zero port", raw: "http://api.example.test:0/v1", want: "between 1 and 65535"},
		{name: "large port", raw: "http://api.example.test:65536/v1", want: "between 1 and 65535"},
		{name: "negative port", raw: "http://api.example.test:-1/v1", want: "absolute http(s) URL"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateEndpointURL(test.raw)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("validateEndpointURL(%q) error = %v, want %q", test.raw, err, test.want)
			}
		})
	}
}

func TestValidateEndpointURLAcceptsIPv6AndTemplates(t *testing.T) {
	for _, raw := range []string{
		"http://[::1]:8000/v1",
		"https://${AZURE_RESOURCE}.openai.azure.com/openai/deployments/${AZURE_DEPLOYMENT}",
	} {
		if err := validateEndpointURL(raw); err != nil {
			t.Fatalf("validateEndpointURL(%q) error = %v", raw, err)
		}
	}
}

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

func TestRegistryFiltersUnscoredIndexCoverageRows(t *testing.T) {
	document := snapshot{
		SchemaVersion: "vllm-sr/model-catalog/v2",
		Catalogs:      []CatalogHeader{{CatalogVersion: "latest"}},
		IndexResults: []IndexResult{
			{Model: "acme/model", Index: "acme/index@1.0.0", Status: "partial", Coverage: 0.4},
			{Model: "acme/model", Index: "acme/index@1.0.0", Status: "missing"},
		},
	}
	registry, err := registryFromSnapshot(document, "sha256:test")
	if err != nil {
		t.Fatalf("valid coverage rows were rejected: %v", err)
	}
	if len(registry.indexResults) != 0 {
		t.Fatalf("unscored rows became routing priors: %+v", registry.indexResults)
	}
}

func TestRegistryRejectsInvalidIndexResultStates(t *testing.T) {
	score := 40.0
	invalid := []IndexResult{
		{Status: "partial", Score: &score, Coverage: 0.4},
		{Status: "partial"},
		{Status: "missing", Coverage: 0.4},
		{Status: "available", Score: &score},
		{Status: "available", Coverage: 1},
	}
	for _, result := range invalid {
		document := snapshot{
			SchemaVersion: "vllm-sr/model-catalog/v2",
			Catalogs:      []CatalogHeader{{CatalogVersion: "latest"}},
			IndexResults:  []IndexResult{result},
		}
		if _, err := registryFromSnapshot(document, "sha256:test"); err == nil ||
			!strings.Contains(err.Error(), "invalid index result") {
			t.Fatalf("invalid index result was accepted: %+v: %v", result, err)
		}
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

func TestBenchmarkLookupReturnsDefensiveTagsAndNormalization(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}

	core, ok := registry.Benchmark("tiger-ai-lab/mmlu-pro@1.0.0")
	if !ok || len(core.Tags) != 1 || core.Tags[0] != "core" {
		t.Fatalf("unexpected core benchmark tags: %+v", core.Tags)
	}
	core.Tags[0] = "mutated"
	reloadedCore, _ := registry.Benchmark("tiger-ai-lab/mmlu-pro@1.0.0")
	if reloadedCore.Tags[0] != "core" {
		t.Fatalf("registry benchmark tags were mutated: %+v", reloadedCore.Tags)
	}

	elo, ok := registry.Benchmark("artificial-analysis/gdpval-aa@2.0.0")
	if !ok || len(elo.Metrics) == 0 || elo.Metrics[0].Normalization == nil ||
		elo.Metrics[0].Normalization.Min == nil {
		t.Fatalf("GDPval display normalization is missing: %+v", elo.Metrics)
	}
	*elo.Metrics[0].Normalization.Min = 999
	reloadedElo, _ := registry.Benchmark("artificial-analysis/gdpval-aa@2.0.0")
	if got := *reloadedElo.Metrics[0].Normalization.Min; got != 500 {
		t.Fatalf("registry metric normalization was mutated: got %v", got)
	}
}

func TestBuiltInEvaluationPreservesOpenSubjectMetadata(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}

	for _, evaluation := range registry.Evaluations() {
		if evaluation.ID != "qwen/qwen3.8-max-model-card-gpqa-diamond@1.0.0" {
			continue
		}
		if evaluation.Subject["variant"] != "Qwen3.8-Max" ||
			evaluation.Subject["source_kind"] != "official_model_card" ||
			evaluation.Subject["reasoning_effort"] != "xhigh" {
			t.Fatalf("benchmark-specific subject metadata was lost: %+v", evaluation.Subject)
		}
		evaluation.Subject["variant"] = "mutated"
		for _, reloaded := range registry.Evaluations() {
			if reloaded.ID == evaluation.ID && reloaded.Subject["variant"] != "Qwen3.8-Max" {
				t.Fatalf("registry subject metadata was mutated: %+v", reloaded.Subject)
			}
		}
		return
	}
	t.Fatal("Qwen3.8 Max evaluation is missing")
}

func TestDeepSeekV4HasEffortIsolatedEvaluationAndProviderBindings(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}

	for _, modelID := range []string{"deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-pro"} {
		assertDeepSeekModelSupport(t, registry, modelID)
	}

	provider, ok := registry.Provider("deepseek")
	if !ok || !containsString(provider.Protocols, "openai/responses@1") {
		t.Fatalf("deepseek provider is missing Responses support: %+v", provider)
	}
}

func assertDeepSeekModelSupport(t *testing.T, registry *Registry, modelID string) {
	t.Helper()
	defaultResult, ok := registry.IndexResult(modelID, "vllm-sr/intelligence@1.0.0")
	if ok {
		t.Fatalf("%s default-effort result must be absent without sufficient evidence: %+v", modelID, defaultResult)
	}
	result, ok := registry.IndexResultForEffort(modelID, "max", "vllm-sr/intelligence@1.0.0")
	if !ok || result.Status != "available" || result.Score == nil || result.Coverage != 1 {
		t.Fatalf("%s max-effort intelligence result is incomplete: %+v", modelID, result)
	}
	for _, providerID := range []string{"deepseek", "vllm"} {
		provider, ok := registry.Provider(providerID)
		if !ok || !providerBindsModel(provider, modelID) {
			t.Fatalf("%s is missing its %s provider binding", modelID, providerID)
		}
	}
}

func providerBindsModel(provider ProviderDefinition, modelID string) bool {
	for _, binding := range provider.Models {
		if binding.Catalog == modelID {
			return true
		}
	}
	return false
}

func TestBuiltInProviderBindingsDeclareRelationships(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	assertProviderBindingRelationshipsValid(t, registry)
	assertProviderBindingRelationships(t, registry, map[string]map[string]CatalogModelRelationship{
		"bedrock": {
			"amazon/nova-2-lite": CatalogModelRelationshipFirstParty,
		},
		"baidu-qianfan": {
			"baidu/ernie-5.0":        CatalogModelRelationshipFirstParty,
			"deepseek/deepseek-v3.2": CatalogModelRelationshipManagedCloud,
		},
		"openrouter": {
			"anthropic/claude-sonnet-5": CatalogModelRelationshipGateway,
		},
		"vllm": {
			"baidu/ernie-4.5-300b-a47b": CatalogModelRelationshipSelfHosted,
		},
	})
}

func assertProviderBindingRelationshipsValid(t *testing.T, registry *Registry) {
	t.Helper()
	valid := map[CatalogModelRelationship]bool{
		CatalogModelRelationshipFirstParty:   true,
		CatalogModelRelationshipManagedCloud: true,
		CatalogModelRelationshipGateway:      true,
		CatalogModelRelationshipSelfHosted:   true,
	}
	for _, provider := range registry.Providers() {
		for _, binding := range provider.Models {
			if !valid[binding.Relationship] {
				t.Fatalf("%s/%s has invalid relationship %q", provider.ID, binding.Catalog, binding.Relationship)
			}
		}
	}
}

func assertProviderBindingRelationships(
	t *testing.T,
	registry *Registry,
	wants map[string]map[string]CatalogModelRelationship,
) {
	t.Helper()
	for providerID, catalogWants := range wants {
		provider, ok := registry.Provider(providerID)
		if !ok || len(provider.Models) == 0 {
			t.Fatalf("%s has no built-in bindings", providerID)
		}
		for catalogID, want := range catalogWants {
			found := false
			for _, binding := range provider.Models {
				if binding.Catalog != catalogID {
					continue
				}
				found = true
				if binding.Relationship != want {
					t.Fatalf("%s/%s relationship = %q, want %q", providerID, binding.Catalog, binding.Relationship, want)
				}
			}
			if !found {
				t.Fatalf("%s has no binding for %s", providerID, catalogID)
			}
		}
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

func TestCompileAcceptsOpenEndedInlineReasoningFamily(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	effective, err := registry.Compile(CompileInput{
		Models: []ModelAlias{{Name: "private", Catalog: "private"}},
		ModelCards: []ModelCardOverlay{{
			Name: "private",
			Reasoning: &ReasoningFamilyDefinition{
				Type: "chat_template_kwargs", Parameter: "enable_thinking",
			},
		}},
	})
	if err != nil {
		t.Fatalf("compile open-ended inline reasoning family: %v", err)
	}
	model, ok := effective.Model("private")
	if !ok {
		t.Fatal("compiled custom model is missing")
	}
	var family ReasoningFamilyDefinition
	found := false
	for _, candidate := range effective.ReasoningFamilies() {
		if candidate.ID == model.Card.Card.ReasoningFamily {
			family = candidate
			found = true
			break
		}
	}
	if !found || family.Parameter != "enable_thinking" || len(family.Levels) != 0 {
		t.Fatalf("open-ended inline reasoning family changed: %+v", family)
	}
}

func TestCompileDefaultsPreserveReasoningEffortOmission(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}

	effective, err := registry.Compile(CompileInput{})
	if err != nil {
		t.Fatal(err)
	}
	if got := effective.Defaults().ReasoningEffort; got != "" {
		t.Fatalf("omitted reasoning effort became %q", got)
	}

	effective, err = registry.Compile(CompileInput{Defaults: Defaults{ReasoningEffort: "high"}})
	if err != nil {
		t.Fatal(err)
	}
	if got := effective.Defaults().ReasoningEffort; got != "high" {
		t.Fatalf("explicit reasoning effort became %q", got)
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

func TestCompileInfersProtocolFromCatalogModelBinding(t *testing.T) {
	registry := protocolInferenceTestRegistry()

	for _, test := range []struct {
		name             string
		catalog          string
		explicitProtocol string
		wantProtocol     string
		wantModelID      string
	}{
		{
			name:         "responses-only model",
			catalog:      "acme/responses-only",
			wantProtocol: "openai/responses@1",
			wantModelID:  "responses-only-v1",
		},
		{
			name:         "provider default is valid for multi-protocol model",
			catalog:      "acme/multi-protocol",
			wantProtocol: "openai/chat-completions@1",
			wantModelID:  "multi-protocol-v1",
		},
		{
			name:             "explicit protocol wins for multi-protocol model",
			catalog:          "acme/multi-protocol",
			explicitProtocol: "openai/responses@1",
			wantProtocol:     "openai/responses@1",
			wantModelID:      "multi-protocol-v1",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			effective, compileErr := registry.Compile(CompileInput{
				Providers: []ProviderInstance{{Name: "primary", Catalog: "test-provider"}},
				Models: []ModelAlias{{
					Name: "production", Catalog: test.catalog,
					Providers: []ModelProviderBinding{{Name: "primary", Protocol: test.explicitProtocol}},
				}},
			})
			if compileErr != nil {
				t.Fatal(compileErr)
			}
			model, ok := effective.Model("production")
			if !ok || len(model.Providers) != 1 {
				t.Fatalf("compiled provider binding is missing: %+v", model)
			}
			binding := model.Providers[0]
			if binding.Binding.Protocol != test.wantProtocol || binding.Binding.ModelID != test.wantModelID {
				t.Fatalf("binding = %+v, want protocol %q and model %q", binding.Binding, test.wantProtocol, test.wantModelID)
			}
			if binding.CatalogBinding == nil {
				t.Fatal("inferred binding lost provider-owned catalog metadata")
			}
		})
	}
}

func TestCompileRejectsExplicitProtocolWithoutCatalogModelBinding(t *testing.T) {
	registry := protocolInferenceTestRegistry()
	_, err := registry.Compile(CompileInput{
		Providers: []ProviderInstance{{Name: "primary", Catalog: "test-provider"}},
		Models: []ModelAlias{{
			Name: "production", Catalog: "acme/responses-only",
			Providers: []ModelProviderBinding{{Name: "primary", Protocol: "openai/chat-completions@1"}},
		}},
	})
	if err == nil || !strings.Contains(err.Error(), "available protocols: openai/responses@1") {
		t.Fatalf("expected actionable protocol mismatch, got %v", err)
	}
}

func TestCompileDeploymentNameBindingRequiresExplicitProviderModelID(t *testing.T) {
	registry := protocolInferenceTestRegistry()
	input := CompileInput{
		Providers: []ProviderInstance{{Name: "primary", Catalog: "test-provider"}},
		Models: []ModelAlias{{
			Name: "production", Catalog: "acme/deployment-name",
			Providers: []ModelProviderBinding{{Name: "primary"}},
		}},
	}

	_, err := registry.Compile(input)
	if err == nil || !strings.Contains(err.Error(), "providers.models[].provider_model_id") ||
		!strings.Contains(err.Error(), "operator-defined deployment name") {
		t.Fatalf("expected an actionable deployment-name error, got %v", err)
	}

	input.Models[0].Providers[0].ModelID = "operator-mai-production"
	effective, err := registry.Compile(input)
	if err != nil {
		t.Fatal(err)
	}
	model, ok := effective.Model("production")
	if !ok || len(model.Providers) != 1 {
		t.Fatalf("compiled provider binding is missing: %+v", model)
	}
	binding := model.Providers[0]
	if binding.Binding.ModelID != "operator-mai-production" || binding.Binding.Protocol != "openai/chat-completions@1" {
		t.Fatalf("deployment binding = %+v", binding.Binding)
	}
	if binding.CatalogBinding == nil || binding.CatalogBinding.ID != "catalog-deployment-placeholder" {
		t.Fatalf("deployment binding lost availability metadata: %+v", binding.CatalogBinding)
	}
}

func protocolInferenceTestRegistry() *Registry {
	const (
		chat      = "openai/chat-completions@1"
		responses = "openai/responses@1"
	)
	return &Registry{
		providers: map[string]ProviderDefinition{
			"test-provider": {
				ID:                  "test-provider",
				DefaultBaseURL:      "https://models.example/v1",
				Protocols:           []string{chat, responses},
				DefaultProtocol:     chat,
				SupportedOperations: []string{chat + "#create", responses + "#create"},
				Models: []CatalogModelBinding{
					{Catalog: "acme/responses-only", ID: "responses-only-v1", Protocols: []string{responses}},
					{Catalog: "acme/multi-protocol", ID: "multi-protocol-v1", Protocols: []string{chat, responses}},
					{
						Catalog: "acme/deployment-name", ID: "catalog-deployment-placeholder", Protocols: []string{chat},
						Restrictions: map[string]any{"provider_model_id_kind": "deployment_name"},
					},
				},
			},
		},
		models: map[string]ModelCard{
			"acme/responses-only":  {ID: "acme/responses-only", Kind: "physical"},
			"acme/multi-protocol":  {ID: "acme/multi-protocol", Kind: "physical"},
			"acme/deployment-name": {ID: "acme/deployment-name", Kind: "physical"},
		},
	}
}

func TestCompileRejectsAmbiguousCatalogModelProtocol(t *testing.T) {
	registry := &Registry{
		providers: map[string]ProviderDefinition{
			"mixed": {
				ID: "mixed", DefaultBaseURL: "https://models.example/v1",
				Protocols: []string{
					"openai/chat-completions@1",
					"openai/responses@1",
					"anthropic/messages@1",
				},
				DefaultProtocol: "openai/chat-completions@1",
				SupportedOperations: []string{
					"openai/chat-completions@1#create",
					"openai/responses@1#create",
					"anthropic/messages@1#create",
				},
				Models: []CatalogModelBinding{{
					Catalog: "acme/ambiguous", ID: "ambiguous-v1",
					Protocols: []string{"openai/responses@1", "anthropic/messages@1"},
				}},
			},
		},
		models: map[string]ModelCard{
			"acme/ambiguous": {
				ID: "acme/ambiguous", DisplayName: "Ambiguous", Kind: "physical",
				Lifecycle: "active", Capabilities: []string{"chat"},
				Modalities: Modalities{Input: []string{"text"}, Output: []string{"text"}},
			},
		},
	}

	_, err := registry.Compile(CompileInput{
		Providers: []ProviderInstance{{Name: "primary", Catalog: "mixed"}},
		Models: []ModelAlias{{
			Name: "production", Catalog: "acme/ambiguous",
			Providers: []ModelProviderBinding{{Name: "primary"}},
		}},
	})
	if err == nil || !strings.Contains(err.Error(), "protocol is ambiguous") ||
		!strings.Contains(err.Error(), "set protocol explicitly") {
		t.Fatalf("expected actionable ambiguity error, got %v", err)
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

func TestExplicitCustomCardMayShareBuiltInName(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	builtIn := false
	capabilities := []string{"chat", "operator-only-capability"}
	effective, err := registry.Compile(CompileInput{
		Models: []ModelAlias{{Name: "local-gpt", Catalog: "openai/gpt-5.4"}},
		ModelCards: []ModelCardOverlay{{
			Name: "openai/gpt-5.4", BuiltIn: &builtIn, Capabilities: &capabilities,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	model, ok := effective.Model("local-gpt")
	if !ok {
		t.Fatal("custom model is missing")
	}
	if model.Card.Provenance["id"] != SourceOperator {
		t.Fatalf("custom card was treated as built-in: %+v", model.Card.Provenance)
	}
	result := model.Indices["vllm-sr/intelligence@1.0.0"]
	if result.Score != nil || result.Status != "missing" || len(result.Provenance) != 0 {
		t.Fatalf("custom card inherited built-in evidence: %+v", result)
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
			Benchmarks: []BenchmarkDefinition{{
				ID: "acme/support@1.0.0", DisplayName: "Support", Domain: "support",
				DefaultProfile: "published", Profiles: []BenchmarkProfile{{
					ID: "published", DisplayName: "Published", Description: "Publisher-reported result",
				}},
				Metrics: []BenchmarkMetric{{ID: "resolution", Unit: "proportion", Direction: "higher_is_better", Range: [2]float64{0, 1}}},
			}},
			Records: []EvaluationRecord{{
				ID: "acme/run-1", Model: "acme/measured", Benchmark: "acme/support@1.0.0",
				BenchmarkProfile: "published", ReasoningEffort: "default", Status: "available",
				Metrics:  map[string]float64{"resolution": 0.82},
				Evidence: EvaluationEvidence{Provenance: "operator", Verification: "reproduced", Redistributable: true},
			}},
			Indices: []IndexDefinition{{
				ID: "acme/readiness@1.0.0", DisplayName: "Readiness", Aggregation: "weighted_mean",
				Scale: [2]float64{0, 100}, Missing: MissingPolicy{Policy: "require_all"},
				Components: []IndexComponent{{
					Benchmark: "acme/support@1.0.0", Metric: "resolution", BenchmarkProfile: "published",
					Weight: 1, Normalization: Normalization{Type: "identity"},
				}},
			}},
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

func TestCustomCardAcceptsOptionalPublisherPresentationAndDistribution(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	publisher := "Acme Research"
	presentation := ProviderPresentation{Logo: "https://models.example/acme.svg", Monogram: "A"}
	distribution := ModelDistribution{
		Type: "open_weights", Source: "https://models.example/acme-reasoner", License: "Apache-2.0",
	}
	effective, err := registry.Compile(CompileInput{
		Models: []ModelAlias{{Name: "private", Catalog: "acme/reasoner"}},
		ModelCards: []ModelCardOverlay{{
			Name: "acme/reasoner", Publisher: &publisher,
			Presentation: &presentation, Distribution: &distribution,
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	model, ok := effective.Model("private")
	if !ok {
		t.Fatal("compiled custom model is missing")
	}
	if model.Card.Card.Publisher != publisher || model.Card.Card.Presentation.Logo != presentation.Logo ||
		model.Card.Card.Distribution.Source != distribution.Source {
		t.Fatalf("custom card metadata was not materialized: %+v", model.Card.Card)
	}
	for _, field := range []string{"publisher", "presentation", "distribution"} {
		if model.Card.Provenance[field] != SourceOperator {
			t.Fatalf("%s provenance = %q, want operator", field, model.Card.Provenance[field])
		}
	}
}

func TestCustomOpenWeightsCardRequiresLicense(t *testing.T) {
	registry, err := BuiltIn()
	if err != nil {
		t.Fatal(err)
	}
	distribution := ModelDistribution{Type: "open_weights", Source: "https://models.example/acme"}
	_, err = registry.Compile(CompileInput{
		Models:     []ModelAlias{{Name: "private", Catalog: "acme/private"}},
		ModelCards: []ModelCardOverlay{{Name: "acme/private", Distribution: &distribution}},
	})
	if err == nil || !strings.Contains(err.Error(), "distribution.license") {
		t.Fatalf("expected missing license error, got %v", err)
	}
}

func TestVirtualModelIndicesRemainMissingWithoutRecipeEvaluation(t *testing.T) {
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
	if result.Status != "missing" || result.Score != nil {
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
