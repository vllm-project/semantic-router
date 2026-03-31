package handlers

import (
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestParseBuilderNLGenerationOutputFromJSONFence(t *testing.T) {
	raw := "```json\n{\"dsl\":\"MODEL \\\"MoM\\\" {\\n  modality: \\\"text\\\"\\n}\\n\\nROUTE default_route (description = \\\"Fallback\\\") {\\n  PRIORITY 100\\n  MODEL \\\"MoM\\\" (reasoning = false)\\n}\",\"summary\":\"Added a fallback route.\",\"suggestedTestQuery\":\"hello world\"}\n```"

	parsed, err := parseBuilderNLGenerationOutput(raw)
	if err != nil {
		t.Fatalf("expected parse to succeed, got error: %v", err)
	}
	if parsed.DSL == "" {
		t.Fatalf("expected non-empty DSL output")
	}
	if parsed.Summary != "Added a fallback route." {
		t.Fatalf("unexpected summary: %q", parsed.Summary)
	}
	if parsed.SuggestedTestQuery != "hello world" {
		t.Fatalf("unexpected suggested test query: %q", parsed.SuggestedTestQuery)
	}
}

func TestApplyBuilderNLCustomConnectionAddsProviderModel(t *testing.T) {
	cfg := &routerconfig.CanonicalConfig{}
	ensureBuilderNLGlobalDefaults(cfg)

	err := applyBuilderNLCustomConnection(cfg, builderNLConnection{
		ProviderKind: builderNLProviderOpenAICompatible,
		ModelName:    "gpt-4o-mini",
		BaseURL:      "https://api.openai.com",
		AccessKey:    "secret",
		EndpointName: "primary",
	})
	if err != nil {
		t.Fatalf("expected custom connection to apply, got error: %v", err)
	}
	if cfg.Global == nil || cfg.Global.Router.AutoModelName != builderNLDefaultModelAlias {
		t.Fatalf("expected global auto model name to default to %q", builderNLDefaultModelAlias)
	}
	if len(cfg.Providers.Models) != 1 {
		t.Fatalf("expected exactly one provider model, got %d", len(cfg.Providers.Models))
	}
	providerModel := cfg.Providers.Models[0]
	if providerModel.Name != "gpt-4o-mini" {
		t.Fatalf("unexpected provider model name: %q", providerModel.Name)
	}
	if len(providerModel.BackendRefs) != 1 {
		t.Fatalf("expected exactly one backend ref, got %d", len(providerModel.BackendRefs))
	}
	backend := providerModel.BackendRefs[0]
	if backend.BaseURL != "https://api.openai.com" {
		t.Fatalf("unexpected backend base url: %q", backend.BaseURL)
	}
	if backend.Provider != "openai" {
		t.Fatalf("unexpected backend provider: %q", backend.Provider)
	}
	if cfg.Providers.Defaults.DefaultModel != "gpt-4o-mini" {
		t.Fatalf("unexpected default model: %q", cfg.Providers.Defaults.DefaultModel)
	}
}
