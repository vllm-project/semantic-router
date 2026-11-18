package config

import "testing"

func TestSparseCustomModelCardCarriesNoDeclaredCapabilities(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.5
providers:
  defaults:
    model: base-model
  models:
    - name: base-model
      backend_refs:
        - name: local
          provider: vllm
          endpoint: 127.0.0.1:8000
          weight: 1
recipes:
  - name: default
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	if got := cfg.ModelConfig["base-model"].Capabilities; len(got) != 0 {
		t.Fatalf("sparse custom card must not declare capabilities, got %v", got)
	}
}

func TestOperatorModelCardCapabilitiesAreDeclared(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.5
providers:
  defaults:
    model: base-model
  models:
    - name: base-model
      backend_refs:
        - name: local
          provider: vllm
          endpoint: 127.0.0.1:8000
          weight: 1
      capabilities: [chat, vision]
recipes:
  - name: default
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	got := cfg.ModelConfig["base-model"].Capabilities
	if len(got) != 2 || got[0] != "chat" || got[1] != "vision" {
		t.Fatalf("operator capabilities must be carried verbatim, got %v", got)
	}
}

// provider_model_id names what the backend serves, not another catalog
// entry, so the Provider mapping's pricing and reasoning stay attached.
func TestCatalogMappingSurvivesProviderModelID(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.5
providers:
  defaults:
    model: palmyra
  models:
    - name: palmyra
      catalog: writer/palmyra-x5
      provider_model_id: palmyra-x5-v2
      backend_refs:
        - provider: writer
          api_key: sk-test
    - name: qwen
      catalog: qwen/qwen3.8-27b
      provider_model_id: qwen38
      backend_refs:
        - provider: vllm
          endpoint: http://127.0.0.1:8000/v1
recipes:
  - name: default
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	palmyra := cfg.ModelConfig["palmyra"]
	if palmyra.ExternalModelIDs["writer"] != "palmyra-x5-v2" ||
		palmyra.Pricing.PromptPer1M != 0.6 || palmyra.Pricing.CompletionPer1M != 6.0 {
		t.Fatalf("palmyra model id %q pricing %+v, want palmyra-x5-v2 at 0.6/6.0",
			palmyra.ExternalModelIDs["writer"], palmyra.Pricing)
	}
	qwen := cfg.ProviderProfiles[cfg.ModelConfig["qwen"].PreferredEndpoints[0]]
	if qwen.ReasoningTransport != "top_level_effort_template_switch" {
		t.Fatalf("qwen reasoning transport = %q, want top_level_effort_template_switch", qwen.ReasoningTransport)
	}
}

func TestGemini31ProCatalogCardMaterializesItsLimits(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
version: v0.5
providers:
  models:
    - name: gemini-pro
      catalog: google/gemini-3.1-pro
      backend_refs:
        - provider: gemini
recipes:
  - name: default
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	params := cfg.ModelConfig["gemini-pro"]
	if params.Catalog != "google/gemini-3.1-pro" ||
		params.ContextWindowSize != 1048576 ||
		params.MaxOutputTokens != 65536 {
		t.Fatalf(
			"catalog=%q context_window=%d max_output=%d",
			params.Catalog,
			params.ContextWindowSize,
			params.MaxOutputTokens,
		)
	}
}
