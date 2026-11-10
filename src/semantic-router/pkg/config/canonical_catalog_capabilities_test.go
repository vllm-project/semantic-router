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
