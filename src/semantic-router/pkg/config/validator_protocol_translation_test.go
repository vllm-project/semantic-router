package config

import (
	"slices"
	"strings"
	"testing"
)

func lossyTranslationConfig(entries string) []byte {
	return []byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: cheap
  models:
    - name: cheap
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing:
  modelCards:
    - name: cheap
  decisions:
    - name: route
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: cheap
global:
  router:
    allow_lossy_translation: ` + entries + `
`)
}

func TestAllowLossyTranslationReachesRuntimeAndExport(t *testing.T) {
	cfg, err := ParseYAMLBytes(lossyTranslationConfig("[tool_result.is_error, reasoning.signature]"))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	want := []string{"tool_result.is_error", "reasoning.signature"}
	if !slices.Equal(cfg.AllowLossyTranslation, want) {
		t.Fatalf("runtime allow_lossy_translation = %v, want %v", cfg.AllowLossyTranslation, want)
	}
	if exported := CanonicalGlobalFromRouterConfig(cfg).Router.AllowLossyTranslation; !slices.Equal(exported, want) {
		t.Fatalf("exported allow_lossy_translation = %v, want %v", exported, want)
	}
}

func TestAllowLossyTranslationRejectsFieldsWithoutOptIn(t *testing.T) {
	_, err := ParseYAMLBytes(lossyTranslationConfig("[context_management]"))
	if err == nil || !strings.Contains(err.Error(), `global.router.allow_lossy_translation: "context_management" is not supported`) {
		t.Fatalf("ParseYAMLBytes() error = %v, want rejection of context_management", err)
	}
}
