package config

import (
	"slices"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func lossyTranslationConfig(entries string) []byte {
	return []byte(`
version: v0.5
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
recipes:
  - name: default
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
	cfg, err := ParseYAMLBytes(lossyTranslationConfig(`[tool_result.is_error, "messages[].role", reasoning.signature, store]`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	want := []string{"tool_result.is_error", "messages[].role", "reasoning.signature", "store"}
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

func entrypointLossyTranslationConfig(claudeCode, codex string) []byte {
	return []byte(strings.Replace(string(lossyTranslationConfig("[tools.defer_loading]")), "recipes:", `entrypoints:
  - model_names: [vllm-sr/claude-code]
    recipe: default
    default: true
`+claudeCode+`
  - model_names: [vllm-sr/codex]
    recipe: default
`+codex+`
recipes:`, 1))
}

func TestEntrypointAllowLossyTranslationReachesRuntimeAndExport(t *testing.T) {
	cfg, err := ParseYAMLBytes(entrypointLossyTranslationConfig(
		`    allow_lossy_translation: [tools.defer_loading, tool_result.is_error]`,
		`    allow_lossy_translation: []`,
	))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	want := map[string][]string{
		"vllm-sr/claude-code": {"tools.defer_loading", "tool_result.is_error"},
		"vllm-sr/codex":       {},
		"cheap":               {"tools.defer_loading"},
	}
	check := func(stage string, cfg *RouterConfig) {
		t.Helper()
		for model, losses := range want {
			entrypoint, ok := cfg.ResolveEntrypoint(ChatAPI, model)
			var got []string
			if ok {
				got = cfg.AcceptedLosses(&entrypoint)
			} else {
				got = cfg.AcceptedLosses(nil)
			}
			if !slices.Equal(got, losses) {
				t.Fatalf("%s: accepted losses for %s = %v, want %v", stage, model, got, losses)
			}
		}
	}
	check("parsed", cfg)

	encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	roundTrip, err := ParseYAMLBytes(encoded)
	if err != nil {
		t.Fatalf("round trip: %v\n%s", err, encoded)
	}
	check("round trip", roundTrip)
}

func TestEntrypointWithoutAllowLossyTranslationUsesGlobal(t *testing.T) {
	cfg, err := ParseYAMLBytes(entrypointLossyTranslationConfig("", ""))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	entrypoint, _ := cfg.ResolveEntrypoint(ChatAPI, "vllm-sr/codex")
	if got := cfg.AcceptedLosses(&entrypoint); !slices.Equal(got, []string{"tools.defer_loading"}) {
		t.Fatalf("accepted losses = %v, want the global list", got)
	}
	encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "allow_lossy_translation: []") {
		t.Fatalf("an unset entrypoint list was exported as empty:\n%s", encoded)
	}
}

func TestEntrypointAllowLossyTranslationRejectsFieldsWithoutOptIn(t *testing.T) {
	_, err := ParseYAMLBytes(entrypointLossyTranslationConfig("", `    allow_lossy_translation: [context_management]`))
	if err == nil || !strings.Contains(err.Error(), `entrypoints[1].allow_lossy_translation: "context_management" is not supported`) {
		t.Fatalf("ParseYAMLBytes() error = %v, want rejection of context_management on entrypoints[1]", err)
	}
}
