package config

import (
	"slices"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestEffectiveAutoModelNamesDefaults(t *testing.T) {
	cfg := &RouterConfig{}
	want := []string{"vllm-sr/auto", "auto", "MoM"}
	if !slices.Equal(cfg.EffectiveAutoModelNames(), want) {
		t.Fatalf("expected default aliases %#v, got %#v", want, cfg.EffectiveAutoModelNames())
	}
}

func TestEffectiveAutoModelNamesExplicitAllowList(t *testing.T) {
	cfg := &RouterConfig{
		RouterOptions: RouterOptions{
			AutoModelName:  "CustomAuto",
			AutoModelNames: []string{"vllm-sr/auto", "custom/auto", "vllm-sr/auto", ""},
		},
	}
	want := []string{"vllm-sr/auto", "custom/auto"}
	if !slices.Equal(cfg.EffectiveAutoModelNames(), want) {
		t.Fatalf("expected explicit aliases %#v, got %#v", want, cfg.EffectiveAutoModelNames())
	}
}

func TestEffectiveAutoModelNamesExplicitEmptyDisablesAliases(t *testing.T) {
	cfg := &RouterConfig{
		RouterOptions: RouterOptions{
			AutoModelNames: []string{},
		},
	}
	if got := cfg.EffectiveAutoModelNames(); len(got) != 0 {
		t.Fatalf("expected no effective aliases, got %#v", got)
	}
}

func TestIsAutoModelNameRecognizesNamespacedAutoAlias(t *testing.T) {
	cfg := &RouterConfig{}
	if !cfg.IsAutoModelName("vllm-sr/auto") {
		t.Fatal("expected vllm-sr/auto to be recognized as an auto alias")
	}
}

func TestIsAutoModelNameHonorsExplicitAllowList(t *testing.T) {
	cfg := &RouterConfig{
		RouterOptions: RouterOptions{
			AutoModelNames: []string{"vllm-sr/auto", "router/auto"},
		},
	}
	for _, alias := range []string{"vllm-sr/auto", "router/auto"} {
		if !cfg.IsAutoModelName(alias) {
			t.Fatalf("expected %q to be recognized as an auto alias", alias)
		}
	}
	for _, alias := range []string{"auto", "MoM"} {
		if cfg.IsAutoModelName(alias) {
			t.Fatalf("expected %q to be excluded by explicit auto_model_names", alias)
		}
	}
}

func TestAutoModelNamesYAML(t *testing.T) {
	yamlContent := []byte(`
auto_model_names:
  - vllm-sr/auto
  - router/auto
default_model: test-model
`)
	var cfg RouterConfig
	if err := yaml.Unmarshal(yamlContent, &cfg); err != nil {
		t.Fatalf("yaml.Unmarshal failed: %v", err)
	}
	want := []string{"vllm-sr/auto", "router/auto"}
	if !slices.Equal(cfg.RouterOptions.AutoModelNames, want) {
		t.Fatalf("expected parsed aliases %#v, got %#v", want, cfg.RouterOptions.AutoModelNames)
	}
	if !slices.Equal(cfg.EffectiveAutoModelNames(), want) {
		t.Fatalf("expected effective aliases %#v, got %#v", want, cfg.EffectiveAutoModelNames())
	}
}

func TestParseYAMLBytesPreservesLegacyAutoModelNameAliasWhenAutoModelNamesOmitted(t *testing.T) {
	canonicalYAML := []byte(`
version: v0.3
listeners: []
providers:
  defaults: {}
routing:
  signals: {}
global:
  router:
    auto_model_name: custom-auto
`)

	cfg, err := ParseYAMLBytes(canonicalYAML)
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	if len(cfg.AutoModelNames) != 0 {
		t.Fatalf("expected auto_model_names to remain omitted, got %#v", cfg.AutoModelNames)
	}
	want := []string{"vllm-sr/auto", "auto", "custom-auto"}
	if !slices.Equal(cfg.EffectiveAutoModelNames(), want) {
		t.Fatalf("expected effective aliases %#v, got %#v", want, cfg.EffectiveAutoModelNames())
	}
}

func TestCanonicalRoundTripPreservesExplicitEmptyAutoModelNames(t *testing.T) {
	canonicalYAML := []byte(`
version: v0.3
routing:
  modelCards:
    - name: model-a
providers:
  defaults:
    default_model: model-a
  models:
    - name: model-a
      backend_refs:
        - endpoint: 127.0.0.1:8000
global:
  router:
    auto_model_names: []
`)
	cfg, err := ParseYAMLBytes(canonicalYAML)
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	if cfg.AutoModelNames == nil || len(cfg.EffectiveAutoModelNames()) != 0 {
		t.Fatalf("expected explicit empty aliases, got %#v", cfg.AutoModelNames)
	}

	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatalf("marshal canonical config: %v", err)
	}
	roundTripped, err := ParseYAMLBytes(exported)
	if err != nil {
		t.Fatalf("reparse canonical config: %v", err)
	}
	if roundTripped.AutoModelNames == nil || len(roundTripped.EffectiveAutoModelNames()) != 0 {
		t.Fatalf("round-trip lost explicit empty aliases: %#v", roundTripped.AutoModelNames)
	}
}
