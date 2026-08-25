package config

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

const implicitAutoAuthoringYAML = `
version: v0.3
providers:
  models:
    - name: model-a
      backend_refs:
        - {provider: private-test, endpoint: http://model-a.example}
routing:
  modelCards:
    - {name: model-a}
  decisions:
    - name: route
      modelRefs: [{model: model-a}]
      rules: {}
global:
  services:
    backend_egress: {policy_file: /app/config/backend-egress-policy.yaml}
`

func TestParseYAMLBytesPreservesExistingRouterControls(t *testing.T) {
	canonicalYAML := []byte(strings.Replace(
		strictV03AuthoringYAML,
		"global:\n",
		"global:\n  router:\n    auto_model_name: custom-auto\n    auto_model_names: []\n    include_config_models_in_list: true\n    strategy: priority\n    model_selection: {enabled: true}\n",
		1,
	))
	cfg, err := testAuthoringParser(t).ParseYAMLBytes(canonicalYAML)
	if err != nil {
		t.Fatalf("existing global.router controls were rejected: %v", err)
	}
	if cfg.AutoModelName != "custom-auto" || cfg.AutoModelNames == nil || len(cfg.AutoModelNames) != 0 ||
		!cfg.IncludeConfigModelsInList || !cfg.ModelSelection.Enabled {
		t.Fatalf("global.router controls were not preserved: %+v", cfg.RouterOptions)
	}
}

func TestCanonicalExportPreservesRuntimeCatalogToggle(t *testing.T) {
	document := strings.Replace(
		strictV03AuthoringYAML,
		"global:\n",
		"global:\n  router:\n    include_config_models_in_list: true\n",
		1,
	)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}

	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatalf("marshal canonical config: %v", err)
	}
	if !strings.Contains(string(exported), "include_config_models_in_list: true") {
		t.Fatalf("canonical export dropped the public model catalog toggle:\n%s", exported)
	}
}

func TestTopLevelRoutingPreservesImplicitAutoEntrypointNames(t *testing.T) {
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(implicitAutoAuthoringYAML))
	if err != nil {
		t.Fatalf("parse implicit v0.3 Entrypoint: %v", err)
	}
	want := []string{DefaultAutoModelName, LegacyAutoModelAlias, DefaultVSRAutoModelName}
	if len(cfg.Entrypoints) != 1 || !reflect.DeepEqual(cfg.Entrypoints[0].ModelNames, want) {
		t.Fatalf("implicit Entrypoint names = %+v, want %v", cfg.Entrypoints, want)
	}
	for _, name := range want {
		if recipe, found := cfg.RecipeForRequestModel(name); !found || recipe.Name != DefaultRecipeName {
			t.Fatalf("implicit model %q did not resolve the default Recipe: %+v, %v", name, recipe, found)
		}
	}
}

func TestExplicitEmptyAutoNamesLetsEntrypointOwnAutoName(t *testing.T) {
	document := strings.Replace(
		implicitAutoAuthoringYAML,
		"global:\n",
		"entrypoints:\n  - model_names: [vllm-sr/auto]\n    recipe: default\n    assignments:\n      route: {models: [{model: model-a}]}\nglobal:\n  router:\n    auto_model_names: []\n",
		1,
	)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("explicit Entrypoint with disabled implicit aliases failed: %v", err)
	}
	if len(cfg.Entrypoints) != 1 || !reflect.DeepEqual(cfg.Entrypoints[0].ModelNames, []string{"vllm-sr/auto"}) {
		t.Fatalf("compiled Entrypoints = %+v", cfg.Entrypoints)
	}

	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatalf("marshal canonical config: %v", err)
	}
	if !strings.Contains(string(exported), "auto_model_names: []") {
		t.Fatalf("canonical export lost the explicit empty alias list:\n%s", exported)
	}
}

func TestExplicitAutoEntrypointSupersedesImplicitAliases(t *testing.T) {
	document := strings.Replace(
		implicitAutoAuthoringYAML,
		"global:\n",
		"entrypoints:\n  - model_names: [vllm-sr/auto]\n    recipe: default\n    assignments:\n      route: {models: [{model: model-a}]}\nglobal:\n",
		1,
	)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("explicit auto Entrypoint should be authoritative: %v", err)
	}
	if len(cfg.Entrypoints) != 1 || !reflect.DeepEqual(cfg.Entrypoints[0].ModelNames, []string{"vllm-sr/auto"}) {
		t.Fatalf("compiled Entrypoints = %+v", cfg.Entrypoints)
	}
}

func TestConfiguredAutoNamesAreNormalized(t *testing.T) {
	document := strings.Replace(
		implicitAutoAuthoringYAML,
		"global:\n",
		"global:\n  router:\n    auto_model_names: [vllm-sr/custom, custom, custom]\n",
		1,
	)
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatalf("parse configured aliases: %v", err)
	}
	want := []string{"custom", "vllm-sr/custom"}
	if len(cfg.Entrypoints) != 1 || !reflect.DeepEqual(cfg.Entrypoints[0].ModelNames, want) {
		t.Fatalf("normalized names = %+v, want %v", cfg.Entrypoints, want)
	}
}
