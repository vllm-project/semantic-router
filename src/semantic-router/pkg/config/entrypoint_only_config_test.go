package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestParseYAMLBytesRejectsRemovedImplicitRoutingFields(t *testing.T) {
	for _, field := range []string{
		"auto_model_name: custom-auto",
		"auto_model_names: []",
		"include_config_models_in_list: true",
		"strategy: priority",
		"model_selection: {enabled: true}",
	} {
		canonicalYAML := []byte(strings.Replace(
			entrypointRulesYAML,
			"global:\n",
			"global:\n  router:\n    "+field+"\n",
			1,
		))
		_, err := testAuthoringParser(t).ParseYAMLBytes(canonicalYAML)
		if err == nil || !strings.Contains(err.Error(), strings.Split(field, ":")[0]) {
			t.Fatalf("expected strict rejection for %q, got %v", field, err)
		}
	}
}

func TestCanonicalExportOmitsRuntimeCatalogToggle(t *testing.T) {
	cfg, err := testAuthoringParser(t).ParseYAMLBytes([]byte(entrypointRulesYAML))
	if err != nil {
		t.Fatalf("ParseYAMLBytes returned error: %v", err)
	}
	cfg.IncludeConfigModelsInList = true

	exported, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatalf("marshal canonical config: %v", err)
	}
	if strings.Contains(string(exported), "include_config_models_in_list") {
		t.Fatalf("canonical export leaked runtime-only model catalog toggle:\n%s", exported)
	}
}
