package aigateway

import (
	"os"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestProfilePinsAcceptanceClassifierModels(t *testing.T) {
	raw, err := os.ReadFile("values.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var values map[string]any
	if err := yaml.Unmarshal(raw, &values); err != nil {
		t.Fatal(err)
	}

	catalog := profileSection(t, values, "config", "global", "model_catalog")
	system := profileSection(t, catalog, "system")
	requireModel(t, system, "prompt_guard", "models/mmbert32k-jailbreak-detector-merged")
	requireModel(t, system, "domain_classifier", "models/mmbert32k-intent-classifier-merged")
	requireModel(t, system, "pii_classifier", "models/mmbert32k-pii-detector-merged")

	modules := profileSection(t, catalog, "modules")
	requireModuleModel(
		t,
		profileSection(t, modules, "prompt_guard"),
		"models/mmbert32k-jailbreak-detector-merged",
		"models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json",
	)
	classifier := profileSection(t, modules, "classifier")
	requireModuleModel(
		t,
		profileSection(t, classifier, "domain"),
		"models/mmbert32k-intent-classifier-merged",
		"models/mmbert32k-intent-classifier-merged/category_mapping.json",
	)
	requireModuleModel(
		t,
		profileSection(t, classifier, "pii"),
		"models/mmbert32k-pii-detector-merged",
		"models/mmbert32k-pii-detector-merged/pii_type_mapping.json",
	)
}

func requireModel(t *testing.T, system map[string]any, name, want string) {
	t.Helper()

	if got := system[name]; got != want {
		t.Fatalf("system model %s = %v, want %s", name, got, want)
	}
}

func requireModuleModel(t *testing.T, module map[string]any, modelID, mappingPath string) {
	t.Helper()

	if module["enabled"] != true || module["model_id"] != modelID {
		t.Fatalf("classifier module is not pinned to %s: %#v", modelID, module)
	}
	if module["max_sequence_length"] != 0 {
		t.Fatalf("classifier module %s must retain the legacy 512-token budget: %#v", modelID, module)
	}
	foundMapping := false
	for _, field := range []string{"jailbreak_mapping_path", "category_mapping_path", "pii_mapping_path"} {
		if value, ok := module[field]; ok {
			foundMapping = value == mappingPath
		}
	}
	if !foundMapping {
		t.Fatalf("classifier module %s does not use mapping %s: %#v", modelID, mappingPath, module)
	}
}

func profileSection(t *testing.T, values map[string]any, keys ...string) map[string]any {
	t.Helper()

	for _, key := range keys {
		section, ok := values[key].(map[string]any)
		if !ok {
			t.Fatalf("profile section %q is missing or is not a mapping", key)
		}
		values = section
	}
	return values
}
