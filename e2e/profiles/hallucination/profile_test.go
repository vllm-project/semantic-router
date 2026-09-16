package hallucination

import (
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestProfileUsesCanonicalFactCheckBindingWithRemoteDetector(t *testing.T) {
	raw, err := os.ReadFile(filepath.Base(valuesFile))
	if err != nil {
		t.Fatal(err)
	}
	var values map[string]any
	if err := yaml.Unmarshal(raw, &values); err != nil {
		t.Fatal(err)
	}
	module := profileSection(t, values, "config", "global", "model_catalog", "modules", "hallucination_mitigation")
	if module["enabled"] != true {
		t.Fatal("hallucination mitigation must remain enabled")
	}
	factCheck := profileSection(t, module, "fact_check")
	if factCheck["model_ref"] != "fact_check_classifier" {
		t.Fatalf("fact-check must use the canonical system binding, got %#v", factCheck)
	}
	if modelID, exists := factCheck["model_id"]; exists && modelID != "" {
		t.Fatalf("fact-check must inherit the catalog model instead of overriding it with %v", modelID)
	}
	if factCheck["threshold"] != 0.65 || factCheck["use_cpu"] != true || factCheck["use_mmbert_32k"] != true {
		t.Fatalf("fact-check execution policy changed: %#v", factCheck)
	}
	detector := profileSection(t, module, "detector")
	if detector["backend"] != "endpoint" || detector["endpoint"] != "http://mock-hallucination-detector.default.svc.cluster.local:8000/v1" {
		t.Fatalf("profile must retain its remote detector: %#v", detector)
	}
	if detector["model_id"] != "KRLabsOrg/lettucedect-v2-qwen-2b" || detector["include_explanation"] != false {
		t.Fatalf("remote detector contract changed: %#v", detector)
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
