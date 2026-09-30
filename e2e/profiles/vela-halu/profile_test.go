package velahalu

import (
	"os"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestProfileUsesCanonicalHaluWithoutAuxiliaryClassifiers(t *testing.T) {
	data, err := os.ReadFile("values.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var values map[string]any
	if err = yaml.Unmarshal(data, &values); err != nil {
		t.Fatal(err)
	}
	section := func(root map[string]any, keys ...string) map[string]any {
		for _, key := range keys {
			value, ok := root[key].(map[string]any)
			if !ok {
				t.Fatalf("missing section %s", key)
			}
			root = value
		}
		return root
	}
	cfg := section(values, "config")
	module := section(cfg, "global", "model_catalog", "modules", "hallucination_mitigation")
	if module["enabled"] != true {
		t.Fatal("Halu detector must be enabled")
	}
	if _, overridden := module["detector"]; overridden {
		t.Fatal("profile must inherit the canonical pinned Halu detector")
	}
	for _, name := range []string{"fact_check", "explainer"} {
		auxiliary := section(module, name)
		if auxiliary["model_ref"] != "" || auxiliary["model_id"] != "" {
			t.Fatalf("profile unexpectedly enables %s", name)
		}
	}
	routing := section(cfg, "routing")
	decisions, ok := routing["decisions"].([]any)
	if !ok || len(decisions) != 1 {
		t.Fatal("expected one grounded decision")
	}
	decision := decisions[0].(map[string]any)
	plugins, ok := decision["plugins"].([]any)
	if !ok || len(plugins) != 1 {
		t.Fatal("expected one real hallucination plugin")
	}
	plugin := plugins[0].(map[string]any)
	policy := section(plugin, "configuration")
	if plugin["type"] != "hallucination" || policy["enabled"] != true || policy["use_nli"] != false || policy["include_hallucination_details"] != true {
		t.Fatalf("published detector policy changed: %+v", plugin)
	}
}
