package velahalu

import (
	"os"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestProfilePinsVela1HaluWithoutAuxiliaryClassifiers(t *testing.T) {
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
	catalog := section(cfg, "global", "model_catalog")
	module := section(catalog, "modules", "hallucination_mitigation")
	if module["enabled"] != true {
		t.Fatal("Halu detector must be enabled")
	}
	name, _ := section(catalog, "bindings")["hallucination"].(string)
	if section(catalog, "deployments", name)["artifact"] != "vllm-sr/Vela-1.0-Encoder-307M-Halu" {
		t.Fatal("profile must bind hallucination to Vela 1.0 Halu; the default detector is vela3-fast, which needs more memory than this profile's limit")
	}
	if section(module, "fact_check")["enabled"] != false {
		t.Fatal("profile unexpectedly enables fact_check")
	}
	if _, retired := module["explainer"]; retired {
		t.Fatal("the NLI explainer is retired; the profile must not configure it")
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
	rules, _ := policy["rules"].([]any)
	if plugin["type"] != "response_guard" || policy["on_unknown"] != "annotate" || len(rules) != 1 {
		t.Fatalf("published detector policy changed: %+v", plugin)
	}
	rule, _ := rules[0].(map[string]any)
	when := section(rule, "when")
	if rule["action"] != "annotate" || when["type"] != "hallucination" || when["name"] != "ungrounded_claims" {
		t.Fatalf("published detector policy changed: %+v", plugin)
	}
	if _, retired := policy["use_nli"]; retired {
		t.Fatal("use_nli is retired; the profile must not set it")
	}
}
