package velashield

import (
	"os"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestProfileSelectsShieldForTheSafetyRuleOnly(t *testing.T) {
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
	if modules, overridden := catalog["modules"]; overridden {
		t.Fatalf("profile must leave the safety and Hazard modules at their defaults: %+v", modules)
	}
	deployments := section(catalog, "deployments")
	for _, name := range []string{"primary", "hazard", "vela-embedding", "omni-nano"} {
		if _, replaced := deployments[name]; replaced {
			t.Fatalf("profile must not replace the pre-declared %s deployment", name)
		}
	}
	if _, bound := section(cfg, "routing")["model_bindings"]; bound {
		t.Fatal("profile must select Shield on the safety rule, not with a binding")
	}
	routing := section(cfg, "routing")
	rules, ok := section(routing, "signals")["safety"].([]any)
	if !ok || len(rules) != 1 {
		t.Fatal("expected one safety rule")
	}
	rule := rules[0].(map[string]any)
	if rule["name"] != "unsafe-content" || rule["threshold"] != 0.5 {
		t.Fatalf("unexpected safety rule: %+v", rule)
	}
	if _, external := rule["model"]; external {
		t.Fatal("safety rule must name its deployment rather than an external model")
	}
	name, _ := rule["deployment"].(string)
	shield := section(deployments, name)
	if shield["provider"] != "model_runtime" || shield["artifact"] != "vllm-sr/Vela-1.0-Encoder-307M-Shield" {
		t.Fatalf("the safety rule must ask Vela 1.0 Shield on the model runtime: %+v", shield)
	}
	decisions, ok := routing["decisions"].([]any)
	if !ok || len(decisions) != 2 {
		t.Fatal("expected a safety decision and a default route")
	}
	decision := decisions[0].(map[string]any)
	conditions := section(decision, "rules")["conditions"].([]any)
	condition := conditions[0].(map[string]any)
	if decision["name"] != "handle-content-risk" || len(conditions) != 1 || condition["type"] != "safety" || condition["name"] != "unsafe-content" {
		t.Fatalf("safety decision does not consume the Shield rule: %+v", decision)
	}
	if section(decision, "rules")["on_unknown"] != "fail_request" {
		t.Fatal("a Shield inference error must fail the request instead of reading as safe")
	}
}
