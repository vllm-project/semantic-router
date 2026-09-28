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
	modules := section(cfg, "global", "model_catalog", "modules")
	if len(modules) != 1 {
		t.Fatalf("profile must override only the safety module: %+v", modules)
	}
	safety := section(modules, "safety")
	if len(safety) != 1 {
		t.Fatalf("profile must leave the Hazard module at its default: %+v", safety)
	}
	head := section(safety, "safety")
	if head["model_id"] != "models/Vela-1.0-Encoder-307M-Shield" || len(head) != 1 {
		t.Fatalf("Shield must be selected explicitly and inherit the module's other settings: %+v", head)
	}
	if _, bound := section(cfg, "routing")["model_bindings"]; bound {
		t.Fatal("profile must select Shield through the module, not a named binding")
	}
	if _, overridden := section(cfg, "global", "model_catalog")["deployments"]; overridden {
		t.Fatal("profile must not replace a default deployment")
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
		t.Fatal("safety rule must use the local module rather than an endpoint")
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
