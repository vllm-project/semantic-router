package hallucination

import (
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestProfileAsksFactCheckOfTheDecisionModelAndHallucinationOfTheFixture(t *testing.T) {
	raw, err := os.ReadFile(filepath.Base(valuesFile))
	if err != nil {
		t.Fatal(err)
	}
	var values map[string]any
	if err := yaml.Unmarshal(raw, &values); err != nil {
		t.Fatal(err)
	}
	recipes, _ := profileSection(t, values, "config")["recipes"].([]any)
	if len(recipes) == 0 {
		t.Fatal("the profile declares no recipe")
	}
	defaultRecipe, _ := recipes[0].(map[string]any)
	signals := profileSection(t, defaultRecipe, "signals")
	rules, _ := signals["fact_check"].([]any)
	var factCheck map[string]any
	for _, rule := range rules {
		if entry, _ := rule.(map[string]any); entry["name"] == "needs_fact_check" {
			factCheck = entry
		}
	}
	if factCheck == nil {
		t.Fatal("the response_guard reads the fact_check rule needs_fact_check")
	}
	if deployment, named := factCheck["deployment"]; named {
		t.Fatalf("fact-check must follow the decision model instead of naming deployment %v", deployment)
	}
	// 0.9997 on vela3-fast keeps Vela 1.0 FactCheck's operating point at 0.65.
	if predicate, _ := factCheck["predicate"].(map[string]any); predicate["gte"] != 0.9997 {
		t.Fatalf("fact-check operating point changed: %#v", factCheck)
	}
	catalog := profileSection(t, values, "config", "global", "model_catalog")
	deploymentName, _ := profileSection(t, catalog, "bindings")["hallucination"].(string)
	deployment := profileSection(t, catalog, "deployments", deploymentName)
	if deployment["provider"] != "systemone" || deployment["model"] != "e2e/fake-systemone" ||
		deployment["endpoint"] != "http://provider-mocker.default.svc.cluster.local:8000/v1/systemone" {
		t.Fatalf("the hallucination binding must name provider-mocker's System One fixture: %#v", deployment)
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
