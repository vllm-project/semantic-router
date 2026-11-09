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
	for _, selector := range []string{"model_ref", "model_id"} {
		if value, exists := factCheck[selector]; exists {
			t.Fatalf("fact-check must follow the decision model instead of naming a model with %s: %v", selector, value)
		}
	}
	// 0.9997 on vela3-fast keeps Vela 1.0 FactCheck's operating point at 0.65.
	if factCheck["threshold"] != 0.9997 || factCheck["use_cpu"] != true {
		t.Fatalf("fact-check execution policy changed: %#v", factCheck)
	}
	if _, retired := factCheck["use_mmbert_32k"]; retired {
		t.Fatal("use_mmbert_32k is retired: the model runtime reads the architecture from the package")
	}
	detector := profileSection(t, module, "detector")
	if _, shorthand := detector["backend"]; shorthand || detector["include_explanation"] != false {
		t.Fatalf("the remote detector is a binding, not the retired endpoint shorthand: %#v", detector)
	}
	catalog := profileSection(t, values, "config", "global", "model_catalog")
	if _, external := catalog["external"]; external {
		t.Fatal("the remote detector is a deployment, not an external model")
	}
	deploymentName, _ := profileSection(t, catalog, "bindings")["hallucination"].(string)
	deployment := profileSection(t, catalog, "deployments", deploymentName)
	if deployment["provider"] != "openai_chat" || deployment["model"] != "KRLabsOrg/lettucedect-v2-qwen-2b" ||
		deployment["endpoint"] != "http://mock-hallucination-detector.default.svc.cluster.local:8000/v1" {
		t.Fatalf("the hallucination binding must name the remote detector served over chat completions: %#v", deployment)
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
