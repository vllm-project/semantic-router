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
	// 0.86 on Vela 2.0 0.3B keeps Vela 1.0 FactCheck's operating point at 0.65.
	if factCheck["threshold"] != 0.86 || factCheck["use_cpu"] != true {
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
	binding := profileSection(t, catalog, "bindings", "hallucination_detector")
	if binding["adapter"] != "http_chat" || binding["contract"] != "token_spans.v1" {
		t.Fatalf("the remote detector must be an http_chat token_spans.v1 binding: %#v", binding)
	}
	deploymentName, _ := binding["deployment"].(string)
	deployment := profileSection(t, catalog, "deployments", deploymentName)
	if deployment["provider"] != "http" || deployment["external_model"] != "hallucination-detector" {
		t.Fatalf("the detector deployment must serve the external model over http: %#v", deployment)
	}
	external, _ := catalog["external"].([]any)
	if len(external) != 1 {
		t.Fatalf("profile must declare exactly its remote detector as an external model: %#v", external)
	}
	model, _ := external[0].(map[string]any)
	endpoint, _ := model["llm_endpoint"].(map[string]any)
	if endpoint["address"] != "mock-hallucination-detector.default.svc.cluster.local" || endpoint["port"] != 8000 ||
		model["llm_model_name"] != "KRLabsOrg/lettucedect-v2-qwen-2b" {
		t.Fatalf("remote detector contract changed: %#v", model)
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
