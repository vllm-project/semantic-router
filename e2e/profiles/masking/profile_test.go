package masking

import (
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
)

// The pii rule asks provider-mocker's System One fixture and nothing else, so
// every span the masking cases see is attributable to its markers.
func TestProfileAsksThePIIRuleOfTheSystemOneFixture(t *testing.T) {
	raw, err := os.ReadFile(filepath.Base(valuesFile))
	if err != nil {
		t.Fatal(err)
	}
	var values map[string]any
	if err := yaml.Unmarshal(raw, &values); err != nil {
		t.Fatal(err)
	}
	catalog := profileSection(t, values, "config", "global", "model_catalog")
	name, _ := profileSection(t, catalog, "bindings")["pii"].(string)
	deployment := profileSection(t, catalog, "deployments", name)
	if deployment["provider"] != "systemone" || deployment["model"] != "e2e/fake-systemone" ||
		deployment["endpoint"] != "http://provider-mocker.default.svc.cluster.local:8000/v1/systemone" {
		t.Fatalf("the PII binding must name provider-mocker's System One fixture: %#v", deployment)
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
