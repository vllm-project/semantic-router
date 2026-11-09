package masking

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// The config loader rejects retired model-catalog fields outright, so one left
// in this profile aborts Router startup with runtime_config_load_failed and no
// masking case ever runs.
func TestProfileCarriesNoRetiredClassifierFields(t *testing.T) {
	raw, err := os.ReadFile(filepath.Base(valuesFile))
	if err != nil {
		t.Fatal(err)
	}
	if field := "use_mmbert_32k"; strings.Contains(string(raw), field) {
		t.Fatalf("%s is retired: the model runtime reads the architecture from the package", field)
	}

	var values map[string]any
	if err := yaml.Unmarshal(raw, &values); err != nil {
		t.Fatal(err)
	}
	catalog := profileSection(t, values, "config", "global", "model_catalog")
	pii := profileSection(t, catalog, "modules", "classifier", "pii")
	// The remote stub is what makes a detection attributable to it, so no
	// local selector may creep back in.
	for _, selector := range []string{"model_ref", "model_id", "backend"} {
		if value, set := pii[selector]; set && value != "" {
			t.Fatalf("pii.%s must stay unset so spans come only from the remote stub, got %v", selector, value)
		}
	}
	// A classify deployment answers the PII signal with token_spans.v1.
	name, _ := profileSection(t, catalog, "bindings")["pii"].(string)
	if deployment := profileSection(t, catalog, "deployments", name); deployment["provider"] != "classify" {
		t.Fatalf("the PII binding must name the remote classify stub: %#v", deployment)
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
