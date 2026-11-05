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
	classifier := profileSection(t, values, "config", "global", "model_catalog", "modules", "classifier")
	pii := profileSection(t, classifier, "pii")
	// The remote backend is what makes a detection attributable to the stub, so
	// no local selector may creep back in.
	for _, selector := range []string{"model_ref", "model_id"} {
		if value, set := pii[selector]; set && value != "" {
			t.Fatalf("pii.%s must stay empty so spans come only from the remote backend, got %v", selector, value)
		}
	}
	backend := profileSection(t, pii, "backend")
	if backend["contract"] != "token_spans.v1" {
		t.Fatalf("the PII backend must stay a token_spans.v1 binding: %#v", backend)
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
