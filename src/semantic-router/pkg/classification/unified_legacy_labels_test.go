package classification

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestLoadLegacyUnifiedLabels(t *testing.T) {
	root := t.TempDir()
	intentDir := filepath.Join(root, "intent")
	piiDir := filepath.Join(root, "pii")
	securityDir := filepath.Join(root, "security")
	mustMkdirAll(t, intentDir, piiDir, securityDir)

	writeMappingFile(t, filepath.Join(intentDir, "category_mapping.json"), `{
		"category_to_idx": {"coding": 0, "math": 1},
		"idx_to_category": {"0": "coding", "1": "math"}
	}`)
	writeMappingFile(t, filepath.Join(piiDir, "pii_type_mapping.json"), `{
		"label_to_idx": {"NO_PII": 0, "PERSON": 1},
		"idx_to_label": {"0": "NO_PII", "1": "PERSON"}
	}`)
	writeMappingFile(t, filepath.Join(securityDir, "jailbreak_type_mapping.json"), `{
		"label_to_id": {"safe": 0, "jailbreak": 1},
		"id_to_label": {"0": "safe", "1": "jailbreak"}
	}`)

	labels, err := loadLegacyUnifiedLabels(&ModelPaths{
		IntentClassifier:   intentDir,
		PIIClassifier:      piiDir,
		SecurityClassifier: securityDir,
	})
	if err != nil {
		t.Fatalf("loadLegacyUnifiedLabels() error = %v", err)
	}
	if want := []string{"coding", "math"}; !reflect.DeepEqual(labels.intent, want) {
		t.Fatalf("intent labels = %v, want %v", labels.intent, want)
	}
	if want := []string{"NO_PII", "PERSON"}; !reflect.DeepEqual(labels.pii, want) {
		t.Fatalf("pii labels = %v, want %v", labels.pii, want)
	}
	if want := []string{"safe", "jailbreak"}; !reflect.DeepEqual(labels.security, want) {
		t.Fatalf("security labels = %v, want %v", labels.security, want)
	}
}

func TestLoadLegacyUnifiedLabelsRejectsSparseMappings(t *testing.T) {
	root := t.TempDir()
	intentDir := filepath.Join(root, "intent")
	piiDir := filepath.Join(root, "pii")
	securityDir := filepath.Join(root, "security")
	mustMkdirAll(t, intentDir, piiDir, securityDir)

	writeMappingFile(t, filepath.Join(intentDir, "category_mapping.json"), `{
		"category_to_idx": {"coding": 0, "math": 2},
		"idx_to_category": {"0": "coding", "2": "math"}
	}`)
	writeMappingFile(t, filepath.Join(piiDir, "pii_type_mapping.json"), `{
		"label_to_idx": {"NO_PII": 0},
		"idx_to_label": {"0": "NO_PII"}
	}`)
	writeMappingFile(t, filepath.Join(securityDir, "jailbreak_type_mapping.json"), `{
		"label_to_idx": {"safe": 0},
		"idx_to_label": {"0": "safe"}
	}`)

	_, err := loadLegacyUnifiedLabels(&ModelPaths{
		IntentClassifier:   intentDir,
		PIIClassifier:      piiDir,
		SecurityClassifier: securityDir,
	})
	if err == nil || !strings.Contains(err.Error(), "missing label for index 1 in category mapping") {
		t.Fatalf("loadLegacyUnifiedLabels() error = %v, want sparse category mapping error", err)
	}
}

func mustMkdirAll(t *testing.T, dirs ...string) {
	t.Helper()
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("MkdirAll(%q): %v", dir, err)
		}
	}
}

func writeMappingFile(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile(%q): %v", path, err)
	}
}
