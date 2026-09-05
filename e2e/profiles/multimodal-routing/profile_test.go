package multimodalrouting

import (
	"encoding/json"
	"os"
	"testing"

	corev1 "k8s.io/api/core/v1"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
)

type environmentValues struct {
	Env      []corev1.EnvVar `json:"env"`
	ExtraEnv []corev1.EnvVar `json:"extraEnv"`
}

func TestProfileRenderPreservesRequiredDefaultEnvironment(t *testing.T) {
	chartDefaults := loadEnvironmentValues(t, "../../../deploy/helm/semantic-router/values.yaml")
	profileValues := loadEnvironmentValues(t, "values.yaml")

	// Helm replaces lists supplied by a values overlay. Profile-only entries must
	// therefore use extraEnv so the chart-owned runtime defaults remain intact.
	if len(profileValues.Env) != 0 {
		t.Fatal("multimodal profile must not replace the chart-owned env list; use extraEnv")
	}
	if len(profileValues.ExtraEnv) != 1 {
		t.Fatalf("profile extraEnv has %d entries, want 1", len(profileValues.ExtraEnv))
	}

	effective := append(append([]corev1.EnvVar{}, chartDefaults.Env...), profileValues.ExtraEnv...)
	environment := environmentByName(t, effective)

	requireLiteralEnvironment(t, environment, "HF_HOME", "/app/models/.cache/huggingface")
	requireSecretEnvironment(t, environment, "HF_TOKEN")
	requireSecretEnvironment(t, environment, "HUGGINGFACE_HUB_TOKEN")
	requireLiteralEnvironment(t, environment, "EMBEDDING_MODEL_OVERRIDE", "multimodal")
}

func loadEnvironmentValues(t *testing.T, path string) environmentValues {
	t.Helper()

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	jsonDocument, err := utilyaml.ToJSON(raw)
	if err != nil {
		t.Fatalf("convert %s to JSON: %v", path, err)
	}
	var values environmentValues
	if err := json.Unmarshal(jsonDocument, &values); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
	return values
}

func environmentByName(t *testing.T, entries []corev1.EnvVar) map[string]corev1.EnvVar {
	t.Helper()

	result := make(map[string]corev1.EnvVar, len(entries))
	for _, entry := range entries {
		if _, exists := result[entry.Name]; exists {
			t.Fatalf("rendered environment contains duplicate entry %q", entry.Name)
		}
		result[entry.Name] = entry
	}
	return result
}

func requireLiteralEnvironment(t *testing.T, environment map[string]corev1.EnvVar, name, want string) {
	t.Helper()

	entry, ok := environment[name]
	if !ok {
		t.Fatalf("rendered environment is missing %s", name)
	}
	if entry.Value != want || entry.ValueFrom != nil {
		t.Fatalf("%s = %#v, want literal value %q", name, entry, want)
	}
}

func requireSecretEnvironment(t *testing.T, environment map[string]corev1.EnvVar, name string) {
	t.Helper()

	entry, ok := environment[name]
	if !ok {
		t.Fatalf("rendered environment is missing %s", name)
	}
	if entry.ValueFrom == nil || entry.ValueFrom.SecretKeyRef == nil {
		t.Fatalf("%s does not reference the Hugging Face token secret: %#v", name, entry)
	}
	secret := entry.ValueFrom.SecretKeyRef
	if secret.Name != "hf-token-secret" || secret.Key != "token" {
		t.Fatalf("%s secret reference = %#v, want hf-token-secret/token", name, secret)
	}
	if secret.Optional == nil || !*secret.Optional {
		t.Fatalf("%s secret reference must remain optional", name)
	}
}
