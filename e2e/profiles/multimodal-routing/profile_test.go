package multimodalrouting

import (
	"encoding/json"
	"os"
	"testing"

	corev1 "k8s.io/api/core/v1"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
)

type profileValues struct {
	Env      []corev1.EnvVar `json:"env"`
	ExtraEnv []corev1.EnvVar `json:"extraEnv"`
	Config   struct {
		Global struct {
			ModelCatalog struct {
				Embeddings struct {
					Semantic struct {
						EmbeddingConfig struct {
							TargetLayer *int `json:"target_layer"`
						} `json:"embedding_config"`
					} `json:"semantic"`
				} `json:"embeddings"`
			} `json:"model_catalog"`
		} `json:"global"`
	} `json:"config"`
}

func TestProfileRenderPreservesRequiredDefaultEnvironment(t *testing.T) {
	chartDefaults := loadProfileValues(t, "../../../deploy/helm/semantic-router/values.yaml")
	profile := loadProfileValues(t, "values.yaml")

	// Helm replaces lists supplied by a values overlay. Profile-only entries must
	// therefore use extraEnv so the chart-owned runtime defaults remain intact.
	if len(profile.Env) != 0 {
		t.Fatal("multimodal profile must not replace the chart-owned env list; use extraEnv")
	}
	if len(profile.ExtraEnv) != 1 {
		t.Fatalf("profile extraEnv has %d entries, want 1", len(profile.ExtraEnv))
	}

	effective := append(append([]corev1.EnvVar{}, chartDefaults.Env...), profile.ExtraEnv...)
	environment := environmentByName(t, effective)

	requireLiteralEnvironment(t, environment, "HF_HOME", "/app/models/.cache/huggingface")
	requireSecretEnvironment(t, environment, "HF_TOKEN")
	requireSecretEnvironment(t, environment, "HUGGINGFACE_HUB_TOKEN")
	requireLiteralEnvironment(t, environment, "EMBEDDING_MODEL_OVERRIDE", "multimodal")

	// Helm deep-merges embedding_config maps. The profile must explicitly select
	// the final multimodal text-encoder layer; omitting it leaves the chart's
	// invalid mmBERT layer-22 default, while zero is lost on canonical marshal.
	embeddingConfig := profile.Config.Global.ModelCatalog.Embeddings.Semantic.EmbeddingConfig
	targetLayer := embeddingConfig.TargetLayer
	if targetLayer == nil || *targetLayer != 6 {
		t.Fatalf("multimodal profile target_layer = %v, want explicit 6", targetLayer)
	}
}

func loadProfileValues(t *testing.T, path string) profileValues {
	t.Helper()

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	jsonDocument, err := utilyaml.ToJSON(raw)
	if err != nil {
		t.Fatalf("convert %s to JSON: %v", path, err)
	}
	var values profileValues
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
