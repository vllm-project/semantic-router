package multimodalrouting

import (
	"encoding/json"
	"os"
	"testing"

	corev1 "k8s.io/api/core/v1"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
)

type profileValues struct {
	Env       []corev1.EnvVar            `json:"env"`
	ExtraEnv  []corev1.EnvVar            `json:"extraEnv"`
	Resources map[string]json.RawMessage `json:"resources"`
	Config    struct {
		Global struct {
			ModelCatalog struct {
				Embeddings struct {
					Semantic struct {
						ModelPath       string `json:"multimodal_model_path"`
						EmbeddingConfig struct {
							ModelType       string `json:"model_type"`
							TargetDimension *int   `json:"target_dimension"`
							TargetLayer     *int   `json:"target_layer"`
							TopK            *int   `json:"top_k"`
						} `json:"embedding_config"`
					} `json:"semantic"`
				} `json:"embeddings"`
			} `json:"model_catalog"`
		} `json:"global"`
	} `json:"config"`
}

type embeddingRule struct {
	Name      string  `json:"name"`
	Threshold float64 `json:"threshold"`
	// A per-rule prototype-scoring override (CRD spelling and pack
	// spelling) replaces the media policy the thresholds were calibrated
	// under; any presence, even an empty block, is a mismatch.
	PrototypeScoringCRD  json.RawMessage `json:"prototypeScoring"`
	PrototypeScoringPack json.RawMessage `json:"prototype_scoring"`
}

func (r embeddingRule) overridesPrototypeScoring() bool {
	return len(r.PrototypeScoringCRD) != 0 || len(r.PrototypeScoringPack) != 0
}

type intelligentRouteManifest struct {
	Spec struct {
		Signals struct {
			Embeddings []embeddingRule `json:"embeddings"`
		} `json:"signals"`
	} `json:"spec"`
}

// imageRoutingPack is the shape of config/fragments/signal/embedding/image-routing.yaml.
type imageRoutingPack struct {
	Routing struct {
		Signals struct {
			Embeddings []embeddingRule `json:"embeddings"`
		} `json:"signals"`
	} `json:"routing"`
}

func TestProfileRenderPreservesRequiredDefaultEnvironment(t *testing.T) {
	chartDefaults := loadProfileValues(t, "../../../deploy/helm/semantic-router/values.yaml")
	profile := loadProfileValues(t, "values.yaml")

	// Helm replaces lists supplied by a values overlay. Profile-only entries must
	// therefore use extraEnv so the chart-owned runtime defaults remain intact.
	if len(profile.Env) != 0 {
		t.Fatal("multimodal profile must not replace the chart-owned env list; use extraEnv")
	}
	if len(profile.ExtraEnv) != 0 {
		t.Fatalf("profile must select its deployment through canonical config, found extraEnv: %+v", profile.ExtraEnv)
	}

	effective := append(append([]corev1.EnvVar{}, chartDefaults.Env...), profile.ExtraEnv...)
	environment := environmentByName(t, effective)

	requireLiteralEnvironment(t, environment, "HF_HOME", "/app/models/.cache/huggingface")
	requireSecretEnvironment(t, environment, "HF_TOKEN")
	requireSecretEnvironment(t, environment, "HUGGINGFACE_HUB_TOKEN")
	if _, exists := environment["EMBEDDING_MODEL_OVERRIDE"]; exists {
		t.Fatal("profile must exercise canonical model selection without an environment override")
	}
	for key := range profile.Resources {
		if key != "limits" && key != "requests" {
			t.Fatalf("container resources contains unsupported chart workaround %q", key)
		}
	}

	// Helm deep-merges embedding_config maps. The profile must explicitly select
	// the final multimodal text-encoder layer; omitting it leaves the chart's
	// invalid mmBERT layer-22 default, zero is preserved on the canonical YAML round trip.
	embeddingConfig := profile.Config.Global.ModelCatalog.Embeddings.Semantic.EmbeddingConfig
	if embeddingConfig.ModelType != "multimodal" || profile.Config.Global.ModelCatalog.Embeddings.Semantic.ModelPath != "models/vela-1.0-omni-nano" {
		t.Fatal("profile must select the prepared Nano deployment through canonical config")
	}
	if embeddingConfig.TargetDimension == nil || *embeddingConfig.TargetDimension != 0 {
		t.Fatal("profile must request the complete manifest embedding dimension")
	}
	targetLayer := embeddingConfig.TargetLayer
	if targetLayer == nil || *targetLayer != 0 {
		t.Fatalf("multimodal profile target_layer = %v, want explicit 0", targetLayer)
	}

	// The image cases assert non-matches from the matched-embeddings header.
	// top_k limits that header to the highest-scoring rules, so anything but
	// unlimited (0) lets a second rule fire on a labelled negative unseen.
	if topK := embeddingConfig.TopK; topK == nil {
		t.Fatal("multimodal profile embedding top_k is unset, want explicit 0 (unlimited) so cross-rule negatives are observable")
	} else if *topK != 0 {
		t.Fatalf("multimodal profile embedding top_k = %d, want 0 (unlimited) so cross-rule negatives are observable", *topK)
	}
}

// The profile thresholds are not tuned to the three fixtures; they mirror the
// calibrated values shipped in the image-routing pack, so the E2E run is an
// acceptance test of what users deploy. tools/calibration/image-routing
// derives the pack values and CI gates them; this keeps the mirror honest.
func TestImageRulesMirrorTheShippedPack(t *testing.T) {
	raw, err := os.ReadFile("crds/intelligentroute.yaml")
	if err != nil {
		t.Fatal(err)
	}
	jsonDocument, err := utilyaml.ToJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	var manifest intelligentRouteManifest
	if err := json.Unmarshal(jsonDocument, &manifest); err != nil {
		t.Fatal(err)
	}

	packRaw, err := os.ReadFile("../../../config/fragments/signal/embedding/image-routing.yaml")
	if err != nil {
		t.Fatal(err)
	}
	packJSON, err := utilyaml.ToJSON(packRaw)
	if err != nil {
		t.Fatal(err)
	}
	var pack imageRoutingPack
	if err := json.Unmarshal(packJSON, &pack); err != nil {
		t.Fatal(err)
	}

	rules := manifest.Spec.Signals.Embeddings
	shipped := pack.Routing.Signals.Embeddings
	if len(rules) != 3 || len(shipped) != 3 {
		t.Fatalf("embedding rules: profile=%d pack=%d, want 3 and 3", len(rules), len(shipped))
	}
	for i, rule := range rules {
		if rule.Name == "" || rule.Threshold <= 0 {
			t.Fatalf("uncalibrated image rule: %+v", rule)
		}
		if rule.Name != shipped[i].Name || rule.Threshold != shipped[i].Threshold {
			t.Fatalf("profile rule %d = %+v does not mirror the shipped pack rule %+v", i, rule, shipped[i])
		}
		// The calibration gate refuses per-rule prototype_scoring overrides on
		// the pack; the CRD the E2E router deploys must not carry one either,
		// or the E2E scores under a blend the thresholds were not selected for.
		if rule.overridesPrototypeScoring() || shipped[i].overridesPrototypeScoring() {
			t.Fatalf("rule %q carries a prototype-scoring override; calibrated thresholds assume the image default", rule.Name)
		}
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
