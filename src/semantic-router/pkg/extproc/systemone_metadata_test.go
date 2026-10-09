package extproc

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

func TestSystemOneAutoMetadataVisibilityCoversEveryState(t *testing.T) {
	identity := systemone.InferenceIdentity{
		ModelID: "kai-native", Revision: "revision-1", ModelSHA256: strings.Repeat("a", 64),
		Engine: "native", Profile: "exact", Numerics: "exact", Accelerator: "cpu",
	}
	policy, err := json.Marshal(map[string]any{
		"schema_version": "systemone-policy/v1", "feature_names": systemone.FeatureNames,
		"actions": map[string]systemone.PolicyActionBinding{"fast": {Model: "kai", Identity: identity}},
		"heads":   map[string]any{"fast": map[string]any{}},
	})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "policy.json")
	if writeErr := os.WriteFile(path, policy, 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	digest := sha256.Sum256(policy)
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`version: v0.3
listeners:
  - name: native
    port: 8801
    systemone: {models: [vllm-sr/auto]}
providers:
  models:
    - name: kai
      api_format: systemone
      provider_model_id: kai-native
      backend_refs: [{provider: systemone-compatible, base_url: http://unused.invalid/v1}]
entrypoints:
  - {api: systemone, model_names: [vllm-sr/auto], recipe: native}
recipes:
  - name: native
    routing:
      budget: {deadline: 2s, max_calls: 1}
      decisions:
        - name: answer
          rules: {}
          modelRefs: [{model: kai}]
          algorithm:
            type: policy
            policy: {source: %q, sha256: %s}
            quality:
              type: uncalibrated
              acceptance:
                rules: [{question_type: noul, field: top_probability, predicate: {gte: 0}}]
            stages: [{name: fast, kind: native, model: kai}]
`, path, hex.EncodeToString(digest[:]))))
	if err != nil {
		t.Fatal(err)
	}
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	executors, err := prepareNativeExecutors(cfg)
	if err != nil {
		t.Fatal(err)
	}
	router := &OpenAIRouter{Config: cfg, RecipeClassifiers: classifiers, nativeExecutors: executors}
	question := `{"meta":{"type":"noul","instructions":"Is this relevant?"}}`
	answer := json.RawMessage(`{"meta":{"type":"noul","noul":0.99,"extension":{"meta":"answer data"}}}`)
	response, err := json.Marshal(map[string]any{
		"model": "kai-native", "meta": identity, "answers": answer,
		"states": map[string]any{"meta": map[string]any{"model": "kai-native", "meta": identity, "answers": answer}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, options := range []string{"", `,"options":{"return_meta":false}`, `,"options":{"return_meta":true}`} {
		t.Run(options, func(t *testing.T) {
			body := json.RawMessage(`{"model":"vllm-sr/auto","state":"first","questions":` + question + `,"states":{"meta":{"state":"second","questions":` + question + `}}` + options + `}`)
			status, result, err := router.RouteSystemOne(t.Context(), "vllm-sr/auto", body, func(_ context.Context, model string, task json.RawMessage) (int, []byte, error) {
				if model != "kai" || !strings.Contains(string(task), `"return_meta":true`) {
					t.Fatal("policy did not request internal provenance")
				}
				return http.StatusOK, response, nil
			})
			if err != nil || status != http.StatusOK {
				t.Fatalf("status=%d error=%v", status, err)
			}
			var envelope struct {
				Meta    json.RawMessage `json:"meta"`
				Answers json.RawMessage `json:"answers"`
				States  map[string]struct {
					Meta    json.RawMessage `json:"meta"`
					Answers json.RawMessage `json:"answers"`
				} `json:"states"`
			}
			if err := json.Unmarshal(result, &envelope); err != nil {
				t.Fatal(err)
			}
			visible := strings.Contains(options, `"return_meta":true`)
			if (len(envelope.Meta) != 0) != visible || (len(envelope.States["meta"].Meta) != 0) != visible {
				t.Fatalf("metadata visibility differs across states: %s", result)
			}
			if string(envelope.Answers) != string(answer) || string(envelope.States["meta"].Answers) != string(answer) {
				t.Fatalf("metadata filtering changed native answers: %s", result)
			}
		})
	}
}
