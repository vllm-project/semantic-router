package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

func TestSystemOneAutoMetadataVisibilityCoversEveryState(t *testing.T) {
	identity := systemone.InferenceIdentity{
		ModelID: "kai-native", Revision: "revision-1", ModelSHA256: strings.Repeat("a", 64),
		Engine: "native", Profile: "exact", Numerics: "exact", Accelerator: "cpu",
	}
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.3
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
      decisions:
        - name: answer
          rules: {}
          modelRefs: [{model: kai}]
          algorithm:
            type: cascade
            budget: {deadline: 2s, max_calls: 1}
            quality:
              type: uncalibrated
              acceptance:
                rules: [{question_type: noul, field: top_probability, predicate: {gte: 0}}]
            stages: [{name: fast, kind: native, model: kai}]
`))
	if err != nil {
		t.Fatal(err)
	}
	classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	recipe, _ := cfg.RecipeByName("native")
	algorithm := recipe.Profile.Decisions[0].Algorithm
	algorithm.Quality = &config.NativeQualityConfig{Type: "calibrated"}
	// The evaluator fixture isolates metadata visibility; artifact admission and
	// statistical acceptance are covered by the systemone calibration tests.
	executor, err := systemone.NewExecutor(algorithm, func(_ *systemone.NativeRequest, candidate systemone.Candidate) (bool, error) {
		observed, valid := api.ResponseInferenceIdentity(candidate.Body)
		return valid && observed == identity, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	router := &OpenAIRouter{Config: cfg, RecipeClassifiers: classifiers, nativeExecutors: map[string]*systemone.Executor{config.RoutingDecisionKey("native", "answer"): executor}}
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
					t.Fatal("calibrated cascade did not request internal provenance")
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
