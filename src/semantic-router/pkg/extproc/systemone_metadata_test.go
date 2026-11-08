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
		Engine: "native", Dtype: "float32", Numerics: "exact", Accelerator: "cpu",
	}
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.5
listeners:
  - name: native
    port: 8801
    systemone: {models: [vllm-sr/auto]}
providers:
  models:
    - name: kai
      api_format: systemone
      provider_model_id: kai-native
      backend_refs: [{provider: systemone-compatible, endpoint: 'http://unused.invalid/v1'}]
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
  - {api: systemone, model_names: [vllm-sr/auto], recipe: native}
recipes:
  - name: default
  - name: native
    decisions:
      - name: answer
        rules: {}
        modelRefs: [{model: kai}]
        algorithm:
          type: escalate
          budget: {deadline: 2s, max_calls: 1}
          escalate:
            quality:
              type: uncalibrated
              accept:
                native: [{question_type: noul, field: top_probability, gte: 0}]
            stages: [{name: fast, model: kai}]
`))
	if err != nil {
		t.Fatal(err)
	}
	classifiers, err := classification.BuildRecipeClassifiers(cfg)
	if err != nil {
		t.Fatal(err)
	}
	recipe, _ := cfg.RecipeByName("native")
	algorithm := recipe.Profile.Decisions[0].Algorithm
	algorithm.Escalate.Quality = &config.EscalateQuality{Type: "calibrated"}
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
		"model": "kai-native", "meta": identity,
		"states": map[string]any{
			"first": map[string]any{"model": "kai-native", "meta": identity, "answers": answer},
			"meta":  map[string]any{"model": "kai-native", "meta": identity, "answers": answer},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, options := range []string{"", `,"options":{"return_meta":false}`, `,"options":{"return_meta":true}`} {
		t.Run(options, func(t *testing.T) {
			body := json.RawMessage(`{"model":"vllm-sr/auto","states":{"first":{"state":"first","questions":` + question + `},"meta":{"state":"second","questions":` + question + `}}` + options + `}`)
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
				Meta   json.RawMessage `json:"meta"`
				States map[string]struct {
					Meta    json.RawMessage `json:"meta"`
					Answers json.RawMessage `json:"answers"`
				} `json:"states"`
			}
			if err := json.Unmarshal(result, &envelope); err != nil {
				t.Fatal(err)
			}
			visible := strings.Contains(options, `"return_meta":true`)
			for name, state := range envelope.States {
				if (len(envelope.Meta) != 0) != visible || (len(state.Meta) != 0) != visible {
					t.Fatalf("metadata visibility differs across states (%s): %s", name, result)
				}
				if string(state.Answers) != string(answer) {
					t.Fatalf("metadata filtering changed native answers: %s", result)
				}
			}
			if len(envelope.States) != 2 {
				t.Fatalf("the batch's entries: %s", result)
			}
		})
	}
}
