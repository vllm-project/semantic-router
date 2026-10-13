package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

// Routing depends on attachment presence, not fixture visual content.
type previewMediaRuntime struct {
	serving.Services
	mu       sync.Mutex
	requests []modelservice.Request
	card     modelservice.ModelCard
}

func (s *previewMediaRuntime) Card(context.Context, string) (modelservice.ModelCard, error) {
	return s.card, nil
}

func (s *previewMediaRuntime) CurrentCard(string) (modelservice.ModelCard, bool) { return s.card, true }

func (s *previewMediaRuntime) Decide(_ context.Context, _ string, req modelservice.Request) (modelservice.Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = append(s.requests, req)
	score := .1
	if len(req.Images)+len(req.Videos) > 0 {
		score = .9
	}
	answers := make(map[string]modelservice.Answer)
	for _, q := range req.Questions {
		answer := modelservice.Answer{Type: q.Type, Noul: score, InputCoverage: "complete"}
		if q.Type == "choice" {
			answer.Choice = "fast"
			if score > .5 {
				answer.Choice = "deep"
			}
			answer.Probabilities = map[string]float64{"fast": 1 - score, "deep": score}
		}
		answers[q.ID] = answer
	}
	return modelservice.Response{Answers: answers}, nil
}

const previewMediaConfig = `version: v0.3
providers:
  defaults: {model: fast}
  models:
    - name: fast
      backend_refs: [{name: fast, endpoint: "127.0.0.1:8000", protocol: http, weight: 1}]
    - name: deep
      backend_refs: [{name: deep, endpoint: "127.0.0.1:8001", protocol: http, weight: 1}]
routing:
  signals:
    decision:
      - name: debug
        predicate: {gte: 0.5}
        question: {type: noul, instructions: "Does the attachment show an error?"}
  decisions:
    - name: debug-route
      priority: 100
      rules: {operator: AND, conditions: [{type: decision, name: debug}]}
      modelRefs: [{model: fast}, {model: deep}]
      algorithm:
        type: decision
        decision:
          instructions: Which model should answer this request?
          candidates: {fast: Routine requests, deep: Debugging requests}
    - name: default-route
      priority: 1
      rules: {operator: AND, conditions: []}
      modelRefs: [{model: fast}]
global:
  model_catalog:
    system:
      decision_model: {deployment: primary}
    deployments:
      primary:
        provider: model_runtime
        endpoint: http://attached.invalid
`

func TestDecisionMediaPreviewMatchesLive(t *testing.T) {
	for _, tc := range []struct {
		name, kind, url, role string
		enabled, wantMedia    bool
	}{
		{"image", "image_url", "data:image/png;base64,iVBORw0KGgo=", "user", true, true},
		{"video", "video_url", "data:video/mp4;base64,AAAAFGZ0eXBpc29t", "user", true, true},
		{"default off", "image_url", "data:image/png;base64,iVBORw0KGgo=", "user", false, false},
		{"remote", "image_url", "https://example.invalid/image.png", "user", true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := config.ParseYAMLBytes([]byte(previewMediaConfig))
			require.NoError(t, err)
			if tc.enabled {
				d := cfg.ModelDeployments["primary"]
				d.Input.Media = &config.DecisionMediaConfig{Images: &config.DecisionMediaLimit{}, Videos: &config.DecisionMediaLimit{}}
				cfg.ModelDeployments["primary"] = d
			}
			runtime := &previewMediaRuntime{card: modelservice.ModelCard{Surfaces: []string{"decisions"}, QuestionTypes: []string{"noul", "choice"}, Modalities: []string{"text", "image", "video"}, ImageMaxBytes: 8000000, VideoMaxBytes: 32000000}}
			graph, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil, classification.RecipeRuntimeOptions{Runtime: serving.New(runtime, nil)})
			require.NoError(t, err)
			require.NoError(t, graph.InitializeRuntime())
			t.Cleanup(func() { require.NoError(t, graph.Close()) })
			router := &OpenAIRouter{Config: cfg, Classifier: graph.Default(), RecipeClassifiers: graph, decisionDecider: runtime, decisionCards: map[string]modelservice.ModelCard{"primary": runtime.card}}
			service := services.NewRecipeClassificationService(graph, cfg)
			service.SetEvalModelSelector(router)
			body := []byte(fmt.Sprintf(`{"model":"vllm-sr/auto","messages":[{"role":%q,"content":[{"type":%q,%q:{"url":%q}}]},{"role":"user","content":"Explain this"}]}`, tc.role, tc.kind, tc.kind, tc.url))
			neutral, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
			require.NoError(t, err)
			ctx := &RequestContext{TraceContext: t.Context(), SemanticRequest: &neutral, RequestModel: "vllm-sr/auto", SourceFormat: llmprotocol.OpenAIChatV1}
			live, _, _, model, err := router.performDecisionEvaluation("vllm-sr/auto", extractSignalConversationHistory(&neutral), ctx)
			require.NoError(t, err)
			wantRoute, wantModel := "default-route", "fast"
			if tc.wantMedia {
				wantRoute, wantModel = "debug-route", "deep"
			}
			require.Equal(t, wantRoute, live)
			require.Equal(t, wantModel, model)
			var req services.IntentRequest
			require.NoError(t, json.Unmarshal(body, &req))
			preview, err := service.ClassifyIntentForEval(t.Context(), req)
			require.NoError(t, err)
			require.NotNil(t, preview.DecisionResult)
			require.Equal(t, live, preview.DecisionResult.DecisionName)
			require.Equal(t, model, preview.SelectedModel)
			serialized, err := json.Marshal(preview)
			require.NoError(t, err)
			require.NotContains(t, string(serialized), tc.url)
		})
	}
}
