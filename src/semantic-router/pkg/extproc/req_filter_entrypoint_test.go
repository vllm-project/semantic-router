package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const entrypointTestConfigYAML = `
version: v0.3
routing:
  modelCards:
    - name: model-a
      description: default tier
    - name: model-b
      description: privacy tier
  signals:
    keywords:
      - name: urgent_keywords
        operator: OR
        keywords: ["urgent"]
  decisions:
    - name: default_route
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: urgent_keywords
      modelRefs:
        - model: model-a
          use_reasoning: false
recipes:
  - name: privacy
    routing:
      signals:
        keywords:
          - name: pii_keywords
            operator: OR
            keywords: ["ssn"]
      decisions:
        - name: privacy_route
          rules:
            operator: AND
            conditions:
              - type: keyword
                name: pii_keywords
          modelRefs:
            - model: model-b
              use_reasoning: false
entrypoints:
  - model_names: ["vllm-sr/privacy"]
    recipe: privacy
  - model_names: ["vllm-sr/default-alias"]
    recipe: default
providers:
  defaults:
    default_model: model-a
  models:
    - name: model-a
      backend_refs:
        - endpoint: 127.0.0.1:8000
    - name: model-b
      backend_refs:
        - endpoint: 127.0.0.1:8001
`

func newEntrypointTestRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(entrypointTestConfigYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	return &OpenAIRouter{Config: cfg}
}

func TestResolveEntrypointForRequest(t *testing.T) {
	router := newEntrypointTestRouter(t)

	ctx := &RequestContext{}
	router.resolveEntrypointForRequest("vllm-sr/privacy", ctx)
	if ctx.EntrypointRecipe == nil || ctx.EntrypointRecipe.Name != "privacy" {
		t.Fatalf("expected the privacy recipe to be resolved, got %+v", ctx.EntrypointRecipe)
	}

	ctx = &RequestContext{}
	router.resolveEntrypointForRequest("model-a", ctx)
	if ctx.EntrypointRecipe != nil {
		t.Fatalf("expected a plain model name to resolve no recipe, got %+v", ctx.EntrypointRecipe)
	}
}

func TestRequestModelActsAsAuto(t *testing.T) {
	router := newEntrypointTestRouter(t)

	cases := []struct {
		model string
		want  bool
	}{
		{model: config.DefaultVSRAutoModelName, want: true},
		{model: "vllm-sr/privacy", want: true},
		{model: "vllm-sr/default-alias", want: true},
		{model: "model-a", want: false},
		{model: "unknown-model", want: false},
	}
	for _, testCase := range cases {
		if got := router.requestModelActsAsAuto(testCase.model); got != testCase.want {
			t.Fatalf("requestModelActsAsAuto(%q) = %v, want %v", testCase.model, got, testCase.want)
		}
	}
}

func TestDecisionCandidatesForRequest(t *testing.T) {
	router := newEntrypointTestRouter(t)

	ctx := &RequestContext{}
	router.resolveEntrypointForRequest("vllm-sr/privacy", ctx)
	candidates := router.decisionCandidatesForRequest("vllm-sr/privacy", ctx)
	if len(candidates) != 1 || candidates[0].Name != "privacy_route" {
		t.Fatalf("expected the privacy recipe's decisions as candidates, got %+v", candidates)
	}

	// An entrypoint alias of the default recipe keeps the engine-default
	// candidate path (nil), which evaluates the flat default decisions.
	ctx = &RequestContext{}
	router.resolveEntrypointForRequest("vllm-sr/default-alias", ctx)
	if ctx.EntrypointRecipe == nil || ctx.EntrypointRecipe.Name != config.DefaultRecipeName {
		t.Fatalf("expected the default recipe to be resolved, got %+v", ctx.EntrypointRecipe)
	}
	if candidates := router.decisionCandidatesForRequest("vllm-sr/default-alias", ctx); candidates != nil {
		t.Fatalf("expected nil candidates for a default-recipe alias, got %+v", candidates)
	}

	ctx = &RequestContext{}
	if candidates := router.decisionCandidatesForRequest(config.DefaultVSRAutoModelName, ctx); candidates != nil {
		t.Fatalf("expected nil candidates for the auto model, got %+v", candidates)
	}
}

// newEntrypointFlowRouter builds a router with a real classifier so decision
// evaluation runs the full signal → decision → model-selection chain. The
// fixture only uses keyword signals, which need no local model artifacts.
func newEntrypointFlowRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(entrypointTestConfigYAML))
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("failed to build classifier: %v", err)
	}
	return &OpenAIRouter{Config: cfg, Classifier: classifier}
}

func TestPerformDecisionEvaluationSelectsRecipeByEntrypoint(t *testing.T) {
	router := newEntrypointFlowRouter(t)

	cases := []struct {
		name         string
		model        string
		message      string
		wantDecision string
		wantModel    string
	}{
		{
			name:         "privacy entrypoint routes through the privacy recipe",
			model:        "vllm-sr/privacy",
			message:      "my ssn is exposed",
			wantDecision: "privacy_route",
			wantModel:    "model-b",
		},
		{
			// The pii signal matches globally, but the default recipe's
			// decisions do not reference it: the request falls back to the
			// default model instead of leaking into another recipe's routes.
			name:         "auto model ignores other recipes' decisions",
			model:        config.DefaultVSRAutoModelName,
			message:      "my ssn is exposed",
			wantDecision: "",
			wantModel:    "model-a",
		},
		{
			name:         "auto model matches the default recipe decision",
			model:        config.DefaultVSRAutoModelName,
			message:      "this is urgent",
			wantDecision: "default_route",
			wantModel:    "model-a",
		},
		{
			name:         "privacy entrypoint falls back to the default model when its recipe matches nothing",
			model:        "vllm-sr/privacy",
			message:      "this is urgent",
			wantDecision: "",
			wantModel:    "model-a",
		},
		{
			name:         "default recipe alias behaves like the auto model",
			model:        "vllm-sr/default-alias",
			message:      "this is urgent",
			wantDecision: "default_route",
			wantModel:    "model-a",
		},
		{
			name:         "explicit model preserves the client selection",
			model:        "model-a",
			message:      "this is urgent",
			wantDecision: "default_route",
			wantModel:    "",
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			ctx := &RequestContext{
				TraceContext: context.Background(),
				Headers:      map[string]string{},
			}
			router.resolveEntrypointForRequest(testCase.model, ctx)

			decisionName, _, _, selectedModel, err := router.performDecisionEvaluation(
				testCase.model,
				signalConversationHistory{currentUserMessage: testCase.message},
				ctx,
			)
			if err != nil {
				t.Fatalf("performDecisionEvaluation failed: %v", err)
			}
			if decisionName != testCase.wantDecision {
				t.Fatalf("expected decision %q, got %q", testCase.wantDecision, decisionName)
			}
			if selectedModel != testCase.wantModel {
				t.Fatalf("expected selected model %q, got %q", testCase.wantModel, selectedModel)
			}
			if testCase.wantDecision != "" && ctx.VSRSelectedDecision != nil && ctx.VSRSelectedDecision.Name != testCase.wantDecision {
				t.Fatalf("expected ctx.VSRSelectedDecision %q, got %q", testCase.wantDecision, ctx.VSRSelectedDecision.Name)
			}
		})
	}
}

func TestModelsListingIncludesEntrypointNames(t *testing.T) {
	router := newEntrypointTestRouter(t)

	response, err := router.handleModelsRequest("/v1/models")
	if err != nil {
		t.Fatalf("handleModelsRequest failed: %v", err)
	}
	immediateResp := response.GetImmediateResponse()
	if immediateResp == nil {
		t.Fatal("expected an immediate response")
	}

	var modelList OpenAIModelList
	if err := json.Unmarshal(immediateResp.Body, &modelList); err != nil {
		t.Fatalf("failed to parse response body: %v", err)
	}

	descriptionByID := make(map[string]string, len(modelList.Data))
	for _, model := range modelList.Data {
		descriptionByID[model.ID] = model.Description
	}
	if _, ok := descriptionByID["vllm-sr/privacy"]; !ok {
		t.Fatalf("expected vllm-sr/privacy in the model list, got %+v", descriptionByID)
	}
	if _, ok := descriptionByID["vllm-sr/default-alias"]; !ok {
		t.Fatalf("expected vllm-sr/default-alias in the model list, got %+v", descriptionByID)
	}
	if got := descriptionByID["vllm-sr/default-alias"]; got != "Entrypoint for the default routing recipe" {
		t.Fatalf("expected the generic entrypoint description for the default alias, got %q", got)
	}
}
